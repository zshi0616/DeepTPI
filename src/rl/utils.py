import numpy as np 
import copy
import torch
import os

from numpy.lib.shape_base import apply_along_axis

# Parameter
# "INPUT": 0, "AND": 1, "NAND": 2, "OR": 3, "NOR": 4, "NOT": 5, "XOR": 6
TOT_GATE_TYPE = 4

def get_gate_type(gate_type, gate_to_idx):
    for symbol in gate_to_idx.keys():
        if gate_to_idx[symbol] == gate_type:
            return symbol
    raise('Unsupport Gate Type: ', gate_type)

def prob_logic(gate_type, signals, gate_to_idx):
    symbol = get_gate_type(gate_type, gate_to_idx)
    one = 0.0
    zero = 0.0

    if symbol == 'AND':  # AND
        mul = 1.0
        for s in signals:
            mul = mul * s[1]
        one = mul
        zero = 1.0 - mul

    elif symbol == 'NAND':  # NAND
        mul = 1.0
        for s in signals:
            mul = mul * s[1]
        zero = mul
        one = 1.0 - mul

    elif symbol == 'OR':  # OR
        mul = 1.0
        for s in signals:
            mul = mul * s[0]
        zero = mul
        one = 1.0 - mul

    elif symbol == 'NOR':  # NOR
        mul = 1.0
        for s in signals:
            mul = mul * s[0]
        one = mul
        zero = 1.0 - mul

    elif symbol == 'NOT':  # NOT
        for s in signals:
            one = s[0]
            zero = s[1]
    
    elif symbol == 'BUFF': # BUFF
        for s in signals:
            one = s[1]
            zero = s[0]

    elif symbol == 'XOR':  # XOR
        mul0 = 1.0
        mul1 = 1.0
        for s in signals:
            mul0 = mul0 * s[0]
        for s in signals:
            mul1 = mul1 * s[1]

        zero = mul0 + mul1
        one = 1.0 - zero

    else:
        raise('Unsupport Gate Type: {:}'.format(gate_type))

    return zero, one

def obs_prob(gate_type, c1, c0, r, y, input_signals, gate_to_idx):
    symbol = get_gate_type(gate_type, gate_to_idx)
    if symbol == 'AND' or symbol == 'NAND':
        obs = y[r]
        for s in input_signals:
            for s1 in input_signals:
                if s != s1:
                    obs = obs * c1[s1]
            if obs < y[s] or y[s] == -1:
                y[s] = obs

    elif symbol == 'OR' or symbol == 'NOR':
        obs = y[r]
        for s in input_signals:
            for s1 in input_signals:
                if s != s1:
                    obs = obs * c0[s1]
            if obs < y[s] or y[s] == -1:
                y[s] = obs

    elif symbol == 'NOT' or symbol == 'BUFF':
        obs = y[r]
        for s in input_signals:
            if obs < y[s] or y[s] == -1:
                y[s] = obs

    elif symbol == 'XOR':
        if len(input_signals) != 2:
            print('Not support non 2-input XOR Gate')
            raise
        # computing for a node
        obs = y[r]
        s = input_signals[1]
        if c1[s] > c0[s]:
            obs = obs * c1[s]
        else:
            obs = obs * c0[s]
        y[input_signals[0]] = obs

        # computing for b node
        obs = y[r]
        s = input_signals[0]
        if c1[s] > c0[s]:
            obs = obs * c1[s]
        else:
            obs = obs * c0[s]
        y[input_signals[1]] = obs

    return y

def one_hot(num, size):
    res = [0] * size
    res[num] = 1
    return res

class Gate:
    def __init__(self, gate_type):
        self.i = []
        self.o = []
        self.gate_level = -1
        self.gate_type = gate_type

class Netlist(object):
    def __init__(self, graph, args) -> None:
        super().__init__()
        self.allGatesVec = []
        self.c1 = graph.c1
        self.c0 = graph.c0
        self.co = graph.co
        self.PI_list = []
        self.PO_list = []
        self.max_level = -1
        self.args = args
        self.tp_record = []
        self.name = graph.name

        # Gate
        for idx in range(len(graph['x'])):
            gate_inst = Gate(int(graph['gate'][idx]))
            gate_inst.gate_level = int(graph['forward_level'][idx])
            if gate_inst.gate_level > self.max_level:
                self.max_level = gate_inst.gate_level
            self.allGatesVec.append(gate_inst)
        self.mask = [1] * len(self.allGatesVec)
        self.init_size = len(self.allGatesVec)
        
        # Edge Connection
        for edge_idx in range(len(graph['edge_index'][-1])):
            src_idx = int(graph['edge_index'][0][edge_idx])
            dst_idx = int(graph['edge_index'][1][edge_idx])
            self.allGatesVec[src_idx].o.append(dst_idx)
            self.allGatesVec[dst_idx].i.append(src_idx)

        # PI_list
        for idx, gate in enumerate(self.allGatesVec):
            if gate.gate_level == 0:
                self.PI_list.append(idx)
                self.mask[idx] = 0
        # PO_list
        for idx, gate in enumerate(self.allGatesVec):
            if len(gate.o) == 0:
                self.PO_list.append(idx)
                self.mask[idx] = 0

    def insert_cp(self, cp_pos, cp_type):
        '''
        Insert Control Point
        '''
        cp_en_gate = Gate(0)
        cp_en_gate_idx = len(self.allGatesVec)
        cp_gate = Gate(cp_type)
        cp_gate_idx = len(self.allGatesVec) + 1
        cp_en_gate.gate_level = 0
        cp_gate.gate_level = self.allGatesVec[cp_pos].gate_level + 1

        # connection 
        cp_gate.i = [cp_en_gate_idx, cp_pos]
        cp_gate.o = copy.deepcopy(self.allGatesVec[cp_pos].o)
        cp_en_gate.o = [cp_gate_idx]
        self.allGatesVec[cp_pos].o = [cp_gate_idx]
        for idx in cp_gate.o:
            for k, fanin_idx in enumerate(self.allGatesVec[idx].i):
                if fanin_idx == cp_pos:
                    self.allGatesVec[idx].i[k] = cp_gate_idx
        
        # Add to allGatesVec
        self.allGatesVec.append(cp_en_gate)
        self.allGatesVec.append(cp_gate)

        # Mask 
        self.mask.append(0)
        self.mask.append(0)
        self.mask[cp_pos] = 0
        
        # Update
        self.update_circuit(cp_gate)
        self.update_graph()

    def insert_cp_aig(self, tp_pos, tp_type, gate2index):
        '''
        Insert Control Point
        '''
        cp_en_gate = Gate(0)
        cp_en_gate_idx = len(self.allGatesVec)
        cp_en_gate.gate_level = 0
        self.allGatesVec.append(cp_en_gate)
        self.mask.append(0)
        self.mask[tp_pos] = 0

        if tp_type == 'AND':
            cp_gate = Gate(gate2index['AND'])
            cp_gate_idx = len(self.allGatesVec)
            cp_gate.gate_level = self.allGatesVec[tp_pos].gate_level + 1
            # connection 
            cp_gate.i = [cp_en_gate_idx, tp_pos]
            cp_gate.o = copy.deepcopy(self.allGatesVec[tp_pos].o)
            cp_en_gate.o = [cp_gate_idx]
            self.allGatesVec[tp_pos].o = [cp_gate_idx]
            for idx in cp_gate.o:
                for k, fanin_idx in enumerate(self.allGatesVec[idx].i):
                    if fanin_idx == tp_pos:
                        self.allGatesVec[idx].i[k] = cp_gate_idx
            # Add to allGatesVec
            self.allGatesVec.append(cp_gate)
            # Mask 
            self.mask.append(0)
            # Update
            self.update_circuit(cp_gate)
        elif tp_type == 'OR':
            cp_gate = Gate(gate2index['NOT'])
            cp_gate_idx = len(self.allGatesVec) + 0
            cp_gate.gate_level = self.allGatesVec[tp_pos].gate_level + 3
            cp_gate_and = Gate(gate2index['AND'])
            cp_gate_and_idx = len(self.allGatesVec) + 1
            cp_gate_and.gate_level = self.allGatesVec[tp_pos].gate_level + 2
            cp_gate_not_1 = Gate(gate2index['NOT'])
            cp_gate_not_1_idx = len(self.allGatesVec) + 2
            cp_gate_not_1.gate_level = self.allGatesVec[tp_pos].gate_level + 1
            cp_gate_not_2 = Gate(gate2index['NOT'])
            cp_gate_not_2_idx = len(self.allGatesVec) + 3
            cp_gate_not_2.gate_level = self.allGatesVec[tp_pos].gate_level + 1
            # Connection
            cp_gate.i = [cp_gate_and_idx]
            cp_gate.o = copy.deepcopy(self.allGatesVec[tp_pos].o)
            cp_gate_and.i = [cp_gate_not_1_idx, cp_gate_not_2_idx]
            cp_gate_and.o = [cp_gate_idx]
            cp_gate_not_1.i = [cp_en_gate_idx]
            cp_gate_not_1.o = [cp_gate_and_idx]
            cp_gate_not_2.i = [tp_pos]
            cp_gate_not_2.o = [cp_gate_and_idx]
            cp_en_gate.o = [cp_gate_not_1_idx]
            self.allGatesVec[tp_pos].o = [cp_gate_not_2_idx]
            for idx in cp_gate.o:
                for k, fanin_idx in enumerate(self.allGatesVec[idx].i):
                    if fanin_idx == tp_pos:
                        self.allGatesVec[idx].i[k] = cp_gate_idx
            # Add to allGatesVec
            self.allGatesVec.append(cp_gate)
            self.allGatesVec.append(cp_gate_and)
            self.allGatesVec.append(cp_gate_not_1)
            self.allGatesVec.append(cp_gate_not_2)
            # Mask 
            self.mask.append(0)
            self.mask.append(0)
            self.mask.append(0)
            self.mask.append(0)
            # Update
            self.update_circuit(cp_gate)
        elif tp_type == 'OP':
            op_gate = Gate(gate2index['BUFF'])
            op_gate_idx = len(self.allGatesVec)
            op_gate.gate_level = self.allGatesVec[tp_pos].gate_level + 1
            # Connection
            op_gate.i = [tp_pos]
            self.allGatesVec[tp_pos].o.append(op_gate_idx)
            # Add to allGatesVec
            self.allGatesVec.append(op_gate)
            self.mask.append(0)
            # Update
            self.update_circuit(op_gate)
        
    def update_circuit(self, cp_gate):
        '''
        Update the list not used in network
        Need Update:
            Logic level 
            C0, C1, CO
        Input: Control Point Instance
        '''
        if self.args.RL_model != 'deepgate' and self.args.ftpt == 'no':
            return

        # Update logic level
        bfs_q = []
        for idx in cp_gate.o:
            if self.allGatesVec[idx].gate_level <= cp_gate.gate_level:
                bfs_q.append(idx)
                self.allGatesVec[idx].gate_level = cp_gate.gate_level + 1
        while len(bfs_q) > 0:
            idx = bfs_q[-1]
            bfs_q.pop()
            if self.allGatesVec[idx].gate_level > self.max_level:
                self.max_level = self.allGatesVec[idx].gate_level
            for fanout_idx in self.allGatesVec[idx].o:
                if self.allGatesVec[fanout_idx].gate_level <= self.allGatesVec[idx].gate_level:
                    bfs_q.append(fanout_idx)
                    self.allGatesVec[fanout_idx].gate_level = self.allGatesVec[idx].gate_level + 1

        # Update COP (C1, C0, CO)
        # Level list 
        for idx, gate in enumerate(self.allGatesVec):
            if gate.gate_level > self.max_level:
                self.max_level = gate.gate_level
        level_list = []
        for level in range(self.max_level+1):
            level_list.append([])
        for idx, gate in enumerate(self.allGatesVec):
            level_list[gate.gate_level].append(idx)
        
        # Update Controlability
        self.c1 = [0] * len(self.allGatesVec)
        self.c0 = [0] * len(self.allGatesVec)
        for idx in self.PI_list:
            self.c1[idx] = 0.5
            self.c0[idx] = 0.5

        for level in range(1, len(level_list), 1):
            for idx in level_list[level]:
                source_node = self.allGatesVec[idx].i
                source_signals = []
                for fanin_idx in source_node:
                    source_signals.append([self.c0[fanin_idx], self.c1[fanin_idx]])
                if len(source_signals) > 0:
                    zero, one = prob_logic(self.allGatesVec[idx].gate_type, source_signals, self.args.gate_to_index)
                    self.c1[idx] = one
                    self.c0[idx] = zero
        
        # Update Observability 
        self.co = [-1] * len(self.allGatesVec)
        for idx, gate in enumerate(self.allGatesVec):
            if len(gate.o) == 0:
                self.co[idx] = 1
        
        for level in range(len(level_list) - 1, -1, -1):
            for idx in level_list[level]:
                source_signals = self.allGatesVec[idx].i
                if len(source_signals) > 0:
                    self.co = obs_prob(self.allGatesVec[idx].gate_type, self.c1, self.c0, idx, 
                                       self.co, source_signals, self.args.gate_to_index)

    def save_fault_list(self, args, faultfile=None):
        if faultfile == None:
            filedir = args.save_dir + '/output'
            if not os.path.exists(filedir):
                os.makedirs(filedir)
            f = open(filedir + '/' + self.name + '_fault_list.txt', 'w')
        else:
            f = open(faultfile, 'w')

        for idx in range(len(self.allGatesVec)):
            f.write('N{:} /0\n'.format(idx))
            f.write('N{:} /1\n'.format(idx))
        f.write('\n')
        f.close()

    def save_bench(self, args, benchfile=None):
        PI_list = []
        PO_list = []
        level_list = []
        for level in range(self.max_level+1):
            level_list.append([])
        for idx, gate in enumerate(self.allGatesVec):
            if len(gate.o) == 0:
                PO_list.append(idx)
            if len(gate.i) == 0:
                PI_list.append(idx)
            level_list[gate.gate_level].append(idx)
        
        # Head
        if benchfile == None:
            filedir = args.save_dir + '/output'
            if not os.path.exists(filedir):
                os.makedirs(filedir)
            f = open(filedir + '/' + self.name + '.bench', 'w')
        else:
            f = open(benchfile, 'w')
        f.write("# {:} inputs \n".format(len(PI_list)))
        f.write("# {:} outputs \n".format(len(PO_list)))
        f.write('\n')

        # PI and PO
        for idx in PI_list:
            f.write('INPUT(N{:})\n'.format(idx))
        f.write('\n')
        for idx in PO_list:
            f.write('OUTPUT(N{:})\n'.format(idx))
        f.write('\n')

        # Gates
        for level in range(len(level_list)):
            for idx in level_list[level]:
                if idx not in PI_list:
                    gate = self.allGatesVec[idx]
                    gate_type = 'None'
                    for key in args.gate_to_index.keys():
                        if args.gate_to_index[key] == gate.gate_type:
                            gate_type = key
                            break
                    newline = 'N{:} = {}('.format(idx, gate_type)
                    for k, fanin_idx in enumerate(gate.i):
                        if k == len(gate.i)-1:
                            newline += 'N{:})\n'.format(fanin_idx)
                        else:
                            newline += 'N{:}, '.format(fanin_idx)
                    f.write(newline)
        f.write('\n')
        f.close()

    def save_verilog(self, args, vfile=None):
        PI_list = []
        PO_list = []
        level_list = []
        for level in range(self.max_level+1):
            level_list.append([])
        for idx, gate in enumerate(self.allGatesVec):
            if len(gate.o) == 0:
                PO_list.append(idx)
            if len(gate.i) == 0:
                PI_list.append(idx)
            level_list[gate.gate_level].append(idx)
        
        # Head
        if vfile == None:
            filedir = args.save_dir + '/output'
            if not os.path.exists(filedir):
                os.makedirs(filedir)
            f = open(filedir + '/' + self.name + '.v', 'w')
        else:
            f = open(vfile, 'w')

        newline = 'module top ('
        for idx in PI_list:
            newline += 'N{:}, '.format(idx)
        for k, idx in enumerate(PO_list):
            if k == len(PO_list) - 1:
                newline += 'N{:}); \n'.format(idx)
            else:
                newline += 'N{:}, '.format(idx)
        f.write(newline)
        f.write('\n')        

        # PI and PO
        for idx in PI_list:
            f.write('input N{:};\n'.format(idx))
        f.write('\n')
        for idx in PO_list:
            f.write('output N{:};\n'.format(idx))
        f.write('\n')

        # Gates
        for level in range(len(level_list)):
            for idx in level_list[level]:
                if idx not in PI_list:
                    gate = self.allGatesVec[idx]
                    newline = 'assign N{:} = ('.format(idx)
                    if args.gate_to_index['AND'] == gate.gate_type:
                        for k, fanin_idx in enumerate(gate.i):
                            if k == len(gate.i)-1:
                                newline += 'N{:});\n'.format(fanin_idx)
                            else:
                                newline += 'N{:} & '.format(fanin_idx)
                    elif args.gate_to_index['NOT'] == gate.gate_type:
                        newline += '~N{:});\n'.format(gate.i[0])
                    elif 'BUFF' in args.gate_to_index and args.gate_to_index['BUFF'] == gate.gate_type:
                        newline += 'N{:});\n'.format(gate.i[0])
                    else:
                        raise('Unsupport')
                    f.write(newline)
        f.write('\n')
        f.write('endmodule \n')
        f.close()

    def save_tp_list(self, args, listfile=None): 
        if listfile == None:
            filedir = args.save_dir + '/output'
            if not os.path.exists(filedir):
                os.makedirs(filedir)
            f = open(filedir + '/' + self.name + '.txt', 'w')
        else:
            f = open(listfile, 'w')
        for tp in self.tp_record:
            f.write('N{:}, {}\n'.format(tp[0], tp[1]))
        f.close()

    def cop_tpi(self, args, cp_idx):
        sort_value = []
        sort_type = []

        for k, ele in enumerate(self.c0[:self.init_size]):
            if self.mask[k] == 1:
                if ele < 0.5:
                    sort_type.append('AND')
                    sort_value.append(float(ele))
                else:
                    sort_type.append('OR')
                    sort_value.append(1-float(ele))
            else:
                sort_type.append('None')
                sort_value.append(0.5)

        sort_idx = np.argsort(sort_value)
        idx = sort_idx[0]
        cp_type = sort_type[idx]
        if cp_type == 'None':
            idx = -1
        return idx, cp_type
        
