from rl.utils import Netlist
from utils import *
import copy
import os
import torch
from datasets.load_data import circuit_parse_pyg
import numpy as np

def one_hot(idx, length):
    if type(idx) is int:
        idx = torch.LongTensor([idx]).unsqueeze(0)
    else:
        idx = torch.LongTensor(idx).unsqueeze(0).t()
    x = torch.zeros((len(idx), length)).scatter_(1, idx, 1)
    return x

def one_hot_vec(num, size):
    res = [0] * size
    res[num] = 1
    return res

class Env(object):
    def __init__(self, graph, config, args) -> None:
        super().__init__()
        self.config = config
        self.args = args
        
        self.netlist = Netlist(graph, args)
        self.graph = copy.deepcopy(graph)

        if args.target == 'LBIST':
            self.tc_baseline = self.get_test_coverage()
        elif args.target == 'ATPG_PC':
            self.faultlist_filepath = self.args.tmp_dir + '/fault_list.txt'
            self.netlist.save_fault_list(self.args, self.faultlist_filepath)
            self.faultlist_baseline_filepath = self.args.tmp_dir + '/fault_baseline_list.txt'
            _, _, tot_cnt = self.get_ATPG_pattern_count(fault_list=self.faultlist_filepath, 
                        detected_fault_list=self.faultlist_baseline_filepath)
            self.pc_baseline, self.dt_cnt_baseline, _ = self.get_ATPG_pattern_count(fault_list=self.faultlist_baseline_filepath)
            
            self.tc_baseline = self.dt_cnt_baseline/tot_cnt*100
            print('Target Coverage: {:} / {:} = {:.2f}%'.format(self.dt_cnt_baseline, tot_cnt, self.tc_baseline))
            print('Pattern Count Baseline: {:}'.format(self.pc_baseline))
        elif args.target == 'ATPG_TC':
            self.tc_baseline = self.get_ATPG_test_coverage()
        else:
            raise('Unknown target: ', self.args.target)

    def rename(self, new_name):
        self.graph.name = new_name
        self.netlist.name = new_name

    def get_test_coverage(self, pc=0):
        netlist = self.netlist
        bench_path = self.args.tmp_dir + '/tmp.bench'
        netlist.save_bench(self.args, bench_path)

        if 'src' in os.getcwd():
            tpg_pathfile = './external/Atalanta_BIST/atalanta'
        else:
            tpg_pathfile = './src/external/Atalanta_BIST/atalanta'

        if pc == 0:
            tmp_info = os.popen('{} {}'.format(tpg_pathfile, bench_path)).readlines()
        else:
            cmd = '{} -p {:} {}'.format(tpg_pathfile, pc, bench_path)
            cmd = '{} -p {:} {}'.format(tpg_pathfile, pc, bench_path)
            tmp_info = os.popen(cmd).readlines()

        coverage = -1
        for line in tmp_info:
            if 'Fault coverage' in line:
                line = line.replace(' ', '').replace('\n', '').replace('%', '')
                coverage = float(line.split(':')[-1])
                break
        if coverage == -1:
            raise('[ERROR] Cannot get test coverage')
        os.remove(bench_path)
        # os.remove('tmp.test')
        return coverage

    def get_ATPG_test_coverage(self):
        netlist = self.netlist
        bench_path = self.args.tmp_dir + '/tmp.bench'
        netlist.save_bench(self.args, bench_path)

        # tpg_pathfile = './external/Atalanta_ATPG/atalanta'
        tpg_pathfile = 'atalanta'
        cmd = '{} {}'.format(tpg_pathfile, bench_path)
        tmp_info = os.popen(cmd).readlines()

        coverage = -1
        for line in tmp_info:
            # if '[TPG-INFO] Test Coverage:' in line:
            #     line = line.replace(' ', '').replace('\n', '').replace('%', '')
            #     coverage = float(line.split('=')[-1])
            #     break
            if 'Fault coverage' in line:
                line = line.replace(' ', '').replace('\n', '').replace('%', '')
                coverage = float(line.split(':')[-1])
                break
        if coverage == -1:
            raise('[ERROR] Cannot get test coverage')
        os.remove(bench_path)
        return coverage

    def get_ATPG_pattern_count(self, fault_list='', detected_fault_list=''):
        netlist = self.netlist
        bench_path = self.args.tmp_dir + '/tmp.bench'
        netlist.save_bench(self.args, bench_path)

        if fault_list == '':
            fault_list = self.faultlist_baseline_filepath

        tpg_pathfile = './external/Atalanta_ATPG/atalanta'
        if detected_fault_list == '':
            cmd = '{} -f {} {}'.format(tpg_pathfile, fault_list, bench_path)
        else:
            cmd = '{} -f {} -F {} {}'.format(tpg_pathfile, fault_list, detected_fault_list, bench_path)
        tmp_info = os.popen(cmd).readlines()

        pattern_count = -1
        total_faults = -1
        detected_faults = -1
        for line in tmp_info:
            if '[TPG-INFO] Total Faults' in line:
                line = line.replace(' ', '').replace('\n', '').replace('%', '')
                total_faults = float(line.split(':')[-1])
            if '[TPG-INFO] ATPG detected fault count' in line:
                line = line.replace(' ', '').replace('\n', '').replace('%', '')
                detected_faults = float(line.split(':')[-1])
            if '[TPG-INFO] ATPG pattern count' in line:
                line = line.replace(' ', '').replace('\n', '')
                pattern_count = float(line.split(':')[-1])

        if pattern_count == -1:
            raise('[ERROR] Cannot get pattern count')
        os.remove(bench_path)
        return pattern_count, detected_faults, total_faults

    def get_reward(self, lbist_pc=0):
        netlist = self.netlist
        if self.args.target == 'LBIST':
            baseline = self.tc_baseline
            coverage = self.get_test_coverage(lbist_pc)
            reward = coverage - baseline
            self.tc_baseline = coverage
            return reward
        elif self.args.target == 'ATPG_PC':
            pattern_count, dt_cnt, tot_cnt = self.get_ATPG_pattern_count()
            # print('DT: {:}, TOT: {:}'.format(dt_cnt, tot_cnt))
            # print('PC: {:}'.format(pattern_count))
            if dt_cnt < self.dt_cnt_baseline:
                reward = -1 * self.pc_baseline
            else:
                reward = self.pc_baseline - pattern_count 
            self.pc_baseline = pattern_count
            return reward
        elif self.args.target == 'ATPG_TC':
            baseline = self.tc_baseline
            coverage = self.get_ATPG_test_coverage()
            reward = coverage - baseline
            self.tc_baseline = coverage
            return reward
        else:
            raise('Unknown target: ', self.args.target)

    def update_graph(self):
        '''
        Update the graph used in network
        Need Update:
            num_nodes
            forward_level (Gate level)
            backward_level
            x (node type one hot encode)
            edge_index
            forward_index backward_index
        '''
        self.graph.forward_level = []
        self.graph.num_nodes = 0
        self.graph.x = []
        self.graph.forward_index = []
        self.graph.edge_index = [[], []]

        allGatesVec = self.netlist.allGatesVec

        self.graph.num_nodes = len(allGatesVec)
        for idx, gate in enumerate(allGatesVec):
            self.graph.forward_level.append(gate.gate_level)
        
        # Backward level 
        backward_level = [-1] * self.graph.num_nodes
        PO_list = []
        bfs_q = []
        for idx, gate in enumerate(allGatesVec):
            if len(gate.o) == 0:
                PO_list.append(idx)
                bfs_q.insert(0, idx)
                backward_level[idx] = 0
        while len(bfs_q) > 0:
            idx = bfs_q[-1]
            bfs_q.pop()
            tmp_level = backward_level[idx] + 1
            for fanin_idx in allGatesVec[idx].i:
                if backward_level[fanin_idx] < tmp_level:
                    backward_level[fanin_idx] = tmp_level
                    bfs_q.insert(0, fanin_idx)

        # x
        for gate in allGatesVec:
            self.graph.x.append(one_hot_vec(gate.gate_type, len(self.args.gate_to_index)))

        # Edge index
        for idx, gate in enumerate(allGatesVec):
            for fanout_idx in gate.o:
                self.graph.edge_index[0].append(idx)
                self.graph.edge_index[1].append(fanout_idx)
        
        # forward_index backward_index
        for idx in range(len(allGatesVec)):
            self.graph.forward_index.append(idx)

        self.graph.forward_level = torch.tensor(self.graph.forward_level)
        self.graph.backward_level = torch.tensor(backward_level)
        self.graph.x = torch.tensor(self.graph.x)
        self.graph.edge_index = torch.tensor(self.graph.edge_index)
        self.graph.forward_index = torch.tensor(self.graph.forward_index)
        self.graph.backward_index = copy.deepcopy(self.graph.forward_index)

    def new_update_graph(self):
        cp_idx = self.graph.cp_idx
        cp_tot = self.graph.cp_tot
        x = []
        edge_index = []
        allGatesVec = self.netlist.allGatesVec
        y = [0] * len(allGatesVec)

        for idx, gate in enumerate(allGatesVec):
            x.append([float(idx), float(gate.gate_type), float(gate.gate_level)])
            for fanout_idx in gate.o:
                edge_index.append([idx, fanout_idx])
        
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        edge_index = np.array(edge_index)

        if self.args.RL_model != 'deepgate' and self.args.ftpt == 'no':
            self.graph.edge_index = torch.zeros([2, len(edge_index)])
            self.graph.forward_index = torch.zeros(len(x))
            self.graph.backward_index = torch.zeros(len(x))
            self.graph.x = torch.zeros([len(x), 4])

            for edge in edge_index:
                self.graph.edge_index[0] = edge[0]
                self.graph.edge_index[1] = edge[1]

            for idx in range(len(self.graph.forward_index)):
                self.graph.forward_index[idx] = idx
                self.graph.backward_index[idx] = idx

            gate_list = x[:, 1]
            self.graph.x = one_hot(gate_list, 4)

            self.graph.edge_index = self.graph.edge_index.long()
            self.graph.forward_index = self.graph.forward_index.float()
            self.graph.backward_index = self.graph.backward_index.float()
            self.graph.x = self.graph.x.long()

        else:
            self.graph = circuit_parse_pyg(x, edge_index, y, self.args.use_edge_attr, \
                self.args.reconv_skip_connection, self.args.logic_diff_embedding, self.args.predict_diff, \
                self.args.diff_multiplier, self.args.no_node_cop, self.args.node_reconv, self.args.un_directed, self.args.num_gate_types, self.args.dim_edge_feature)
            
        
        
        self.graph.len = len(x)
        self.graph.cp_idx = cp_idx
        self.graph.cp_tot = cp_tot

    def step_frame(self, cp_pos, cp_type_name, cp_idx, insert_cp_cnt):
        if cp_pos > 0:
            if self.args.aig:
                self.netlist.insert_cp_aig(cp_pos, cp_type_name, self.args.gate_to_index)
            else:
                self.netlist.insert_cp(cp_pos, cp_type_name)
    
        if self.args.reward == 'cont':
            reward = self.get_reward()
        elif self.args.reward == 'sparse':
            if cp_idx == insert_cp_cnt or cp_pos < 0: reward = self.get_reward()
            else: reward = 0
        else:
            reward = 0

        self.graph.cp_idx = cp_idx
        self.graph.cp_tot = insert_cp_cnt
        # self.new_update_graph()
        return self.graph, reward

    def update_tp_list(self, tp_pos, tp_type):
        self.netlist.tp_record.append([tp_pos, tp_type])

    
