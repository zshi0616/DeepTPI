from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy


import os
from progress.bar import Bar
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime
from torch._C import Graph
import time

from config import get_parse_args
from utils.logger import Logger
from utils.random_seed import set_seed
from datasets.dataset_factory import dataset_factory
from rl.model import load_model, save_model

from rl.env import Env
from rl.agent import Agent
from rl.config import RL_Config

selected_name_list = ['b04_C_000', 'b04_C_003', 'b04_C_005', 'b04_C_006', 
                      'b04_opt_C_002', 'b04_opt_C_006', 'b04_opt_C_007', 
                      'b07_C_000', 'b07_C_004', 'b07_C_011']

def train_RL(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str

    print(args)
    args.num_rounds = args.test_num_rounds
    Logger(args)

    ####################################
    # 01 - Prepare data
    ####################################
    dataset = dataset_factory[args.dataset](args.data_dir, args)

    ####################################
    # 02 - Value Network
    ####################################
    config = RL_Config(args)
    agent = Agent(args, config)

    if args.resume:
        agent.Q_netT, _, start_epoch = load_model(
        agent.Q_netT, args.load_model, agent.optimizer, args.resume, args.lr, args.lr_step)
        agent.Q_netT.args.pretrain = False
        print(start_epoch)
        agent.Q_net.load_state_dict(agent.Q_netT.state_dict())

    if args.ftpt != 'no':
        if args.feature_pretrain_model == '':
            raise('No feature_pretrain_model filepath')
        agent.Q_netT.ft_net, _, ft_net_round = load_model(
        agent.Q_netT.ft_net, args.feature_pretrain_model, agent.optimizer, args.resume, args.lr, args.lr_step)
        agent.Q_netT.args.pretrain = False
        agent.Q_net.ft_net.load_state_dict(agent.Q_netT.ft_net.state_dict())
        print('[INFO] Read feature network with pre-trained round: ', ft_net_round)

    baseline_tc_list = []
    cop_tc_list = []
    dg_tc_list = []

    tot_begin_time = time.time()
    tmp_tp_each_round = args.no_tp_each_round

    for circuit_idx, g in enumerate(dataset):
        # if 'b17_C' not in g.name and 'mem_ctrl' not in g.name:
        #     continue
        # if 'b17' in g.name or 'mem_ctrl' in g.name:
        #     continue

        if 'b15' not in g.name:
            continue
        
        print(g.name)
        baseline_env = Env(g, config, args)
        dg_env = copy.deepcopy(baseline_env)
        cop_env = copy.deepcopy(baseline_env)

        # if baseline_env.baseline < 70:
        #     continue

        # Circuit information
        if args.circuit_info:
            print('[INFO] Circuit {}, # Nodes {:}, # PI {:}, # PO {:},  # Levels {:}'.format(
                g.name, len(g.forward_level), int(torch.sum(g.forward_level == 0)), int(torch.sum(g.backward_level == 0)), 
                int(max(g.forward_level))))
        
        # Agent test
        if args.no_cp != -1:
            insert_cp_cnt = args.no_cp
        else:
            insert_cp_cnt = int(len(g.x) / 100)

        if (insert_cp_cnt / tmp_tp_each_round) > 20:
            args.no_tp_each_round = int(insert_cp_cnt / 20)
        else:
            args.no_tp_each_round = tmp_tp_each_round
        
        if tmp_tp_each_round == -1:
            args.no_tp_each_round = insert_cp_cnt

        print('\n===========================')
        print('Circuit {}'.format(g.name))
        print('Insert {:} Control Points'.format(insert_cp_cnt))
        print('Insert {:} TPs, {:} TPs / Step'.format(insert_cp_cnt, args.no_tp_each_round))
        agent.setInitState(dg_env.graph)
        terminal = 0
        dg_env.rename(dg_env.graph.name + '_dg')
        cp_idx = 1
        for cp_group in range(int(insert_cp_cnt / args.no_tp_each_round)):
            begin_time = time.time()
            cp_pos_list, cp_type_list = agent.getAction(dg_env.netlist, cp_idx, insert_cp_cnt)
            end_time = time.time()
            begin_cp_idx = cp_idx
            for idx in range(len(cp_pos_list)):
                cp_pos = cp_pos_list[idx]
                cp_type = cp_type_list[idx]
                dg_env.update_tp_list(cp_pos, cp_type)
                if not args.ignore_action:
                    print('[Action] Insert {:} CP on Pos {:}, Time: {:}s'.format(cp_type, cp_pos, end_time-begin_time))
                if len(cp_pos_list) + begin_cp_idx < insert_cp_cnt:
                    nextState, reward = dg_env.step_frame(cp_pos, cp_type, cp_idx, insert_cp_cnt)
                cp_idx += 1

        # COP
        cop_env.rename(cop_env.graph.name + '_cop')
        if not args.no_cop_tpi: 
            for cp_idx in range(insert_cp_cnt):
                cp_pos, cp_type = cop_env.netlist.cop_tpi(args, cp_idx)
                cop_env.update_tp_list(cp_pos, cp_type)
                nextState, reward = cop_env.step_frame(cp_pos, cp_type, cp_idx, insert_cp_cnt)

        # Print
        baseline_tc = baseline_env.get_test_coverage()
        dg_tc = dg_env.get_test_coverage()
        cop_tc = cop_env.get_test_coverage()
        print('Baseline: {:}%, COP: {:}%, DGRL: {:}%'.format(baseline_tc, cop_tc, dg_tc))

        # if baseline_tc < 99:
        baseline_tc_list.append(baseline_tc)
        dg_tc_list.append(dg_tc)
        cop_tc_list.append(cop_tc)

        # Save
        if args.save_bench:
            baseline_env.netlist.save_bench(args)
            cop_env.netlist.save_bench(args)
            dg_env.netlist.save_bench(args)

            baseline_env.netlist.save_tp_list(args)
            cop_env.netlist.save_tp_list(args)
            dg_env.netlist.save_tp_list(args)
            
            baseline_env.netlist.save_verilog(args)
            cop_env.netlist.save_verilog(args)
            dg_env.netlist.save_verilog(args)
            print('[INFO] Save {}'.format(args.bench_dir + '/' + g.name + '.bench'))

    tot_end_time = time.time()

    # Average
    print('=======================================')
    print('Average Baseline TC: {:}%'.format(np.sum(baseline_tc_list)/len(baseline_tc_list)))
    print('Average COP TC: {:}%'.format(np.sum(cop_tc_list)/len(cop_tc_list)))
    print('Average DGRL TC: {:}%'.format(np.sum(dg_tc_list)/len(dg_tc_list)))

    print('*** Baseline ***')
    print(baseline_tc_list)
    print('*** COP ***')
    print(cop_tc_list)
    print('*** DGRL ***')
    print(dg_tc_list)

    print('Baseline, COP, DGRL')
    for idx in range(len(baseline_tc_list)):
        print('{:.4f}, {:.4f}, {:.4f}'.format(baseline_tc_list[idx]/100, cop_tc_list[idx]/100, dg_tc_list[idx]/100))

    print('*** DGRL Improvement ***')
    print(np.average(np.array(dg_tc_list) - np.array(baseline_tc_list)), '%')

    print('Total Run Time: {:}s'.format(tot_end_time-tot_begin_time))

if __name__ == '__main__':
    args = get_parse_args()
    set_seed(args)
    random.seed(datetime.now())

    train_RL(args)

