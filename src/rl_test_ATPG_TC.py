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
            raise ('No feature_pretrain_model filepath')
        agent.Q_netT.ft_net, _, ft_net_round = load_model(
            agent.Q_netT.ft_net, args.feature_pretrain_model, agent.optimizer, args.resume, args.lr, args.lr_step)
        agent.Q_netT.args.pretrain = False
        agent.Q_net.ft_net.load_state_dict(agent.Q_netT.ft_net.state_dict())
        print('[INFO] Read feature network with pre-trained round: ', ft_net_round)

    baseline_tc_list = []
    cop_tc_list = []
    dg_tc_list = []
    name_list = []

    tot_begin_time = time.time()

    # for circuit_idx, g in enumerate(dataset):
    #     print(g.name)
    #     if circuit_idx > 100:
    #         break

    for circuit_name in selected_name_list:
        g = None
        for circuit_idx, g_tmp in enumerate(dataset):
            if g_tmp.name == circuit_name:
                g = g_tmp
                break
        print(g.name)

        baseline_env = Env(g, config, args)
        dg_env = copy.deepcopy(baseline_env)
        cop_env = copy.deepcopy(baseline_env)

        # if baseline_env.tc_baseline < 70> :
        #     continue

        # Circuit information
        if args.circuit_info:
            print('[INFO] Circuit {}, # Nodes {:}, # PI {:}, # Levels {:}'.format(
                g.name, len(g.forward_level), int(torch.sum(g.forward_level == 0)), int(max(g.forward_level))))

        # Agent test
        insert_cp_cnt = args.no_cp
        print('===========================')
        print('Circuit {}'.format(g.name))
        print('Insert {:} Control Points'.format(insert_cp_cnt))
        agent.setInitState(dg_env.graph)
        terminal = 0
        dg_env.rename(dg_env.graph.name + '_dg')
        cp_idx = 1
        for cp_group in range(int(insert_cp_cnt / args.no_tp_each_round)):
            begin_time = time.time()
            cp_pos_list, cp_type_list = agent.getAction(dg_env.netlist, cp_idx, insert_cp_cnt)
            end_time = time.time()
            for idx in range(len(cp_pos_list)):
                cp_pos = cp_pos_list[idx]
                cp_type = cp_type_list[idx]
                if not args.ignore_action:
                    print(
                        '[Action] Insert {:} CP on Pos {:}, Time: {:}s'.format(cp_type, cp_pos, end_time - begin_time))
                nextState, reward = dg_env.step_frame(cp_pos, cp_type, cp_idx, insert_cp_cnt)
                cp_idx += 1

        # COP
        cop_env.rename(cop_env.graph.name + '_cop')
        for cp_idx in range(insert_cp_cnt):
            cp_pos, cp_type = cop_env.netlist.cop_tpi(args, cp_idx)
            nextState, reward = cop_env.step_frame(cp_pos, cp_type, cp_idx, insert_cp_cnt)
            # print('[INFO] Reward: {:}'.format(reward))

        # Print
        baseline_tc = baseline_env.get_ATPG_test_coverage()
        dg_tc = dg_env.get_ATPG_test_coverage()
        cop_tc = cop_env.get_ATPG_test_coverage()
        print('Baseline: {:}%, COP: {:}%, DGRL: {:}%'.format(baseline_tc, cop_tc, dg_tc))

        if baseline_tc < 99:
            baseline_tc_list.append(baseline_tc)
            dg_tc_list.append(dg_tc)
            cop_tc_list.append(cop_tc)
            name_list.append(g.name)

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
    print('Average Baseline TC: {:}%'.format(np.sum(baseline_tc_list) / len(baseline_tc_list)))
    print('Average COP TC: {:}%'.format(np.sum(cop_tc_list) / len(cop_tc_list)))
    print('Average DGRL TC: {:}%'.format(np.sum(dg_tc_list) / len(dg_tc_list)))

    print('Baseline, COP, DGRL')
    for idx in range(len(baseline_tc_list)):
        print('{}, {:.4f}, {:.4f}, {:.4f}'.format(name_list[idx], baseline_tc_list[idx] / 100,
                                                  cop_tc_list[idx] / 100, dg_tc_list[idx] / 100))

    print('*** DGRL Improvement ***')
    print(np.average(np.array(dg_tc_list) - np.array(baseline_tc_list)), '%')

    print('Total Run Time: {:}s'.format(tot_end_time - tot_begin_time))


if __name__ == '__main__':
    args = get_parse_args()
    set_seed(args)
    random.seed(datetime.now())

    train_RL(args)

