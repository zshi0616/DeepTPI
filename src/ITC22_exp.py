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
        # cp_pos_list = [23206, 22779, 22504, 22510, 5865, 5867, 17240, 5809, 21287, 17139, 18801, 21926, 18178, 21005, 21022, 21027, 18820, 18234, 18872, 18885, 21012, 18416, 18898, 18911, 18924, 16888, 18846, 18859, 18833, 18960, 17984, 18391, 18421, 18475, 18735, 18793, 18529, 18502, 18448, 18364, 18134, 18692, 18556, 18638, 18611, 18665, 18583, 18807, 18742, 18745, 18233, 18131, 18181, 21924, 25676, 18585, 18942, 16904, 16920, 16928, 16944, 16960, 16896, 16912, 18936, 16936, 16952, 16968, 16984, 16992, 17000, 16976, 19351, 19383, 19359, 19367, 19375, 18954, 18948, 18426, 25384, 25391, 25417, 25398, 25558, 25565, 25407, 25531, 25538, 25378, 25477, 25483, 25471, 25442, 25552, 25545, 25494, 25514, 25465, 25459, 25432, 25371, 25421, 25436, 19138, 19122, 25453, 25525, 25502, 25508, 25764, 25763, 25765, 19126, 19130, 19134, 19142, 19146, 19150, 25768, 25767, 25752, 25769, 25771, 25751, 25753, 25772, 25766, 25773, 19094, 19106, 19114, 26712, 25770, 25754, 21925, 25756, 25755, 25757, 25779, 25780, 21931, 25781, 21927, 19118, 25760, 25761, 25759, 25758, 21285, 21284, 25782, 25762, 25776, 25775, 25777, 19098, 19110, 21014, 21015, 21013, 21932, 21028, 21029, 21030, 21023, 21024, 21025, 21006, 21007, 21008, 21933, 21928, 25748, 26711, 21930, 27721, 25408, 25399, 25392, 21929, 27720, 25385, 27719, 26664, 27282, 25414, 25429, 27280, 25428, 22236, 21940, 26668, 21286, 21063, 16872, 16868, 19102, 19729, 19732, 19733, 19734, 19731, 19727, 19728, 19730, 19735, 19736, 19725, 19717, 19719, 19713, 19711, 19714, 26665, 19718, 19723, 19722, 19724, 26705, 16839, 25733, 26670, 16874, 16870, 26669, 21143, 21249, 21941, 19721, 25734, 25749, 21248, 21250, 21142, 20846, 25418, 19720, 25379, 25750, 21144, 25566, 19726]
        cp_pos_list = [23206, 22779, 22504, 22510, 18171, 18126, 18125, 18169, 17240, 17139, 18801, 18178, 21005, 21022, 21027, 18820, 18234, 18038, 18039, 18872, 18885, 21012, 18416, 18898, 18911, 18924, 16888, 18846, 18859, 18833, 18960, 17984, 18391, 18421, 18093, 18094, 18203, 18147, 18201, 18149, 18475, 18735, 18793, 18529, 18502, 18448, 18364, 18134, 18692, 18556, 18638, 18611, 18665, 18583, 18807, 18742, 18745, 18233, 18131, 18181, 18585, 19295, 18942, 16904, 16920, 16928, 16944, 16960, 19311, 16896, 16912, 18936, 16936, 16952, 16968, 16984, 16992, 17000, 19319, 16976, 19335, 19327, 19343, 19351, 19383, 19359, 19367, 19375, 19303, 18954, 18948, 18396, 18399, 18429, 18426, 18385, 18442, 19138, 19122, 19126, 19130, 19134, 19142, 19146, 19150, 19094, 19106, 19114, 19118, 19098, 19110, 21011, 21014, 21015, 21013, 21026, 21028, 21029, 21030, 21023, 21021, 21024, 21025, 21004, 21006, 21007, 21008, 16872, 16868, 19102, 21042, 21043, 16839, 16874, 16870, 21143, 21249, 21248, 21250, 21142, 20846, 21144, ]
        cp_type_list = []
        for idx in cp_pos_list:
            cp_type_list.append('OP')

        for idx in range(len(cp_pos_list)):
            cp_pos = cp_pos_list[idx]
            cp_type = cp_type_list[idx]
            dg_env.update_tp_list(cp_pos, cp_type)
            nextState, reward = dg_env.step_frame(cp_pos, cp_type, cp_idx, insert_cp_cnt)

        # COP
        cop_env.rename(cop_env.graph.name + '_cop')
        cp_pos_list = [5546, 18356, 18343, 16517, 8675, 27782, 27724, 8763, 18372, 8764, 16518, 27784, 27726, 8677, 19279, 19271, 16157, 27366, 8125, 27424, 8213, 19287, 8765, 8679, 16519, 27786, 27728, 27368, 8214, 27426, 16158, 8127, 27730, 8681, 8766, 27788, 16520, 8129, 16159, 27370, 8215, 27428, 18414, 27372, 8131, 16160, 27430, 8216, 27790, 8767, 8683, 27732, 16521, 16161, 27432, 27374, 8133, 8217, 27792, 8685, 27734, 8768, 16522, 16162, 27434, 27376, 8135, 8218, 8688, 27736, 8769, 27794, 16523, 18456, 18340, 18325, 8580, 16454, 8546, 18354, 8138, 8219, 27378, 16163, 27436, 8576, 27659, 16452, 18369, 8582, 27604, 27664, 16396, 8492, 18231, 16637, 18175, 11149, 18382, 11152, 18198, 8770, 27796, 8689, 16524, 27738, 18129, 16636, 18200, 11151, 18143, 27666, 8583, 27606, 8494, 16397, 18469, 9794, 16638, 11148, 18145, 18124, 18225, 18197, 1385, 22601, 18170, 1384, 18142, 18226, 22582, 1383, 18227, 22561, 18090, 1626, 18204, 11340, 18097, 2498, 1388, 23821, 22620, 11153, 11338, 1624, 18148, 11150, 23835, 22617, 1387, 2500, 11336, 18202, 1622, 18146, 1620, 11334, 23828, 18127, 11332, 18092, 23298, 1618, 18069, 18070, 18173, 22603, 10974, 18087, 18071, 18229, 22584, 18072, 18096, 11339, 23836, 18040, 11337, 18095, 18034, 18174, 22625, 1389, 18230, 18032, 18037, 11335, 18036, 11333, 22624, 18035, 11331, 8857, 8859, 10973, 18033, 8861, 8863, 1625, 22621, 23308, 1627, 1623, 18128, 1621, 1619, 1617, 8584, 8496, 27608, 27668, 16398, 8858, 8860, 602, 8862, 8864, 9791, 8867, 8139, 8220, 27380, 16164, 27438, 18411, 586, 20959, 20973, 588, 8868, 590, 20987, 8869, 20997, 592, 594, 20992, 591, 596, 9792, 597, 21016, 598, 599, 600, 21031, 601]
        cp_type_list = []
        for idx in cp_pos_list:
            cp_type_list.append('OP')
        for idx in range(len(cp_pos_list)):
            cp_pos = cp_pos_list[idx]
            cp_type = cp_type_list[idx]
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

