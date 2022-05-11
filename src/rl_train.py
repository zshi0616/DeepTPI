from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
from progress.bar import Bar
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime
from torch._C import Graph
from torch_geometric.loader import DataLoader, DataListLoader
import copy
import time

from config import get_parse_args
from utils.logger import Logger
from utils.random_seed import set_seed
from datasets.dataset_factory import dataset_factory
from rl.model import load_model, save_model

from rl.env import Env
from rl.agent import Agent
from rl.config import RL_Config

def train_RL(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str

    print(args)
    logger = Logger(args)
    config = RL_Config(args)

    ####################################
    # 01 - Prepare data
    ####################################
    dataset = dataset_factory[args.dataset](args.data_dir, args)
    # dataset = dataset[0: 300]
    # dataset = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, drop_last=True, num_workers=args.num_workers)
    
    ####################################
    # 02 - Value Network
    ####################################
    agent = Agent(args, config)
    train_rounds = 0
    start_circuit_idx = -1

    if args.resume:
        # config.RANDOM_ACTION = 0
        agent.Q_netT, _, start_circuit_idx = load_model(
        agent.Q_netT, args.load_model, agent.optimizer, args.resume, args.lr, args.lr_step)
        agent.Q_netT.args.pretrain = False
        agent.Q_net.load_state_dict(agent.Q_netT.state_dict())
        print('[INFO] Load network, Start Epoch: {:}'.format(start_circuit_idx))
    else:
        agent.Q_netT.args.pretrain = False
        print('[INFO] Create new networks')

    if args.ftpt != 'no':
        if args.feature_pretrain_model == '':
            raise('No feature_pretrain_model filepath')
        agent.Q_netT.ft_net, _, ft_net_round = load_model(
        agent.Q_netT.ft_net, args.feature_pretrain_model, agent.optimizer, args.resume, args.lr, args.lr_step)
        agent.Q_netT.args.pretrain = False
        agent.Q_net.ft_net.load_state_dict(agent.Q_netT.ft_net.state_dict())
        print('[INFO] Read feature network with pre-trained round: ', ft_net_round)

    for circuit_idx, g in enumerate(dataset):
        if circuit_idx <= start_circuit_idx:
            continue
        if circuit_idx < 5:
            continue
        if circuit_idx == args.RL_max_times:
            print('[INFO] Reach Maximum Times: ', circuit_idx)
            break

        # if len(g.x) < 500:
        #     continue

        is_vaild_circuit = False
        logger.write('==========================\n')
        logger.write('Circuit: {} , Idx: {:}\n'.format(g.name, circuit_idx))
        agent.action_times = 0
        for train_times in range(config.EACH_TRAIN_TIMES):
            logger.write('Round: {:}\n'.format(train_times))
            env = Env(g, config, args)
            # insert_tp_cnt = random.randint(1, min(config.MAX_CP_CNT, len(g.x)))
            insert_tp_cnt = 10
            print('===========================')
            print('Round {:} with Circuit {}'.format(train_times, g.name))
            print('Insert {:} Control Points'.format(insert_tp_cnt))
            agent.setInitState(env.graph)
            if env.tc_baseline == 100:
                break
            terminal = 0
            is_vaild_circuit = True
            for tp_idx in range(1, insert_tp_cnt+1, 1):
                tp_pos, tp_type = agent.getAction(env.netlist, tp_idx, insert_tp_cnt)
                tp_pos = tp_pos[0]
                tp_type = tp_type[0]
                if tp_pos < 0:
                    print('[INFO] No insert position, Exit')
                    break
                nextState, reward = env.step_frame(tp_pos, tp_type, tp_idx, insert_tp_cnt)
                if tp_idx == insert_tp_cnt or tp_pos == -1:
                    terminal = 1
                loss = agent.setMemory([tp_pos, tp_type], reward, nextState, terminal)
                print('[Action] Insert {:} TP on Pos {:}, with Reward: {:}'.format(tp_type, tp_pos, reward))
                logger.write('[Action] Insert {:} TP on Pos {:}| Reward: {:}| Loss: {:}'.format(tp_type, tp_pos, reward, loss))
                logger.write('\n')
                if terminal == 1:
                    break
            train_rounds += 1
            if config.SAVE_ROUND_GAP > 0 and train_rounds % config.SAVE_ROUND_GAP == 0:
                save_model(os.path.join(args.save_dir, 'model_round_{}.pth'.format(train_rounds)), 
                    circuit_idx, agent.Q_netT, agent.optimizer)
            logger.write('\n')
        if is_vaild_circuit:
            logger.write('\n')
            save_model(os.path.join(args.save_dir, 'model_circuit_{}.pth'.format(circuit_idx)), 
                        circuit_idx, agent.Q_netT, agent.optimizer)
            # agent.save_loss_figure(loss_figure_folder)


if __name__ == '__main__':
    args = get_parse_args()
    set_seed(args)
    random.seed(datetime.now())

    train_RL(args)

