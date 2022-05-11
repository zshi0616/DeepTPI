from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from argparse import Action
from collections import deque

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

import torch
from torch_geometric.loader import DataLoader, DataListLoader
from torch.autograd import Variable
import torch.nn as nn 
from progress.bar import Bar

from config import get_parse_args
from utils.logger import Logger

from rl.old_deepgate import get_recurrent_gnn as old_model
from rl.new_deepgate import get_recurrent_gnn as new_model


class Agent(object):
    def __init__(self, args, config) -> None:
        super().__init__()

        print('==> Using settings {}'.format(args))

        logger = Logger(args)
        self.args = args
        self.replayMemory = deque()
        self.config = config
        self.train_times = 0
        self.action_times = 0
        self.average_loss = []

        # GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str
        args.device = torch.device('cuda' if args.gpus[0] >= 0 else 'cpu')
        
        # if torch.cuda.is_available():
        #     args.device = 'cuda'
        # else:
        #     args.device = 'cpu'

        print('Using device: ', args.device)


        # Model
        print('==> Creating model...')
        if self.args.RL_model == 'deepgate':
            self.Q_net = old_model(args)
            self.Q_netT = old_model(args)
            self.Q_net = self.Q_net.to(self.args.device)
            self.Q_netT = self.Q_netT.to(self.args.device)
        elif self.args.RL_model == 'non_level' or self.args.RL_model == 'non_level_nonattn':
            self.Q_net = new_model(args)
            self.Q_netT = new_model(args)
            self.Q_net = self.Q_net.to(self.args.device)
            self.Q_netT = self.Q_netT.to(self.args.device)
        else:
            raise ("No model")
        print(self.Q_net)
        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), args.lr, weight_decay=args.weight_decay)
        # self.optimizer = torch.optim.SGD(self.Q_net.parameters(), args.lr)
        self.loss_func = nn.MSELoss()
        
    def train(self):
        config = self.config
        begin_time = time.time()
        bar = Bar()

        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, config.BATCH_SIZE)
        state_batch_all = [data[0] for data in minibatch]
        action_batch_all = [data[1] for data in minibatch]
        reward_batch_all = [data[2] for data in minibatch]
        nextState_batch_all = [data[3] for data in minibatch]

        # Trick A. Normalize Reward:
        # reward_batch_all = (reward_batch_all - np.mean(reward_batch_all)) / (np.std(reward_batch_all) + 1e-7)
        # print(reward_batch_all)

        # Step 2: calculate y (Q_real)
        state_batch = None
        nextState_batch = None
        for batch_idx in range(config.BATCH_SIZE):
            if state_batch == None:
                state_batch = copy.deepcopy(state_batch_all[batch_idx])
                nextState_batch = copy.deepcopy(nextState_batch_all[batch_idx])
            else:
                state_batch.forward_level = torch.cat((state_batch.forward_level, state_batch_all[batch_idx].forward_level), 0)
                state_batch.backward_level = torch.cat((state_batch.backward_level, state_batch_all[batch_idx].backward_level), 0)
                state_batch.forward_index = torch.cat((state_batch.forward_index, state_batch_all[batch_idx].forward_index), 0)
                state_batch.backward_index = torch.cat((state_batch.backward_index, state_batch_all[batch_idx].backward_index), 0)
                edge_index = copy.deepcopy(state_batch_all[batch_idx].edge_index)
                for k in range(len(edge_index[0])):
                    edge_index[0][k] += len(state_batch.x)
                    edge_index[1][k] += len(state_batch.x)
                state_batch.edge_index = torch.cat((state_batch.edge_index, edge_index), 1)
                state_batch.x = torch.cat((state_batch.x, state_batch_all[batch_idx].x), 0)

                nextState_batch.forward_level = torch.cat((nextState_batch.forward_level, nextState_batch_all[batch_idx].forward_level), 0)
                nextState_batch.backward_level = torch.cat((nextState_batch.backward_level, nextState_batch_all[batch_idx].backward_level), 0)
                nextState_batch.forward_index = torch.cat((nextState_batch.forward_index, nextState_batch_all[batch_idx].forward_index), 0)
                nextState_batch.backward_index = torch.cat((nextState_batch.backward_index, nextState_batch_all[batch_idx].backward_index), 0)
                edge_index = copy.deepcopy(nextState_batch_all[batch_idx].edge_index)
                for k in range(len(edge_index[0])):
                    edge_index[0][k] += len(nextState_batch.x)
                    edge_index[1][k] += len(nextState_batch.x)
                nextState_batch.edge_index = torch.cat((nextState_batch.edge_index, edge_index), 1)
                nextState_batch.x = torch.cat((nextState_batch.x, nextState_batch_all[batch_idx].x), 0)
        state_batch.num_nodes = len(state_batch.x)
        nextState_batch.num_nodes = len(nextState_batch.x)

        reward_batch = torch.tensor(reward_batch_all)
        # normalization 
        max_reward = torch.abs(torch.max(reward_batch))
        min_reward = torch.abs(torch.min(reward_batch))
        for ele in reward_batch:
            if ele > 0:
                ele /= max_reward
            elif ele < 0: 
                ele /= min_reward

            # mean_a = torch.mean(reward_batch)
            # std_a = torch.std(reward_batch)
            # reward_batch = (reward_batch - mean_a) / std_a
        
        QValue_batch = self.Q_netT(nextState_batch)[0]
        QValue_batch = QValue_batch.to('cpu')
        QValue_batch = QValue_batch.detach().numpy()
        terminal = minibatch[0][4]
        if terminal:
            y_real = reward_batch
        else:
            y_real = reward_batch + config.GAMMA * np.max(QValue_batch)

        # Step 3: calcuate y (Q_pred)
        y_predict_all = self.Q_net(state_batch)[0]
        y_predict_all = y_predict_all.to('cpu')
        y_predict = None
        begin_idx = 0
        for batch_idx in range(config.BATCH_SIZE):
            if action_batch_all[batch_idx][1] == 'AND':
                new_y = y_predict_all[action_batch_all[batch_idx][0] + begin_idx][0]
            elif action_batch_all[batch_idx][1] == 'OR':
                new_y = y_predict_all[action_batch_all[batch_idx][0] + begin_idx][1]
            elif action_batch_all[batch_idx][1] == 'OP':
                new_y = y_predict_all[action_batch_all[batch_idx][0] + begin_idx][2]
            
            begin_idx += len(state_batch_all[batch_idx].x)
            if y_predict == None:
                y_predict = new_y.unsqueeze(0)
            else:
                y_predict = torch.cat((y_predict, new_y.unsqueeze(0)),0)
        
        # Step 4: Loss
        loss = self.loss_func(y_predict, y_real)
        if loss != loss:
            print('[Warning] Loss: nan')
        else:
            assert torch.isnan(loss).sum() == 0, print(loss)
            self.optimizer.zero_grad()
            assert torch.isnan(list(self.Q_net.parameters())[0]).sum() == 0, print(list(self.Q_net.parameters())[0])
            loss.backward()
            assert torch.isnan(list(self.Q_net.parameters())[0]).sum() == 0, print(list(self.Q_net.parameters())[0])
            self.optimizer.step()
            assert torch.isnan(list(self.Q_net.parameters())[0]).sum() == 0, print(list(self.Q_net.parameters())[0])

        # Bar
        end_time = time.time()
        Bar.suffix = '[{:}]|Tot: {}| ETA: {}| Loss: {:}| Time: {:}'.format(self.train_times,  
                                                                bar.elapsed_td, bar.eta_td, 
                                                                float(loss), end_time-begin_time)
        bar.next()
        print('\n')

        self.average_loss.append(float(loss))
        self.train_times += 1
        if self.train_times % config.UPDATE_TIME == 0:
            self.Q_netT.load_state_dict(self.Q_net.state_dict())
            print('[INFO] Update Q and Q star')
        return float(loss)

    def getAction(self, netlist, cp_idx, cp_tot):
        cand_score = []
        cand_idx = []
        cand_type = []
        self.action_times += 1

        # Random Action
        if self.action_times < self.config.RANDOM_ACTION and self.args.RL_mode == 'train':
            print("[INFO] Random Action")
            cand_idx = []
            for idx in range(netlist.init_size):
                if netlist.mask[idx] == 1:
                    if self.args.target == 'ATPG_PC' and netlist.allGatesVec[idx].gate_type != self.args.gate_to_index['BUFF']:
                        continue
                    cand_idx.append(idx)
                    cand_idx.append(idx)
                    cand_idx.append(idx)
                    cand_type.append('AND')
                    cand_type.append('OR')
                    cand_type.append('OP')

            if len(cand_idx) < self.args.no_tp_each_round:
                return [-1], [-1]

            sorted_id = list(range(len(cand_idx)))
            random.shuffle(sorted_id)

        # Get Action from model
        else:
            graph = self.currentState.to(self.args.device)
            graph.cp_idx = cp_idx
            graph.cp_tot = cp_tot
            QValue = self.Q_net(graph)[0].detach().cpu().numpy()

            for idx in range(netlist.init_size):
                if netlist.mask[idx] == 1:
                    if self.args.target == 'ATPG_PC' and netlist.allGatesVec[idx].gate_type != self.args.gate_to_index['BUFF']:
                        continue
                    cand_score.append(QValue[idx][0])
                    cand_score.append(QValue[idx][1])
                    cand_score.append(QValue[idx][2])
                    cand_idx.append(idx)
                    cand_idx.append(idx)
                    cand_idx.append(idx)
                    cand_type.append('AND')
                    cand_type.append('OR')
                    cand_type.append('OP')

            if len(cand_score) < self.args.no_tp_each_round:
                return [-1], [-1]

            sorted_id = sorted(range(len(cand_score)), key=lambda k: cand_score[k], reverse=True)

        tp_pos = []
        tp_type = []

        k = 0
        while len(tp_pos) < self.args.no_tp_each_round:
            if cand_idx[sorted_id[k]] not in tp_pos:
                tp_pos.append(cand_idx[sorted_id[k]])
                tp_type.append(cand_type[sorted_id[k]])
            k += 1

        return tp_pos, tp_type

    def setMemory(self, action, reward, nextState, terminal):
        config = self.config
        loss = 0
        self.replayMemory.append((self.currentState, action, reward, nextState, terminal))
        if len(self.replayMemory) > config.REPLAY_MEMORY:
            self.replayMemory.popleft()
        if len(self.replayMemory) > config.OBSERVE:  # Train the network
            loss = self.train()

        self.currentState = nextState
        return loss

    def setInitState(self, initState):
        self.currentState = initState

    def save_loss_figure(self, fig_path):
        y = []
        for ele in self.average_loss:
            y.append(float(ele))
        plt.plot(range(len(y)), y)
        plt.savefig(fig_path+'/Agent_QNet_loss_{:}_Rounds.jpg'.format(self.train_times))
