from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from typing import Optional
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax
from torch_geometric.nn.glob import *
from torch_geometric.nn import MessagePassing
from utils.dag_utils import subgraph, custom_backward_subgraph

from models.mlp import MLP
from models.gat_conv import AGNNConv
from models.gcn_conv import AggConv
from .model import get_recurrent_gnn as deepgate

from torch.nn import LSTM, GRU
import time

_update_function_factory = {
    'lstm': LSTM,
    'gru': GRU,
}

class nonlevel_attn_GNN(nn.Module):
    '''
    Recurrent Graph Neural Networks for Circuits.
    '''
    def __init__(self, args):
        super(nonlevel_attn_GNN, self).__init__()
        
        self.args = args
        if self.args.ftpt != 'no':
            self.ft_net = deepgate(args)

        # configuration
        self.num_rounds = args.num_rounds
        self.device = args.device
        self.predict_diff = args.predict_diff
        self.intermediate_supervision = args.intermediate_supervision
        self.reverse = args.reverse
        self.custom_backward = args.custom_backward
        self.use_edge_attr = args.use_edge_attr

        # dimensions
        self.num_aggr = args.num_aggr
        self.dim_node_feature = args.dim_node_feature
        self.dim_hidden = args.dim_hidden
        self.dim_mlp = args.dim_mlp
        self.dim_pred = args.dim_pred
        self.num_fc = args.num_fc
        self.wx_update = args.wx_update
        self.wx_mlp = args.wx_mlp
        self.dim_edge_feature = args.dim_edge_feature

        # 1. message/aggr-related
        dim_aggr = self.dim_hidden# + self.dim_edge_feature if self.use_edge_attr else self.dim_hidden
        # if self.use_edge_attr:
        #     aggr_forward_pre = MLP(self.dim_hidden, self.dim_hidden, self.dim_hidden, num_layer=3, p_drop=0.2)
        # else:
        aggr_forward_pre = nn.Linear(dim_aggr, self.dim_hidden)
        if self.args.aggr_function == 'deepset':
            raise()
        else:
            if self.args.RL_model == 'non_level':
                self.aggr_forward = AGNNConv(dim_aggr, self.dim_hidden, mlp=aggr_forward_pre, wea=self.use_edge_attr)
                self.aggr_backward = AGNNConv(dim_aggr, self.dim_hidden, mlp=aggr_forward_pre, reverse=True, wea=self.use_edge_attr)
            elif self.args.RL_model == 'non_level_nonattn':
                self.aggr_forward = AggConv(dim_aggr, self.dim_hidden, mlp=aggr_forward_pre, wea=self.use_edge_attr)
                self.aggr_backward = AggConv(dim_aggr, self.dim_hidden, mlp=aggr_forward_pre, reverse=True, wea=self.use_edge_attr)
            else:
                raise KeyError('no support {} RL model'.format(self.args.RL_model))

        # 2. update-related
        if self.args.update_function in _update_function_factory.keys():
            # Here only consider the inputs as the concatenated vector from embedding and feature vector.
            if self.wx_update:
                if self.args.ftpt == 'update_with':
                    self.update_layer = _update_function_factory[self.args.update_function](
                        self.dim_node_feature + self.dim_hidden * 3, self.dim_hidden)
                else:
                    self.update_layer = _update_function_factory[self.args.update_function](
                        self.dim_node_feature + self.dim_hidden * 2, self.dim_hidden)
        else:
            raise KeyError('no support {} update function.'.format(self.args.update_function))
        # consider the embedding for the LSTM/GRU model initialized by non-zeros
        self.one = torch.ones(1).to(self.device)
        self.emd_int = nn.Linear(1, self.dim_hidden).to(self.device)
        self.one.requires_grad = False


        # 3.1 Value Network
        output_dim = 2
        if self.args.op:
            output_dim = 3

        # self.value_network = MLP(self.dim_hidden + 2, self.dim_mlp, output_dim, 
        #     num_layer=self.num_fc, norm_layer=args.norm_layer, act_layer=args.activation_layer, sigmoid=False, tanh=False)
        
        self.value_network = nn.Sequential(
            nn.Linear(self.dim_hidden + 2, 128), 
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )    

        # 3.2 Probability 
        self.predictor = MLP(self.dim_hidden, self.dim_mlp, self.dim_pred, 
            num_layer=self.num_fc, norm_layer=args.norm_layer, act_layer=args.activation_layer, sigmoid=False, tanh=False)

    def forward(self, G):
        num_nodes = G.num_nodes
        num_layers_f = max(G.forward_level).item() + 1
        num_layers_b = max(G.backward_level).item() + 1
        one = self.one
        h_init = self.emd_int(one).view(1, 1, -1) # (1 x 1 x dim_hidden)
        h_init = h_init.repeat(1, num_nodes, 1) # (1 x num_nodes x dim_hidden)

        if self.args.update_function == 'lstm':
            raise()
        elif self.args.update_function == 'gru':
            preds = self._gru_forward(G, h_init, num_layers_f, num_layers_b, num_nodes)
        else:
            raise NotImplementedError('The update function should be specified as one of lstm and gru.')
        
        return preds
            
       
    def _gru_forward(self, G, h_init, num_layers_f, num_layers_b, num_nodes):
        x, edge_index = G.x, G.edge_index
        x = x.to(self.args.device)
        edge_index = edge_index.to(self.args.device)
        edge_attr = G.edge_attr if self.use_edge_attr else None

        if self.args.ftpt != 'no':
            node_emb = self.ft_net.get_emb(G.to(self.args.device)).detach()

        if self.args.ftpt == 'init_feature':
            node_state = node_emb.unsqueeze(0)
        else:
            node_state = h_init # (h_0). here we initialize h_0. TODO: option of not initializing the hidden state of GRU.

        preds = []
        for _ in range(self.num_rounds):
            # From fanin to node 
            fw_edge_index, fw_edge_attr = subgraph(G.forward_index, edge_index, edge_attr, dim=1)
            fw_msg = self.aggr_forward(node_state.squeeze(0), fw_edge_index, fw_edge_attr)

            # From fanout to node 
            bw_edge_index, bw_edge_attr = subgraph(G.backward_index, edge_index, edge_attr, dim=0)
            bw_msg = self.aggr_backward(node_state.squeeze(0), bw_edge_index, bw_edge_attr)

            if self.args.wx_update:
                if self.args.ftpt == 'update_with':
                    _, node_state = self.update_layer(torch.cat([fw_msg, bw_msg, x, node_emb], dim=1).unsqueeze(0), node_state)
                else:
                    _, node_state = self.update_layer(torch.cat([fw_msg, bw_msg, x], dim=1).unsqueeze(0), node_state)
            else:
                raise('unsupport no wx_update')

        node_embedding = node_state.squeeze(0).to(self.args.device)
        if self.args.pretrain:
            pred = self.predictor(node_embedding)
        else:
            extra_feature = torch.tensor([G.cp_idx, G.cp_tot]).repeat(len(node_embedding), 1).to(self.args.device)
            node_embedding = torch.cat([node_embedding, extra_feature], dim=1)
            pred = self.value_network(node_embedding)    
        preds.append(pred)

        return preds
    
def get_recurrent_gnn(args):
    return nonlevel_attn_GNN(args)

def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None):
    start_epoch = 0
    checkpoint = torch.load(
        model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '
                      'loaded shape{}. {}'.format(
                          k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            # optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = start_lr
            # print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        # return model, optimizer, start_epoch
        return model, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)
