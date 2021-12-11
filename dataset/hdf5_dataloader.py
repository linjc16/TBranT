""" Data loader definition. """

import h5py
import torch
from torch._C import Graph
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import torch_geometric
from utils import TreeNodeData, GraphDataset
import pdb

class dataset_h5_ori(Dataset):
    def __init__(self, h5_file, node_dim, mip_dim, var_dim):
        """
        :param h5_file: str, pathway to the data H5 file
        :param node_dim: int, dimension of node state
        :param mip_dim: int, dimension of mip state
        :param var_dim: int, dimension of variable state
        """
        super(dataset_h5_ori, self).__init__()

        # load the h5 file
        self.h5_file = h5py.File(h5_file, 'r')

        # define the dimensions of each feature
        self.node_dim = node_dim
        self.mip_dim = mip_dim
        self.var_dim = var_dim

        # define the number of data points
        self.n_data = len(self.h5_file['dataset'])

    def __getitem__(self, index):
        x = self.h5_file['dataset'][index]
        # pdb.set_trace()
        cand_state_cnt = x[1 + self.node_dim + self.mip_dim].astype(np.int32) # modified
        return [torch.LongTensor([x[0]]),
                torch.FloatTensor(x[1:1 + self.node_dim]),
                torch.FloatTensor(x[1 + self.node_dim:1 + self.node_dim + self.mip_dim]),
                torch.FloatTensor(x[2 + self.node_dim + self.mip_dim : 2 + self.node_dim + self.mip_dim + cand_state_cnt].reshape(-1, self.var_dim)) # modified
                ]

    def __len__(self):
        return self.n_data

class dataset_h5_graph(Dataset):
    def __init__(self, h5_file, node_dim, mip_dim, var_dim):
        """
        :param h5_file: str, pathway to the data H5 file
        :param node_dim: int, dimension of node state
        :param mip_dim: int, dimension of mip state
        :param var_dim: int, dimension of variable state
        """
        super(dataset_h5_graph, self).__init__()

        # load the h5 file
        self.h5_file = h5py.File(h5_file, 'r')

        # define the dimensions of each feature
        self.node_dim = node_dim
        self.mip_dim = mip_dim
        self.var_dim = var_dim

        # define the number of data points
        self.n_data = len(self.h5_file['dataset'])

    def __getitem__(self, index):
        x = self.h5_file['dataset'][index]
        cand_state_cnt = x[1 + self.node_dim + self.mip_dim].astype(np.int32)
        tree_feature_cnt = x[2 + self.node_dim + self.mip_dim + cand_state_cnt].astype(np.int32)

        mip_state = x[1 + self.node_dim:1 + self.node_dim + self.mip_dim]

        tree_feature = x[3 + self.node_dim + self.mip_dim + cand_state_cnt : 3 + self.node_dim + self.mip_dim + cand_state_cnt + tree_feature_cnt].reshape(-1, self.node_dim + self.mip_dim + self.var_dim)

        return [torch.LongTensor([x[0]]),
                torch.FloatTensor(x[1:1 + self.node_dim]),
                torch.FloatTensor(mip_state),
                torch.FloatTensor(x[2 + self.node_dim + self.mip_dim : 2 + self.node_dim + self.mip_dim + cand_state_cnt].reshape(-1, self.var_dim)),
                torch.FloatTensor(tree_feature),
                torch.LongTensor(x[4 + self.node_dim + self.mip_dim + cand_state_cnt + tree_feature_cnt:].reshape(2, -1))
                ]

    def __len__(self):
        return self.n_data


def collate_fn_original(batch):
    batch_list = [item for item in batch]
    return batch_list

def collate_fn_transformer(batch):

    # batch_padding = batch.copy()
    # pdb.set_trace()
    max_len = 0
    # find the max length
    for _, data_tuple in enumerate(batch):
        grid = data_tuple[-1]
        if grid.size(0) > max_len:
            max_len = grid.size(0)
    
    target_list = []
    node_list = []
    var_list = []
    mip_list = []
    padding_mask_list = []
    # padding
    for idx, data_tuple in enumerate(batch):
        grid = data_tuple[-1]
        pad_module = nn.ConstantPad2d((0, 0, 0, max_len - grid.size(0)), 0)
        var_list.append(pad_module(grid).tolist())
        
        padding_mask = torch.zeros((grid.size(0)))
        pad_mask_module = nn.ConstantPad1d((0, max_len - grid.size(0)), 1)
        padding_mask = pad_mask_module(padding_mask)
        padding_mask_list.append(padding_mask.tolist())

        target_list.append(data_tuple[0])
        node_list.append(data_tuple[1].tolist())
        mip_list.append(data_tuple[2].tolist())
    
    target = torch.tensor(target_list) 
    node = torch.tensor(node_list)
    mip = torch.tensor(mip_list)
    var = torch.tensor(var_list) # batch_sz max_len feature
    padding_mask = torch.tensor(padding_mask_list) # batch_sz max_len

    # pdb.set_trace()

    return target, node, mip, var, padding_mask


def collate_fn_transformer_graph(batch):

    max_len = 0
    # find the max length
    for _, data_tuple in enumerate(batch):
        grid = data_tuple[3]
        if grid.size(0) > max_len:
            max_len = grid.size(0)
    
    target_list = []
    node_list = []
    var_list = []
    mip_list = []
    padding_mask_list = []
    tree_list = []
    # padding
    for idx, data_tuple in enumerate(batch):

        grid = data_tuple[3]
        pad_module = nn.ConstantPad2d((0, 0, 0, max_len - grid.size(0)), 0)
        var_list.append(pad_module(grid).tolist())
        
        padding_mask = torch.zeros((grid.size(0)))
        pad_mask_module = nn.ConstantPad1d((0, max_len - grid.size(0)), 1)
        padding_mask = pad_mask_module(padding_mask)
        padding_mask_list.append(padding_mask.tolist())

        target_list.append(data_tuple[0])
        node_list.append(data_tuple[1].tolist())
        mip_list.append(data_tuple[2].tolist())
        tree_list.append(TreeNodeData(tree_feature=data_tuple[4], edge_index=data_tuple[5], num_nodes=data_tuple[4].size(0)))
    
    target = torch.tensor(target_list) 
    node = torch.tensor(node_list)
    mip = torch.tensor(mip_list)
    var = torch.tensor(var_list) # batch_sz max_len feature
    padding_mask = torch.tensor(padding_mask_list) # batch_sz max_len

    tree_batch = list2Graph(tree_list, target.size(0))

    return target, node, mip, var, padding_mask, tree_batch

def list2Graph(tree_list, batch_size):
    tree_dataset = GraphDataset(tree_list)
    graphDataLoader = torch_geometric.loader.DataLoader(tree_dataset, batch_size=batch_size, shuffle=False)
    assert len(graphDataLoader) == 1
    tree_batch = [tree for _, tree in enumerate(graphDataLoader)][0]
    
    return tree_batch