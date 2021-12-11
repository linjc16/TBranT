import random
import os
import numpy as np
import torch
import torch_geometric

# state dimensions
# var_dim is the dimension of each candidate variable's input, i.e., the fixed dimension of matrix C_t
# Tree_t is given by concatenation of two states, for a total dimension node_dim + mip_dim
STATE_DIMS = {
    'var_dim': 25,
    'node_dim': 8,
    'mip_dim': 53
}

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())    



class TreeNodeData(torch_geometric.data.Data):

    def __init__(self, tree_feature, edge_index, num_nodes):
        super().__init__()
        self.tree_feature = tree_feature
        self.edge_index = edge_index
        self.num_nodes = num_nodes 

class GraphDataset(torch_geometric.data.Dataset):

    def __init__(self, data_list):
        super().__init__(root=None, transform = None, pre_transform=None)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, index):
        graph = self.data_list[index]
        graph.num_tree_nodes = graph.tree_feature.size(0)
        return graph