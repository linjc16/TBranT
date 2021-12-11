""" Definitions of the IL modules. """

import torch
import torch.nn as nn
from torch.nn import functional as F
import functools
import pdb



def get_norm_layer(norm_type='none'):
    """
    :param norm_type: str, the name of the normalization layer: batch | instance | layer | none
    :return:
        norm_layer, a normalization layer
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm1d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm1d, affine=False, track_running_stats=False)
    elif norm_type == 'layer':
        norm_layer = functools.partial(nn.LayerNorm)
    elif norm_type == 'none':
        norm_layer = functools.partial(nn.Identity)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


# TreeGate
class TreeGateBranchingNet(nn.Module):
    """
    TreeGate specific network.
    """
    def __init__(self, branch_size, tree_state_size, dim_reduce_factor, infimum=8, norm='none', depth=2,
                 hidden_size=128):
        super(TreeGateBranchingNet, self).__init__()
        norm_layer = get_norm_layer(norm)
        self.norm = norm
        self.branch_size = branch_size
        self.tree_state_size = tree_state_size
        self.dim_reduce_factor = dim_reduce_factor
        self.infimum = infimum
        self.n_layers = 0
        self.depth = depth
        self.hidden_size = hidden_size
        unit_count = infimum
        while unit_count < branch_size:
            unit_count *= dim_reduce_factor
            self.n_layers += 1
        self.n_units_dict = dict.fromkeys(range(self.n_layers))    
        self.BranchingNet = nn.ModuleList()
        input_dim = hidden_size
        for i in range(self.n_layers):
            output_dim = int(input_dim / dim_reduce_factor)
            self.n_units_dict[i] = input_dim
            if i < self.n_layers - 1:
                layer = [nn.Linear(input_dim, output_dim),
                         norm_layer(output_dim),
                         nn.ReLU(True)]
            elif i == self.n_layers - 1:
                layer = [nn.Linear(input_dim, output_dim)]  # Dense output
            input_dim = output_dim
            self.BranchingNet.append(nn.Sequential(*layer))

        # define the GatingNet
        self.GatingNet = []
        self.n_attentional_units = sum(self.n_units_dict.values())
        if depth == 1:
            self.GatingNet += [nn.Linear(tree_state_size, self.n_attentional_units),
                               nn.Sigmoid()]
        else:
            self.GatingNet += [nn.Linear(tree_state_size, hidden_size),
                               nn.ReLU(True)]
            for i in range(depth - 2):
                self.GatingNet += [nn.Linear(hidden_size, hidden_size),
                                   nn.ReLU(True)]
            self.GatingNet += [nn.Linear(hidden_size, self.n_attentional_units),
                               nn.Sigmoid()]
            self.GatingNet = nn.Sequential(*self.GatingNet)

    def forward(self, cands_state_mat, node_state, mip_state):
        tree_state = torch.cat([node_state, mip_state])
        attn_weights = self.GatingNet(tree_state)
        start_slice_idx = 0
        for index, layer in enumerate(self.BranchingNet):
            end_slice_idx = start_slice_idx + self.n_units_dict[index]
            attn_slice = attn_weights[start_slice_idx:end_slice_idx]
            cands_state_mat = cands_state_mat * attn_slice  # No in-place operations, bad for .backward()
            cands_state_mat = layer(cands_state_mat)
            start_slice_idx = end_slice_idx
        cands_prob = cands_state_mat.mean(dim=1, keepdim=True)  # Keep the axis
        return cands_prob


# Bidirection Matching
class BiMatchingNet(nn.Module):
    def __init__(self, hidden_size):
        super(BiMatchingNet, self).__init__()
        self.linear1_1 = nn.Linear(hidden_size, hidden_size)
        self.linear1_2 = nn.Linear(hidden_size, hidden_size)

        self.linear2_1 = nn.Linear(hidden_size, hidden_size)
        self.linear2_2 = nn.Linear(hidden_size, hidden_size)

        self.linear3_1 = nn.Linear(hidden_size, hidden_size)
        self.linear3_2 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, tree_feat, var_feat, padding_mask):
        """
            dim:
                tree_feat: N * E
                var_feat: N * L * E
                padding_mask: N * L
        """
        tree_feat = tree_feat.unsqueeze(1)
        G_tc = torch.bmm(self.linear1_1(tree_feat), var_feat.transpose(1, 2))
        G_tc = torch.squeeze(G_tc, dim=1)
        G_tc.masked_fill_(
            padding_mask,
            float('-inf')
        )
        G_tc = F.softmax(G_tc, dim=1).unsqueeze(1) # N * 1 * L

        G_ct = torch.bmm(self.linear1_2(var_feat), tree_feat.transpose(1, 2))
        G_ct = torch.squeeze(G_ct, dim=2)
        G_ct.masked_fill_(
            padding_mask,
            float('-inf')
        )
        G_ct = F.softmax(G_ct, dim=1).unsqueeze(2) # N * L * 1

        E_t = torch.bmm(G_tc, var_feat) # N * 1 * E
        E_c = torch.bmm(G_ct, tree_feat) # N * L * E

        S_tc = F.relu(self.linear2_1(E_t))
        S_ct = F.relu(self.linear2_2(E_c))

        attn_weight = torch.sigmoid(self.linear3_1(S_tc) + self.linear3_2(S_ct)) # N * L * E
        M_tc = attn_weight * S_tc + (1 - attn_weight) * S_ct

        return M_tc

 
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))