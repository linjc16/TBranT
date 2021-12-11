import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import pdb

    
class TreeGNN(nn.Module):

    def __init__(self, feat_size, hidden_size=64, layer_num=2, heads=2, dropout=0.1):
        super(TreeGNN, self).__init__()

        self.hidden_size = hidden_size
        self.layer_num = layer_num
        
        self.tree_node_embedding = nn.Sequential(
            nn.LayerNorm(feat_size),
            nn.Linear(feat_size, hidden_size)
        )
        self.var_embedding = nn.Sequential(
            nn.LayerNorm(25),
            nn.Linear(25, hidden_size)
        )
        self.global_embedding = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        self.GAT1 = GATConv(hidden_size, hidden_size, heads=heads, dropout=dropout)
        self.GAT2 = GATConv(hidden_size * heads, hidden_size, heads=1, dropout=dropout)

        self.glbNode = nn.Parameter(torch.randn(hidden_size))

        
    def forward(self, data):
        """
            dim:
                tree_feature: num_ndoe * E
                edge_index: 2 * num_edge
        """
        
        tree_feature, edge_index, num_tree_nodes, ptr = data.tree_feature, data.edge_index, data.num_tree_nodes, data.ptr
        var_feature = tree_feature[:, -25:]
        tree_feature = tree_feature[:, :-25]

        var_feature = self.var_embedding(var_feature)
        tree_feature = self.tree_node_embedding(tree_feature)

        tree_feature = self.global_embedding(torch.cat((var_feature, tree_feature), dim=1))
        # pdb.set_trace()
        # add GLB node
        batch_size = len(num_tree_nodes)
        sum_tree_nodes = sum(num_tree_nodes.tolist())
        glb_nodes_embedding = torch.cat([self.glbNode.clone().unsqueeze(0)] * batch_size, dim=0)
        tree_feature = torch.cat((tree_feature, glb_nodes_embedding), dim=0)

        with torch.no_grad():
            glb_edges = torch.LongTensor([[sum_tree_nodes + 0] * num_tree_nodes[0], torch.arange(ptr[0], ptr[1]).tolist()]).cuda()
            glb_edges = torch.cat((glb_edges, torch.LongTensor([torch.arange(ptr[0], ptr[1]).tolist(), [sum_tree_nodes + 0] * num_tree_nodes[0]]).cuda()), dim=1)
            for i in range(1, batch_size):
                tmp1 = torch.LongTensor([[sum_tree_nodes + i] * num_tree_nodes[i], torch.arange(ptr[i], ptr[i + 1]).tolist()]).cuda()
                tmp2 = torch.LongTensor([torch.arange(ptr[i], ptr[i + 1]).tolist(), [sum_tree_nodes + i] * num_tree_nodes[i]]).cuda()
                glb_edges = torch.cat((glb_edges, tmp1, tmp2), dim=1)
            edge_index = torch.cat((edge_index, glb_edges), dim=1)
        
        tree_feature = F.relu(self.GAT1(tree_feature, edge_index))
        tree_feature, att = self.GAT2(tree_feature, edge_index, return_attention_weights=True)

        tree_glb = tree_feature[sum_tree_nodes:, :]
        
        return tree_glb, att