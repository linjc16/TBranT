from platform import node
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules import padding
from torch.nn.modules.normalization import LayerNorm
from models.modules import BiMatchingNet
from models.treeGNN import TreeGNN
import pdb

class BranT(nn.Module):

    def __init__(self, var_dim, node_dim, mip_dim, hidden_size=256, dropout_rate=0.1, nhead=1, num_encoder_layers=1, dim_feedforward=256, tree_gate=True):
        super(BranT, self).__init__()
        print('BranT cat')
        # define the dimensionality of the features and the hidden states
        self.var_dim = var_dim
        self.node_dim = node_dim
        self.mip_dim = mip_dim
        self.hidden_size = hidden_size
        self.tree_gate = tree_gate

        # define CandidateEmbeddingNet
        self.CandidateEmbeddingNet = [LayerNorm(var_dim), nn.Linear(var_dim, hidden_size)]
        self.CandidateEmbeddingNet = nn.Sequential(*self.CandidateEmbeddingNet)

        self.TreeEmbeddingNet = [LayerNorm(node_dim + mip_dim), nn.Linear(node_dim + mip_dim, hidden_size)]
        self.TreeEmbeddingNet = nn.Sequential(*self.TreeEmbeddingNet)
        
        self.globalEmbeddingNet = [nn.Linear(hidden_size * 2, hidden_size)]
        self.globalEmbeddingNet = nn.Sequential(*self.globalEmbeddingNet)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=dim_feedforward, activation='gelu')
        encoder_norm = LayerNorm(hidden_size)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_size, 1)


        self.classifier = nn.Linear(hidden_size, 1)

        # do the Xavier initialization for the linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(tensor=m.weight, gain=nn.init.calculate_gain('relu'))

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, cands_state_mat, padding_mask, node_state=None, mip_state=None):
        '''
            dim:
                cands_state_mat: N * S * E
                padding_mask: N * S
                node_state: N * E
                mip_state: N * E
        '''

        # go through the CandidateEmbeddingNet
        cands_state_mat = self.CandidateEmbeddingNet(cands_state_mat) 
        tree_state = self.TreeEmbeddingNet(torch.cat((node_state, mip_state), dim=1))
        
        if self.tree_gate:
            repeat_count = cands_state_mat.size(1)
            cands_state_mat = torch.cat((cands_state_mat, tree_state.unsqueeze(1).repeat(1, repeat_count, 1)), dim=2)
            cands_state_mat = self.globalEmbeddingNet(cands_state_mat)

        cands_state_mat = cands_state_mat.transpose(0, 1) # S N E

        padding_mask = (padding_mask == 1)
        cands_embedding = self.transformer(cands_state_mat, src_key_padding_mask=padding_mask)

        cands_embedding = cands_embedding.transpose(0, 1)
        

        output = self.linear1(cands_embedding)

        output = self.dropout(output)
        output = self.linear2(output)

        output = torch.squeeze(output, dim=-1)

        output.masked_fill_(
            padding_mask,
            float('-inf')
        )

        raw_output = self.classifier(cands_embedding).squeeze(dim=-1)
        raw_output.masked_fill_(
            padding_mask,
            float('-inf')
        )
        return output, raw_output


class BranchFormer(nn.Module):

    def __init__(self, var_dim, node_dim, mip_dim, hidden_size=256, dropout_rate=0.1, nhead=1, num_encoder_layers=1, dim_feedforward=256, tree_gate=True, graph=False):
        super(BranchFormer, self).__init__()
        print('Bidirection Matching G+l_ori')
        # define the dimensionality of the features and the hidden states
        self.var_dim = var_dim
        self.node_dim = node_dim
        self.mip_dim = mip_dim
        self.hidden_size = hidden_size
        self.tree_gate = tree_gate
        self.graph = graph

        # define CandidateEmbeddingNet
        self.CandidateEmbeddingNet = [LayerNorm(var_dim), nn.Linear(var_dim, hidden_size)]
        self.CandidateEmbeddingNet = nn.Sequential(*self.CandidateEmbeddingNet)

        self.TreeEmbeddingNet = [LayerNorm(node_dim + mip_dim), nn.Linear(node_dim + mip_dim, hidden_size)]
        self.TreeEmbeddingNet = nn.Sequential(*self.TreeEmbeddingNet)
        
        self.globalEmbeddingNet = [nn.Linear(hidden_size * 2, hidden_size)]
        self.globalEmbeddingNet = nn.Sequential(*self.globalEmbeddingNet)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=dim_feedforward, activation='gelu')
        encoder_norm = LayerNorm(hidden_size)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.BiMatchingNet = BiMatchingNet(hidden_size)

        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_size, 1)

        if graph:
            self.BABSTreeGNNNet = TreeGNN(node_dim + mip_dim, hidden_size)
            self.BiMatchingNet2 = BiMatchingNet(hidden_size)
            self.linear3 = nn.Linear(hidden_size * 2, hidden_size)


        self.classifier = nn.Linear(hidden_size, 1)

        # do the Xavier initialization for the linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(tensor=m.weight, gain=nn.init.calculate_gain('relu'))

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, cands_state_mat, padding_mask, node_state=None, mip_state=None, tree_batch=None):
        '''
            dim:
                cands_state_mat: N * S * E
                padding_mask: N * S
                node_state: N * E
                mip_state: N * E
        '''
        # pdb.set_trace()
        # go through the CandidateEmbeddingNet
        cands_state_mat = self.CandidateEmbeddingNet(cands_state_mat) 
        # pdb.set_trace()
        tree_state = self.TreeEmbeddingNet(torch.cat((node_state, mip_state), dim=1))
        
        if self.tree_gate:
            repeat_count = cands_state_mat.size(1)
            cands_state_mat = torch.cat((cands_state_mat, tree_state.unsqueeze(1).repeat(1, repeat_count, 1)), dim=2)
            cands_state_mat = self.globalEmbeddingNet(cands_state_mat)

        cands_state_mat = cands_state_mat.transpose(0, 1) # S N E

        padding_mask = (padding_mask == 1)
        cands_embedding = self.transformer(cands_state_mat, src_key_padding_mask=padding_mask)

        cands_embedding = cands_embedding.transpose(0, 1)
        
        # pdb.set_trace()
        if self.graph:
            tree_state_avg, _ = self.BABSTreeGNNNet(tree_batch)
            output = self.BiMatchingNet(tree_state_avg, cands_embedding, padding_mask)
            output2 = self.BiMatchingNet2(tree_state, cands_embedding, padding_mask)
            output = self.linear3(torch.cat((output, output2), dim=-1))
        else:
            output = self.BiMatchingNet(tree_state, cands_embedding, padding_mask)
            output = self.linear1(output)

        output = self.dropout(output)
        output = self.linear2(output)
        
        output = torch.squeeze(output, dim=-1)

        output.masked_fill_(
            padding_mask,
            float('-inf')
        )

        raw_output = self.classifier(cands_embedding).squeeze(dim=-1)
        raw_output.masked_fill_(
            padding_mask,
            float('-inf')
        )

        return output, raw_output
