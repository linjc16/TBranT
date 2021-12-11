""" Brancher classes. """

from collections import OrderedDict
import pyscipopt as scip
import numpy as np

import torch

import torch_geometric

import copy
import pdb
import time

from utils import TreeNodeData, GraphDataset


class Tree:
    """
    Container for a B&B search Tree
    """
    def __init__(self):
        self.node_feature = []
        self.mip_feature = []
        self.var_feature = [] # Branching Decision
        self.edge_index = []
        self.cand_features = []
        self.num_node = 0
        self.scipIdMapping = {}
        self.scip_id_list = []

    def update(self, curr_scip_id, parent_scip_id, curr_node_feature, curr_mip_feature, curr_var_feature, curr_cand_features, branch_status=None):
        # pdb.set_trace()
        if curr_scip_id in self.scip_id_list: 
            if branch_status == scip.SCIP_RESULT.BRANCHED:
                # replace
                # pdb.set_trace()
                id = self.scipIdMapping[str(curr_scip_id)]
                self.node_feature[id] = curr_node_feature
                self.mip_feature[id] = curr_mip_feature
                self.var_feature[id] = curr_var_feature
                self.cand_features[id] = curr_cand_features
                
        else:
            id = self.num_node
            self.num_node = self.num_node + 1
            self.node_feature.append(curr_node_feature)
            self.mip_feature.append(curr_mip_feature)
            self.var_feature.append(curr_var_feature)
            self.cand_features.append(curr_cand_features)

            self.scipIdMapping[str(curr_scip_id)] = id
            self.scip_id_list.append(curr_scip_id)
            
            if parent_scip_id is not None:
                parent_id = self.scipIdMapping[str(parent_scip_id)]
                self.edge_index.append([id, parent_id])
                self.edge_index.append([parent_id, id])
    
    def getTreeNodeFeature(self, scip_idx):
        index = self.scipIdMapping[str(scip_idx)]
        return self.node_feature[index], self.mip_feature[index], self.var_feature[index]

class Brancher(scip.Branchrule):
    """
    Base class for scip.Branchrule subclasses.
    Callback method branchexeclp is customized in each subclass.
    """
    def initialize(self):
        pass

    def branchinit(self):
        pass


""" IL branchers """


class ILEvalBrancher(Brancher):
    """
    Brancher using trained Imitation Learning policy and ILEvalEnv.
    Evaluation mode is deterministic.
    """

    def __init__(self, model, device, policy, state_dims, verbose, policy_name):
        super(ILEvalBrancher, self).__init__()

        self.model = model
        self.device = device
        self.policy = policy.to(device)
        self.var_dim = state_dims['var_dim']
        self.node_dim = state_dims['node_dim']
        self.mip_dim = state_dims['mip_dim']
        self.verbose = verbose
        self.policy_name = policy_name

        self.branch_count = 0
        self.branchexec_count = 0
        self.episode_rewards = []

    def choose(self, probs):
        if len(probs.size()) == 0:
            probs = probs.unsqueeze(0)
        confidence_score, branch_decision = probs.max(0)
        return confidence_score, branch_decision

    def branchexeclp(self, allowaddcons):
        if self.model.getNNodes() == 1:
            result = self.model.executeBranchRule('relpscost', allowaddcons)
            if result == scip.SCIP_RESULT.BRANCHED:
                _, chosen_variable, *_ = self.model.getChildren()[0].getBranchInfos()
                assert chosen_variable is not None
                assert chosen_variable.isInLP()
        else:
            # get state representations
            cands, cands_pos, cands_state_mat = self.model.getCandsState(self.var_dim, self.branchexec_count)
            node_state = self.model.getNodeState(self.node_dim)
            mip_state = self.model.getMIPState(self.mip_dim)

            self.branchexec_count += 1

            # torchify states
            cands_state_mat = torch.from_numpy(cands_state_mat.astype('float32')).to(self.device)
            node_state = torch.from_numpy(node_state.astype('float32')).to(self.device)
            mip_state = torch.from_numpy(mip_state.astype('float32')).to(self.device)


            # select action from the policy probs
            if self.policy_name == 'Transformer':
                probs, _ = self.policy(cands_state_mat.unsqueeze(0), torch.zeros((1, cands_state_mat.size(0))).cuda(), node_state.unsqueeze(0), mip_state.unsqueeze(0))
            else:
                probs = self.policy(cands_state_mat, node_state, mip_state)
            
            
            probs = probs.squeeze()
            _, action = self.choose(probs)  # the chosen variable
            
            # define the SCIP branch var
            var = cands[action.item()]
            # branch on the selected variable (SCIP Variable object)
            self.model.branchVar(var)
            self.branch_count += 1

            if self.verbose:
                print('\tBranch count: {}. Selected var: {}.'.format(
                    self.branch_count, cands_pos[action.item()]))

            result = scip.SCIP_RESULT.BRANCHED
            if result == scip.SCIP_RESULT.BRANCHED:
                _, chosen_variable, *_ = self.model.getChildren()[0].getBranchInfos()
                assert chosen_variable is not None
                assert chosen_variable.isInLP()

        return {'result': result}

    def finalize(self):
        pass

    def finalize_zero_branch(self):
        pass

class ILEvalGraphBrancher(Brancher):
    """
    Brancher using trained Imitation Learning policy and ILEvalEnv.
    Evaluation mode is deterministic.
    """

    def __init__(self, model, device, policy, state_dims, verbose):
        super(ILEvalGraphBrancher, self).__init__()

        self.model = model
        self.device = device
        self.policy = policy.to(device)
        self.var_dim = state_dims['var_dim']
        self.node_dim = state_dims['node_dim']
        self.mip_dim = state_dims['mip_dim']
        self.verbose = verbose

        self.branch_count = 0
        self.branchexec_count = 0
        self.episode_rewards = []
        
        self.searchTree = Tree()

    def choose(self, probs):
        if len(probs.size()) == 0:
            probs = probs.unsqueeze(0)
        confidence_score, branch_decision = probs.max(0)
        return confidence_score, branch_decision

    def branchexeclp(self, allowaddcons):
        
        # self.branchexec_count += 1
        # get state representations
        cands, cands_pos, cands_state_mat = self.model.getCandsState(self.var_dim, self.branchexec_count)
        node_state = self.model.getNodeState(self.node_dim)
        mip_state = self.model.getMIPState(self.mip_dim)

        self.branchexec_count += 1
        # pdb.set_trace()
        curr_scip_id = self.model.getCurrentNode().getNumber()
        parent_scip_id = None if self.model.getNNodes() == 1 else self.model.getCurrentNode().getParent().getNumber()
        
        if self.model.getNNodes() == 1:
            result = self.model.executeBranchRule('relpscost', allowaddcons)
            if result == scip.SCIP_RESULT.BRANCHED:
                _, chosen_variable, *_ = self.model.getChildren()[0].getBranchInfos()
                # chosen_variable is a SCIP Variable object
                assert chosen_variable is not None
                assert chosen_variable.isInLP()
                branch_cand_state = copy.deepcopy(cands_state_mat[cands_pos.index(chosen_variable.getCol().getLPPos()), :])

                self.searchTree.update(curr_scip_id, parent_scip_id, node_state, mip_state, branch_cand_state, cands_state_mat, result)
        else:
            # torchify states
            cands_state_mat = torch.from_numpy(cands_state_mat.astype('float32')).to(self.device)
            node_state = torch.from_numpy(node_state.astype('float32')).to(self.device)
            mip_state = torch.from_numpy(mip_state.astype('float32')).to(self.device)

            mip_feature = np.array(self.searchTree.mip_feature)
            node_feature = np.array(self.searchTree.node_feature)
            var_feature = np.array(self.searchTree.var_feature)

            tree_feature = np.concatenate((node_feature, mip_feature, var_feature), axis=1).astype('float32')
            
            tree_feature = torch.from_numpy(tree_feature).to(self.device)

            edge_index = np.array(self.searchTree.edge_index).transpose().astype('float32').reshape(2, -1)
            edge_index = torch.LongTensor(edge_index).to(self.device)
            
            tree_data = TreeNodeData(tree_feature, edge_index, tree_feature.size(0))

            tree_dataset = GraphDataset([tree_data])
            graphDataLoader = torch_geometric.loader.DataLoader(tree_dataset, batch_size=1, shuffle=False)
            assert len(graphDataLoader) == 1
            tree_data = [tree for _, tree in enumerate(graphDataLoader)][0]

            # select action from the policy probs
            probs, _ = self.policy(cands_state_mat.unsqueeze(0), torch.zeros((1, cands_state_mat.size(0))).cuda(), node_state.unsqueeze(0), mip_state.unsqueeze(0), 
                                tree_batch=tree_data)
            probs = probs.squeeze()
            _, action = self.choose(probs)  # the chosen variable
            
            # define the SCIP branch var
            var = cands[action.item()]
            # branch on the selected variable (SCIP Variable object)
            self.model.branchVar(var)
            self.branch_count += 1


            if self.verbose:
                print('\tBranch count: {}. Selected var: {}.'.format(
                    self.branch_count, cands_pos[action.item()]))

            result = scip.SCIP_RESULT.BRANCHED
            if result == scip.SCIP_RESULT.BRANCHED:
                _, chosen_variable, *_ = self.model.getChildren()[0].getBranchInfos()
                assert chosen_variable is not None
                assert chosen_variable.isInLP()
                branch_cand_state = copy.deepcopy(cands_state_mat[cands_pos.index(chosen_variable.getCol().getLPPos()), :])
                
                self.searchTree.update(curr_scip_id, parent_scip_id, node_state.cpu().numpy(), mip_state.cpu().numpy(), branch_cand_state.cpu().numpy(), cands_state_mat.cpu().numpy(), result)

        return {'result': result}

    def finalize(self):
        pass

    def finalize_zero_branch(self):
        pass

""" SCIP branchers """


class SCIPCollectBrancher(Brancher):
    """
    Brancher to run SCIP data collection for imitation learning, with SCIPCollectEnv class.
    Instead of a single policy, 'explorer' and 'expert' rules are specified
    (each should be a string corresponding to a SCIP branching rule).
    The explorer policy runs for the top k branching decisions, then the expert takes over.
    Data is collected from expert decisions only.
    """
    def __init__(self, model, explorer, expert, k, state_dims, verbose):
        super(SCIPCollectBrancher, self).__init__()

        self.model = model
        self.explorer = explorer
        self.expert = expert
        self.k = k
        self.var_dim = state_dims['var_dim']
        self.node_dim = state_dims['node_dim']
        self.mip_dim = state_dims['mip_dim']
        self.verbose = verbose

        # counters and data structures
        self.branchexec_count = 0
        self.branch_count = 0
        self.explore = True
        self.explorer_count = 0
        self.collect_count = 0  # data collect counter
        self.collect_dict = OrderedDict()  # data dictionary to be filled with states and labels
        self.nnodes_list = []
        self.nnodesleft_list = []

        self.searchTree = Tree()
        
        self.count = 0

    def branchexeclp(self, allowaddcons):

        # determine whether explorer or expert should be run
        if self.branch_count < self.k:
            self.explore = True
        else:
            self.explore = False

        # get state representations
        node_state = self.model.getNodeState(self.node_dim).astype('float32')
        mip_state = self.model.getMIPState(self.mip_dim).astype('float32')
        cands, cands_pos, cands_state_mat = self.model.getCandsState(self.var_dim, self.branchexec_count)
        cands_state_mat.astype('float32')

        if self.explore:
            # branch with explorer
            assert isinstance(self.explorer, str)
            self.branchexec_count += 1
            self.nnodes_list.append(self.model.getNNodes())  # Retrieve the total number of processed nodes.
            self.nnodesleft_list.append(self.model.getNNodesLeft()) #  Return the number of nodes left (leaves + children + siblings).
            result = self.model.executeBranchRule(self.explorer, allowaddcons)
            
            if result == scip.SCIP_RESULT.BRANCHED:
                self.explorer_count += 1
                self.branch_count += 1
                _, chosen_variable, *_ = self.model.getChildren()[0].getBranchInfos()
                # chosen_variable is a SCIP Variable object
                assert chosen_variable is not None
                assert chosen_variable.isInLP()
                branch_cand_state = cands_state_mat[cands_pos.index(chosen_variable.getCol().getLPPos()), :]

                curr_scip_id = self.model.getCurrentNode().getNumber()
                parent_scip_id = None if self.model.getNNodes() == 1 else self.model.getCurrentNode().getParent().getNumber()

                self.searchTree.update(curr_scip_id, parent_scip_id, node_state, mip_state, branch_cand_state, cands_state_mat, result)

                if self.verbose:
                    print(curr_scip_id, parent_scip_id)
                    print('\tExplore count: {} (exec. {}).'.format(self.explorer_count, self.branchexec_count))
        
        else:
            # branch with expert
            assert isinstance(self.expert, str)
            self.branchexec_count += 1
            self.nnodes_list.append(self.model.getNNodes())
            self.nnodesleft_list.append(self.model.getNNodesLeft())
            result = self.model.executeBranchRule(self.expert, allowaddcons)

            if result == scip.SCIP_RESULT.BRANCHED:
                self.collect_count += 1
                self.branch_count += 1
                _, chosen_variable, *_ = self.model.getChildren()[0].getBranchInfos()
                # chosen_variable is a SCIP Variable object
                assert chosen_variable is not None
                assert chosen_variable.isInLP()

                branch_cand_state = cands_state_mat[cands_pos.index(chosen_variable.getCol().getLPPos()), :]

                curr_scip_id = self.model.getCurrentNode().getNumber()
                parent_scip_id = None if self.model.getNNodes() == 1 else self.model.getCurrentNode().getParent().getNumber()

                if self.verbose:
                    print(curr_scip_id, parent_scip_id)

                # print(self.searchTree.edge_index)
                # pdb.set_trace()
                self.collect_dict[self.collect_count] = {
                    'cands_state_mat': cands_state_mat,
                    'mip_state': mip_state,
                    'node_state': node_state,
                    'varLPpos': chosen_variable.getCol().getLPPos(),
                    'varRELpos': cands_pos.index(chosen_variable.getCol().getLPPos()),
                    'global_node_feature': copy.deepcopy(self.searchTree.node_feature),
                    'global_mip_feature': copy.deepcopy(self.searchTree.mip_feature),
                    'global_var_feature': copy.deepcopy(self.searchTree.var_feature),
                    'edge_index': copy.deepcopy(self.searchTree.edge_index)
                }
                
                self.searchTree.update(curr_scip_id, parent_scip_id, node_state, mip_state, branch_cand_state, cands_state_mat, result)

                if self.verbose:
                    print('\tBranch count: {} (exec. {}). '
                          'Selected varLPpos: {}. '
                          'Selected varRELpos: {}. '
                          'Num cands: {}'.format(self.branch_count, self.branchexec_count,
                                                 chosen_variable.getCol().getLPPos(),
                                                 cands_pos.index(chosen_variable.getCol().getLPPos()),
                                                 len(cands),
                                                 ))
                    
                    num_current_node = self.model.getCurrentNode().getNumber()
                    num_total = self.model.getNNodes()
                    print('branchext:{}, num_curr:{}, num_total:{}'.format(self.branchexec_count, num_current_node, num_total))
                    
        return {'result': result}

    def finalize(self):
        pass


class SCIPEvalBrancher(Brancher):
    """
    Brancher for SCIP evaluation run, with SCIPEvalEnv class.
    A single branching policy is specified (a string corresponding to a SCIP branching rule).
    """
    def __init__(self, model, policy, verbose):
        super(SCIPEvalBrancher, self).__init__()

        self.model = model
        self.policy = policy
        self.verbose = verbose

        # counters and data structures
        self.branchexec_count = 0
        self.branch_count = 0
        self.nnodes_list = []
        self.nnodesleft_list = []

    def branchexeclp(self, allowaddcons):

        # SCIP branching rule
        assert isinstance(self.policy, str)
        self.branchexec_count += 1
        self.nnodes_list.append(self.model.getNNodes())
        self.nnodesleft_list.append(self.model.getNNodesLeft())
        result = self.model.executeBranchRule(self.policy, allowaddcons)
        if result == scip.SCIP_RESULT.BRANCHED:
            self.branch_count += 1
            _, chosen_variable, *_ = self.model.getChildren()[0].getBranchInfos()
            assert chosen_variable is not None
            assert chosen_variable.isInLP()
            
            curr_scip_id = self.model.getCurrentNode().getNumber()
            parent_scip_id = None if self.model.getNNodes() == 1 else self.model.getCurrentNode().getParent().getNumber()
            # print(curr_scip_id, parent_scip_id)
            # pdb.set_trace()

            if self.verbose:
                print('\tBranch count: {} (exec. {}).'.format(self.branch_count, self.branchexec_count))

        return {'result': result}

    def finalize(self):
        pass
