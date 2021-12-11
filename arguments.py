"""argparser configuration"""

import argparse
import os

def add_TreeGate_config_args(parser):
    """TreeGate arguments"""

    group = parser.add_argument_group('model_treegate', 'TreeGate model configuration')
    group.add_argument('--depth', type=int, help='Depth of the TreeGate Network.')
    group.add_argument('--dim_reduce_factor', type=int, help='Dimension reduce factor of the branching policy network.')
    group.add_argument('--infimum', type=int, help='Infimum parameter of the branching policy network.')
    group.add_argument('--norm', type=str, default='none', help='Normalization type of the branching policy network.')
    
    return parser

def add_TBranT_config_args(parser):
    """TBranT arguments"""

    group = parser.add_argument_group('model_TBranT', 'TBranT model configuration')
    group.add_argument('--head_num', type=int, default=1, help='Number of heads of Transformer Encoder in T-BranT')
    group.add_argument('--layer_num', type=int, default=1, help='Number of Transformer Encoder layers in T-BranT')
    group.add_argument('--tree_gate', default=False, action='store_true', 
        help='Concatenate the tree features and candidate variable features.')
    group.add_argument('--graph', default=False, action='store_true', 
        help='Use the history search tree to obtain global tree features.')
    
    return parser
    

def add_training_args(parser):
    """Training arguments."""

    group = parser.add_argument_group('train', 'training configurations')
    group.add_argument('--train_batchsize', type=int, default=32, help='Training batchsize.')
    group.add_argument('--seed', type=int, default=0, help='Random seed for IL training experiment.')
    group.add_argument('--opt', default='adam', type=str, help='Type of optimizer to use.')
    group.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
    group.add_argument('--momentum', default=0.9, type=float, help='Momentum optimization parameter.')
    group.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay optimization parameter.')
    group.add_argument('--num_epochs', type=int, default=40, help='Number of training epochs.')
    group.add_argument('--top_k', type=int, nargs='+', default=[2, 5],
        help='In addition to top-1 generalization accuracy, we track top-k.'
    )
    group.add_argument('--use_gpu', default=False, action='store_true', help='Use gpu or not.')
    group.add_argument('--out_dir', type=str, default='./output', help='Directory to save the experimental results.')

    group.add_argument('--hidden_size', type=int, help='Hidden size of the branching policy network.')
    group.add_argument('--dropout', type=float, default=0.0, help='Dropout parameter for the branching policy network.')

    # TreeGate
    group.add_argument('--lr_decay_schedule', type=int, nargs='+', default=[20, 30],
        help='Learning rate decay schedule.')
    group.add_argument('--lr_decay_factor', type=float, default=0.1, help='LR decay factor.')

    # T-BranT
    group.add_argument('--noam', default=False, action='store_true', help='Use the Noam Scheduler.')
    group.add_argument('--warm_epochs', type=int, default=5, help='Warm epochs for Noam Scheduler.')

    return parser

def add_evaluation_args(parser):
    """Evaluation arguments."""

    group = parser.add_argument_group('validation', 'validation configurations')
    group.add_argument('--eval_batchsize', type=int, default=128, help='Evaluation batchsize.')

    return parser

def add_data_args(parser):
    """Train/valid/test data arguments."""

    group = parser.add_argument_group('data', 'data configurations')

    group.add_argument('--train_h5_path', type=str, help='Pathway to the train H5 file.')
    group.add_argument('--val_h5_path', type=str, help='Pathway to the val H5 file.')
    group.add_argument('--test_h5_path', type=str, help='Pathway to the test H5 file.')
    group.add_argument('--policy_type',
        type=str,
        choices=['TreeGatePolicy', 'TBranT', 'BranT'],
        help='Type of policy to use.')
    
    return parser

def get_args():
    """Parse all the args"""
    # parser definition
    parser = argparse.ArgumentParser(description='Parser for IL training experiments.')
    parser = add_TreeGate_config_args(parser)
    parser = add_TBranT_config_args(parser)
    parser = add_training_args(parser)
    parser = add_evaluation_args(parser)
    parser = add_data_args(parser)

    args = parser.parse_args()

    return args