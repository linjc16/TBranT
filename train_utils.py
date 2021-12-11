
import numpy as np
import torch.optim as optim
from models.feedforward import TreeGatePolicy
from models.transformer import BranchFormer, BranT
from utils import STATE_DIMS

def get_scheduler(args, optimizer):
    # specify a learning rate scheduler
    if args.lr_decay_schedule:
        if args.noam:
            from noam import NoamLR
            scheduler = NoamLR(optimizer, args.warm_epochs)
        else:
            lr_decay_schedule = args.lr_decay_schedule
            lr_decay_factor = args.lr_decay_factor
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_decay_schedule, lr_decay_factor)
        use_scheduler = True
    else:
        use_scheduler = False
    
    return scheduler, use_scheduler

def get_optimizer(args, policy):
    if args.opt == 'adam':
        optimizer = optim.Adam(
            policy.parameters(),
            lr=args.lr,
            betas=(args.momentum, 0.999),
            weight_decay=args.weight_decay
        )
        eps = np.finfo(np.float32).eps.item()
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(
            policy.parameters(),
            lr=args.lr,
            betas=(args.momentum, 0.999),
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError('A valid optimizer should be set.')

    return optimizer

def get_dataset(args, dataset_h5):

    train_h5 = dataset_h5(
        h5_file=args.train_h5_path,
        node_dim=STATE_DIMS['node_dim'],
        mip_dim=STATE_DIMS['mip_dim'],
        var_dim=STATE_DIMS['var_dim']
    )

    val_h5 = dataset_h5(
        h5_file=args.val_h5_path,
        node_dim=STATE_DIMS['node_dim'],
        mip_dim=STATE_DIMS['mip_dim'],
        var_dim=STATE_DIMS['var_dim']
    )

    test_h5 = dataset_h5(
        h5_file=args.test_h5_path,
        node_dim=STATE_DIMS['node_dim'],
        mip_dim=STATE_DIMS['mip_dim'],
        var_dim=STATE_DIMS['var_dim']
    )
    
    return train_h5, val_h5, test_h5


def get_policy(args):
    # setup the policy
    if args.policy_type == 'TreeGatePolicy':
        policy = TreeGatePolicy(
            var_dim=STATE_DIMS['var_dim'],
            node_dim=STATE_DIMS['node_dim'],
            mip_dim=STATE_DIMS['mip_dim'],
            hidden_size=args.hidden_size,
            depth=args.depth,
            dropout=args.dropout,
            dim_reduce_factor=args.dim_reduce_factor,
            infimum=args.infimum,
            norm=args.norm,
        )
        policy_name = 'TreeGatePolicy'
    elif args.policy_type == 'TBranT':
        policy = BranchFormer(
            var_dim=STATE_DIMS['var_dim'],
            node_dim=STATE_DIMS['node_dim'],
            mip_dim=STATE_DIMS['mip_dim'],
            hidden_size=args.hidden_size,
            dim_feedforward=args.hidden_size,
            nhead=args.head_num,
            num_encoder_layers=args.layer_num,
            tree_gate=args.tree_gate,
            graph=args.graph,
        )
        policy_name = 'Transformer'
    elif args.policy_type == 'BranT':
        policy = BranT(
            var_dim=STATE_DIMS['var_dim'],
            node_dim=STATE_DIMS['node_dim'],
            mip_dim=STATE_DIMS['mip_dim'],
            hidden_size=args.hidden_size,
            dim_feedforward=args.hidden_size,
            nhead=args.head_num,
            num_encoder_layers=args.layer_num,
            tree_gate=args.tree_gate,
        )
        policy_name = 'Transformer'
    else:
        raise ValueError('A valid policy should be set.')
    
    return policy, policy_name