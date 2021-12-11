""" Imitation Learning (IL) evaluations, using SCIP. """

import os
import argparse
import pickle

import numpy as np
import torch
import pyscipopt as scip

from src.environments import ILEvalEnv
from models.feedforward import TreeGatePolicy
from models.transformer import BranchFormer, BranT

import multiprocessing as mp

from utils import STATE_DIMS

import pdb

import faulthandler
faulthandler.enable()

# solver parametric setting, key ('sandbox' or 'default') to be specified in argparse --setting
SETTINGS = {
    'sandbox': {
        'heuristics': False,        # enable primal heuristics
        'cutoff': True,             # provide cutoff (value needs to be passed to the environment)
        'conflict_usesb': False,    # use SB conflict analysis
        'probing_bounds': False,    # use probing bounds identified during SB
        'checksol': False,
        'reevalage': 0,
    },
    'default': {
        'heuristics': True,
        'cutoff': False,
        'conflict_usesb': True,
        'probing_bounds': True,
        'checksol': True,
        'reevalage': 10,
    },
}

# limits in solvers
LIMITS = {
    'node_limit': -1,
    'time_limit': 3600.,
}

def get_policy(args):
    # load a checkpoint path (cpu load of a gpu checkpoint)
    chkpnt = torch.load(args.checkpoint, map_location='cpu')
    print('Checkpoint loaded from path {}...'.format(args.checkpoint))
    
    # read config from checkpoint: the policy parameters are inferred from the checkpoint args
    checkpoint_args = chkpnt['args']

    print(checkpoint_args)
    if checkpoint_args.policy_type == 'TreeGatePolicy':
        policy = TreeGatePolicy(
            var_dim=STATE_DIMS['var_dim'],
            node_dim=STATE_DIMS['node_dim'],
            mip_dim=STATE_DIMS['mip_dim'],
            hidden_size=checkpoint_args.hidden_size,
            depth=checkpoint_args.depth,
            dropout=checkpoint_args.dropout,
            dim_reduce_factor=checkpoint_args.dim_reduce_factor,
            infimum=checkpoint_args.infimum,
            norm=checkpoint_args.norm,
        )
        policy_name = 'TreeGatePolicy'
    elif checkpoint_args.policy_type == 'TBranT':
        policy = BranchFormer(
            var_dim=STATE_DIMS['var_dim'],
            node_dim=STATE_DIMS['node_dim'],
            mip_dim=STATE_DIMS['mip_dim'],
            hidden_size=checkpoint_args.hidden_size,
            dim_feedforward=checkpoint_args.hidden_size,
            nhead=checkpoint_args.head_num,
            num_encoder_layers=checkpoint_args.layer_num,
            tree_gate=checkpoint_args.tree_gate,
            graph=checkpoint_args.graph,
        )
        policy_name = 'Transformer'
    elif checkpoint_args.policy_type == 'BranT':
        policy = BranT(
            var_dim=STATE_DIMS['var_dim'],
            node_dim=STATE_DIMS['node_dim'],
            mip_dim=STATE_DIMS['mip_dim'],
            hidden_size=checkpoint_args.hidden_size,
            dim_feedforward=checkpoint_args.hidden_size,
            nhead=checkpoint_args.head_num,
            num_encoder_layers=checkpoint_args.layer_num,
            tree_gate=checkpoint_args.tree_gate,
        )
        policy_name = 'Transformer'
    else:
        raise ValueError('A valid policy should be set.')
    
    policy.eval()
    policy.load_state_dict(chkpnt['state_dict'])
    policy = policy.to(args.device)

    return policy, policy_name, checkpoint_args

def sub_worker(args, instance_names):

    policy, policy_name, checkpoint_args = get_policy(args)
    # torch.cuda.set_device(args.gpu_id)
    for seed in args.seeds:
        args.seed = seed
        for instance_name in instance_names:
            neural_eval_instances(args, instance_name, policy, policy_name, checkpoint_args)

# Neural Policies
def neural_eval_instances(args, instance_name, policy, policy_name, checkpoint_args):

    instance_file_path = os.path.join(args.instances_dir, instance_name)  # name contains extension mps.gz
    name = instance_name.split('.')[0]

    # get cutoff
    cutoff_dict = pickle.load(open(args.cutoff_dict, 'rb'))
    assert name in cutoff_dict

    # setup the environment and a policy within it
    # pdb.set_trace()
    env = ILEvalEnv(device=args.device, graph=checkpoint_args.graph)

    # main evaluation
    with torch.no_grad():
        exp_dict = env.run_episode(
            instance=instance_file_path,
            name=name,
            policy=policy,
            policy_name=policy_name,
            state_dims=STATE_DIMS,
            scip_seed=args.seed,
            cutoff_value=cutoff_dict[name],
            scip_limits=LIMITS,
            scip_params=SETTINGS[args.setting],
            verbose=args.verbose,
        )

    # dump the exp_dict
    f = open(os.path.join(args.out_dir, '{}_{}_ILEval_info.pkl'.format(name, args.seed)), 'wb')
    pickle.dump(exp_dict, f)
    f.close()

if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn')
    # parser definition
    parser = argparse.ArgumentParser(description='Parser for IL evaluation experiments.')
    parser.add_argument(
        '-c',
        '--checkpoint',
        type=str,
        help='Pathway to torch checkpoint to be loaded.'
    )
    parser.add_argument(
        '--cutoff_dict',
        type=str,
        help='Pathway to pickled dictionary containing cutoff values.'
    )
    parser.add_argument(
        '--instances_dir',
        type=str,
        help='Pathway to the MILP instances.'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='./output_IL_transformer_graph',
        help='Pathway to save all the SCIP eval pickle files.'
    )
    parser.add_argument(
        '-s',
        '--seeds',
        type=int,
        nargs='+',
        default=[0, 1, 2, 3, 4],
        help='Random seed for SCIP solver.'
    )
    parser.add_argument(
        '--setting',
        type=str,
        default='sandbox',
        help='Solver parameters setting.'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        default=False,
        help='Flag on verbosity.'
    )
    parser.add_argument(
        '-j', '--njobs',
        type=int,
        help='Number of parallel jobs.',
    )

    args = parser.parse_args()

    # set device (cpu for eval)
    args.device = torch.device('cuda')

    instance_names = os.listdir(args.instances_dir)

    mp = mp.get_context('spawn')
    pool = []
    total_num = len(instance_names)
    interval = total_num // args.njobs + 1
    # pdb.set_trace()
    for i in range(args.njobs):
        instances_names_sub = instance_names[i * interval:(i + 1) * interval]
        process = mp.Process(target=sub_worker, args=(args, instances_names_sub)) 
        process.start()
        pool.append(process)

    for p in pool:
        p.join()
    
    for p in pool:
        p.terminate()

    # sub_worker(args, instance_names)
    # n_gpu = torch.cuda.device_count()
