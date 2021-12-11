""" Collect MILP data for Imitation Learning (IL), by performing SCIP roll-outs. """

import os
import argparse
import pickle
import pyscipopt as scip

from src.environments import SCIPCollectEnv

from utils import STATE_DIMS

import multiprocessing as mp

import faulthandler
faulthandler.enable()

# solver parametric setting, key ('sandbox' or 'default') to be specified in argparse --setting
SETTINGS = {
    'sandbox': {
        'heuristics': False,        # enable primal heuristics
        'cutoff': True,             # provide cutoff (value needs to be passed to the environment)
        'conflict_usesb': False,    # use SB conflict analysis
        'probing_bounds': False,    # use probing bounds identified during SB
        'checksol': False,          # check LP solutions found during strong branching with propagation
        'reevalage': 0,             # number of intermediate LPs solved to trigger reevaluation of SB value
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

# collection branching rules
COLLECTORS = {
    'explorer': 'random',
    'expert': 'relpscost'
}



def make_samples(args, instance_name):
    # setup output directory and path to instance
    outfile_dir = os.path.join(args.out_dir, 'SCIPCollect_{}_{}_{}'.format(
        args.setting, args.seed, args.k_node
    ))

    os.makedirs(outfile_dir, exist_ok=True)
    
    instance_file_path = os.path.join(args.instances_dir, instance_name)  # name contains extension mps.gz
    name = instance_name.split('.')[0]

    # get cutoff
    cutoff_dict = pickle.load(open(args.cutoff_dict, 'rb'))
    assert name in cutoff_dict

    # setup the environment and collect data
    env = SCIPCollectEnv()
    exp_dict, collect_dict = env.run_episode(
        instance=instance_file_path,
        name=name,
        explorer=COLLECTORS['explorer'],
        expert=COLLECTORS['expert'],
        k=args.k_node,
        state_dims=STATE_DIMS,
        scip_seed=args.seed,
        cutoff_value=cutoff_dict[name],
        scip_limits=LIMITS,
        scip_params=SETTINGS[args.setting],
        verbose=args.verbose,
    )

    # dump the dictionaries
    f = open(os.path.join(outfile_dir, '{}_{}_{}_info.pkl'.format(name, args.seed, args.k_node)), 'wb')
    pickle.dump(exp_dict, f)
    f.close()

    ff = open(os.path.join(outfile_dir, '{}_{}_{}_data.pkl'.format(name, args.seed, args.k_node)), 'wb')
    pickle.dump(collect_dict, ff)
    ff.close()


if __name__ == '__main__':

    # parser definition
    parser = argparse.ArgumentParser(description='Parser for SCIP data collection.')
    parser.add_argument(
        '-s',
        '--seeds',
        type=int,
        nargs='+',
        default=[0, 1, 2, 3, 4],
        help='Random seed for SCIP solver.'
    )
    parser.add_argument(
        '-k',
        '--k_nodes',
        type=int,
        nargs='+',
        default=[0,1,5,10,15],
        help='Number of initial nodes to be explored randomly, before starting data collection.'
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
        '--out_dir',
        type=str,
        help='Path to output directory',
    )
    parser.add_argument(
        '--instances_dir',
        type=str,
        help='Path to MILP instances',
    )
    parser.add_argument(
        '--cutoff_dict',
        type=str,
        help='Path to pickled dictionary containing cutoff values'
    )
    parser.add_argument(
        '-j', '--njobs',
        type=int,
        help='Number of parallel jobs.',
    )

    args = parser.parse_args()

    instance_names = os.listdir(args.instances_dir)
    
    pool = mp.Pool(processes=args.njobs)

    for k_node in args.k_nodes:
        args.k_node = k_node
        for seed in args.seeds:
            args.seed = seed
            for instance_name in instance_names:
                pool.apply_async(make_samples, (args, instance_name))

    pool.close()
    pool.join()
    pool.terminate()
