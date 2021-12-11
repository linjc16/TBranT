""" Converter to transform collected pickle data files to IL inputs. """

import h5py
import numpy as np
import pickle
import argparse
import os
import glob
import torch
import pdb

from utils import STATE_DIMS


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Converter to transform collected pickle data files to IL inputs.')
    parser.add_argument(
        '--pkl_file_dir',
        type=str,
        default='./final_train/',
        help='Pathway to the directory containing all the pkl data collect files.'
    )
    parser.add_argument(
        '--dataset_mode',
        type=str,
        default='train',
        help='Denotes train or val/test data.'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='./data',
        help='Directory to save the h5 and pkl files.'
    )
    parser.add_argument(
        '-s',
        '--seed', 
        type=int, 
        default=0, 
        help='Random seed.'
    )
    args = parser.parse_args()

    # set the NumPy random seed
    np.random.seed(args.seed)
    
    # stack all the state_vectors into a single matrix
    pkl_paths = sorted(glob.glob(args.pkl_file_dir + '/**/*_data.pkl',  recursive=True))

    state_vectors = np.zeros((0, STATE_DIMS['node_dim'] + STATE_DIMS['mip_dim'])).astype('float32')
    num_data = 0
    for index, pkl in enumerate(pkl_paths):
        with open(pkl, 'rb') as f:
            ins_name=pkl.split('/')[-1][:-13]
            print('\tProcessing {:s}, {:d} of {:d}...'.format(pkl.split('/')[-1], index + 1, len(pkl_paths)))
            # pdb.set_trace()
            try:
                D = pickle.load(f)
                if len(D) > 0:  # We only collect data with more than 1 candidate
                    non_trivial_cands_keys = [key for key in D if D[key]['cands_state_mat'].shape[0] > 1 and len(D[key]['global_mip_feature']) > 0]
                    if len(non_trivial_cands_keys) <= 500:
                        num_data += len(non_trivial_cands_keys)
                    else:
                        num_data += 500
            except:
                pass
    print('Processed {:d} datapoints...'.format(num_data))

    # create an h5 file
    print('Creating an h5 file...')
    f = h5py.File(os.path.join(args.out_dir, '{}.h5'.format(args.dataset_mode)), 'w')
    dt = h5py.special_dtype(vlen=np.dtype('float32'))
    dataset = f.create_dataset('dataset', (num_data,), dtype=dt)
    counter = 0
    
    for index, pkl in enumerate(pkl_paths):
        with open(pkl, 'rb') as f:
            ins_name=pkl.split('/')[-1][:-13]
            print('\tProcessing {:s}, {:d} of {:d}...'.format(pkl.split('/')[-1], index + 1, len(pkl_paths)))
            try:
                D = pickle.load(f)
                non_trivial_cands_keys = [key for key in D if D[key]['cands_state_mat'].shape[0] > 1 and len(D[key]['global_mip_feature']) > 0]
                print(pkl.split('/')[-1])
                
                if len(non_trivial_cands_keys)>0 and len(non_trivial_cands_keys) <= 500:
                    lend=non_trivial_cands_keys
                elif len(non_trivial_cands_keys) > 500:
                    lend=np.array(non_trivial_cands_keys)
                    # print(lend)
                    np.random.shuffle(lend)
                    lend=lend[:500]

                for idx in D:
                    if idx in lend:
                        if D[idx]['cands_state_mat'].shape[0] > 1 and len(D[idx]['global_mip_feature']) > 0:  # We only collect data with more than 1 candidate
                            # flat_vector is always [target, node, mip, grid_flattened]
                            flat_vector = np.hstack([D[idx]['varRELpos'], D[idx]['node_state'], D[idx]['mip_state'], D[idx]['cands_state_mat'].flatten().shape[0],
                                                    D[idx]['cands_state_mat'].flatten()]).astype('float32')
                            edge_index = np.array(D[idx]['edge_index']).transpose().astype('float32').reshape(2, -1)
                            
                            mip_feature = np.array(D[idx]['global_mip_feature']).reshape(-1, STATE_DIMS['mip_dim'])
                            node_feature = np.array(D[idx]['global_node_feature']).reshape(-1, STATE_DIMS['node_dim'])
                            var_feature = np.array(D[idx]['global_var_feature']).reshape(-1, STATE_DIMS['var_dim'])

                            tree_feature = np.concatenate((node_feature, mip_feature, var_feature), axis=1).astype('float32')
                            
                            flat_vector = np.hstack([flat_vector, tree_feature.flatten().shape[0], tree_feature.flatten(), edge_index.shape[1], edge_index.flatten()])
                            # pdb.set_trace()
                            dataset[counter] = flat_vector
                            counter += 1
                        # pdb.set_trace()
            except:
                pass
    print('Processed {:d} datapoints.'.format(counter))