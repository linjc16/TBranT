# Learning to Branch with Tree-aware Branching Transformers
This repository is the official implementation of [Learning to Branch with Tree-aware Branching Transformers]().
## Requirements
- We use [SCIP]((https://scip.zib.de/index.php#download)) as the backend solver. To install SCIP, see installation instructions [here](SCIP_INSTALL.md). 
- All other requirements are in `conda_requirements.txt`.
## Dataset
The `T-BranT` dataset can be downloaded [here](https://data.mendeley.com/datasets/8msnxmvdgp/1).

Our dataset consists of the following files:
- `train.h5`: a H5 file containing all the training samples.
- `val.h5`: a H5 file containing all the validation samples.
- `test.h5`: a H5 containing all the testing samples.
- `train_instances/`: a directory containing the 25 training MILP instances.
- `test_instances/`: a directory containing 66 testing MILP instances.
- `cutoff_train.pkl`: a pickle file containing the cutoff values for the training instances.
- `cutoff_test.pkl`: a pickle file containing the cutoff values for the testing instances.

### Data collection
- Download the `T-BranT` dataset. 
- Run the following script for collecting training samples. Note that `out_dir, instances_dir, cutoff_dict` need to be changed to your local path. You may also change the `njobs` according to your available hardware.
```
$ bash scripts/run_collect_train.sh
```
- Likewise, run the following scripts for collecting validation and testing samples.
```
$ bash scripts/run_collect_val.sh
$ bash scripts/run_collect_test.sh
```
### HDF5 creation
Once we collect all train/val/test expert samples, we convert all the collected pickle files into a single H5 file. Run the following script:
```convert_to_h5
$ bash scripts/generate_hdf5.sh
```

## Training

- To train our `T-BranT` models in the paper, run the following script for training. Note that `TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, OUT_DIR` need to be changed to your local path. You may also change the `train_batchsize` and `eval_batchsize` according to your available hardware.

```train_tbrant
$ bash scripts/train_TBranT.sh
```
- Similary, run the following scripts for training `LT-BranT`, `BranT` and `TreeGate`.

```train_treegate
$ bash scripts/train_LTBranT.sh
$ bash scripts/train_BranT.sh
$ bash scripts/train_TreeGate.sh
```

## Evaluation

- To evaluate the models on the MILP datasets, for SCIP policies, run the following script. Note that `policy, out_dir, instances_dir, cutoff_dict` need to be modified adaptively.

```eval_SCIP
$ bash scripts/eval_scip.sh
```

- For Neural policies, run the following script. Change `checkpoint` according to the policies.
```eval_neural
$ bash scripts/eval_neural.sh
```
## Results
See more experimental details in our paper. For instance-specific results, refer to folder `results/`.
### 48 easier instances

The performance on 48 easier instances are shown as follows. Bold numbers denote the best results of the neural policies.

|                    | Nodes  | Fair Nodes |
| ------------------ |---------------- | -------------- |
| T-BranT   |     **1886.08**         |      **1944.02**       |
| [TreeGate]((https://github.com/ds4dm/branch-search-trees))  | 2371.81 | 2442.86|
| pscost    | 2857.16 | 2857.16|
| relpscost | 930.46  | 1617.82|
| random    | 12844.99| 16205.81|



### 18 harder instances
The performance on 18 harder instances are shown as follows. Bold numbers denote the best results of the neural policies.
|                    | Integral  | Gap |
| ------------------ |---------------- | -------------- |
| T-BranT   |     **9606.06**         |      **0.0684**       |
| TreeGate  | 10929.07 | 0.1139|
| pscost    | 16445.60 | 0.4490|
| relpscost | 7254.43  | 0.0679|
| random    | 21695.67| 0.4711|


## Acknowledgement
- Our implementation is partly based on Zarpellon's [code](https://github.com/ds4dm/branch-search-trees).
- We use [SCIP 6.0.1](https://scip.zib.de/index.php#download) and further a customized version of [PySCIPOpt](https://github.com/ds4dm/PySCIPOpt/tree/branch-search-trees) as our backend solver.

## Contact
Please feel free to submit a Github issue if you have any questions or find any bugs. We do not guarantee any support, but will do our best if we can help.
