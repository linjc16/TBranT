python evaluate_SCIP.py \
    --seeds 0 1 2 3 4 \
    --policy relpscost \
    --setting sandbox \
    --out_dir ./data/train \
    --instances_dir /data/TBranT-dataset/train_instances/ \
    --cutoff_dict /data/TBranT-dataset/cutoff_train.pkl \
    --njobs 8