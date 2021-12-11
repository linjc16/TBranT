python collect.py \
    --seeds 0 1 2 3 \
    --k_nodes 0 1 5 10 15 \
    --setting sandbox \
    --out_dir ./data/train \
    --instances_dir /data/TBranT-dataset/train_instances/ \
    --cutoff_dict /data/TBranT-dataset/cutoff_train.pkl \
    --njobs 8