CUDA_VISIBLE_DEVICES=1 python evaluate_IL.py \
    --seeds 0 1 2 3 4 \
    --checkpoint checkpoints/LTBranT/final_checkpoint.pth.tar \
    --setting sandbox \
    --out_dir ./data/train \
    --instances_dir /data/TBranT-dataset/train_instances/ \
    --cutoff_dict /data/TBranT-dataset/cutoff_train.pkl \
    --njobs 2