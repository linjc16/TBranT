TRAIN_DATA_PATH="/data/zjl/data_graph-para-add/train.h5"
VAL_DATA_PATH="/data/zjl/data_graph-para-add/valid.h5"
TEST_DATA_PATH="/data/zjl/66_test/test_500_50.h5"
OUT_DIR="checkpoints/TBranT"

CUDA_VISIBLE_DEVICES=5 python train_test_IL.py \
    --policy_type TBranT \
    --seed 777 \
    --train_batchsize 16 \
    --eval_batchsize 20 \
    --num_epochs 100 \
    --use_gpu  \
    --noam \
    --warm_epochs 5 \
    --head_num 8 \
    --lr 5e-4 \
    --tree_gate \
    --layer_num 2 \
    --opt adamw \
    --graph \
    --hidden_size 80 \
    --train_h5_path  ${TRAIN_DATA_PATH} \
    --val_h5_path  ${VAL_DATA_PATH} \
    --test_h5_path ${TEST_DATA_PATH} \
    --out_dir ${OUT_DIR}