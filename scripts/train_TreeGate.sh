TRAIN_DATA_PATH="/data/zjl/data_graph-para-add/train.h5"
VAL_DATA_PATH="/data/zjl/data_graph-para-add/valid.h5"
TEST_DATA_PATH="/data/zjl/66_test/test_500_50.h5"
OUT_DIR="/data/ljc/pf_output_treegate_s0_5e-3_64"


CUDA_VISIBLE_DEVICES=6 python train_test_IL.py \
    --policy_type TreeGatePolicy \
    --depth 5 \
    --dim_reduce_factor 2 \
    --infimum 8 \
    --seed 777 \
    --train_batchsize 32 \
    --eval_batchsize 128 \
    --num_epochs 40 \
    --use_gpu  \
    --lr 5e-3 \
    --lr_decay_factor 0.1 \
    --lr_decay_schedule 20 30 \
    --opt adam \
    --hidden_size 64 \
    --train_h5_path  ${TRAIN_DATA_PATH} \
    --val_h5_path  ${VAL_DATA_PATH} \
    --test_h5_path ${TEST_DATA_PATH} \
    --out_dir ${OUT_DIR}