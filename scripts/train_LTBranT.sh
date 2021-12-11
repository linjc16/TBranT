TRAIN_DATA_PATH="/data/zjl/data_graph-para-add/train.h5"
VAL_DATA_PATH="/data/zjl/data_graph-para-add/valid.h5"
TEST_DATA_PATH="/data/zjl/66_test/test_500_50.h5"
OUT_DIR="/data/ljc/pf_exp/pf_transformer_80_8_s0_gathead2_galmatch_2layer_var_noam5_nograph"

CUDA_VISIBLE_DEVICES=5 python train_test_IL.py \
    --policy_type TBranT \
    --seed 0 \
    --train_batchsize 20 \
    --eval_batchsize 20 \
    --num_epochs 100 \
    --use_gpu  \
    --noam \
    --warm_epochs 5 \
    --head_num 8 \
    --lr 1e-3 \
    --tree_gate \
    --layer_num 2 \
    --opt adamw \
    --hidden_size 80 \
    --train_h5_path  ${TRAIN_DATA_PATH} \
    --val_h5_path  ${VAL_DATA_PATH} \
    --test_h5_path ${TEST_DATA_PATH} \
    --out_dir ${OUT_DIR}