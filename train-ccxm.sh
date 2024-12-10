#!/bin/bash
cd Training

# specify which GPUs you want to use.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ROOT_DATA_PATH=/scratch/datasets  # cbb24
MODEL_CACHE_PATH=/fast_scratch/datasets/model-zoo  # cbb24
# CC12M
# CCxM_PATH=$ROOT_DATA_PATH"/cc12m/{000000000..000010503}.tar"
# CCxM_NUM_SAMPLES=
# CC3M
CCxM_PATH=$ROOT_DATA_PATH"/cc3m/{000000000..000002876}.tar"
CCxM_NUM_SAMPLES=2876999
TRAIN_DATA_PARAMS="--train-data $CCxM_PATH --train-num-samples $CCxM_NUM_SAMPLES --dataset-type webdataset"

TRAIN_PARAMS="--epochs 5 --lr-start 1e-6 --lr 5e-6 --lr-end 1e-7 --resume-only-weights 1 --batch-size 512"
TRAIN_PARAMS="--epochs 1 --lr-start 1e-6 --lr 5e-6 --lr-end 1e-7 --resume-only-weights 1 --batch-size 512"
VAL_DATA_PARAMS="--imagenet-val-dir $ROOT_DATA_PATH/imagenet1k/val_per-class"

torchrun --nnodes=1 --nproc_per_node 8 --nnodes=1 main.py \
    $TRAIN_DATA_PARAMS $TRAIN_PARAMS $VAL_DATA_PARAMS \
    --resume $MODEL_CACHE_PATH/synthclip-30m/checkpoint_best.pt \
    --output-dir /fast_scratch/datasets/slip_logs/vitb16-synthclip-ft

# Debug
# CUDA_VISIBLE_DEVICES=0 python -m ipdb main.py \
#     $TRAIN_DATA_PARAMS $TRAIN_PARAMS $VAL_DATA_PARAMS \
#     --resume $MODEL_CACHE_PATH/synthclip-30m/checkpoint_best.pt \
#     --output-dir /fast_scratch/datasets/slip_logs/vitb16-synthclip-ft \
#     --batch-size 64 # -j 0
