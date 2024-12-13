#!/bin/bash
cd Training
export WANDB_MODE=offline

# specify which GPUs you want to use.
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# Get the hostname
hostname=$(hostname)

# Check if 'cbb24' is in the hostname
if [[ $hostname == *"cbb24"* ]]; then
    # conda activate synthclip
    num_gpus=8
    batch_size=64
    ROOT_DATA_PATH=/scratch/datasets
    MODEL_CACHE_PATH=/fast_scratch/datasets/model-zoo
    IMAGENET_VAL_ROOT=$ROOT_DATA_PATH/imagenet1k/val_per-class
    LOG_DIR=/fast_scratch/datasets

# Check if 'dgx' is in the hostname
elif [[ $hostname == *"dgx"* ]]; then
    # conda activate synthclip-train
    num_gpus=8
    batch_size=256
    ROOT_DATA_PATH=/home/nfs/y.ouali2022/data
    MODEL_CACHE_PATH=$HOME/model-zoo
    IMAGENET_VAL_ROOT=/nfs/datasets/imagenet1k/val
    LOG_DIR=../

# Check if 'ccn2' is in the hostname
elif [[ $hostname == *"ccn2"* ]]; then
    # conda activate synthclip-train
    num_gpus=10
    batch_size=416
    ROOT_DATA_PATH=/scratch2/datasets
    CCXM_SUFIX=_wd
    MODEL_CACHE_PATH=/scratch2/model-zoo
    IMAGENET_VAL_ROOT=/scratch1/datasets/imagenet/images/val
    LOG_DIR=/scratch2
else
    echo "Hostname does not match any known patterns. Please check the hostname and update the script accordingly."
    exit 1
fi

# Print the environment variables for verification
echo "ROOT_DATA_PATH: $ROOT_DATA_PATH"
echo "MODEL_CACHE_PATH: $MODEL_CACHE_PATH"
echo "IMAGENET_VAL_ROOT: $IMAGENET_VAL_ROOT"
echo "LOG_DIR: $LOG_DIR"
if [[ -n $CCXM_SUFIX ]]; then
    echo "CCXM_SUFIX: $CCXM_SUFIX"
fi

# CC12M
# CCxM_PATH=$ROOT_DATA_PATH"/cc12m/{000000000..000010503}.tar"
# CCxM_NUM_SAMPLES=
# CC3M
CCxM_PATH=$ROOT_DATA_PATH"/cc3m${CCXM_SUFIX}/{000000000..000002876}.tar"
CCxM_NUM_SAMPLES=2876999
TRAIN_DATA_PARAMS="--train-data $CCxM_PATH --train-num-samples $CCxM_NUM_SAMPLES --dataset-type webdataset"

TRAIN_PARAMS="--epochs 5 --lr-start 1e-6 --lr 5e-6 --lr-end 1e-7 --resume-only-weights 1 --batch-size $batch_size"
# 8x RTX 2080 Ti (debuggin multiprocessing)
# TRAIN_PARAMS="--epochs 1 --lr-start 1e-6 --lr 5e-6 --lr-end 1e-7 --resume-only-weights 1 --batch-size 64"

VAL_DATA_PARAMS="--imagenet-val-dir $IMAGENET_VAL_ROOT"

torchrun --nnodes=1 --nproc_per_node $num_gpus main.py \
    $TRAIN_DATA_PARAMS $TRAIN_PARAMS $VAL_DATA_PARAMS \
    --resume $MODEL_CACHE_PATH/synthclip-30m/checkpoint_best.pt \
    --output-dir $LOG_DIR/slip_logs/vitb16-synthclip-ft --wandb

# Debug
# CUDA_VISIBLE_DEVICES=0 python -m ipdb main.py \
#     $TRAIN_DATA_PARAMS $TRAIN_PARAMS $VAL_DATA_PARAMS \
#     --resume $MODEL_CACHE_PATH/synthclip-30m/checkpoint_best.pt \
#     --output-dir /fast_scratch/datasets/slip_logs/vitb16-synthclip-ft \
#     --batch-size 64 # -j 0
