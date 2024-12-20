#!/bin/bash

# export WANDB_MODE=offline
# specify which GPUs you want to use
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Get the hostname
hostname=$(hostname)

# Check if 'cbb24' is in the hostname
if [[ $hostname == *"cbb24"* ]]; then
    # conda activate synthclip
    num_gpus=8
    batch_size="--batch-size 64"
    ROOT_DATA_PATH=/scratch/datasets
    MODEL_CACHE_PATH=/fast_scratch/datasets/model-zoo
    IMAGENET_VAL_ROOT=$ROOT_DATA_PATH/imagenet1k/val_per-class
    LOG_DIR=/fast_scratch/datasets

# Check if 'dgx' is in the hostname
elif [[ $hostname == *"dgx"* ]]; then
    # conda activate synthclip-train
    num_gpus=4
    batch_size="--batch-size 256 --update-freq 4"
    ROOT_DATA_PATH=/home/nfs/y.ouali2022/data
    MODEL_CACHE_PATH=$HOME/model-zoo
    IMAGENET_VAL_ROOT=/nfs/datasets/imagenet1k/val
    LOG_DIR=~/slip_logs

# Check if 'ccn2' is in the hostname
elif [[ $hostname == *"ccn2"* ]]; then
    # conda activate synthclip-train
    num_gpus=8
    batch_size="--batch-size 256 --update-freq 2"
    ROOT_DATA_PATH=/scratch2/datasets
    CCXM_SUFIX=_wd
    MODEL_CACHE_PATH=/scratch2/model-zoo
    IMAGENET_VAL_ROOT=/scratch1/datasets/imagenet/images/val
    LOG_DIR=/scratch2
# Check if lambda nodes
elif [[ $hostname == *"gpu-node"* ]]; then
    # ln -s /fss/shared_saic-futureinteraction/users/v.castillo $HOME/fss-me
    # ln -s /fss/shared_saic-futureinteraction/users/data $HOME/fss-data
    # ln -s /fss/shared_saic-futureinteraction/users/v.castillo/miniconda3 $HOME/
    # source $HOME/miniconda3/bin/activate
    # conda activate synthclip-train
    WORK_DIR=$HOME/fss-me/synthclip
    num_gpus=4
    batch_size="4096 --update-freq 1"
    ROOT_DATA_PATH=/ephemeral/persistent-data
    MODEL_CACHE_PATH=/fss-me/model-zoo
    IMAGENET_VAL_ROOT=$ROOT_DATA_PATH/clip_eval/imagenet1k/val
    LOG_DIR=$HOME/fss-me/slip_logs
else
    echo "Hostname does not match any known patterns. Please check the hostname and update the script accordingly."
    exit 1
fi

if [[ -n $WORK_DIR ]]; then
    cd $WORK_DIR
fi
cd Training
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
# CCxM_NUM_SAMPLES=10500000
# CC3M
# CCxM_PATH=$ROOT_DATA_PATH"/cc3m${CCXM_SUFIX}/{000000000..000002876}.tar"
# CCxM_NUM_SAMPLES=2876999
# TRAIN_DATA_PARAMS="--train-data $CCxM_PATH --train-num-samples $CCxM_NUM_SAMPLES --dataset-type webdataset"
# EXP_NAME=vitb16-synthclip-ft
# TRAIN_PARAMS="--epochs 5 --lr-start 1e-6 --lr 5e-6 --lr-end 1e-7 --resume-only-weights 1 $batch_size"
# 8x RTX 2080 Ti (debuggin multiprocessing)
# TRAIN_PARAMS="--epochs 1 --lr-start 1e-6 --lr 5e-6 --lr-end 1e-7 --resume-only-weights 1 --batch-size 64"

# SynthCI
DATA_PATH=$ROOT_DATA_PATH/SynthCI-30/3m.csv
TRAIN_DATA_PARAMS="--train-data $DATA_PATH --dataset-type csv --workers 6"
EXP_NAME=vitb16-synthclip-scratch-3m
TRAIN_PARAMS="--epochs 40 --warmup-epochs 1 --batch-size $batch_size --lr 5e-4 --wd 0.5"

# Validation data
VAL_DATA_PARAMS="--imagenet-val-dir $IMAGENET_VAL_ROOT"

set -x
torchrun --nnodes=1 --nproc_per_node $num_gpus main.py \
    $TRAIN_DATA_PARAMS $TRAIN_PARAMS $VAL_DATA_PARAMS \
    --resume $MODEL_CACHE_PATH/synthclip-30m/checkpoint_best.pt \
    --output-dir $LOG_DIR/$EXP_NAME --wandb
    # --output-dir $LOG_DIR/$(date +%Y-%m-%d-%H-%M-%S)_$EXP_NAME --save-images

# Debug
# CUDA_VISIBLE_DEVICES=0 python -m ipdb main.py \
#     $TRAIN_DATA_PARAMS $TRAIN_PARAMS $VAL_DATA_PARAMS \
#     --resume $MODEL_CACHE_PATH/synthclip-30m/checkpoint_best.pt \
#     --output-dir $LOG_DIR/debug_$EXP_NAME\
#     --batch-size 64 # -j 0
