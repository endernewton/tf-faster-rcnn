#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

ROOT_FASTER=$1
GPU_ID=$2
DATASET=$3
NET=$4
CLASSES=$5
TRAIN_IMDB="${DATASET}_trainval"
TEST_IMDB="${DATASET}_test"
ITERS=$6
STEPSIZE=$7
ANCHORS=$8
RATIOS=$9
CFG_FILE=${10}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:10:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

LOG="${ROOT_FASTER}/experiments/logs/test_${NET}_${TRAIN_IMDB}_${CFG_FILE}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  NET_FINAL=${ROOT_FASTER}/output/${CFG_FILE}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.ckpt
else
  NET_FINAL=${ROOT_FASTER}/output/${CFG_FILE}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_iter_${ITERS}.ckpt
fi
set -x

if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} python ${ROOT_FASTER}/tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg ${ROOT_FASTER}/experiments/cfgs/${CFG_FILE}.yml \
    --classes ${CLASSES} \
    --tag ${EXTRA_ARGS_SLUG} \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} ${EXTRA_ARGS}
else
  CUDA_VISIBLE_DEVICES=${GPU_ID} python ${ROOT_FASTER}/tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg ${ROOT_FASTER}/experiments/cfgs/${CFG_FILE}.yml \
    --classes ${CLASSES} \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} ${EXTRA_ARGS}
fi
