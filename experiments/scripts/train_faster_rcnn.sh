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

LOG="${ROOT_FASTER}/experiments/logs/${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo ${EXTRA_ARGS}

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
    NET_FINAL=${ROOT_FASTER}/output/${CFG_FILE}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.ckpt
else
    NET_FINAL=${ROOT_FASTER}/output/${CFG_FILE}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_iter_${ITERS}.ckpt
fi
set -x

if [ ! -f ${NET_FINAL}.index ]; then
    if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
        CUDA_VISIBLE_DEVICES=${GPU_ID} python ${ROOT_FASTER}/tools/trainval_net.py \
            --weight ${ROOT_FASTER}/data/imagenet_weights/${NET}.ckpt \
            --imdb ${TRAIN_IMDB} \
            --imdbval ${TEST_IMDB} \
            --iters ${ITERS} \
            --cfg ${ROOT_FASTER}/experiments/cfgs/${CFG_FILE}.yml \
            --dataset_name ${DATASET} \
            --classes ${CLASSES} \
            --tag ${EXTRA_ARGS_SLUG} \
            --net ${NET} \
            --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
    else
        CUDA_VISIBLE_DEVICES=${GPU_ID} python ${ROOT_FASTER}/tools/trainval_net.py \
            --weight ${ROOT_FASTER}/data/imagenet_weights/${NET}.ckpt \
            --imdb ${TRAIN_IMDB} \
            --imdbval ${TEST_IMDB} \
            --iters ${ITERS} \
            --cfg ${ROOT_FASTER}/experiments/cfgs/${CFG_FILE}.yml \
            --dataset_name ${DATASET} \
            --classes ${CLASSES} \
            --net ${NET} \
            --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
    fi
fi

${ROOT_FASTER}/experiments/scripts/test_faster_rcnn.sh $@
