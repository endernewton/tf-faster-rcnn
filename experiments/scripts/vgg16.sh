#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE=50000
    ITERS=70000
    ANCHORS="[8,16,32]"
    ;;
  pascal_voc_0712)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE=80000
    ITERS=110000
    ANCHORS="[8,16,32]"
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train+coco_2014_valminusminival"
    TEST_IMDB="coco_2014_minival"
    STEPSIZE=350000
    ITERS=490000
    ANCHORS="[4,8,16,32]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/vgg16_depre_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}_vgg16.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
    NET_FINAL=output/vgg16_depre/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/vgg16_faster_rcnn_iter_${ITERS}.ckpt
else
    NET_FINAL=output/vgg16_depre/${TRAIN_IMDB}/default/vgg16_faster_rcnn_iter_${ITERS}.ckpt
fi
set -x

if [ ! -f ${NET_FINAL}.index ]; then
    if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
        CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_vgg16_net.py \
            --weight data/imagenet_weights/vgg16.weights \
            --imdb ${TRAIN_IMDB} \
            --imdbval ${TEST_IMDB} \
            --iters ${ITERS} \
            --cfg experiments/cfgs/vgg16_depre.yml \
            --tag ${EXTRA_ARGS_SLUG} \
            --set ANCHOR_SCALES ${ANCHORS} TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
    else
        CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_vgg16_net.py \
            --weight data/imagenet_weights/vgg16.weights \
            --imdb ${TRAIN_IMDB} \
            --imdbval ${TEST_IMDB} \
            --iters ${ITERS} \
            --cfg experiments/cfgs/vgg16_depre.yml \
            --set ANCHOR_SCALES ${ANCHORS} TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
    fi
fi

./experiments/scripts/test_vgg16.sh $@
