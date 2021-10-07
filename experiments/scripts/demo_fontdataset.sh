#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1

CUDA_VISIBLE_DEVICES=${GPU_ID} time python tools/demo_fontdataset.py --testimg data/fontdataset --net res101 --model output/res101/fontdataset_trainval/default/res101_faster_rcnn_iter_490000.ckpt --dataset fontdataset_test --index data/fontdataset/test.txt
