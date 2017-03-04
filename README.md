# tf-faster-rcnn
A Tensorflow implementation of faster RCNN detection framework by Xinlei Chen (xinleic@cs.cmu.edu). This repository is based on the python Caffe implementation of faster RCNN available [here](https://github.com/rbgirshick/py-faster-rcnn).

**Note**: Several minor modifications are made when reimplementing the framework, which give potential improvements. For details about the modifications and ablative analysis, please refer to the technical report [An Implementation of Faster RCNN with Study for Region Sampling](https://arxiv.org/pdf/1702.02138.pdf). If you are seeking to reproduce the results in the original paper, please use the [official code](https://github.com/ShaoqingRen/faster_rcnn) or maybe the [semi-official code](https://github.com/rbgirshick/py-faster-rcnn). For details about the faster RCNN architecture please refer to the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497.pdf). 

### Detection Performance
We only tested it on plain VGG16 and Resnet101 (experimental, thank you @philokey!) architecture so far. As the baseline, we report numbers using a single model on a single convolution layer, so no multi-scale, no multi-stage bounding box regression, no skip-connection, no extra input is used. The only data augmentation technique is left-right flipping during training following the original Faster RCNN. 

With VGG16 (``conv5_3``):
  - Train on VOC 2007 trainval and test on VOC 2007 test, **71.2**.
  - Train on COCO 2014 [trainval-minival](https://github.com/rbgirshick/py-faster-rcnn/tree/master/models) and test on [minival](https://github.com/rbgirshick/py-faster-rcnn/tree/master/models) (longer), **29.5**. 
  
With Resnet101 (experimental, last ``conv4`` layer):
  - Train on VOC 2007 trainval and test on VOC 2007 test, **74.1**.
  - Train on VOC 2007 trainval + VOC 2012 trainval and test on VOC 2007 test, **77.8**. 

**Note**:
  - The VGG16 numbers are obtained with a different testing scheme without selecting region proposals using non-maximal suppression (TEST.MODE top), the default and original testing scheme (TEST.MODE nms) will result in slightly worse performance (see [report](https://arxiv.org/pdf/1702.02138.pdf), for COCO it drops 0.3 - 0.4 AP). 
  - Since we keep the small proposals (\< 16 pixels width/height), our performance is especially good for small objects.
  - For other minor modifications, please check the [report](https://arxiv.org/pdf/1702.02138.pdf).
  - For COCO, we find the performance improving with more iterations (VGG16 350k/490k: 26.9, 600k/790k: 28.3, 900k/1190k: 29.5), and potentially better performance can be achieved with even more iterations. Check out [here](http://ladoga.graphics.cs.cmu.edu/xinleic/tf-faster-rcnn/coco_longer/) or ([here](http://gs11655.sp.cs.cmu.edu/xinleic/tf-faster-rcnn/coco_longer/)/[here](https://drive.google.com/open?id=0B1_fAEgxdnvJSmF3YUlZcHFqWTQ)) for the latest models.
  - For Resnet101 on PASCAL VOC, check out [here](https://drive.google.com/drive/folders/0B2Zdmaho7pB7dXJORFRaRmhjVFU?usp=sharing) for the latest models.

### Additional Features
Additional features not mentioned in the [report](https://arxiv.org/pdf/1702.02138.pdf) are added to make research life easier:
  - **Support for train-and-validation**. During training, the validation data will also be tested from time to time to monitor the process and check potential overfitting. Ideally training and validation should be separate, where the model is loaded everytime to test on validation. However I have implemented it in a joint way to save time and GPU memory. Though in the default setup the testing data is used for validation, no special attempts is made to overfit on testing set.
  - **Support for resuming training**. I tried to store as much information as possible when snapshoting, with the purpose to resume training from the lateset snapshot properly. The meta information includes current image index, permutation of images, and random state of numpy. However, when you resume training the random seed for tensorflow will be reset (not sure how to save the random state of tensorflow now), so it will result in a difference. **Note** that, the current implementation still cannot force the model to behave deterministically even with the random seeds set. Suggestion/solution is welcome and much appreciated.
  - **Support for visualization**. The current implementation will summarize statistics of losses, activations and variables during training, and dump it to a separate folder for tensorboard visualization. The computing graph is also saved for debugging.

### Prerequisites
  - A basic Tensorflow installation. The code follows **r1.0** format now. The released model follows the r0.12 format. If you are using an order version (r0.1-r0.12), please check out the v0.12 release. While it is not required, for experimenting the original RoI pooling (which requires modification of the C++ code in tensorflow), you can check out my tensorflow [fork](https://github.com/endernewton/tensorflow) and look for ``tf.image.roi_pooling``.
  - Python packages you might not have: `cython`, `python-opencv`, `easydict` (similar to [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)).
  - Docker users: A Docker image containing all of the required dependencies can be found in Docker hub at mbuckler/tf-faster-rcnn-deps. The Docker file used to create this image can be found in the docker directory of this repository.

### Installation
1. Clone the repository
  ```Shell
  git clone https://github.com/endernewton/tf-faster-rcnn.git
  ```

2. Update your -arch in setup script to match your GPU
  ```Shell
  cd tf-faster-rcnn/lib
  vim setup.py
  # Check the GPU architecture, if you are using Maxwell arch, please switch to sm_61
  ```

3. Build the Cython modules
  ```Shell
  make clean
  make
  cd ..
  ```
  
4. Download pre-trained models and weights. Pre-trained models is provided by slim, you can get the pre-trained models [here](https://github.com/tensorflow/models/tree/master/slim#pre-trained-models) and set them in the ``data/imagenet_weights`` folder. For example for VGG16 model, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
   tar -xzvf vgg_16_2016_08_28.tar.gz
   mv vgg_16.ckpt vgg16.ckpt
   cd ../..
   ```
 For Resnet101, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   wget -v http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
   tar -xzvf resnet_v1_101_2016_08_28.tar.gz
   mv resnet_v1_101.ckpt res101.ckpt
   cd ../..
   ```

5. Install the [Python COCO API](https://github.com/pdollar/coco). And create a symbolic link to it within ``tf-faster-rcnn/data``, The code requires the API to access COCO dataset.

### Setup data
Please follow the instructions of py-faster-rcnn [here](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to setup VOC and COCO datasets. The steps involve downloading data and creating softlinks in the ``data`` folder. Since faster RCNN does not rely on pre-computed proposals, it is safe to ignore the steps that setup proposals.

If you find it useful, the ``data/cache`` folder created on my side is also shared [here](http://ladoga.graphics.cs.cmu.edu/xinleic/tf-faster-rcnn/cache.tgz). 

### Test with pre-trained models
1. Download pre-trained models and weights (VGG16)
  ```Shell
  # return to the repository root
  cd ..
  # VGG16 for both voc and coco using default training scheme
  ./data/scripts/fetch_faster_rcnn_models.sh
  # VGG16 for coco using longer training scheme (600k/790k)
  ./data/scripts/fetch_coco_long_models.sh
  # VGG16 weights for imagenet pretrained model, extracted from released caffe model
  ./data/scripts/fetch_imagenet_weights.sh
  ```
  **Note**: if you cannot download the models through the link. You can check out the following solutions:
  - Another server [here](http://gs11655.sp.cs.cmu.edu/xinleic/tf-faster-rcnn/).
  - Google drive [here](https://drive.google.com/open?id=0B1_fAEgxdnvJSmF3YUlZcHFqWTQ).

2. Create a folder and a softlink to use the pretrained model
  ```Shell
  NET=vgg16
  mkdir -p output/${NET}
  cd output/${NET}
  ln -s ../../data/faster_rcnn_models/voc_2007_trainval ./
  ln -s ../../data/faster_rcnn_models/coco_2014_train+coco_2014_valminusminival ./
  cd ../..
  ```

3. Test with pre-trained VGG16 models
  ```Shell
  GPU_ID=0
  ./experiments/scripts/test_vgg16.sh $GPU_ID pascal_voc
  ./experiments/scripts/test_vgg16.sh $GPU_ID coco
  ```

### Train your own model 
1. (Optional) If you have just tested the models, first remove the link to the pre-trained models
  ```Shell
  NET=vgg16
  # NET in {vgg16, res101} is the network arch to use
  rm -v output/${NET}/voc_2007_trainval
  rm -v output/${NET}/coco_2014_train+coco_2014_valminusminival
  ```
  
2. Train (and test, evaluation)
  ```Shell
  ./experiments/scripts/train_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res101} is the network arch to use
  # DATASET {pascal_voc, coco} is defined in train_faster_rcnn.sh
  # Examples:
  ./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16
  ./experiments/scripts/train_faster_rcnn.sh 1 coco res101
  ```

3. Visualization with Tensorboard
  ```Shell
  tensorboard --logdir=tensorboard/vgg16/voc_2007_trainval/ --port=7001 &
  tensorboard --logdir=tensorboard/vgg16/coco_2014_train+coco_2014_valminusminival/ --port=7002 &
  ```
  
4. Test and evaluate
  ```Shell
  ./experiments/scripts/test_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res101} is the network arch to use
  # DATASET {pascal_voc, coco} is defined in test_faster_rcnn.sh
  # Examples:
  ./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
  ./experiments/scripts/test_faster_rcnn.sh 1 coco res101
  ```
  
5. You can use ``tools/reval.sh`` for re-evaluation


By default, trained networks are saved under:

```
output/[NET]/[DATASET]/default/
```

Test outputs are saved under:

```
output/[NET]/[DATASET]/default/[SNAPSHOT]/
```

Tensorboard information for train and validation is saved under:

```
tensorboard/[NET]/[DATASET]/default/
tensorboard/[NET]/[DATASET]/default_val/
```

The default number of training iterations is kept the same to the original faster RCNN, however I find it is beneficial to train longer for COCO (see [report](https://arxiv.org/pdf/1702.02138.pdf)). Also note that due to the nondeterministic nature of the current implementation, the performance can vary a bit, but in general it should be within 1% of the reported numbers. Solutions are welcome.

### Citation
If you find this implementation or the analysis conducted in our report helpful, please consider citing:

    @article{chen17implementation,
        Author = {Xinlei Chen and Abhinav Gupta},
        Title = {An Implementation of Faster RCNN with Study for Region Sampling},
        Journal = {arXiv preprint arXiv:1702.02138},
        Year = {2017}
    }
    
For convenience, here is the faster RCNN citation:

    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }

### Detailed Numbers

VGG16 COCO 2015 test-dev (900k/1190k):
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.297
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.504
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.312
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.128
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.325
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.421
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.272
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.399
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.409
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.187
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.451
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.591
 ```
 
VGG16 COCO 2015 test-std (900k/1190k):
 ```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.295
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.501
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.312
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.119
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.327
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.418
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.273
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.409
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.179
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.455
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.586
 ```
