# tf-faster-rcnn
A Tensorflow implementation of faster RCNN detection framework by Xinlei Chen (xinleic@cs.cmu.edu). This repository is based on the python Caffe implementation of faster RCNN available [here](https://github.com/rbgirshick/py-faster-rcnn). For details about the faster RCNN architecture please refer to the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497.pdf) by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. 

**Note**: Several modifications are made when reimplementing the framework, which gives potential improvements. For details about the modifications and ablative analysis, please refer to the technical report [An Implementation of Faster RCNN with Study for Region Sampling](http://arxiv.org/pdf/). If you are seeking to reproduce the results in the original paper, please use the [official code](https://github.com/ShaoqingRen/faster_rcnn) and [semi-official code](https://github.com/rbgirshick/py-faster-rcnn).

### Detection Performance
We only tested it on VGG16 architecture so far. Our best performance as of January 2017:
  - Train on VOC 2017 trainval and test on VOC 2017 test, **71.2**.
  - Train on COCO 2014 [trainval-minival](https://github.com/rbgirshick/py-faster-rcnn/tree/master/models) and test on [minival](https://github.com/rbgirshick/py-faster-rcnn/tree/master/models), **28.3**. 
  
### Additional Features
Additional features are added to make research life easier:
  - Support for train and validation. During training, the validation data will also be tested from time to time to monitor the process and check potential overfitting. Ideally training and validation should be separate, where the model is loaded everytime to test on validation. However I have implemented it in a joint way to save time and GPU memory. 
  - Support for stop and retrain. I tried to store as much information as possible when snapshoting, with the purpose to resume training from the lateset snapshot properly. The meta information includes current image index, permutation of images, and random state of numpy. However, when you resume training the random seed for tensorflow will be reset (not sure how to save the random state of tensorflow now), so it will result in a difference. **Note** that, the current implementation still cannot force the model to behave deterministically even with the random seed set. Suggestion/solution is welcome and much appreciated.
  - Support for visualization. The current implementation will summarize statistics of losses, activations and variables during training, and dump it to a separate folder for tensorboard visualization. The computing graph is also saved.

### Prerequisites
  - A basic Tensorflow installation. r0.12 is fully tested. r0.10+ should in general be fine.
  - Python packages you might not have: `cython`, `python-opencv`, `easydict` (similar to py-faster-rcnn).

### Installation

### Testing

### Training

### 

## Citation


