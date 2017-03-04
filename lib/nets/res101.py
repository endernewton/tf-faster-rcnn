# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
import numpy as np

try:
  import cPickle as pickle
except ImportError:
  import pickle
from layer_utils.snippets import generate_anchors_pre
from layer_utils.proposal_layer import proposal_layer
from layer_utils.proposal_top_layer import proposal_top_layer
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from nets.network import Network
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from model.config import cfg

def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  batch_norm_params = {
      'is_training': False,
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': ops.GraphKeys.UPDATE_OPS,
  }

  with arg_scope(
      [slim.conv2d],
      weights_regularizer=regularizers.l2_regularizer(weight_decay),
      weights_initializer=initializers.variance_scaling_initializer(),
      trainable=is_training,
      activation_fn=nn_ops.relu,
      normalizer_fn=layers.batch_norm,
      normalizer_params=batch_norm_params):
    with arg_scope([layers.batch_norm], **batch_norm_params):
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      # tf.contrib.framework.arg_scope([tf.contrib.layers.max_pool2d], padding='VALID').
      with arg_scope([layers.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc

class Resnet101(Network):
  def __init__(self, batch_size=1):
    Network.__init__(self, batch_size=batch_size)

  def _crop_pool_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      height = tf.ceil(self._im_info[0, 0] / 16. - 1.) * 16.
      width = tf.ceil(self._im_info[0, 1] / 16. - 1.) * 16.
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
      bboxes = tf.concat(axis=1, values=[y1, x1, y2, x2])
      crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [14, 14], name="crops")

      # Change it back to the Google RoI pooling layer, more experiments needed.
    return slim.max_pool2d(crops, [2, 2], padding='SAME')

  def build_network(self, sess, is_training=True):
    # select initializers
    if cfg.TRAIN.TRUNCATED:
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    else:
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
    bottleneck = resnet_v1.bottleneck
    blocks_fixed = [
      resnet_utils.Block('block1', bottleneck,
                         [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      resnet_utils.Block('block2', bottleneck,
                         [(512, 128, 1)] * 3 + [(512, 128, 2)])
    ]
    blocks_train = [
      resnet_utils.Block('block3', bottleneck,
                         [(1024, 256, 1)] * 22 + [(1024, 256, 1)]),
      resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    with slim.arg_scope(resnet_arg_scope(is_training=False)):
      net, end_points = resnet_v1.resnet_v1(self._image,
                                            blocks_fixed,
                                            global_pool=False,
                                            include_root_block=True,
                                            scope='resnet_v1_101')
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
      net_conv5, end_points = resnet_v1.resnet_v1(net,
                                            [blocks_train[0]],
                                            global_pool=False,
                                            include_root_block=False,
                                            scope='resnet_v1_101')

    self._act_summaries.append(net)
    self._layers['conv5_3'] = net
    with tf.variable_scope('resnet_v1_101', 'resnet_v1_101',
                           regularizer=tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)):
      # build the anchors for the image
      self._anchor_component()

      # rpn
      # rpn = self._conv_layer_shape(net, [3, 3], 512, "rpn_conv/3x3", initializer, train)
      if cfg.TRAIN.BIAS_DECAY:
        biases_regularizer = None
      else:
        biases_regularizer = tf.no_regularizer
      rpn = slim.conv2d(net_conv5, 512, [3, 3], trainable=is_training, weights_initializer=initializer,
                        biases_regularizer= biases_regularizer,
                        biases_initializer=tf.constant_initializer(0.0), scope="rpn_conv/3x3")
      self._act_summaries.append(rpn)
      rpn_cls_score = slim.conv2d(rpn, self._num_scales * 6, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  biases_regularizer=biases_regularizer,
                                  biases_initializer=tf.constant_initializer(0.0),
                                  padding='VALID', activation_fn=None, scope='rpn_cls_score')
      # change it so that the score has 2 as its channel size
      rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
      rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
      rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_scales * 6, "rpn_cls_prob")
      rpn_bbox_pred = slim.conv2d(rpn, self._num_scales * 12, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  biases_regularizer=biases_regularizer,
                                  biases_initializer=tf.constant_initializer(0.0),
                                  padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
      if is_training:
        rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
        # Try to have a determinestic order for the computing graph, for reproducibility
        with tf.control_dependencies([rpn_labels]):
          rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
      else:
        if cfg.TEST.MODE == 'nms':
          rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        elif cfg.TEST.MODE == 'top':
          rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        else:
          raise NotImplementedError

      # rcnn
      if cfg.POOLING_MODE == 'crop':
        pool5 = self._crop_pool_layer(net_conv5, rois, "pool5")
      else:
        raise NotImplementedError

    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
      fc7, end_points = resnet_v1.resnet_v1(pool5,
                                            [blocks_train[1]],
                                            global_pool=False,
                                            include_root_block=False,
                                            scope='resnet_v1_101')
      # Use conv2d instead of fully_connected layers.
      # fc7 = slim.avg_pool2d(pool5, [7, 7], stride=7,  padding='VALID', scope='fc7')
      # fc6 = slim.dropout(fc6, is_training=is_training,
      #                    scope='dropout6')
      # fc7 = slim.conv2d(fc6, 4096, [1, 1], scope='fc7')
      # fc7 = slim.dropout(fc7, is_training=is_training,
      #                    scope='dropout7')
    with tf.variable_scope('resnet_v1_101', 'resnet_v1_101',
                           regularizer=tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)):
      fc7 = slim.flatten(fc7, scope='flatten')
      cls_score = slim.fully_connected(fc7, self._num_classes, weights_initializer=initializer, trainable=is_training,
                              biases_regularizer=biases_regularizer,
                              activation_fn=None, scope='cls_score')
      cls_prob = self._softmax_layer(cls_score, "cls_prob")
      bbox_pred = slim.fully_connected(fc7, self._num_classes * 4, weights_initializer=initializer_bbox,
                              trainable=is_training, biases_regularizer=biases_regularizer,
                              activation_fn=None, scope='bbox_pred')
    self._predictions["rpn_cls_score"] = rpn_cls_score
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    self._predictions["rpn_cls_prob"] = rpn_cls_prob
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    self._predictions["cls_score"] = cls_score
    self._predictions["cls_prob"] = cls_prob
    self._predictions["bbox_pred"] = bbox_pred
    self._predictions["rois"] = rois

    self._score_summaries.update(self._predictions)

    return rois, cls_prob, bbox_pred

