# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np
import cPickle as pickle

from layer_utils.snippets import generate_anchors_pre
from layer_utils.proposal_layer import proposal_layer
from layer_utils.proposal_top_layer import proposal_top_layer
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer

from model.config import cfg

class vgg16(object):
  def __init__(self, batch_size=1):
    self._feat_stride = [16,]
    self._feat_compress = [1./16.,]
    self._batch_size = batch_size
    self._predictions = {}
    self._losses={}
    self._anchor_targets={}
    self._proposal_targets={}
    self._layers = {}
    self._act_summaries = []
    self._score_summaries = {}
    self._train_summaries = []
    self._event_summaries = {}
    self._initialized = []

  def _add_act_summary(self, tensor):
    tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
    tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                    tf.nn.zero_fraction(tensor))

  def _add_score_summary(self, key, tensor):
    tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

  def _add_train_summary(self, var):
    tf.summary.histogram('TRAIN/' + var.op.name, var)

  def _caffe_weights(self, layer_name):
    layer=self._caffe_layers[layer_name]
    return layer['weights']

  def _caffe_bias(self, layer_name):
    layer=self._caffe_layers[layer_name]
    return layer['bias']

  def _caffe2tf_filter(self, name):
    f=self._caffe_weights(name)
    return f.transpose((2, 3, 1, 0))

  # Session is used to assign initial values, so that the big constant is not stored in the graph
  def _get_conv_filter(self, sess, name, trainable):
    w=self._caffe2tf_filter(name)
    phw=tf.placeholder(tf.float32, shape=w.shape)
    conv=tf.get_variable("weight", initializer=phw, dtype=tf.float32, trainable=trainable)
    sess.run(conv.initializer, feed_dict={phw: w})
    self._initialized.append(conv)

    return conv

  def _get_bias(self, sess, name, trainable):
    b=self._caffe_bias(name)
    if name == "bbox_pred":
      stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
      means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
      b -= means
      b /= stds
    phb=tf.placeholder(tf.float32, shape=b.shape)
    if cfg.TRAIN.BIAS_DECAY:
      bias = tf.get_variable("bias", initializer=phb, dtype=tf.float32, trainable=trainable)
    else:
      bias = tf.get_variable("bias", initializer=phb, regularizer=tf.no_regularizer, dtype=tf.float32, trainable=trainable)
    sess.run(bias.initializer, feed_dict={phb: b})
    self._initialized.append(bias)

    return bias

  def _get_fc_weight(self, sess, name, trainable):
    cw = self._caffe_weights(name)
    if name == "fc6":
      assert cw.shape == (4096, 25088)
      cw = cw.reshape((4096, 512, 7, 7)) 
      cw = cw.transpose((2, 3, 1, 0))
      cw = cw.reshape(25088, 4096)
    elif name == "bbox_pred":
      cw = cw.transpose((1, 0))
      stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
      cw /= stds
    else:
      cw = cw.transpose((1, 0))
    phcw=tf.placeholder(tf.float32, shape=cw.shape)
    weight = tf.get_variable("weight", initializer=phcw, dtype=tf.float32, trainable=trainable)
    sess.run(weight.initializer, feed_dict={phcw: cw})
    self._initialized.append(weight)
    
    return weight

  def _conv_layer(self, sess, bottom, name, trainable=True, padding='SAME', relu=True):
    with tf.variable_scope(name) as scope:
      filt=self._get_conv_filter(sess, name, trainable=trainable)
      conv_biases=self._get_bias(sess, name, trainable=trainable)

      conv=tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding=padding)
      bias=tf.nn.bias_add(conv, conv_biases)

      if relu:
        bias=tf.nn.relu(bias)
      return bias

  def _fc_layer(self, sess, bottom, name, trainable=True, relu=True):
    with tf.variable_scope(name) as scope:
      shape = bottom.get_shape().as_list()
      dim = 1
      for d in shape[1:]:
        dim *= d
      x = tf.reshape(bottom, [-1, dim])

      weight = self._get_fc_weight(sess, name, trainable=trainable)
      bias = self._get_bias(sess, name, trainable=trainable)

      fc=tf.nn.bias_add(tf.matmul(x, weight), bias)

      if relu:
        fc=tf.nn.relu(fc)
      return fc

  def _conv_layer_shape(self, bottom, size, channels, name, initializer=None, trainable=True, padding='SAME', relu=True):
    bottom_shape = bottom.get_shape().as_list()
    size.extend([bottom_shape[3],channels])
    with tf.variable_scope(name) as scope:
      filt=tf.get_variable('weight', size, initializer=initializer, trainable=trainable)
      if cfg.TRAIN.BIAS_DECAY:
        conv_biases=tf.get_variable('bias', [channels], initializer=tf.constant_initializer(0.0), trainable=trainable)
      else:
        conv_biases=tf.get_variable('bias', [channels], initializer=tf.constant_initializer(0.0), regularizer=tf.no_regularizer, trainable=trainable)

      conv=tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding=padding)
      bias=tf.nn.bias_add(conv, conv_biases)

      if relu:
        bias=tf.nn.relu(bias)
      return bias

  def _fc_layer_shape(self, bottom, channels, name, initializer=None, trainable=True, relu=True):
    with tf.variable_scope(name) as scope:
      shape = bottom.get_shape().as_list()
      dim = 1
      for d in shape[1:]:
        dim *= d
      x = tf.reshape(bottom, [-1, dim])

      weight = tf.get_variable('weight', [dim, channels], initializer=initializer, trainable=trainable)
      if cfg.TRAIN.BIAS_DECAY:
        bias = tf.get_variable('bias', [channels], initializer=tf.constant_initializer(0.0), trainable=trainable)
      else:
        bias = tf.get_variable('bias', [channels], initializer=tf.constant_initializer(0.0), regularizer=tf.no_regularizer, trainable=trainable)

      fc=tf.nn.bias_add(tf.matmul(x, weight), bias)

      if relu:
        fc=tf.nn.relu(fc)
      return fc

  def _reshape_layer(self, bottom, num_dim, name):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name) as scope:
      # change the channel to the caffe format
      to_caffe = tf.transpose(bottom, [0,3,1,2])
      # then force it to have channel 2
      reshaped = tf.reshape(to_caffe, tf.concat(0, [[self._batch_size], [num_dim, -1], [input_shape[2]]]))
      # then swap the channel back
      to_tf = tf.transpose(reshaped, [0,2,3,1])
      return to_tf

  def _softmax_layer(self, bottom, name):
    if name == 'rpn_cls_prob_reshape':
      input_shape = tf.shape(bottom)
      bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
      reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
      return tf.reshape(reshaped_score, input_shape)
    return tf.nn.softmax(bottom, name=name)

  def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      rois, rpn_scores = tf.py_func(proposal_top_layer,
                                [rpn_cls_prob, rpn_bbox_pred, self._im_info, 
                                self._feat_stride, self._anchors, self._anchor_scales], [tf.float32, tf.float32])
      rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
      rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])

    return rois, rpn_scores

  def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      rois, rpn_scores = tf.py_func(proposal_layer,
                                [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode, 
                                self._feat_stride, self._anchors, self._anchor_scales], [tf.float32, tf.float32])
      rois.set_shape([None, 5])
      rpn_scores.set_shape([None, 1])

    return rois, rpn_scores

  # Only use it if you have roi_pooling op written in tf.image
  def _roi_pool_layer(self, bootom, rois, name):
    with tf.variable_scope(name) as scope:
      return tf.image.roi_pooling(bootom, rois,
                                    pooled_height=7,
                                    pooled_width=7,
                                    spatial_scale=1./16)[0]

  def _crop_pool_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      height = tf.ceil(self._im_info[0, 0] / 16. - 1.) * 16.
      width = tf.ceil(self._im_info[0, 1] / 16. - 1.) * 16.
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
      bboxes = tf.concat(1, [y1, x1, y2, x2])
      crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [14, 14], name="crops")

    return slim.max_pool2d(crops, [2, 2], padding='SAME')

  def _dropout_layer(self, bottom, name, ratio=0.5):
    return tf.nn.dropout(bottom, ratio, name=name)

  def _anchor_target_layer(self, rpn_cls_score, name):
    with tf.variable_scope(name) as scope:
      rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(anchor_target_layer, 
                          [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._anchor_scales], 
                          [tf.float32, tf.float32, tf.float32, tf.float32])

      rpn_labels.set_shape([1, 1, None, None])
      rpn_bbox_targets.set_shape([1, None, None, self._num_scales*12])
      rpn_bbox_inside_weights.set_shape([1, None, None, self._num_scales*12])
      rpn_bbox_outside_weights.set_shape([1, None, None, self._num_scales*12])

      rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
      self._anchor_targets['rpn_labels'] = rpn_labels
      self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
      self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
      self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

      self._score_summaries.update(self._anchor_targets)

    return rpn_labels

  def _proposal_target_layer(self, rois, roi_scores, name):
    with tf.variable_scope(name) as scope:
      rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(proposal_target_layer, 
                                        [rois, roi_scores, self._gt_boxes, self._num_classes], 
                                        [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

      rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
      roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
      labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
      bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes*4])
      bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes*4])
      bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes*4])

      self._proposal_targets['rois'] = rois
      self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
      self._proposal_targets['bbox_targets'] = bbox_targets
      self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
      self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

      self._score_summaries.update(self._proposal_targets)

      return rois, roi_scores

  def _anchor_component(self):
    with tf.variable_scope('ANCHOR_' + self._tag) as scope:
      height = tf.to_int32(tf.ceil(self._im_info[0, 0] / 16.))
      width = tf.to_int32(tf.ceil(self._im_info[0, 1] / 16.))
      anchors, anchor_length = tf.py_func(generate_anchors_pre,
                          [height, width, 
                          self._feat_stride, self._anchor_scales], 
                          [tf.float32, tf.int32], name="generate_anchors")
      anchors.set_shape([None, 4])
      anchor_length.set_shape([])
      self._anchors = anchors
      self._anchor_length = anchor_length

  def _vgg16_from_imagenet(self, sess, train=True):
    with tf.variable_scope('vgg16_' + self._tag, regularizer=tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)) as scope:
      # select initializers
      if cfg.TRAIN.TRUNCATED:
        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox=tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
      else:
        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox=tf.random_normal_initializer(mean=0.0, stddev=0.001)
      # first layer
      net = self._conv_layer(sess, self._image, "conv1_1", False)
      # self._act_summaries.append(net)
      net = self._conv_layer(sess, net, "conv1_2", False)
      # self._act_summaries.append(net)
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
      # second layer
      net = self._conv_layer(sess, net, "conv2_1", False)
      # self._act_summaries.append(net)
      net = self._conv_layer(sess, net, "conv2_2", False)
      # self._act_summaries.append(net)
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
      # third layer
      net = self._conv_layer(sess, net, "conv3_1", train)
      # self._act_summaries.append(net)
      net = self._conv_layer(sess, net, "conv3_2", train)
      # self._act_summaries.append(net)
      net = self._conv_layer(sess, net, "conv3_3", train)
      # self._act_summaries.append(net)
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
      # fourth layer
      net = self._conv_layer(sess, net, "conv4_1", train)
      # self._act_summaries.append(net)
      net = self._conv_layer(sess, net, "conv4_2", train)
      # self._act_summaries.append(net)
      net = self._conv_layer(sess, net, "conv4_3", train)
      # self._act_summaries.append(net)
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
      # fifth layer
      net = self._conv_layer(sess, net, "conv5_1", train)
      # self._act_summaries.append(net)
      net = self._conv_layer(sess, net, "conv5_2", train)
      # self._act_summaries.append(net)
      net = self._conv_layer(sess, net, "conv5_3", train)
      self._act_summaries.append(net)

      self._layers['conv5_3'] = net
      # build the anchors for the image
      self._anchor_component()

      # rpn
      rpn = self._conv_layer_shape(net, [3,3], 512, "rpn_conv/3x3", initializer, train)
      self._act_summaries.append(rpn)
      rpn_cls_score = self._conv_layer_shape(rpn, [1,1], self._num_scales * 6, "rpn_cls_score", initializer, train, 'VALID', False)
      # change it so that the score has 2 as its channel size
      rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, "rpn_cls_score_reshape")
      rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
      rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_scales * 6, "rpn_cls_prob")
      rpn_bbox_pred = self._conv_layer_shape(rpn, [1,1], self._num_scales * 12, "rpn_bbox_pred", initializer, train, 'VALID', False)

      if train:
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
        pool5 = self._crop_pool_layer(net, rois, "pool5")
      else:
        raise NotImplementedError

      fc6 = self._fc_layer(sess, pool5, "fc6", train)
      self._act_summaries.append(fc6)
      if train:
        fc6 = self._dropout_layer(fc6, "dropout6")
      fc7 = self._fc_layer(sess, fc6, "fc7", train)
      self._act_summaries.append(fc7)
      if train:
        fc7 = self._dropout_layer(fc7, "dropout7")
      cls_score = self._fc_layer_shape(fc7, self._num_classes, "cls_score", initializer, train, False)
      cls_prob = self._softmax_layer(cls_score, "cls_prob")
      bbox_pred = self._fc_layer_shape(fc7, self._num_classes * 4, "bbox_pred", initializer_bbox, train, False)

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

  def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma**2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1./sigma_2)))
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.0) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = tf.reduce_mean(tf.reduce_sum(
                        out_loss_box, 
                        reduction_indices=dim
                )) 
    return loss_box

  def _add_losses(self, sigma_rpn=3.0):
    with tf.variable_scope('vgg16-loss_' + self._tag) as scope:
      # RPN, class loss
      rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1,2])
      rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
      rpn_select = tf.where(tf.not_equal(rpn_label,-1))
      rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select),[-1,2])
      rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select),[-1])
      rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(rpn_cls_score, rpn_label))

      # RPN, bbox loss
      rpn_bbox_pred = self._predictions['rpn_bbox_pred']
      rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
      rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
      rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']

      rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1,2,3])

      # RCNN, class loss
      cls_score = self._predictions["cls_score"]
      label = tf.reshape(self._proposal_targets["labels"],[-1])
      cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(cls_score, label))

      # RCNN, bbox loss
      bbox_pred = self._predictions['bbox_pred']
      bbox_targets = self._proposal_targets['bbox_targets']
      bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
      bbox_outside_weights = self._proposal_targets['bbox_outside_weights']

      loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

      self._losses['cross_entropy'] = cross_entropy
      self._losses['loss_box'] = loss_box
      self._losses['rpn_cross_entropy'] = rpn_cross_entropy
      self._losses['rpn_loss_box'] = rpn_loss_box

      loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
      self._losses['total_loss'] = loss

      self._event_summaries.update(self._losses)

    return loss

  def create_architecture(self, sess, mode, num_classes, 
                          caffe_weight_path=None, 
                          tag=None, anchor_scales=[8, 16, 32]):
    self._image = tf.placeholder(tf.float32, shape=[self._batch_size, None, None, 3])
    self._im_info = tf.placeholder(tf.float32, shape=[self._batch_size, 3])
    self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
    self._caffe_weight_path=caffe_weight_path
    self._tag=tag
    
    self._num_classes = num_classes
    self._mode = mode
    self._anchor_scales = anchor_scales
    self._num_scales = len(anchor_scales)

    training = mode == 'TRAIN'
    testing = mode == 'TEST'

    assert tag != None
    print 'Loading caffe weights...'
    with open(self._caffe_weight_path, 'r') as f:
      self._caffe_layers = pickle.load(f)
    print 'Done!'

    rois, cls_prob, bbox_pred = self._vgg16_from_imagenet(sess, training)

    layers_to_output = {'rois': rois}
    layers_to_output.update(self._predictions)

    for var in tf.trainable_variables():
      self._train_summaries.append(var)

    if mode == 'TEST':
      stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
      means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
      self._predictions["bbox_pred"] *= stds
      self._predictions["bbox_pred"] += means
    else:
      self._add_losses()
      layers_to_output.update(self._losses)

    val_summaries = []
    with tf.device("/cpu:0"):
      for key, var in self._event_summaries.items():
        val_summaries.append(tf.summary.scalar(key, var))
      for key, var in self._score_summaries.items():
        self._add_score_summary(key, var)
      for var in self._act_summaries:
        self._add_act_summary(var)
      for var in self._train_summaries:
        self._add_train_summary(var)

    self._summary_op = tf.summary.merge_all()
    if not testing:
      self._summary_op_val = tf.summary.merge(val_summaries)

    return layers_to_output

  # only useful during testing mode
  def extract_conv5(self, sess, image):
    feed_dict={self._image: image}
    feat = sess.run(self._layers["conv5_3"], feed_dict=feed_dict)
    return feat

  # only useful during testing mode
  def test_image(self, sess, image, im_info):
    feed_dict={self._image: image,
              self._im_info: im_info}
    cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"], 
                                                    self._predictions['cls_prob'], 
                                                    self._predictions['bbox_pred'], 
                                                    self._predictions['rois']], 
                                                    feed_dict=feed_dict)
    return cls_score, cls_prob, bbox_pred, rois

  def get_summary(self, sess, blobs):
    feed_dict={self._image: blobs['data'], self._im_info: blobs['im_info'], \
             self._gt_boxes: blobs['gt_boxes']}
    summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

    return summary

  def train_step(self, sess, blobs, train_op):
    feed_dict={self._image: blobs['data'], self._im_info: blobs['im_info'], \
             self._gt_boxes: blobs['gt_boxes']}
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, _ = sess.run([self._losses["rpn_cross_entropy"], 
                                                    self._losses['rpn_loss_box'], 
                                                    self._losses['cross_entropy'], 
                                                    self._losses['loss_box'],
                                                    self._losses['total_loss'],
                                                    train_op], 
                                                    feed_dict=feed_dict)
    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

  def train_step_with_summary(self, sess, blobs, train_op):
    feed_dict={self._image: blobs['data'], self._im_info: blobs['im_info'], \
             self._gt_boxes: blobs['gt_boxes']}
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary, _ = sess.run([self._losses["rpn_cross_entropy"], 
                                                    self._losses['rpn_loss_box'], 
                                                    self._losses['cross_entropy'], 
                                                    self._losses['loss_box'],
                                                    self._losses['total_loss'],
                                                    self._summary_op,
                                                    train_op], 
                                                    feed_dict=feed_dict)
    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary

  def train_step_no_return(self, sess, blobs, train_op):
    feed_dict={self._image: blobs['data'], self._im_info: blobs['im_info'], \
             self._gt_boxes: blobs['gt_boxes']}
    sess.run([train_op], feed_dict=feed_dict)

