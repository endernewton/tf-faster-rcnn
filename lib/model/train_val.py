# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------

from model.config import cfg
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
import cPickle
import numpy as np
import os
import sys
import glob
import time

import tensorflow as tf

class SolverWrapper(object):
  """
    A wrapper class for the training process
  """
  def __init__(self, sess, network, imdb, roidb, valroidb, output_dir, tbdir, pretrained_model=None):
    self.net = network
    self.imdb = imdb
    self.roidb = roidb
    self.valroidb = valroidb
    self.output_dir = output_dir
    self.tbdir = tbdir
    # Simply put '_val' at the end to save the summaries from the validation set
    self.tbvaldir = tbdir + '_val'
    if not os.path.exists(self.tbvaldir):
      os.makedirs(self.tbvaldir)
    self.pretrained_model = pretrained_model

  def snapshot(self, sess, iter):
    net = self.net

    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    # Store the model snapshot
    filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.ckpt'
    filename = os.path.join(self.output_dir, filename)
    self.saver.save(sess, filename)
    print 'Wrote snapshot to: {:s}'.format(filename)

    # Also store some meta information, random state, etc.
    nfilename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pkl'
    nfilename = os.path.join(self.output_dir, nfilename)
    # current state of numpy random
    st0 = np.random.get_state()
    # current position in the database
    cur = self.data_layer._cur
    # current shuffled indeces of the database
    perm = self.data_layer._perm
    # current position in the validation database
    cur_val = self.data_layer_val._cur
    # current shuffled indeces of the validation database
    perm_val = self.data_layer_val._perm

    # Dump the meta info
    with open(nfilename, 'wb') as fid:
      cPickle.dump(st0, fid, cPickle.HIGHEST_PROTOCOL)
      cPickle.dump(cur, fid, cPickle.HIGHEST_PROTOCOL)
      cPickle.dump(perm, fid, cPickle.HIGHEST_PROTOCOL)
      cPickle.dump(cur_val, fid, cPickle.HIGHEST_PROTOCOL)
      cPickle.dump(perm_val, fid, cPickle.HIGHEST_PROTOCOL)
      cPickle.dump(iter, fid, cPickle.HIGHEST_PROTOCOL)

    return filename, nfilename

  def train_model(self, sess, max_iters):
    # Build data layers for both training and validation set
    self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)
    self.data_layer_val = RoIDataLayer(self.valroidb, self.imdb.num_classes, random=True)

    # Determine different scales for anchors, see paper
    if self.imdb.name.startswith('voc'):
      anchors = [8, 16, 32]
    else:
      anchors = [4, 8, 16, 32]

    with sess.graph.as_default():
      # Set the random seed for tensorflow
      tf.set_random_seed(cfg.RNG_SEED)
      # Build the main computation graph
      layers = self.net.create_architecture(sess, "TRAIN", self.imdb.num_classes,
                                            caffe_weight_path=self.pretrained_model, 
                                            tag='default', anchor_scales=anchors)
      # Define the loss
      loss = layers['total_loss']

      # Set learning rate and momentum
      lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
      momentum = cfg.TRAIN.MOMENTUM
      self.optimizer = tf.train.MomentumOptimizer(lr, momentum)
      # Compute the gradients wrt the loss
      gvs = self.optimizer.compute_gradients(loss)
      # Double the gradient of the bias if set
      if cfg.TRAIN.DOUBLE_BIAS:
        final_gvs = []
        with tf.variable_scope('Gradient_Mult') as scope:
          for grad, var in gvs:
            scale = 1.
            if cfg.TRAIN.DOUBLE_BIAS and '/bias:' in var.name:
              scale *= 2.
            if not np.allclose(scale, 1.0):
              grad = tf.mul(grad, scale)
            final_gvs.append((grad, var))
        train_op = self.optimizer.apply_gradients(final_gvs)
      else:
        train_op = self.optimizer.apply_gradients(gvs)

      # We will handle the snapshots ourselves
      self.saver = tf.train.Saver(max_to_keep=100000)
      # Write the train and validation information to tensorboard
      self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)
      self.valwriter = tf.summary.FileWriter(self.tbvaldir)

    # Find previous snapshots if there is any to restore from
    sfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.ckpt.meta')
    sfiles = glob.glob(sfiles)
    sfiles.sort(key=os.path.getmtime)
    # Get the snapshot name in TensorFlow
    sfiles = [ss.replace('.meta', '') for ss in sfiles]

    nfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pkl')
    nfiles = glob.glob(nfiles)
    nfiles.sort(key=os.path.getmtime)

    lsf = len(sfiles)
    assert len(nfiles) == lsf

    np_paths = nfiles
    ss_paths = sfiles

    if lsf == 0:
      # Fresh train directly from VGG weights
      print ('Loading initial model weights from {:s}').format(self.pretrained_model)
      variables = tf.global_variables()
      # Only initialize the variables that were not initialized when the graph was built
      for vbs in self.net._initialized:
        variables.remove(vbs)
      sess.run(tf.variables_initializer(variables, name='init'))
      print 'Loaded.'
      sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE))
      last_snapshot_iter = 0
    else:
      # Get the most recent snapshot and restore
      ss_paths = [ss_paths[-1]]
      np_paths = [np_paths[-1]]

      print ('Restorining model snapshots from {:s}').format(sfiles[-1])
      self.saver.restore(sess, str(sfiles[-1]))
      print 'Restored.'
      # Needs to restore the other hyperparameters/states for training, (TODO xinlei) I have
      # tried my best to find the random states so that it can be recovered exactly
      # However the Tensorflow state is currently not available
      with open(str(nfiles[-1]), 'rb') as fid:
        st0 = cPickle.load(fid)
        cur = cPickle.load(fid)
        perm = cPickle.load(fid)
        cur_val = cPickle.load(fid)
        perm_val = cPickle.load(fid)
        last_snapshot_iter = cPickle.load(fid)

        np.random.set_state(st0)
        self.data_layer._cur = cur
        self.data_layer._perm = perm
        self.data_layer_val._cur = cur_val
        self.data_layer_val._perm = perm_val
        
        # Set the learning rate, only reduce once
        if last_snapshot_iter >= cfg.TRAIN.STEPSIZE:
          sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE * cfg.TRAIN.GAMMA))
        else:
          sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE))

    timer = Timer()
    iter = last_snapshot_iter+1
    last_summary_time = time.time()
    while iter < max_iters+1:
      # Learning rate
      if iter == cfg.TRAIN.STEPSIZE:
        sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE * cfg.TRAIN.GAMMA))

      timer.tic()
      # Get training data, one batch at a time
      blobs = self.data_layer.forward()

      now = time.time()
      if now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
        # Compute the graph with summary
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, summary = \
                        self.net.train_step_with_summary(sess, blobs, train_op)
        self.writer.add_summary(summary, float(iter))
        # Also check the summary on the validation set
        blobs_val = self.data_layer_val.forward()
        summary_val = self.net.get_summary(sess, blobs_val)
        self.valwriter.add_summary(summary_val, float(iter))
        last_summary_time = now
      else:
        # Compute the graph without summary
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss = \
                        self.net.train_step(sess, blobs, train_op)
      timer.toc()

      # Display training information
      if iter % (cfg.TRAIN.DISPLAY) == 0:
        print 'iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n >>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n >>> lr: %f'%\
              (iter, max_iters, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, lr.eval())
        print 'speed: {:.3f}s / iter'.format(timer.average_time)

      if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
        last_snapshot_iter = iter
        snapshot_path, np_path = self.snapshot(sess, iter)
        np_paths.append(np_path)
        ss_paths.append(snapshot_path)

        # Remove the old snapshots if there are too many
        if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
          to_remove = len(np_paths) - cfg.TRAIN.SNAPSHOT_KEPT
          for c in xrange(to_remove):
            nfile = np_paths[0]
            os.remove(str(nfile))
            np_paths.remove(nfile)

        if len(ss_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
          to_remove = len(ss_paths) - cfg.TRAIN.SNAPSHOT_KEPT
          for c in xrange(to_remove):
            sfile = ss_paths[0]
            # To make the code compatible to earlier versions of Tensorflow,
            # where the naming tradition for checkpoints are different
            if os.path.exists(str(sfile)):
              os.remove(str(sfile))
            else:
              os.remove(str(sfile + '.data-00000-of-00001'))
              os.remove(str(sfile + '.index'))
            sfile_meta = sfile + '.meta'
            os.remove(str(sfile_meta))
            ss_paths.remove(sfile)

      iter += 1

    if last_snapshot_iter != iter - 1:
      self.snapshot(sess, iter - 1)

    self.writer.close()
    self.valwriter.close()

def get_training_roidb(imdb):
  """Returns a roidb (Region of Interest database) for use in training."""
  if cfg.TRAIN.USE_FLIPPED:
    print 'Appending horizontally-flipped training examples...'
    imdb.append_flipped_images()
    print 'done'

  print 'Preparing training data...'
  rdl_roidb.prepare_roidb(imdb)
  print 'done'

  return imdb.roidb

def filter_roidb(roidb):
  """Remove roidb entries that have no usable RoIs."""

  def is_valid(entry):
    # Valid images have:
    #   (1) At least one foreground RoI OR
    #   (2) At least one background RoI
    overlaps = entry['max_overlaps']
    # find boxes with sufficient overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # image is only valid if such boxes exist
    valid = len(fg_inds) > 0 or len(bg_inds) > 0
    return valid

  num = len(roidb)
  filtered_roidb = [entry for entry in roidb if is_valid(entry)]
  num_after = len(filtered_roidb)
  print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                     num, num_after)
  return filtered_roidb

def train_net(network, imdb, roidb, valroidb, output_dir, tb_dir,
              pretrained_model=None, 
              max_iters=40000):
  """Train a Fast R-CNN network."""
  roidb = filter_roidb(roidb)
  valroidb = filter_roidb(valroidb)

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth=True

  with tf.Session(config=tfconfig) as sess:
    sw = SolverWrapper(sess, network, imdb, roidb, valroidb, output_dir, tb_dir,
                      pretrained_model=pretrained_model)
    print 'Solving...'
    sw.train_model(sess, max_iters)
    print 'done solving'
