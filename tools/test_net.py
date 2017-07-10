# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi he, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test import test_net
from model.config import cfg, cfg_from_file, cfg_from_list
from datasets.dataset import dataset
#from datasets.factory import get_imdb

import pprint
import time, os, sys

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

def get_imdb(name, __sets):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()

def testing(imdbval_name, classes, cfg_file, model, weights, tag, net, max_per_image):

  __sets = {}

  for split in ['train', 'val', 'trainval', 'test']:
    name = imdbval_name.split('_')[0] + '_{}'.format(split)
    __sets[name] = (lambda split=split: dataset(split, classes, name.split('_')[0]))

  if cfg_file is not None:
    cfg_from_file(cfg_file)

  print('Using config:')
  pprint.pprint(cfg)

  # if has model, get the name from it
  # if does not, then just use the inialization weights
  if model:
    filename = os.path.splitext(os.path.basename(model))[0]
  else:
    filename = os.path.splitext(os.path.basename(weights))[0]

  tag = tag if tag else 'default'
  filename = tag + '/' + filename
  imdb = get_imdb(imdbval_name, __sets)

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth=True

  # init session
  sess = tf.Session(config=tfconfig)
  # load network
  if net == 'vgg16':
    net = vgg16(batch_size=1)
  elif net == 'res50':
    net = resnetv1(batch_size=1, num_layers=50)
  elif net == 'res101':
    net = resnetv1(batch_size=1, num_layers=101)
  elif net == 'res152':
    net = resnetv1(batch_size=1, num_layers=152)
  else:
    raise NotImplementedError

  # load model
  net.create_architecture(sess, "TEST", imdb.num_classes, tag='default',
                          anchor_scales=cfg.ANCHOR_SCALES,
                          anchor_ratios=cfg.ANCHOR_RATIOS)

  if model:
    print(('Loading model check point from {:s}').format(model))
    saver = tf.train.Saver()
    saver.restore(sess, model)
    print('Loaded.')
  else:
    print(('Loading initial weights from {:s}').format(weights))
    sess.run(tf.global_variables_initializer())
    print('Loaded.')

  test_net(sess, net, imdb, filename, max_per_image=max_per_image)

  sess.close()
