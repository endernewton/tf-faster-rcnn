# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.train_val import get_training_roidb, train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
#from datasets.factory import get_imdb
from datasets.dataset import dataset
import datasets.imdb
import pprint
import numpy as np

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

def get_imdb(name, __sets):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()

def combined_roidb(imdb_names, __sets):
  """
  Combine multiple roidbs
  """

  def get_roidb(imdb_name, __sets):
    imdb = get_imdb(imdb_name, __sets)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    return roidb

  roidbs = [get_roidb(s, __sets) for s in imdb_names.split('+')]
  roidb = roidbs[0]
  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names, __sets)
  return imdb, roidb

'''
def trainval_net(cfg_file, dataset_name, classes, weight, imdb_name, imdbval_name, max_iters, tag, net):

  __sets = {}

  for split in ['train', 'val', 'trainval', 'test']:
    name = dataset_name + '_{}'.format(split)
    __sets[name] = (lambda split=split: dataset(split, classes, dataset_name))

  print('Called with parameters from configuration file:')
  print('cfg_file:', cfg_file)

  if cfg_file is not None:
    cfg_from_file(cfg_file)

  print('Using config:')
  pprint.pprint(cfg)

  np.random.seed(cfg.RNG_SEED)

  # train set
  imdb, roidb = combined_roidb(imdb_name, __sets)
  print('{:d} roidb entries'.format(len(roidb)))

  # output directory where the models are saved
  output_dir = get_output_dir(imdb, None)
  print('Output will be saved to `{:s}`'.format(output_dir))

  # tensorboard directory where the summaries are saved during training
  tb_dir = get_output_tb_dir(imdb, tag)
  print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

  # also add the validation set, but with no flipping images
  orgflip = cfg.TRAIN.USE_FLIPPED
  cfg.TRAIN.USE_FLIPPED = False
  _, valroidb = combined_roidb(imdbval_name, __sets)
  print('{:d} validation roidb entries'.format(len(valroidb)))
  cfg.TRAIN.USE_FLIPPED = orgflip

  # load network
  if net == 'vgg16':
    net = vgg16(batch_size=cfg.TRAIN.IMS_PER_BATCH)
  elif net == 'res50':
    net = resnetv1(batch_size=cfg.TRAIN.IMS_PER_BATCH, num_layers=50)
  elif net == 'res101':
    net = resnetv1(batch_size=cfg.TRAIN.IMS_PER_BATCH, num_layers=101)
  elif net == 'res152':
    net = resnetv1(batch_size=cfg.TRAIN.IMS_PER_BATCH, num_layers=152)
  else:
    raise NotImplementedError

  train_net(net, imdb, roidb, valroidb, output_dir, tb_dir,
            pretrained_model=weight,
            max_iters=max_iters)'''