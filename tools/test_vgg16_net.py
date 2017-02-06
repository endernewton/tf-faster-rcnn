import _init_paths
from model.test import test_net
from model.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys

import tensorflow as tf
from nets.vgg16 import vgg16

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
            help='optional config file', default=None, type=str)
  parser.add_argument('--model', dest='model',
            help='model to test',
            default=None, type=str)
  parser.add_argument('--weight', dest='weight',
            help='weight to test',
            default=None, type=str)
  parser.add_argument('--imdb', dest='imdb_name',
            help='dataset to test',
            default='voc_2007_test', type=str)
  parser.add_argument('--comp', dest='comp_mode', help='competition mode',
            action='store_true')
  parser.add_argument('--num_dets', dest='max_per_image',
            help='max number of detections per image',
            default=100, type=int)
  parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = parse_args()

  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  # if has model, get the name from it
  # if does not, then just use the inialization weights
  if args.model:
    filename = os.path.splitext(os.path.basename(args.model))[0]
  else:
    filename = os.path.splitext(os.path.basename(args.weight))[0]

  tag = args.tag
  tag = tag if tag else 'default'
  filename = tag + '/' + filename

  imdb = get_imdb(args.imdb_name)
  imdb.competition_mode(args.comp_mode)

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth=True

  # init session
  sess = tf.Session(config=tfconfig)
  # load network
  net = vgg16(batch_size=1)
  # load model
  if imdb.name.startswith('voc'):
    anchors = [8, 16, 32]
  else:
    anchors = [4, 8, 16, 32]

  net.create_architecture(sess, "TEST", imdb.num_classes, caffe_weight_path=args.weight, 
                          tag='default', anchor_scales=anchors)

  if args.model:
    print ('Loading model check point from {:s}').format(args.model)
    saver = tf.train.Saver()
    saver.restore(sess, args.model)
    print 'Loaded.'
  else:
    print ('Loading initial weights from {:s}').format(args.weight)
    sess.run(tf.global_variables_initializer())
    print 'Loaded.'

  test_net(sess, net, imdb, filename, max_per_image=args.max_per_image)

  sess.close()
