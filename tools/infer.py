#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test_vgg16 import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.res101 import Resnet101

CLASSES = ('__background__',
           'new_classes')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_70000.ckpt',)}

DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),'new_dataset': ('new_dataset_trainval',)}

COLORMAP = ('#000000','#ff0000','#06ff00','#ff00e4','#ff9600','#0024ff')  #If we train for more than 5 classes we must to add some colors

def save_detections(im, fig,ax,list_data,class_name, dets, imgName, outputDir,bboxes_list, color,thresh=0.3): #Add bboxes.txt
    """Draw detected bounding boxes."""
    import matplotlib.pyplot as plt
    inds = np.where(dets[:, -1] >= thresh)[0]
    
    for i in inds:

        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=color, linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.3),
                fontsize=10, color='white')
	
        bbox_data = imgName[imgName.rfind('/')+1:] 
    	bbox_data = bbox_data+' '+class_name+' '+str(score)
    	for i in range(len(bbox)):    
    		bbox_data = bbox_data+' '+str(bbox[i])
    	bbox_data = bbox_data+'\n'
	bboxes_list.append(bbox_data) 

    list_data.append([str(len(inds)), class_name, class_name,str(thresh)])
   
    outfile = os.path.join(outputDir, imgName)

    if not os.path.isdir(args.outfolder):
        os.makedirs(args.outfolder)
     
    with open(args.outfolder+'/bboxes.txt','w') as bboxes_file:
    	for i in range(len(bboxes_list)):
    		bboxes_file.write(bboxes_list[i])
        
    return fig,ax, list_data

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    plt.use('Agg')
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def process_image(sess, net, im_file, outfolder,bboxes_list): #Add bboxes.txt
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(im_file)
    output = np.zeros((1200,1200,3))+255
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3

    fig, ax = plt.subplots(figsize=(12, 12))
    im = im[:, :, (2, 1, 0)]
    ax.imshow(im, aspect='equal')
    list_data=[]

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
	color = COLORMAP[cls_ind + 1]
    	fig,ax,list_data = save_detections(im, fig,ax,list_data,cls, dets, im_file, outfolder, bboxes_list,color, thresh=CONF_THRESH) #Add bboxes.txt

    title =''
    for data in list_data:
	text = '{} {} detections with p({} | box) >= {}'.format(data[0], data[1], data[1],data[3])
	title+=str(text)+'\n'

    ax.set_title(title,fontsize=14)
    
    (ignore, filename) = os.path.split(im_file)
    #outputDir = os.path.join(outputDir,class_name) #Changed to one dir per class
    outfile = os.path.join(outfolder, filename)

    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)

    print ("Saving test image with boxes in {}".format(outfile))

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net',
                        help='model to test',
                        default='res101', type=str)
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), 
			default=None, type=str)
    parser.add_argument('--imgfolder', dest='imgfolder',
                        help='Folder with images to use for inferece',
                        default=None, type=str)
    parser.add_argument('--outfolder', dest='outfolder',
                        help='Folder to use for storing resultant infereced images',
                        default=None, type=str)
    parser.add_argument('--imgsetfile', dest='imgsetfile',
                        help='Use a image set file (test.txt or so) to filter for files in imgfolder.',
                        default=None, type=str)

    args = parser.parse_args()

    return args

def ensure_file_exists(file):
    if not os.path.isfile(file):
        raise IOError("File {:s} not found.".format(file))

def ensure_dir_exists(dir):
    if not os.path.isdir(dir):
        raise IOError("Folder {:s} not found.".format(dir))


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('../output',demonet,DATASETS[dataset][0], 'default',
                              NETS[demonet][0])

    bboxes_list = [] #Add bboxes.txt

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly? If you want something '
                       'simple and handy, try ./tools/demo_depre.py first.').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    elif demonet == 'res101':
        net = Resnet101(batch_size=1)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", NUM_CLASSES,
                          tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    image_names = []
    if not args.imgsetfile == None:
        assert os.path.exists(args.imgsetfile), \
                'Image set file does not exist: {}'.format(args.imgsetfile)
        with open(args.imgsetfile) as f:
            im_names = [x.strip() + ".jpg" for x in f.readlines()]
    else:
        im_names = [f for f in os.listdir(args.imgfolder)
                    if isfile(os.path.join(args.imgfolder, f)) and f.lower().endswith(".jpg")]

    for im_name in im_names:
        print ('Processing {} from {} and saving into {}'.format(im_name, args.imgfolder, args.outfolder))
        process_image(sess,net, os.path.join(args.imgfolder, im_name), args.outfolder, bboxes_list) #Add bboxes.txt
