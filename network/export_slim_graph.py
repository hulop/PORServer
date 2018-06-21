##############################################################################
#The MIT License (MIT)
#
#Copyright (c) 2018 IBM Corporation, Carnegie Mellon University and others
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import tensorflow as tf
import tensorflow.contrib.image
from nets.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from nets.inception_v3 import inception_v3, inception_v3_arg_scope
from nets.resnet_v2 import resnet_v2_50, resnet_v2_152, resnet_arg_scope
from nets.vgg import vgg_16, vgg_arg_scope
from nets.mobilenet_v1 import mobilenet_v1, mobilenet_v1_arg_scope
from nets.nasnet import nasnet
import argparse
import os
import sys

slim = tf.contrib.slim

def main(args):
    network_model = args.network_model
    num_classes = args.num_classes

    #Set image size
    if network_model=='inception-v4' or network_model=='inception-v3' or network_model=='resnet-v2-50' or network_model=='resnet-v2-152':
        image_size = 299
    elif network_model=='vgg-16' or network_model=='mobilenet-v1' or network_model=='nasnet-mobile':
        image_size = 224
    elif network_model=='nasnet-large':
        image_size = 331
    else:
        print("invalid network model : " + network_model)
        sys.exit()
    
    with tf.Graph().as_default() as graph:
        images = tf.placeholder(name='input', dtype=tf.float32,
                                shape=[None, image_size, image_size, 3])
        
        if network_model=='inception-v4':
            with slim.arg_scope(inception_resnet_v2_arg_scope()):
                logits, end_points = inception_resnet_v2(images, num_classes = num_classes, is_training = False, create_aux_logits=False)
        elif network_model=='inception-v3':
            with slim.arg_scope(inception_v3_arg_scope()):
                logits, end_points = inception_v3(images, num_classes = num_classes, is_training = False)
        elif network_model=='resnet-v2-50':
            with slim.arg_scope(resnet_arg_scope()):
                logits, end_points = resnet_v2_50(images, num_classes = num_classes, is_training = False)
        elif network_model=='resnet-v2-152':
            with slim.arg_scope(resnet_arg_scope()):
                logits, end_points = resnet_v2_152(images, num_classes = num_classes, is_training = False)
        elif network_model=='vgg-16':
            with slim.arg_scope(vgg_arg_scope()):
                logits, _ = vgg_16(images, num_classes = num_classes, is_training = False)
        elif network_model=='mobilenet-v1':
            with slim.arg_scope(mobilenet_v1_arg_scope()):
                logits, end_points = mobilenet_v1(images, num_classes = num_classes, is_training = False)
        elif network_model=='nasnet-large':
            with slim.arg_scope(nasnet.nasnet_large_arg_scope()):            
                logits, end_points = nasnet.build_nasnet_large(images, num_classes = num_classes, is_training = False)
        elif network_model=='nasnet-mobile':
            with slim.arg_scope(nasnet.nasnet_mobile_arg_scope()):            
                logits, end_points = nasnet.build_nasnet_mobile(images, num_classes = num_classes, is_training = False)
        else:
            print("Invalid network model : " + network_model)
            sys.exit()
        
        graph_def = graph.as_graph_def()
        with gfile.GFile(args.output_file, 'wb') as f:
            f.write(graph_def.SerializeToString())
  
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('network_model', type=str, 
                        help='Network model')
    parser.add_argument('num_classes', type=int, 
                        help='Number of classes')
    parser.add_argument('output_file', type=str, 
                        help='Filename for the exported graphdef protobuf (.pb)')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
