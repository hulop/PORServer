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

import os
import time
import numpy as np
from utils import gpu_utils
import tensorflow as tf
from nets.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from nets.inception_v3 import inception_v3, inception_v3_arg_scope
from nets.resnet_v2 import resnet_v2_50, resnet_v2_152, resnet_arg_scope
from nets.vgg import vgg_16, vgg_arg_scope
from nets.mobilenet_v1 import mobilenet_v1, mobilenet_v1_arg_scope
from nets.nasnet import nasnet
import inception_preprocessing
from network.portrainer_slim import load_labels

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    # print imported graph for debug
    imported_graph_nodes = [n.name for n in graph_def.node]
    for node in imported_graph_nodes:
        print("Imported node : " + str(node))
    
    return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels = 3,
                                           name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                            name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)
    
    return result

def run_inference_on_image(imagePath, model_file, labels_file, 
                           network_model='inception-v3'):
    start_time = time.time()
    
    #Image and model type
    image_size = 299
    
    if network_model!='inception-v4' and network_model!='inception-v3' and network_model!='resnet-v2-50' and \
       network_model!='resnet-v2-152' and network_model!='vgg-16' and network_model!='mobilenet-v1' and \
       network_model!='nasnet-large' and network_model!='nasnet-mobile':
        print("invalid network model : " + network_model)
        sys.exit()
    
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
    
    #Set input output layer name
    input_layer = "input"    
    if network_model=='inception-v4':
        output_layer = "InceptionResnetV2/Logits/Predictions"
    elif network_model=='inception-v3':
        output_layer = "InceptionV3/Predictions/Reshape_1"
    elif network_model=='resnet-v2-50':
        output_layer = "resnet_v2_50/predictions/Reshape_1"
    elif network_model=='resnet-v2-152':
        output_layer = "resnet_v2_152/predictions/Reshape_1"
    elif network_model=='vgg-16':
        output_layer = "vgg_16/fc8/squeezed"
    elif network_model=='mobilenet-v1':
        output_layer = "MobilenetV1/Predictions/Reshape_1"
    elif network_model=='nasnet-large' or network_model=='nasnet-mobile':
        output_layer = "final_layer/predictions"
    else:
        print("Invalid network model : " + network_model)
        sys.exit()

    # Load labels
    labels_to_name, label_list = load_labels(labels_file)
    num_classes = len(label_list)
    
    # Load model
    graph = load_graph(model_file)
    t = read_tensor_from_image_file(imagePath,
                                    input_height=image_size,
                                    input_width=image_size)
    
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);
    
    time_elapsed = time.time() - start_time
    print('time elapsed after restore : ' + str(time_elapsed) + ' sec')
    
    # Run inference
    env_cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
    if env_cuda_visible_devices is not None and len(env_cuda_visible_devices)==0:
        print('CUDA_VISIBLE_DEVICES is empty, set cpu inference mode')
        gpu_options = tf.ConfigProto(device_count={'GPU': 0})
    else:
        gpu_id = gpu_utils.pick_gpu_lowest_memory()
        print('found lowest memory gpu id : ' + str(gpu_id))
        gpu_options = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list=str(gpu_id),
                                                               per_process_gpu_memory_fraction=0.4))
    with tf.Session(config=gpu_options, graph=graph) as sess:
        prob_values = sess.run(output_operation.outputs[0],
                               {input_operation.outputs[0]: t})
    prob_values = np.squeeze(prob_values)
    
    time_elapsed = time.time() - start_time
    print('time elapsed after inference : ' + str(time_elapsed) + ' sec')

    # Getting top 5 predictions
    top_k = prob_values.argsort()[-5:][::-1]
    pred_dict = {}
    for node_id in top_k:
        human_string = str((labels_to_name[node_id]+'\n').encode('utf-8'))
        score = prob_values[node_id]
        pred_dict[human_string] = float(score)
        print('%s (score = %.5f)' % (human_string, score))
    
    return pred_dict

def run_inference_on_multi_images(multiImagePath, model_file, labels_file, 
                                  network_model='inception-v3'):
    start_time = time.time()
    
    #Image and model type
    image_size = 299
    
    if network_model!='inception-v4' and network_model!='inception-v3' and network_model!='resnet-v2-50' and network_model!='resnet-v2-152' and \
       network_model!='vgg-16' and network_model!='mobilenet-v1' and network_model!='nasnet-large' and network_model!='nasnet-mobile':
        print("invalid network model : " + network_model)
        sys.exit()
    
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

    #Set input output layer name
    input_layer = "input"    
    if network_model=='inception-v4':
        output_layer = "InceptionResnetV2/Logits/Predictions"
    elif network_model=='inception-v3':
        output_layer = "InceptionV3/Predictions/Reshape_1"
    elif network_model=='resnet-v2-50':
        output_layer = "resnet_v2_50/predictions/Reshape_1"
    elif network_model=='resnet-v2-152':
        output_layer = "resnet_v2_152/predictions/Reshape_1"
    elif network_model=='vgg-16':
        output_layer = "vgg_16/fc8/squeezed"
    elif network_model=='mobilenet-v1':
        output_layer = "MobilenetV1/Predictions/Reshape_1"
    elif network_model=='nasnet-large' or network_model=='nasnet-mobile':
        output_layer = "final_layer/predictions"
    else:
        print("Invalid network model : " + network_model)
        sys.exit()

    # Load labels
    labels_to_name, label_list = load_labels(labels_file)
    num_classes = len(label_list)
    
    # Load model
    graph = load_graph(model_file)
    tensor_images = None
    for idx, imagePath in enumerate(multiImagePath):
        t = read_tensor_from_image_file(imagePath,
                                        input_height=image_size,
                                        input_width=image_size)
        if idx==0:
            tensor_images = t
        else:
            tensor_images = np.append(tensor_images, t, axis=0)
    
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);
    
    time_elapsed = time.time() - start_time
    print('time elapsed after restore : ' + str(time_elapsed) + ' sec')
    
    # Run inference
    env_cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
    if env_cuda_visible_devices is not None and len(env_cuda_visible_devices)==0:
        print('CUDA_VISIBLE_DEVICES is empty, set cpu inference mode')
        gpu_options = tf.ConfigProto(device_count={'GPU': 0})
    else:
        gpu_id = gpu_utils.pick_gpu_lowest_memory()
        print('found lowest memory gpu id : ' + str(gpu_id))
        gpu_options = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list=str(gpu_id),
                                                               per_process_gpu_memory_fraction=0.4))
    with tf.Session(config=gpu_options, graph=graph) as sess:
        prob_values = sess.run(output_operation.outputs[0],
                               {input_operation.outputs[0]: tensor_images})
    print('prob_values=' + str(prob_values))
    prob_values = np.average(prob_values, axis=0)
    print('average prob_values=' + str(prob_values))
    
    time_elapsed = time.time() - start_time
    print('time elapsed after inference : ' + str(time_elapsed) + ' sec')
    
    # Getting top 5 predictions
    top_k = prob_values.argsort()[-5:][::-1]
    pred_dict = {}
    for node_id in top_k:
        human_string = str((labels_to_name[node_id]+'\n').encode('utf-8'))
        score = prob_values[node_id]
        pred_dict[human_string] = float(score)
        print('%s (score = %.5f)' % (human_string, score))
    
    return pred_dict
