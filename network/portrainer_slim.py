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

import random
import os
import shutil
import time
import sys
import subprocess
from utils.dataset_utils import _dataset_exists, _get_filenames_and_classes, write_label_file, _convert_dataset
from utils import gpu_utils
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import freeze_graph
import inception_preprocessing_noaug
import inception_preprocessing_rotaug
from nets.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from nets.inception_v3 import inception_v3, inception_v3_arg_scope
from nets.resnet_v2 import resnet_v2_50, resnet_v2_152, resnet_arg_scope
from nets.vgg import vgg_16, vgg_arg_scope
from nets.mobilenet_v1 import mobilenet_v1, mobilenet_v1_arg_scope
from nets.nasnet import nasnet
from por_server import controller, exc

slim = tf.contrib.slim

def load_labels(labels_file):
    labels = open(labels_file, 'r')
    
    labels_to_name = {}
    label_list = []
    for line in labels:
        label, string_name = line.split(':')
        string_name = string_name[:-1]
        labels_to_name[int(label)] = string_name
        label_list.append(string_name)
    return labels_to_name, label_list

def get_split(split_name, dataset_dir, file_pattern, file_pattern_for_counting, labels_to_name, num_classes, items_to_descriptions):
    if split_name not in ['train', 'validation']:
        raise exc.InvalidData('Invalid split name for training', status_code=404)
    
    # file path for tfrecord_files
    file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))
    
    # get number of examples
    num_samples = 0
    file_pattern_for_counting = file_pattern_for_counting + '_' + split_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.startswith(file_pattern_for_counting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1
    
    # create keys_to_features for the decoder
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }
    
    # create items_to_handlers for the decoder.
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }
    
    # create dataset
    reader = tf.TFRecordReader
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    dataset = slim.dataset.Dataset(
        data_sources = file_pattern_path,
        decoder = decoder,
        reader = reader,
        num_readers = 4,
        num_samples = num_samples,
        num_classes = num_classes,
        labels_to_name = labels_to_name,
        items_to_descriptions = items_to_descriptions)

    return dataset

def load_batch(dataset, batch_size, data_augmentation, mix_up, height, width, is_training=True):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity = 24 + 3 * batch_size,
        common_queue_min = 24)
    
    raw_image, label = data_provider.get(['image', 'label'])
    
    if data_augmentation:
        image = inception_preprocessing_rotaug.preprocess_image(raw_image, height, width, is_training)
    else:
        image = inception_preprocessing_noaug.preprocess_image(raw_image, height, width, is_training)
    
    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
    raw_image = tf.squeeze(raw_image)
    
    images, raw_images, labels = tf.train.batch(
        [image, raw_image, label],
        batch_size = batch_size,
        num_threads = 4,
        capacity = 4 * batch_size,
        allow_smaller_final_batch = True)
    
    # Experimental function for Mixup
    if mix_up:
        # parameters
        alpha = 0.2
        
        # circular shift in batch dimension
        def cshift(values):
            return tf.concat([values[-1:, ...], values[:-1, ...]], 0)
        
        orig_images_dtype = images.dtype
        orig_raw_images_dtype = raw_images.dtype
        
        images = tf.image.convert_image_dtype(images, tf.float32)
        raw_images = tf.image.convert_image_dtype(raw_images, tf.float32)
        labels = slim.one_hot_encoding(labels, dataset.num_classes)
        labels = tf.image.convert_image_dtype(labels, tf.float32)
        
        beta = tf.distributions.Beta(alpha, alpha)
        lam = beta.sample(batch_size)
        ll_image = tf.expand_dims(tf.expand_dims(tf.expand_dims(lam, -1), -1), -1)
        ll_label = tf.expand_dims(lam, -1)
        images = ll_image * images + (1 - ll_image) * cshift(images)
        raw_images = ll_image * raw_images + (1 - ll_image) * cshift(raw_images)
        labels = ll_label * labels + (1 - ll_label) * cshift(labels)
        
        images = tf.image.convert_image_dtype(images, orig_images_dtype)
        raw_images = tf.image.convert_image_dtype(raw_images, orig_raw_images_dtype)
        
        tf.summary.image('mixup_image', images)
    
    return images, raw_images, labels

def get_variables_to_train_by_scopes(scopes):
    variables_to_train = []
    for scope in scopes:
        variables = slim.get_trainable_variables(scope=scope)
        variables_to_train.extend(variables)
    return variables_to_train

def run_training(path_db, pid, category, task_id, path_unknown,
                 pretrained_dir, tensorflow_dir, path_save,
                 num_epochs = 1000,
                 batch_size = 32,
                 finetune_last_layer=False,
                 data_augmentation=True,
                 mix_up=False,
                 network_model='inception-v3',
                 restore_all_parameters=False,
                 initial_learning_rate=0.0002,
                 learning_rate_decay_factor=0.7,
                 num_epochs_before_decay=2):
    ##### start parameters for creating TFRecord files #####
    #validation_size = 0.1
    validation_size = 0.0
    num_shards = 2
    random_seed = 0
    ##### end parameters for creating TFRecord files #####
    
    dataset_dir = os.path.join(path_db, pid, category)
    log_dir = path_save
    tfrecord_filename = pid + '_' + category
    
    if _dataset_exists(dataset_dir = dataset_dir, _NUM_SHARDS = num_shards, output_filename = tfrecord_filename):
        print('Dataset files already exist. Overwrite them.')
  
    photo_filenames, class_names = _get_filenames_and_classes(dataset_dir, path_unknown)
    
    # dictionary for class name and class ID
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    
    # number of validation examples
    num_validation = int(validation_size * len(photo_filenames))
    
    # divide to training and validation data
    random.seed(random_seed)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[num_validation:]
    validation_filenames = photo_filenames[:num_validation]
    
    # find available GPU ID
    gpu_id = gpu_utils.pick_gpu_lowest_memory()
    
    # if log directory does not exist, create log directory and dataset
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
      
        print('found lowest memory gpu id : ' + str(gpu_id))  
        _convert_dataset(gpu_id, 'train', training_filenames, class_names_to_ids,
                         dataset_dir = dataset_dir, tfrecord_filename = tfrecord_filename, _NUM_SHARDS = num_shards)
        _convert_dataset(gpu_id, 'validation', validation_filenames, class_names_to_ids,
                         dataset_dir = dataset_dir, tfrecord_filename = tfrecord_filename, _NUM_SHARDS = num_shards)
      
        labels_to_class_names = dict(zip(range(len(class_names)), class_names))
        write_label_file(labels_to_class_names, dataset_dir)
  
    print('finished creating dataset ' + tfrecord_filename)
  
    # start training
    output_label_filepath = os.path.join(dataset_dir, 'labels.txt')

    if network_model!='inception-v4' and network_model!='inception-v3' and network_model!='resnet-v2-50' and network_model!='resnet-v2-152' and \
       network_model!='vgg-16' and network_model!='mobilenet-v1' and network_model!='nasnet-large' and network_model!='nasnet-mobile':
        print("invalid network model : " + network_model)
        sys.exit()
    
    # find pretrained model
    if os.path.exists(os.path.join(log_dir, 'model.ckpt')):
        checkpoint_file = os.path.join(log_dir, 'model.ckpt')
    else:
        if network_model=='inception-v4':
            checkpoint_file = os.path.join(pretrained_dir, 'inception_resnet_v2_2016_08_30.ckpt')
        elif network_model=='inception-v3':
            checkpoint_file = os.path.join(pretrained_dir, 'inception_v3.ckpt')
        elif network_model=='resnet-v2-50':
            checkpoint_file = os.path.join(pretrained_dir, 'resnet_v2_50.ckpt')
        elif network_model=='resnet-v2-152':
            checkpoint_file = os.path.join(pretrained_dir, 'resnet_v2_152.ckpt')
        elif network_model=='vgg-16':
            checkpoint_file = os.path.join(pretrained_dir, 'vgg_16.ckpt')
        elif network_model=='mobilenet-v1':
            checkpoint_file = os.path.join(pretrained_dir, 'mobilenet_v1_1.0_224.ckpt')
        elif network_model=='nasnet-large':
            checkpoint_file = os.path.join(pretrained_dir, 'nasnet-a_large_04_10_2017', 'model.ckpt')
        elif network_model=='nasnet-mobile':
            checkpoint_file = os.path.join(pretrained_dir, 'nasnet-a_mobile_04_10_2017', 'model.ckpt')
        else:
            print("invalid network model : " + network_model)
            sys.exit()
    
    # set image size
    if network_model=='inception-v4' or network_model=='inception-v3' or network_model=='resnet-v2-50' or network_model=='resnet-v2-152':
        image_size = 299
    elif network_model=='vgg-16' or network_model=='mobilenet-v1' or network_model=='nasnet-mobile':
        image_size = 224
    elif network_model=='nasnet-large':
        image_size = 331
    else:
        print("invalid network model : " + network_model)
        sys.exit()
    
    # create the file pattern of TFRecord files
    file_pattern = tfrecord_filename + '_%s_*.tfrecord'
    file_pattern_for_counting = tfrecord_filename
    
    labels_to_name, label_list = load_labels(output_label_filepath)
    num_classes = len(label_list)
    
    # create a dataset discription
    items_to_descriptions = {
        'image': 'A 3-channel RGB coloured image that is either ' + ','.join(label_list),
        'label': 'A label that is as such -- ' + ','.join([str(key)+':'+labels_to_name[key] for key in labels_to_name.keys()])
    }
    
    # start training
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)
    
        # create dataset and load one batch
        dataset = get_split('train', dataset_dir, file_pattern, file_pattern_for_counting, labels_to_name, num_classes, items_to_descriptions)
        images, _, labels = load_batch(dataset, batch_size=batch_size, data_augmentation=data_augmentation, mix_up=mix_up,
                                       height=image_size, width=image_size)
    
        # number of steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = int(dataset.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch # because one step is one batch processed
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)
        
        # create model for inference
        finetune_vars = []
        if network_model=='inception-v4':
            with slim.arg_scope(inception_resnet_v2_arg_scope()):
                logits, end_points = inception_resnet_v2(images, num_classes = dataset.num_classes, is_training = True)

            finetune_vars = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
        elif network_model=='inception-v3':
            with slim.arg_scope(inception_v3_arg_scope()):
                logits, end_points = inception_v3(images, num_classes = dataset.num_classes, is_training = True)

            finetune_vars = ['InceptionV3/Logits', 'InceptionV3/AuxLogits']
        elif network_model=='resnet-v2-50':
            with slim.arg_scope(resnet_arg_scope()):
                logits, end_points = resnet_v2_50(images, num_classes = dataset.num_classes, is_training = True)

            finetune_vars = ['resnet_v2_50/logits']
        elif network_model=='resnet-v2-152':
            with slim.arg_scope(resnet_arg_scope()):
                logits, end_points = resnet_v2_152(images, num_classes = dataset.num_classes, is_training = True)
                
            finetune_vars = ['resnet_v2_152/logits']
        elif network_model=='vgg-16':
            with slim.arg_scope(vgg_arg_scope()):
                logits, _ = vgg_16(images, num_classes = dataset.num_classes, is_training = True)

            finetune_vars = ['vgg_16/fc8']
        elif network_model=='mobilenet-v1':
            with slim.arg_scope(mobilenet_v1_arg_scope()):
                logits, end_points = mobilenet_v1(images, num_classes = dataset.num_classes, is_training = True)

            finetune_vars = ['MobilenetV1/Logits']
        elif network_model=='nasnet-large':
            with slim.arg_scope(nasnet.nasnet_large_arg_scope()):
                logits, end_points = nasnet.build_nasnet_large(images, dataset.num_classes)
    
            finetune_vars = ['final_layer', 'aux_11', 'cell_stem_0/comb_iter_0/left/global_step']
        elif network_model=='nasnet-mobile':
            with slim.arg_scope(nasnet.nasnet_mobile_arg_scope()):
                logits, end_points = nasnet.build_nasnet_mobile(images, dataset.num_classes)
    
            finetune_vars = ['final_layer', 'aux_7']
        else:
            print("Invalid network model : " + network_model)
            sys.exit()
        
        # define the scopes that you want to exclude for restoration
        exclude = []
        if not restore_all_parameters:
            exclude = finetune_vars
        variables_to_restore = slim.get_variables_to_restore(exclude = exclude)

        if mix_up:
            labels.set_shape([batch_size, dataset.num_classes])
            logits.set_shape([batch_size, dataset.num_classes])
            loss = tf.losses.sigmoid_cross_entropy(labels, logits)
        else:
            # perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!)
            one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
            
            # performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
            loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = logits)
        total_loss = tf.losses.get_total_loss()    #obtain the regularization losses as well
    
        # create the global step for monitoring the learning_rate and training.
        global_step = tf.train.get_or_create_global_step()
    
        # define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(
            learning_rate = initial_learning_rate,
            global_step = global_step,
            decay_steps = decay_steps,
            decay_rate = learning_rate_decay_factor,
            staircase = True)
        
        # define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate = lr)
    
        # create train_op
        if finetune_last_layer:
            variables_to_train = get_variables_to_train_by_scopes(finetune_vars)
            print("finetune variables : " + str(variables_to_train))
            train_op = slim.learning.create_train_op(total_loss, optimizer, variables_to_train=variables_to_train)
        else:        
            train_op = slim.learning.create_train_op(total_loss, optimizer)
    
        # define prediction matrix
        if network_model=='inception-v4' or network_model=='inception-v3' or network_model=='mobilenet-v1' or \
           network_model=='nasnet-large' or network_model=='nasnet-mobile':
            predictions = tf.argmax(end_points['Predictions'], 1)
            probabilities = end_points['Predictions']
        elif network_model=='resnet-v2-50' or network_model=='resnet-v2-152':
            predictions = tf.argmax(end_points['predictions'], 1)
            probabilities = end_points['predictions']
        elif network_model=='vgg-16':
            predictions = tf.argmax(logits, 1)
            probabilities = tf.nn.softmax(logits)
        else:
            print("Invalid network model : " + network_model)
            sys.exit()
        if mix_up:
            argmax_labels = tf.argmax(labels, 1)
            accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, argmax_labels)
        else:
            accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        metrics_op = tf.group(accuracy_update, probabilities)
    
        # create summaries
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', lr)
        my_summary_op = tf.summary.merge_all()
    
        # defube training step function that runs both the train_op, metrics_op and updates the global_step concurrently
        def train_step(sess, train_op, global_step):
            # check the time for each sess run
            start_time = time.time()
            total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
            time_elapsed = time.time() - start_time
            
            # run the logging to print some results
            logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)
            
            return total_loss, int(global_step_count)

        # create a saver function that actually restores the variables from a checkpoint file
        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        # define your supervisor for running a managed session
        sv = tf.train.Supervisor(logdir = log_dir, summary_op = None, init_fn = restore_fn)
        
        # run the managed session
        start_train_time = time.time()
    
        gpu_options = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list=str(gpu_id),
                                                               per_process_gpu_memory_fraction=0.4))
    
        with sv.prepare_or_wait_for_session(config=gpu_options) as sess:
            for step in range(num_steps_per_epoch * num_epochs):
                # check if training task is not canceled
                if not controller.check_train_task_alive(pid, category, task_id):
                    print('Training task is canceled.')
                    sv.stop()
                    return False, "", "", output_label_filepath, global_step_count
        
                # at the start of every epoch, show the vital information:
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, num_epochs)
                    learning_rate_value, accuracy_value = sess.run([lr, accuracy])
                    logging.info('Current Learning Rate: %s', learning_rate_value)
                    logging.info('Current Streaming Accuracy: %s', accuracy_value)
          
                    # optionally, print your logits and predictions for a sanity check that things are going fine.
                    logits_value, probabilities_value, predictions_value, labels_value = sess.run([logits, probabilities, predictions, labels])
                    print('logits: \n', logits_value)
                    print('Probabilities: \n', probabilities_value)
                    print('predictions: \n', predictions_value)
                    print('Labels:\n:', labels_value)
        
                # log the summaries every 10 step.
                if step % 10 == 0:
                    loss, global_step_count = train_step(sess, train_op, sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)
        
                # if not, simply run the training step
                else:
                    loss, global_step_count = train_step(sess, train_op, sv.global_step)
        
                # if specific time passes, save model for evaluation
                time_elapsed_train = time.time() - start_train_time
                print('training time : ' + str(time_elapsed_train))
      
            # log the final training loss and accuracy
            logging.info('Training Progress : %.2f %% ', 100.0*step/float(num_steps_per_epoch * num_epochs))
            logging.info('Final Loss: %s', loss)
            logging.info('Global Step: %s', global_step_count)
            logging.info('Final Accuracy: %s', sess.run(accuracy))
            
            # after all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            sv.saver.save(sess, sv.save_path, global_step = sv.global_step)
      
            # save graph definition file
            output_graph_filepath = os.path.join(log_dir, 'graph.pb')
            export_graph_command_exec = "./network/export_slim_graph.py"
            if not os.path.exists(export_graph_command_exec):
                print("fatal error, cannot find command : " + export_graph_command_exec)
                sys.exit()
            export_graph_command_env = os.environ.copy()
            export_graph_command_env["CUDA_VISIBLE_DEVICES"] = ''
            export_graph_command = []
            export_graph_command.append(sys.executable)
            export_graph_command.append(export_graph_command_exec)
            export_graph_command.append(network_model)
            export_graph_command.append(str(dataset.num_classes))
            export_graph_command.append(output_graph_filepath)
            print("start exec:" + " ".join(export_graph_command))
            proc = subprocess.Popen(export_graph_command, env=export_graph_command_env)
            print("export graph process ID=" + str(proc.pid))
            controller.upsert_train_child_process(task_id, proc.pid)
            proc.communicate()
            controller.delete_train_child_process(task_id, proc.pid)
            print("finish exec:" + " ".join(export_graph_command))
            if not controller.check_train_task_alive(pid, category, task_id):
                print('Training task is canceled.')
                sv.stop()
                return False, "", "", output_label_filepath, global_step_count
      
            # save frozon graph, optimized graph, and quantized graph from graph definition and checkpoint
            latest_checkpoint_filepath = tf.train.latest_checkpoint(log_dir)
      
            # you can check output node name by tensorflow/tools/graph_transforms::summarize_graph
            # https://github.com/tensorflow/models/tree/master/research/slim#Export
            output_node_names = ""
            if network_model=='inception-v4':          
                output_node_names = "InceptionResnetV2/Logits/Predictions"
            elif network_model=='inception-v3':
                output_node_names = "InceptionV3/AuxLogits/SpatialSqueeze,InceptionV3/Predictions/Reshape_1"
            elif network_model=='resnet-v2-50':
                output_node_names = "resnet_v2_50/predictions/Reshape_1"
            elif network_model=='resnet-v2-152':
                output_node_names = "resnet_v2_152/predictions/Reshape_1"
            elif network_model=='vgg-16':
                output_node_names = "vgg_16/fc8/squeezed"
            elif network_model=='mobilenet-v1':
                output_node_names = "MobilenetV1/Predictions/Reshape_1"
            elif network_model=='nasnet-large' or network_model=='nasnet-mobile':
                output_node_names = "final_layer/predictions"
            else:
                print("Invalid network model : " + network_model)
                sys.exit()
      
            output_frozen_graph_filepath = os.path.join(log_dir, 'frozen_graph.pb')
            freeze_graph_command_exec = os.path.join(tensorflow_dir, "bazel-bin/tensorflow/python/tools/freeze_graph")
            if not os.path.exists(freeze_graph_command_exec):
                print("fatal error, cannot find command : " + freeze_graph_command_exec)
                sys.exit()
            freeze_graph_command_env = os.environ.copy()
            freeze_graph_command_env["CUDA_VISIBLE_DEVICES"] = ''
            freeze_graph_command = []
            freeze_graph_command.append(freeze_graph_command_exec)
            freeze_graph_command.append("--input_graph=" + output_graph_filepath)
            freeze_graph_command.append("--input_checkpoint=" + latest_checkpoint_filepath)
            freeze_graph_command.append("--input_binary=true")
            freeze_graph_command.append("--output_graph=" + output_frozen_graph_filepath)
            freeze_graph_command.append("--output_node_names=" + output_node_names)
            print("start exec:" + " ".join(freeze_graph_command))
            proc = subprocess.Popen(freeze_graph_command, env=freeze_graph_command_env)
            print("freeze graph process ID=" + str(proc.pid))
            controller.upsert_train_child_process(task_id, proc.pid)
            proc.communicate()
            controller.delete_train_child_process(task_id, proc.pid)
            print("finish exec:" + " ".join(freeze_graph_command))
            if not controller.check_train_task_alive(pid, category, task_id):
                print('Training task is canceled.')
                sv.stop()
                return False, "", "", output_label_filepath, global_step_count
      
            output_optimized_graph_filepath = os.path.join(log_dir, 'optimized_graph.pb')
            optimize_graph_command_exec = os.path.join(tensorflow_dir, "bazel-bin/tensorflow/python/tools/optimize_for_inference")
            if not os.path.exists(optimize_graph_command_exec):
                print("fatal error, cannot find command : " + optimize_graph_command_exec)
                sys.exit()
            optimize_graph_command_env = os.environ.copy()
            optimize_graph_command_env["CUDA_VISIBLE_DEVICES"] = ''
            optimize_graph_command = []
            optimize_graph_command.append(optimize_graph_command_exec)
            optimize_graph_command.append("--input=" + output_frozen_graph_filepath)
            optimize_graph_command.append("--output=" + output_optimized_graph_filepath)
            optimize_graph_command.append("--input_names=input")
            optimize_graph_command.append("--output_names=" + output_node_names)
            optimize_graph_command.append("--frozen_graph=true")
            print("start exec:" + " ".join(optimize_graph_command))
            proc = subprocess.Popen(optimize_graph_command, env=optimize_graph_command_env)
            print("optimize graph process ID=" + str(proc.pid))
            controller.upsert_train_child_process(task_id, proc.pid)
            proc.communicate()
            controller.delete_train_child_process(task_id, proc.pid)
            print("finish exec:" + " ".join(optimize_graph_command))
            if not controller.check_train_task_alive(pid, category, task_id):
                print('Training task is canceled.')
                sv.stop()
                return False, "", "", output_label_filepath, global_step_count
      
            output_quantized_graph_filepath = os.path.join(log_dir, 'quantized_graph.pb')
            quantize_graph_command_exec = os.path.join(tensorflow_dir, "bazel-bin/tensorflow/tools/quantization/quantize_graph")
            if not os.path.exists(quantize_graph_command_exec):
                print("fatal error, cannot find command : " + quantize_graph_command_exec)
                sys.exit()
            quantize_graph_command_env = os.environ.copy()
            quantize_graph_command_env["CUDA_VISIBLE_DEVICES"] = ''
            quantize_graph_command = []
            quantize_graph_command.append(quantize_graph_command_exec)
            quantize_graph_command.append("--input=" + output_optimized_graph_filepath)
            quantize_graph_command.append("--output=" + output_quantized_graph_filepath)
            quantize_graph_command.append("--input_node_names=input")
            quantize_graph_command.append("--output_node_names=" + output_node_names)
            quantize_graph_command.append("--mode=eightbit")
            print("start exec:" + " ".join(quantize_graph_command))
            proc = subprocess.Popen(quantize_graph_command, env=quantize_graph_command_env)
            print("quantize graph process ID=" + str(proc.pid))
            controller.upsert_train_child_process(task_id, proc.pid)
            proc.communicate()
            controller.delete_train_child_process(task_id, proc.pid)
            print("finish exec:" + " ".join(quantize_graph_command))
            if not controller.check_train_task_alive(pid, category, task_id):
                print('Training task is canceled.')
                sv.stop()
                return False, "", "", output_label_filepath, global_step_count
  
    return True, output_optimized_graph_filepath, output_quantized_graph_filepath, output_label_filepath, global_step_count
