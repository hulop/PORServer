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
import shutil
import distutils.dir_util
import datetime
from por_server.models import models
from por_server import utils
from network import portester_slim, portrainer_slim
from enum import Enum
from multiprocessing import Process, Queue

class TrainingStatus(Enum):
    not_trained = 0
    training = 1
    basic_trained = 2
    final_trained = 3
    error = 4

class TrainingMode(Enum):
    finetune = "finetune"
    full_train = "full_train"

def _create_model_record(PID, Category, TrainingStatus):
    model = {
        "PID": PID,
        "Category": Category,
        "TrainingStatus": TrainingStatus.value,
        "UpdateTime": datetime.datetime.utcnow()
    }
    return model

def _create_record(PID, Category, Label, DirPath):
    record = {
        "PID": PID,
        "Category": Category,
        "Label": Label,
        "DirPath": DirPath,
        "UpdateTime": datetime.datetime.utcnow()
    }
    return record

## Need to launch subprocess to free GPU memory
def run_inference(PID, Category, filepath):
    print('Run inference is called. PID=' + PID + ', Category=' + Category + ', filepath=' + filepath)
    q = Queue()
    p = Process(target=_run_inference, args=(PID, Category, filepath, q))
    p.start()
    p.join()
    return q.get()

def _run_inference(PID, Category, filepath, queue):
    print('Run inference subprocess is called. PID=' + PID + ', Category=' + Category + ', filepath=' + filepath)
    query_dict = {
        "PID": PID,
        "Category": Category
    }
    returned_models = models().get_models_sorted(query_dict)
    ModelPath = None
    LabelsPath = None
    NetworkModel = None
    if returned_models:
        for model in returned_models:
            if 'ModelPath' in model and 'LabelsPath' in model and 'NetworkModel' in model:
                ModelPath = model['ModelPath']
                LabelsPath = model['LabelsPath']
                NetworkModel = model['NetworkModel']
                break
    
    result = None
    if ModelPath and LabelsPath and NetworkModel:
        print("Model file=%s, Label file=%s, Network model=%s" % (ModelPath, LabelsPath, NetworkModel))
        result = portester_slim.run_inference_on_image(filepath, ModelPath, LabelsPath, network_model=NetworkModel)
    
    queue.put(result)

## Need to launch subprocess to free GPU memory
def run_inference_multi(PID, Category, multi_filepath):
    print('Run inference multi is called. PID=' + PID + ', Category=' + Category + ', multi_filepath=' + str(multi_filepath))
    q = Queue()
    p = Process(target=_run_inference_multi, args=(PID, Category, multi_filepath, q))
    p.start()
    p.join()
    return q.get()

def _run_inference_multi(PID, Category, multi_filepath, queue):
    print('Run inference multi subprocess is called. PID=' + PID + ', Category=' + Category + ', multi_filepath=' + str(multi_filepath))
    query_dict = {
        "PID": PID,
        "Category": Category
    }
    returned_models = models().get_models_sorted(query_dict)
    ModelPath = None
    LabelsPath = None
    NetworkModel = None
    if returned_models:
        for model in returned_models:
            if 'ModelPath' in model and 'LabelsPath' in model and 'NetworkModel' in model:
                ModelPath = model['ModelPath']
                LabelsPath = model['LabelsPath']
                NetworkModel = model['NetworkModel']
                break
    
    result = None
    if ModelPath and LabelsPath and NetworkModel:
        print("Model file=%s, Label file=%s, Network model=%s" % (ModelPath, LabelsPath, NetworkModel))
        result = portester_slim.run_inference_on_multi_images(multi_filepath, ModelPath, LabelsPath, network_model=NetworkModel)
    
    queue.put(result)

def run_training(PID, Category, task_id, upload_folder, unknown_folder, pretrained_folder,
                 tensorflow_folder, model_folder, network_model, num_epochs, finetune_last_layer):
    if finetune_last_layer:
        finish_training, output_graph, quantized_graph, output_labels, global_step_count = portrainer_slim.run_training(upload_folder, PID, Category, task_id,
                                                                                                                        unknown_folder, pretrained_folder, tensorflow_folder,
                                                                                                                        model_folder, num_epochs = num_epochs,
                                                                                                                        network_model=network_model,
                                                                                                                        finetune_last_layer=True, data_augmentation=True,
                                                                                                                        initial_learning_rate=0.01, learning_rate_decay_factor=1.0)
    else:
        finish_training, output_graph, quantized_graph, output_labels, global_step_count = portrainer_slim.run_training(upload_folder, PID, Category, task_id,
                                                                                                                        unknown_folder, pretrained_folder, tensorflow_folder,
                                                                                                                        model_folder, num_epochs = num_epochs,
                                                                                                                        network_model=network_model,
                                                                                                                        finetune_last_layer=False, data_augmentation=True,
                                                                                                                        initial_learning_rate=0.0002, learning_rate_decay_factor=0.7)
    return finish_training, output_graph, quantized_graph, output_labels, global_step_count

def insert_record(PID, Category, Label, DirPath):
    record = _create_record(PID, Category, Label, DirPath)
    result = models().save_record(record)
    return result

def insert_model(PID, Category):
    query_dict = {
        "PID": PID,
        "Category": Category
    }
    output = models().ifModelExists(query_dict)
    if output:
        record = models().get_one_model(query_dict)
        return record['_id']
    else:
        model = _create_model_record(PID, Category, TrainingStatus.not_trained)
        result = models().save_model(model)
        return result

def update_model_status(PID, Category, status, train_mode=None, global_step_count=None):
    query_dict = {
        "PID": PID,
        "Category": Category
    }
    model = models().get_one_model(query_dict)
    if model:
        query_dict = {
            "_id": model['_id']
        }
        update_dict = {
            "TrainingStatus": status.value,
            "UpdateTime": datetime.datetime.utcnow()
        }
        if train_mode is not None:
            update_dict["TrainingMode"] = train_mode.value
        if global_step_count is not None:
            update_dict["GlobalStepCount"] = global_step_count
        models().update_one_model(query_dict, update_dict)

def delete_category(PID, Category, image_directory, model_directory, keep_deleted_image_directory=None):
    query_dict = {
        "PID": PID,
        "Category": Category
    }
    
    if os.path.exists(image_directory):
        if keep_deleted_image_directory is not None:
            print("[delete_category] keep deleting image directory : " + keep_deleted_image_directory)
            distutils.dir_util.copy_tree(image_directory, keep_deleted_image_directory)
        print("[delete_category] removing image directory : " + image_directory)
        shutil.rmtree(image_directory)
    else:
        print("[delete_category] cannot find image directory : " + image_directory)
            
    if os.path.exists(model_directory):
        print("[delete_category] removing model directory : " + model_directory)
        shutil.rmtree(model_directory)
    else:
        print("[delete_category] cannot find model directory : " + model_directory)
    
    models().delete_records(query_dict)
    models().delete_models(query_dict)
    return True

def delete_label(PID, Category, Label, keep_deleted_image_directory=None):
    query_dict = {
        "PID": PID,
        "Category": Category,
        "Label": Label
    }
    record = models().get_one_record(query_dict, "_id", -1)
    if record and ('DirPath' in record or 'ImagePath' in record):
        if 'DirPath' in record:
            if os.path.exists(record['DirPath']):
                if keep_deleted_image_directory is not None:
                    print("[delete_category] keep deleting image directory : " + keep_deleted_image_directory)
                    distutils.dir_util.copy_tree(record['DirPath'], keep_deleted_image_directory)
                print("[delete_label] removing directory : " + record['DirPath'])
                shutil.rmtree(record['DirPath'])
            else:
                print("[delete_label] cannot find directory : " + record['DirPath'])
        else:
            if os.path.exists(os.path.dirname(record['ImagePath'])):
                if keep_deleted_image_directory is not None:
                    print("[delete_category] keep deleting image directory : " + keep_deleted_image_directory)
                    distutils.dir_util.copy_tree(os.path.dirname(record['ImagePath']), keep_deleted_image_directory)
                print("[delete_label] remove image path : " + os.path.dirname(record['ImagePath']))
                shutil.rmtree(os.path.dirname(record['ImagePath']))
            else:
                print("[delete_label] cannot find image path : " + os.path.dirname(record['ImagePath']))
        models().delete_records(query_dict)
        update_model_status(PID, Category, TrainingStatus.not_trained)
        return True
    else:
        print("[delete_label] cannot find record. PID=" + PID + ", Category=" + Category + ", Label=" + Label)
        return False

def list_categories(PID):
    query_dict = {
        'PID': PID
    }
    result = models().list_record(query_dict, "Category")
    if result:
        return result
    else:
        return []

def list_labels(PID, Category):
    query_dict = {
        'PID': PID,
        'Category': Category
    }
    result = models().list_record(query_dict, "Label")
    if result:
        return result
    else:
        return []

def get_training_status(PID, Category):
    query_dict = {
        "PID": PID,
        "Category": Category
    }
    model = models().get_one_model(query_dict)
    if model and 'TrainingStatus' in model:
        return model['TrainingStatus']
    else:
        return None

def get_training_mode(PID, Category):
    query_dict = {
        "PID": PID,
        "Category": Category
    }
    model = models().get_one_model(query_dict)
    if ('NetworkModel' in model) and ('TrainingMode' in model) and ('GlobalStepCount' in model):
        output = {'NetworkModel': model['NetworkModel'],
                  'TrainingMode': model['TrainingMode'],
                  'GlobalStepCount': model['GlobalStepCount']}
        return output
    else:
        return None

def update_model_path(PID, Category, ModelPath, QuantizedModelPath, LabelsPath, NetworkModel):
    query_dict = {
        "PID": PID,
        "Category": Category
    }
    model = models().get_one_model(query_dict)
    if model:
        query_dict = {
            "_id": model['_id']
        }
        update_dict = {
            "ModelPath": ModelPath,
            "QuantizedModelPath": QuantizedModelPath,
            "LabelsPath": LabelsPath,
            "NetworkModel": NetworkModel,
            "UpdateTime": datetime.datetime.utcnow()
        }
        models().update_one_model(query_dict, update_dict)

def get_model_path(PID, Category):
    query_dict = {
        "PID": PID,
        "Category": Category
    }
    model = models().get_one_model(query_dict)
    if model and ('ModelPath' in model) and ('QuantizedModelPath' in model) and ('LabelsPath' in model) and ('NetworkModel' in model) and \
       os.path.exists(model['ModelPath']) and os.path.exists(model['QuantizedModelPath']) and os.path.exists(model['LabelsPath']):
        output = {'ModelPath': model['ModelPath'],
                  'QuantizedModelPath': model['QuantizedModelPath'],
                  'LabelsPath': model['LabelsPath'],
                  'NetworkModel': model['NetworkModel']}
        return output
    else:
        return None

def update_category(PID, old_category, new_category):
    query_dict = {
        "PID": PID,
        "Category": old_category
    }
    update_dict = {
        "Category": new_category,
        "UpdateTime": datetime.datetime.utcnow()
    }
    models().update_records(query_dict, update_dict)
    models().update_models(query_dict, update_dict)

def update_label(PID, Category, old_label, new_label):
    query_dict = {
        "PID": PID,
        "Category": Category,
        "Label": old_label
    }
    update_dict = {
        "Label": new_label,
        "UpdateTime": datetime.datetime.utcnow()
    }
    models().update_records(query_dict, update_dict)

def upsert_train_task(PID, Category, task_id):
    query_dict = {
        "PID": PID,
        "Category": Category,
        "TaskID": task_id
    }
    update_dict = {
        "UpdateTime": datetime.datetime.utcnow()
    }
    print("upsert_train_task is called. PID=" + PID + ", Category=" + Category + ", task_id=" + task_id)
    result = models().upsert_train_task(query_dict, update_dict)
    return result

def delete_train_task(PID, Category, task_id):
    query_dict = {
        "PID": PID,
        "Category": Category,
        "TaskID": task_id
    }
    print("delete_train_task is called. PID=" + PID + ", Category=" + Category + ", task_id=" + task_id)
    result = models().delete_train_task(query_dict)
    return result

def get_train_task_ids(PID, Category):
    query_dict = {
        "PID": PID,
        "Category": Category
    }
    returned_tasks = models().get_train_tasks(query_dict)
    task_ids = []
    if returned_tasks:
        for task in returned_tasks:
            if 'TaskID' in task:
                task_ids.append(task['TaskID'])
    return task_ids

def check_train_task_alive(PID, Category, task_id):
    query_dict = {
        "PID": PID,
        "Category": Category,
        "TaskID": task_id
    }
    returned_tasks = models().get_train_tasks(query_dict)
    if returned_tasks.count()>0:
        print("check_train_task_alive is called. PID=" + PID + ", Category=" + Category + ", task_id=" + task_id + ", Alive")
        return True
    else:
        print("check_train_task_alive is called. PID=" + PID + ", Category=" + Category + ", task_id=" + task_id + ", Not Alive")
        return False

def upsert_train_task_process(task_id, process_id):
    query_dict = {
        "TaskID": task_id,
        "ProcessID": process_id
    }
    update_dict = {
        "UpdateTime": datetime.datetime.utcnow()
    }
    print("upsert_train_task_process is called. task_id=" + task_id + ", process_id=" + str(process_id))
    result = models().upsert_train_task_process(query_dict, update_dict)
    return result

def delete_train_task_process(task_id, process_id):
    query_dict = {
        "TaskID": task_id,
        "ProcessID": process_id
    }
    print("delete_train_task_process is called. task_id=" + task_id + ", process_id=" + str(process_id))
    result = models().delete_train_task_process(query_dict)
    return result

def get_train_task_process_ids(task_id):
    query_dict = {
        "TaskID": task_id
    }
    returned_processes = models().get_train_task_processes(query_dict)
    process_ids = []
    if returned_processes:
        for process in returned_processes:
            if 'ProcessID' in process:
                process_ids.append(process['ProcessID'])
    return process_ids

def upsert_train_child_process(task_id, process_id):
    query_dict = {
        "TaskID": task_id,
        "ProcessID": process_id
    }
    update_dict = {
        "UpdateTime": datetime.datetime.utcnow()
    }
    result = models().upsert_train_child_process(query_dict, update_dict)
    return result

def delete_train_child_process(task_id, process_id):
    query_dict = {
        "TaskID": task_id,
        "ProcessID": process_id
    }
    result = models().delete_train_child_process(query_dict)
    return result

def get_train_child_process_ids(task_id):
    query_dict = {
        "TaskID": task_id
    }
    returned_processes = models().get_train_child_processes(query_dict)
    process_ids = []
    if returned_processes:
        for process in returned_processes:
            if 'ProcessID' in process:
                process_ids.append(process['ProcessID'])
    return process_ids

def upsert_train_wait_device(task_id, bundle_id, device_token):
    query_dict = {
        "TaskID": task_id,
        "BundleID": bundle_id,
        "DeviceToken": device_token
    }
    update_dict = {
        "UpdateTime": datetime.datetime.utcnow()
    }
    result = models().upsert_train_wait_device(query_dict, update_dict)
    return result

def delete_train_wait_device(task_id, bundle_id, device_token):
    device_dict = {
        "TaskID": task_id,
        "BundleID": bundle_id,
        "DeviceToken": device_token
    }
    result = models().delete_train_wait_device(device_dict)
    return result

def delete_train_wait_devices(task_id):
    device_dict = {
        "TaskID": task_id
    }
    result = models().delete_train_wait_devices(device_dict)
    return result

def get_train_wait_devices(task_id):
    query_dict = {
        "TaskID": task_id
    }
    returned_devices = models().get_train_wait_devices(query_dict)
    bundle_ids = []
    device_tokens = []
    if returned_devices:
        for device in returned_devices:
            if 'BundleID' in device and 'DeviceToken' in device:
                bundle_ids.append(device['BundleID'])
                device_tokens.append(device['DeviceToken'])
    return bundle_ids, device_tokens

def insert_log_flask_api(api_path, request_form, ElapsedTime, response=None, misc=None):
    log = {
        "Path": api_path,
        "Form": request_form,
        "ElapsedTime": ElapsedTime,
        "UpdateTime": datetime.datetime.utcnow()
    }
    if response is not None:
        log["Response"] = response
    if misc is not None:
        log["Misc"] = misc
    result = models().save_log_flask_api(log)
    return result
