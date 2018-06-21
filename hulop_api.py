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
import sys
import uuid
from flask import Flask, jsonify, abort, request, redirect, make_response, flash, url_for, send_file
from werkzeug.utils import secure_filename
import yaml
from por_server import controller, exc, utils
import logging
import traceback
from celery import Celery
from celery_once import QueueOnce, AlreadyQueued
from celery.result import AsyncResult
from billiard.exceptions import Terminated
from train_thread import TrainThread
from multiprocessing import current_process
import threading
import psutil
import time

# Init Server
app = Flask(__name__, static_url_path="")

# Read config file
dir = os.path.dirname(__file__)
filename = os.path.join(dir, 'config.yaml')
config = yaml.safe_load(open(filename))

# Init celery distributed queue
celery_app = Celery('hulop_api', backend='redis://'+str(config['redis_host'])+':'+str(config['redis_port'])+'/0',
                    broker='redis://'+str(config['redis_host'])+':'+str(config['redis_port'])+'/0')
celery_app.conf.ONCE = {
    'backend': 'celery_once.backends.Redis',
    'settings': {
        'url': 'redis://'+str(config['redis_host'])+':'+str(config['redis_port'])+'/0',
        'default_timeout': 60 * 60
    }
}

# Load and set MongoDB details
app.config['UPLOAD_FOLDER'] = config['upload_dir']
app.config['MODEL_FOLDER'] = config['model_dir']
app.config['INFERENCE_FOLDER'] = config['inference_dir']
app.config['DELETED_IMAGE_FOLDER'] = config['deleted_image_dir']
app.config['UNKNOWN_FOLDER'] = config['unknown_dir']
app.config['PRETRAINED_FOLDER'] = config['pretrained_dir']
app.config['TENSORFLOW_FOLDER'] = config['tensorflow_dir']
app.config['NETWORK_MODEL'] = config['network_model']
app.config['APNS_SETTINGS'] = {}
for apns_setting in config['apns_settings']:
    app.config['APNS_SETTINGS'][apns_setting["apns_topic"]] = {}
    app.config['APNS_SETTINGS'][apns_setting["apns_topic"]]["APNS_USE_SANDBOX"] = apns_setting['apns_use_sandbox']
    app.config['APNS_SETTINGS'][apns_setting["apns_topic"]]["APNS_KEY_FILE"] = apns_setting['apns_key_file']
    print("APNS setting : bundle ID = " + apns_setting["apns_topic"] + ", " + str(app.config['APNS_SETTINGS'][apns_setting["apns_topic"]]))
app.config['LOG_DETAIL'] = config['log_detail']

os.makedirs(app.config['UNKNOWN_FOLDER'], mode=0o777, exist_ok=True)

app.secret_key = "super secret key"

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.DEBUG)
app.logger.debug("Server started")
app.debug = True

def cancel_training(PID, Category):
    print("cancel_training is called. PID=" + PID + ", Category=" + Category)
    
    old_train_task_ids = controller.get_train_task_ids(PID, Category)
    
    # delete training flag to cancel training child process
    for old_train_task_id in old_train_task_ids:
        print('[cancel_training] delete train task. task_id=' + old_train_task_id)
        controller.delete_train_task(PID, Category, old_train_task_id)
        print('[cancel_training] delete train wait devices. task_id=' + old_train_task_id)
        controller.delete_train_wait_devices(old_train_task_id)
    # confirm all training child processes stopped
    for old_train_task_id in old_train_task_ids:
        print('[cancel_training] wait child process stopped for old train task. task_id=' + old_train_task_id)
        old_train_child_process_ids = controller.get_train_child_process_ids(old_train_task_id)
        for old_train_process_id in old_train_child_process_ids:
            try:
                print('[cancel_training] wait child process stopped. process_id=' + str(old_train_process_id))
                old_train_process = psutil.Process(old_train_process_id)
                old_train_process.wait(timeout=60 * 60)
                print('[cancel_training] child process stopped. process_id=' + str(old_train_process_id))
            except psutil.NoSuchProcess:
                print('cannot find old training process : PID=' + PID + ', Category=' + Category + ', process ID=' + str(old_train_process_id))
            controller.delete_train_child_process(old_train_task_id, old_train_process_id)
        print('[cancel_training] all child process stopped for old train task. task_id=' + old_train_task_id)
    
    # revoke training task
    for old_train_task_id in old_train_task_ids:
        print('start revoke train tasks ' + old_train_task_id + '....')
        try:
            celery_app.control.revoke(old_train_task_id, terminate=True)
        except Terminated:
            traceback.print_exc()
        print('finish revoke train tasks ' + old_train_task_id + '.')
    # confirm all training task processes stopped
    for old_train_task_id in old_train_task_ids:
        print('[cancel_training] wait task process stopped for old train task. task_id=' + old_train_task_id)
        old_train_task_process_ids = controller.get_train_task_process_ids(old_train_task_id)
        for old_train_process_id in old_train_task_process_ids:
            try:
                print('[cancel_training] wait task process stopped. process_id=' + str(old_train_process_id))
                old_train_process = psutil.Process(old_train_process_id)
                old_train_process.wait(timeout=60 * 60)
                print('[cancel_training] task process stopped. process_id=' + str(old_train_process_id))
            except psutil.NoSuchProcess:
                print('cannot find old training process : PID=' + PID + ', Category=' + Category + ', process ID=' + str(old_train_process_id))
            controller.delete_train_task_process(old_train_task_id, old_train_process_id)
        print('[cancel_training] all task process stopped for old train task. task_id=' + old_train_task_id)

    print("cancel_training is finished. PID=" + PID + ", Category=" + Category)
    return old_train_task_ids
    
@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify({'error': 'Bad request'}), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.errorhandler(exc.InvalidData)
def invalid_data(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@app.route('/api/train_status', methods=['POST'])
def get_training_status():
    start_time = time.time()
    print('API called ' + str(request.path) + ' : ' + str(request.form))
    
    if 'PID' not in request.form or 'Category' not in request.form:
        raise exc.InvalidData('Missing Data for training status', status_code=404)
    training_status = controller.get_training_status(request.form['PID'], request.form['Category'])
    
    if app.config['LOG_DETAIL']:
        time_elapsed = time.time() - start_time
        controller.insert_log_flask_api(request.path, request.form, time_elapsed, response=training_status)
    
    if training_status is None:
        raise exc.InvalidData('No model trained for this data', status_code=404)
    else:
        print('training status : ' + str(training_status))
        return make_response(jsonify(training_status))

@app.route('/api/deletecategory', methods=['POST'])
def delete_category():
    start_time = time.time()
    print('API called ' + str(request.path) + ' : ' + str(request.form))
    
    image_dirpath = utils.create_model_filepath(request.form['PID'], request.form['Category'], app.config['UPLOAD_FOLDER'])
    model_dirpath = utils.create_model_filepath(request.form['PID'], request.form['Category'], app.config['MODEL_FOLDER'])

    if app.config['LOG_DETAIL']:
        # keep deleting image if detail log option is set
        keep_deleted_image_dirpath = utils.create_model_filepath(request.form['PID'], request.form['Category'], app.config['DELETED_IMAGE_FOLDER'])
        status = controller.delete_category(request.form['PID'], request.form['Category'], image_dirpath, model_dirpath,
                                            keep_deleted_image_directory=keep_deleted_image_dirpath)
    else:
        status = controller.delete_category(request.form['PID'], request.form['Category'], image_dirpath, model_dirpath)
    
    if app.config['LOG_DETAIL']:
        time_elapsed = time.time() - start_time
        controller.insert_log_flask_api(request.path, request.form, time_elapsed, response=status)
    
    if status:
        return "Category Deleted"
    else:
        raise exc.InvalidData('Error deleting category. Please check input data.', status_code=404)

@app.route('/api/deletelabel', methods=['POST'])
def delete_label():
    start_time = time.time()
    print('API called ' + str(request.path) + ' : ' + str(request.form))
    
    if app.config['LOG_DETAIL']:
        # keep deleting image if detail log option is set
        keep_deleted_image_dirpath = utils.create_dirpath(request.form, app.config['DELETED_IMAGE_FOLDER'])
        status = controller.delete_label(request.form['PID'], request.form['Category'], request.form['Label'],
                                         keep_deleted_image_directory=keep_deleted_image_dirpath)
    else:
        status = controller.delete_label(request.form['PID'], request.form['Category'], request.form['Label'])
    
    if app.config['LOG_DETAIL']:
        time_elapsed = time.time() - start_time
        controller.insert_log_flask_api(request.path, request.form, time_elapsed, response=status)
    
    if status:
        return "Label Deleted"
    else:
        raise exc.InvalidData('Error deleting label. Please check input data.', status_code=404)

@app.route('/api/categories', methods=['POST'])
def list_categories():
    start_time = time.time()
    print('API called ' + str(request.path) + ' : ' + str(request.form))
    
    if 'PID' not in request.form:
        raise exc.InvalidData('Missing Data for training', status_code=404)
    result = controller.list_categories(request.form['PID'])

    if app.config['LOG_DETAIL']:
        time_elapsed = time.time() - start_time
        controller.insert_log_flask_api(request.path, request.form, time_elapsed, response=result)
    
    return make_response(jsonify(result))

@app.route('/api/labels', methods=['POST'])
def list_labels():
    start_time = time.time()
    print('API called ' + str(request.path) + ' : ' + str(request.form))
    
    if 'PID' not in request.form or 'Category' not in request.form:
        raise exc.InvalidData('Missing Data for training', status_code=404)
    result = controller.list_labels(request.form['PID'], request.form['Category'])

    if app.config['LOG_DETAIL']:
        time_elapsed = time.time() - start_time
        controller.insert_log_flask_api(request.path, request.form, time_elapsed, response=result)
    
    return make_response(jsonify(result))

@app.route('/api/update_category', methods=['POST'])
def update_category():
    start_time = time.time()
    print('API called ' + str(request.path) + ' : ' + str(request.form))
    
    if 'PID' not in request.form or 'Category' not in request.form or 'NewCategory' not in request.form:
        raise exc.InvalidData('Missing Data for training', status_code=404)
    controller.update_category(request.form['PID'], request.form['Category'], request.form['NewCategory'])

    if app.config['LOG_DETAIL']:
        time_elapsed = time.time() - start_time
        controller.insert_log_flask_api(request.path, request.form, time_elapsed)
    
    return "Category Updated!"

@app.route('/api/update_label', methods=['POST'])
def update_label():
    start_time = time.time()
    print('API called ' + str(request.path) + ' : ' + str(request.form))
    
    if 'PID' not in request.form or 'Category' not in request.form or 'Label' not in request.form or 'NewLabel' not in request.form:
        raise exc.InvalidData('Missing Data for training', status_code=404)
    controller.update_label(request.form['PID'], request.form['Category'], request.form['Label'], request.form['NewLabel'])

    if app.config['LOG_DETAIL']:
        time_elapsed = time.time() - start_time
        controller.insert_log_flask_api(request.path, request.form, time_elapsed)
    
    return "Label Updated!"

@app.route('/api/train', methods=['POST'])
def start_training():
    start_time = time.time()
    print('API called ' + str(request.path) + ' : ' + str(request.form))
    
    thread = threading.Thread(target=_start_training, args=(request.form,))
    thread.start()

    if app.config['LOG_DETAIL']:
        time_elapsed = time.time() - start_time
        controller.insert_log_flask_api(request.path, request.form, time_elapsed)
    
    return "Model Queued"

def _start_training(request_form):
    print('API called /api/train : ' + str(request_form))
    
    if 'PID' not in request_form or 'Category' not in request_form:
        raise exc.InvalidData('Missing Data for training', status_code=404)
    quick_train = False
    if 'QuickTrain' in request_form:
        quick_train = True
    
    # cancel old train tasks
    print('[start_training] Start cancel training...')
    old_train_task_ids = cancel_training(request_form['PID'], request_form['Category'])
    print('[start_training] Finish cancel training')
    
    try :
        # add training process to queue
        print('[start_training] Start add training queue...')
        result = queue_training.apply_async(queue='queue_train', args=[request_form['PID'], request_form['Category'], quick_train])
        print('[start_training] Finish add training queue.')
        
        # update devices that wait push message
        if 'BundleID' in request_form and len(request_form['BundleID'])>0 and 'DeviceToken' in request_form and len(request_form['DeviceToken'])>0:
            print('Added train wait device : task=' + result.task_id + ', bunlde ID=' + request_form['BundleID'] + ', device=' + request_form['DeviceToken'])
            controller.upsert_train_wait_device(result.task_id, request_form['BundleID'], request_form['DeviceToken'])
            for old_train_task_id in old_train_task_ids:
                old_train_wait_bundle_ids, old_train_wait_devices = controller.get_train_wait_devices(old_train_task_id)
                for idx in range(len(old_train_wait_bundle_ids)):
                    controller.delete_train_wait_device(old_train_task_id, old_train_wait_bundle_ids[idx], old_train_wait_devices[idx])
                    controller.upsert_train_wait_device(result.task_id, old_train_wait_bundle_ids[idx], old_train_wait_devices[idx])
        print('Added training queue : PID=' + request_form['PID'] + ', Category=' + request_form['Category'] + ', task ID=' + result.task_id + ', process ID=' + str(os.getpid()))
    except AlreadyQueued:
        print('Already queued : PID=' + request_form['PID'] + ', Category=' + request_form['Category'] + ', process ID=' + str(os.getpid()))

@celery_app.task(base=QueueOnce, once={'unlock_before_run': True})
def queue_training(pid, category, quick_train):
    # set as non-daemon process to create sub process from this celery task
    # https://github.com/celery/celery/issues/1709
    current_process()._config['daemon'] = False
    
    task_id = queue_training.request.id
    print('Queue training is called. PID=' + pid + ', Category=' + category + ', task ID=' + task_id + ', process ID=' + str(os.getpid()))
    
    # update training flag to cancel task
    controller.upsert_train_task(pid, category, task_id)
    controller.upsert_train_task_process(task_id, os.getpid())
    
    # start training
    try:
        output_graph_filepath = utils.create_model_filepath(pid, category, app.config['MODEL_FOLDER'])
        if output_graph_filepath:
            controller.update_model_status(pid, category, controller.TrainingStatus.training)
            output_model_dir = utils.create_model_filepath(pid, category, app.config['MODEL_FOLDER'])
            train_thread = TrainThread(app.config['UPLOAD_FOLDER'], app.config['UNKNOWN_FOLDER'], app.config['PRETRAINED_FOLDER'],
                                       app.config['TENSORFLOW_FOLDER'], app.config['APNS_SETTINGS'])
            train_thread.run_train(pid, category, app.config['NETWORK_MODEL'], task_id, output_model_dir, quick_train=quick_train)
        else:
            raise exc.InvalidData("Couldn't create output graph file path.", status_code=404)
    except Exception:
        traceback.print_exc()
        controller.update_model_status(pid, category, controller.TrainingStatus.error)
        controller.delete_train_task_process(task_id, os.getpid())
        return
    controller.delete_train_task_process(task_id, os.getpid())
    return

@app.route('/api/upload', methods=['POST'])
def receive_training_data():
    start_time = time.time()
    print('API called ' + str(request.path) + ' : ' + str(request.form))
    
    if 'file' not in request.files:
        raise exc.InvalidData('No file part', status_code=404)
    if not request.form['PID'] or not request.form['Category'] or not request.form['Label']:
        raise exc.InvalidData('Missing metadata', status_code=404)
    saved_files = []
    uploaded_files = request.files.getlist('file')
    print('uploaded files : ' + str(uploaded_files))
    for file in uploaded_files:
        if file.filename == '':
            raise exc.InvalidData('No selected file', status_code=404)
        if not file or not utils.allowed_file(file.filename):
            raise exc.InvalidData('Image Data invalid', status_code=404)
        filename = secure_filename(file.filename)
        filepath = utils.create_filepath(request.form, app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        saved_files.append(filepath)
    DirPath = os.path.dirname(filepath)
    controller.insert_record(request.form['PID'], request.form['Category'], request.form['Label'], DirPath)
    inserted_id = controller.insert_model(request.form['PID'], request.form['Category'])
    controller.update_model_status(request.form['PID'], request.form['Category'], controller.TrainingStatus.not_trained)

    if app.config['LOG_DETAIL']:
        log_misc = {}
        log_misc["ImagesFilePath"] = []
        for saved_file in saved_files:
            log_misc["ImagesFilePath"].append(saved_file)
        time_elapsed = time.time() - start_time
        controller.insert_log_flask_api(request.path, request.form, time_elapsed, misc=log_misc)
    
    return "Images Uploaded!"

@app.route('/api/inference', methods=['POST'])
def run_inference():
    start_time = time.time()
    print('API called ' + str(request.path) + ' : ' + str(request.form))
    
    if 'file' not in request.files:
        raise exc.InvalidData('No file part', status_code=404)
    file = request.files['file']
    print('uploaded file : ' + str(file))
    if file.filename == '':
        raise exc.InvalidData('No selected file', status_code=404)
    if not file or not utils.allowed_file(file.filename):
        raise exc.InvalidData('Image Data invalid', status_code=404)
    filename = secure_filename(file.filename)
    filepath = utils.create_filepath(request.form, app.config['INFERENCE_FOLDER'], filename)
    print('Saved image file for inference : ' + filepath)
    file.save(filepath)
    
    async_result = queue_inference.apply_async(queue='queue_inference', args=[request.form['PID'], request.form['Category'], filepath, None])
    result = async_result.get(interval=0.005)
    
    if not app.config['LOG_DETAIL']:
        os.remove(filepath)
    
    if app.config['LOG_DETAIL']:
        time_elapsed = time.time() - start_time
        log_misc = {}
        log_misc["ImageFilePath"] = filepath
        train_mode = controller.get_training_mode(request.form['PID'], request.form['Category'])
        if train_mode is not None and 'NetworkModel' in train_mode:
            log_misc["NetworkModel"] = train_mode["NetworkModel"]
        if train_mode is not None and 'TrainingMode' in train_mode:
            log_misc["TrainingMode"] = train_mode["TrainingMode"]
        if train_mode is not None and 'GlobalStepCount' in train_mode:
            log_misc["GlobalStepCount"] = train_mode["GlobalStepCount"]
        log_result = {}
        for key in result:
            encode_key = key.replace("\\", "\\\\").replace("\$", "\\u0024").replace(".", "\\u002e")
            log_result[encode_key] = result[key]
        controller.insert_log_flask_api(request.path, request.form, time_elapsed, response=log_result, misc=log_misc)
    
    if result is not None:
        return make_response(jsonify(result))
    else:
        raise exc.InvalidData('No model available for the specified data.', status_code=404)

@app.route('/api/inference_multi', methods=['POST'])
def run_inference_multi():
    start_time = time.time()
    print('API called ' + str(request.path) + ' : ' + str(request.form))
    
    if 'file' not in request.files:
        raise exc.InvalidData('No file part', status_code=404)
    
    saved_files = []
    upload_dir = str(uuid.uuid4())
    uploaded_files = request.files.getlist('file')
    print('uploaded files : ' + str(uploaded_files))
    for file in uploaded_files:
        if file.filename == '':
            raise exc.InvalidData('No selected file', status_code=404)
        if not file or not utils.allowed_file(file.filename):
            raise exc.InvalidData('Image Data invalid', status_code=404)
        filename = secure_filename(file.filename)
        filename = os.path.join(upload_dir, filename)
        filepath = utils.create_filepath(request.form, app.config['INFERENCE_FOLDER'], filename)
        print('Saved image file for inference : ' + filepath)
        file.save(filepath)
        saved_files.append(filepath)
    
    async_result = queue_inference.apply_async(queue='queue_inference', args=[request.form['PID'], request.form['Category'], None, saved_files])
    result = async_result.get(interval=0.005)
    
    if not app.config['LOG_DETAIL']:
        shutil.rmtree(uplod_dir)
    
    if app.config['LOG_DETAIL']:
        time_elapsed = time.time() - start_time
        log_misc = {}
        log_misc["ImagesFilePath"] = saved_files
        train_mode = controller.get_training_mode(request.form['PID'], request.form['Category'])
        if train_mode is not None and 'NetworkModel' in train_mode:
            log_misc["NetworkModel"] = train_mode["NetworkModel"]
        if train_mode is not None and 'TrainingMode' in train_mode:
            log_misc["TrainingMode"] = train_mode["TrainingMode"]
        if train_mode is not None and 'GlobalStepCount' in train_mode:
            log_misc["GlobalStepCount"] = train_mode["GlobalStepCount"]
        log_result = {}
        for key in result:
            encode_key = key.replace("\\", "\\\\").replace("\$", "\\u0024").replace(".", "\\u002e")
            log_result[encode_key] = result[key]
        controller.insert_log_flask_api(request.path, request.form, time_elapsed, response=log_result, misc=log_misc)
    
    if result is not None:
        return make_response(jsonify(result))
    else:
        raise exc.InvalidData('No model available for the specified data.', status_code=404)

@celery_app.task
def queue_inference(pid, category, filepath, multi_filepath):
    if (filepath is not None and multi_filepath is not None) or (filepath is None and multi_filepath is None):
        print('Please input single input image or multiple images')
        return None
    
    if multi_filepath is not None:
        print('Queue inference is called. PID=' + pid + ', Category=' + category + ', multi_filepath=' + str(multi_filepath) + ', task ID=' + queue_inference.request.id)
    else:
        print('Queue inference is called. PID=' + pid + ', Category=' + category + ', filepath=' + filepath + ', task ID=' + queue_inference.request.id)
    
    # set as non-daemon process to create sub process from this celery task
    # https://github.com/celery/celery/issues/1709
    current_process()._config['daemon'] = False
    
    # execute inference
    if multi_filepath is not None:
        result = controller.run_inference_multi(pid, category, multi_filepath)
    else:
        result = controller.run_inference(pid, category, filepath)
    print('Inference result : ' + str(result))
    
    return result
    
@app.route('/api/label', methods=['POST'])
def create_label():
    start_time = time.time()
    print('API called ' + str(request.path) + ' : ' + str(request.form))
    
    if 'PID' not in request.form or 'Category' not in request.form or 'Label' not in request.form:
        raise exc.InvalidData('Missing Data for training', status_code=404)
    controller.insert_record(request.form['PID'], request.form['Category'], request.form['Label'], "")
    controller.insert_model(request.form['PID'], request.form['Category'])
    
    if app.config['LOG_DETAIL']:
        time_elapsed = time.time() - start_time
        controller.insert_log_flask_api(request.path, request.form, time_elapsed)
    
    return "Label created"

@app.route('/api/model', methods=['POST'])
def send_model():
    start_time = time.time()
    print('API called ' + str(request.path) + ' : ' + str(request.form))
    
    if 'PID' not in request.form or 'Category' not in request.form:
        raise exc.InvalidData('Missing Data for training', status_code=404)
    output = controller.get_model_path(request.form['PID'], request.form['Category'])

    if app.config['LOG_DETAIL']:
        time_elapsed = time.time() - start_time
        controller.insert_log_flask_api(request.path, request.form, time_elapsed, response=output)
    
    if output:
        return make_response(jsonify(output))
    else:
        raise exc.InvalidData('No model available for the specified data.', status_code=404)

@app.route('/api/download_model/<pid>/<category>/<filename>')
def send_model_file(pid, category, filename):
    start_time = time.time()
    print('API called ' + str(request.path) + ' : ' + str(request.form))

    if app.config['LOG_DETAIL']:
        time_elapsed = time.time() - start_time
        controller.insert_log_flask_api(request.path, request.form, time_elapsed)
    
    try:
        path = os.path.join(pid, category, filename)
        return send_file(os.path.join(app.config['MODEL_FOLDER'], path))
    except Exception as e:
        raise exc.InvalidData('Failed to send file', status_code=404)
    
with app.test_request_context():
    print("list of API")
    print(url_for("get_training_status") + " -> get_training_status")
    print(url_for("delete_category") + " -> delete_category")
    print(url_for("delete_label") + " -> delete_label")
    print(url_for("list_categories") + " -> list_categories")
    print(url_for("list_labels") + " -> list_labels")
    print(url_for("update_category") + " -> update_category")
    print(url_for("update_label") + " -> update_label")
    print(url_for("start_training") + " -> start_training")
    print(url_for("receive_training_data") + " -> receive_training_data")
    print(url_for("run_inference") + " -> run_inference")
    print(url_for("run_inference_multi") + " -> run_inference_multi")
    print(url_for("send_model") + " -> send_model")

# Main
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
