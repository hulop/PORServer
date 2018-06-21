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
import traceback
from por_server import controller, utils
from por_server.controller import TrainingStatus
from multiprocessing import Process, Queue
from apns2.client import APNsClient
from apns2.payload import Payload

class TrainThread():
    def __init__(self, upload_folder, unknown_folder, pretrained_folder, tensorflow_folder, apns_settings):
        self.upload_folder = upload_folder
        self.unknown_folder = unknown_folder
        self.pretrained_folder = pretrained_folder
        self.tensorflow_folder = tensorflow_folder
        self.apns_settings = apns_settings
    
    def run_train(self, pid, category, network_model, task_id, output_model_dir, quick_train=False):
        # backup previously trained model if it exists, and change model path to backuped model until next training finishes
        old_model = controller.get_model_path(pid, category)
        if old_model:
            old_model_path = old_model['ModelPath']
            old_label_path = old_model['LabelsPath']
            old_quantized_model_path = old_model['QuantizedModelPath']
            if os.path.exists(old_model_path) and os.path.exists(old_label_path) and os.path.exists(old_quantized_model_path) and 'NetworkModel' in old_model:
                backup_dir = os.path.join(output_model_dir, 'backup')
                if not os.path.exists(backup_dir):
                    os.mkdir(backup_dir)
                
                backup_model_path = os.path.join(backup_dir, os.path.basename(os.path.normpath(old_model_path)))
                backup_label_path = os.path.join(backup_dir, os.path.basename(os.path.normpath(old_label_path)))
                backup_quantized_model_path = os.path.join(backup_dir, os.path.basename(os.path.normpath(old_quantized_model_path)))
                backup_network_model = old_model['NetworkModel']
                print("[TrainThread] backup previous model file : " + old_model_path + ' -> ' + backup_model_path)
                shutil.move(old_model_path, backup_model_path)
                print("[TrainThread] backup previous quantized model file : " + old_quantized_model_path + ' -> ' + backup_quantized_model_path)
                shutil.move(old_quantized_model_path, backup_quantized_model_path)
                print("[TrainThread] backup previous label file : " + old_label_path + ' -> ' + backup_label_path)
                shutil.move(old_label_path, backup_label_path)
                controller.update_model_path(pid, category, backup_model_path, backup_quantized_model_path, backup_label_path, backup_network_model)
            else:
                print("Warning : DB entry for previously trained model exists, but files do not exists")
                print("[TrainThread] previous model file : " + old_model_path)
                print("[TrainThread] previous quantized model file : " + old_quantized_model_path)
                print("[TrainThread] previous label file : " + old_label_path)
        else:
            print("[TrainThread] cannot find previous model, skip backup : pid=" + pid + ", category=" + category)
        
        # first run finetuning
        finetune_epochs = None
        if quick_train:
            finetune_epochs = [50]
        else:
            finetune_epochs = [50, 50]
        # clear finetune log
        output_finetune_model_dir = os.path.join(output_model_dir, 'finetune')
        if os.path.exists(output_finetune_model_dir):
            shutil.rmtree(output_finetune_model_dir)
        # run finetune
        for idx, num_epoch in enumerate(finetune_epochs):
            # send push message to client only first training of finetuning
            push_message = False
            if idx==0:
                push_message = True
            
            q = Queue()
            p = Process(target=self._run_train, args=(pid, category, network_model, task_id, output_finetune_model_dir, num_epoch, True, push_message, False, q))
            p.start()
            print("[TrainThread] start new thread for finetune. thread name : " + p.name)
            p.join()
            finish_finetune = q.get()
            print("[TrainThread] finish new thread for finetune. thread name : " + p.name + '. training result : ' + str(finish_finetune))
            if not finish_finetune:
                break
        
        if finish_finetune and not quick_train:
            # if finetune finishes, run full layer training
            fulltrain_epochs = [100, 400, 500]
            # clear full train log
            output_full_model_dir = os.path.join(output_model_dir, 'full_train')
            if os.path.exists(output_full_model_dir):
                shutil.rmtree(output_full_model_dir)
            # run full train
            for idx, num_epoch in enumerate(fulltrain_epochs):
                # send push message to client only last training of full training
                push_message = False
                if idx==len(fulltrain_epochs)-1:
                    push_message = True
                
                final_train = False
                if idx==len(fulltrain_epochs)-1:
                    final_train = True
                
                q = Queue()
                p = Process(target=self._run_train, args=(pid, category, network_model, task_id, output_full_model_dir, num_epoch, False, push_message, final_train, q))
                p.start()
                print("[TrainThread] start new thread for full training. thread name : " + p.name)
                p.join()
                finish_full = q.get()
                print("[TrainThread] finish new thread for full training. thread name : " + p.name + '. training result : ' + str(finish_full))
                if not finish_full:
                    break
        
        # delete train task flag that is used for canceling task
        controller.delete_train_task(pid, category, task_id)
        # delete all devices that wait push messages
        controller.delete_train_wait_devices(task_id)
    
    def _run_train(self, pid, category, network_model, task_id, output_model_dir, num_epochs, finetune_last_layer, push_message, final_train, queue):
        result = False
        try:
            finish, output_graph, quantized_graph, output_labels, global_step_count = controller.run_training(pid, category, task_id, self.upload_folder,
                                                                                                              self.unknown_folder, self.pretrained_folder,
                                                                                                              self.tensorflow_folder, output_model_dir,
                                                                                                              network_model, num_epochs, finetune_last_layer)
            
            if finish:
                # update model status
                controller.update_model_path(pid, category, output_graph, quantized_graph, output_labels, network_model)
                if final_train:
                    train_status = controller.TrainingStatus.final_trained
                else:
                    train_status = controller.TrainingStatus.basic_trained
                if finetune_last_layer:
                    train_mode = controller.TrainingMode.finetune
                else:
                    train_mode = controller.TrainingMode.full_train
                controller.update_model_status(pid, category, train_status, train_mode=train_mode, global_step_count=global_step_count)
                
                # send push message to client
                if push_message:
                    bundle_ids, device_tokens = controller.get_train_wait_devices(task_id)
                    for idx in range(len(bundle_ids)):
                        bundle_id = bundle_ids[idx]
                        device_token = device_tokens[idx]
                        if bundle_id in self.apns_settings:
                            apns_key_file = self.apns_settings[bundle_id]["APNS_KEY_FILE"]
                            apns_use_sandbox = self.apns_settings[bundle_id]["APNS_USE_SANDBOX"]
                            if os.path.exists(apns_key_file):
                                print("device waited for training : " + device_token)
                                if final_train:
                                    push_message = "Training finished completely."
                                else:
                                    push_message = "Initial training completed."
                                payload = Payload(alert=push_message, sound="default", badge=1)
                                client = APNsClient(apns_key_file, use_sandbox=apns_use_sandbox, use_alternative_port=False)
                                client.send_notification(device_token, payload, topic=bundle_id)
                                print("sent push message for device " + device_token)
                            else:
                                print("APNS key does not exists : " + apns_key_file)
                        else:
                            print("Bundle ID does not exists in settings : " + bundle_id)
                
                result = True
            else:
                controller.delete_train_task(pid, category, task_id)
                controller.delete_train_wait_devices(task_id)
        except Exception:
            controller.update_model_status(pid, category, controller.TrainingStatus.error)
            controller.delete_train_task(pid, category, task_id)
            controller.delete_train_wait_devices(task_id)
            traceback.print_exc()
            queue.put(result)
            return
        queue.put(result)
        return
