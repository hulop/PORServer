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
import pymongo
import yaml
import threading

class models():
    __instance = None
    _lock = threading.Lock()    
    
    def __new__(cls):
        with cls._lock:
            if cls.__instance is None:
                cls.__instance = object.__new__(cls)
                
                # initialize class
                dir = os.path.dirname(__file__)
                parent_path = os.path.abspath(os.path.join(dir, os.pardir))
                filename = os.path.join(parent_path, 'config.yaml')

                config = yaml.safe_load(open(filename))
                client = pymongo.MongoClient(config['host'], config['port'])
                cls.__instance.db = client[config['db']]
        return cls.__instance
    
    def __init__(self):
        print('init models')
    
    def save_record(self, record):
        print(record)
        record_id = self.db.record.insert_one(record)
        return record_id.inserted_id
    
    def save_model(self, model):
        print(model)
        model_id = self.db.model.insert_one(model)
        return model_id.inserted_id

    def get_record_by_ID(self, ID):
        record = self.db.record.find_one({"_id":ID})
        if record:
            return record
        else:
            return None

    def get_model_by_ID(self, ID):
        record = self.db.record.find_one({"_id":ID})
        if record:
            return record
        else:
            return None

    def get_records(self, dict):
        records = self.db.record.find(dict)
        return records

    def get_models(self, dict):
        models = self.db.model.find(dict)
        return models

    def get_records_sorted(self, dict, sort_field='_id', sort_direction=pymongo.DESCENDING):
        records = self.db.record.find(dict).sort(sort_field, sort_direction)
        print(records.count())
        return records

    def get_models_sorted(self, dict, sort_field='_id', sort_direction=pymongo.DESCENDING):
        models = self.db.model.find(dict).sort(sort_field, sort_direction)
        print(models.count())
        return models

    def get_one_record(self, dict, sort_field='_id', sort_direction=pymongo.DESCENDING):
        record = self.db.record.find_one(dict, sort=[(sort_field, sort_direction)])
        return record

    def get_one_model(self, dict, sort_field='_id', sort_direction=pymongo.DESCENDING):
        model = self.db.model.find_one(dict, sort=[(sort_field, sort_direction)])
        return model

    def update_one_model(self, query_dict, update_dict):
        self.db.model.update_one(query_dict, {
            '$set': update_dict
        }, upsert=False)

    def update_one_record(self, query_dict, update_dict):
        self.db.record.update_one(query_dict, {
            '$set': update_dict
        }, upsert=False)

    def update_models(self, query_dict, update_dict):
        self.db.model.update(query_dict, {
            '$set': update_dict
        }, upsert=False, multi=True)

    def update_records(self, query_dict, update_dict):
        self.db.record.update(query_dict, {
            '$set': update_dict
        }, upsert=False, multi=True)

    def ifRecordExists(self, dict):
        record = self.get_records(dict)
        if not record.count():
            return False
        else:
            return True

    def ifModelExists(self, dict):
        model = self.get_models(dict)
        if not model.count():
            return False
        else:
            return True

    def delete_records(self, dict):
        result = self.db.record.delete_many(dict)

    def delete_one_record(self, dict):
        result = self.db.record.delete_one(dict)

    def delete_models(self, dict):
        result = self.db.model.delete_many(dict)

    def list_record(self, dict, field):
        items = self.db.record.find(dict).distinct(field)
        return items

    def list_model(self, dict, field):
        items = self.db.model.find(dict).distinct(field)
        return items

    def upsert_train_task(self, query_dict, update_dict):
        task_id = self.db.trainTask.update_one(query_dict, {
            '$set': update_dict
        }, upsert=True)
        return task_id.upserted_id
    
    def delete_train_task(self, dict):
        result = self.db.trainTask.delete_one(dict)
    
    def get_train_tasks(self, dict):
        tasks = self.db.trainTask.find(dict)
        return tasks

    def upsert_train_task_process(self, query_dict, update_dict):
        process_id = self.db.trainTaskProcess.update_one(query_dict, {
            '$set': update_dict
        }, upsert=True)
        return process_id.upserted_id

    def delete_train_task_process(self, dict):
        result = self.db.trainTaskProcess.delete_one(dict)

    def get_train_task_processes(self, dict):
        processes = self.db.trainTaskProcess.find(dict)
        return processes

    def upsert_train_child_process(self, query_dict, update_dict):
        process_id = self.db.trainChildProcess.update_one(query_dict, {
            '$set': update_dict
        }, upsert=True)
        return process_id.upserted_id

    def delete_train_child_process(self, dict):
        result = self.db.trainChildProcess.delete_one(dict)

    def get_train_child_processes(self, dict):
        processes = self.db.trainChildProcess.find(dict)
        return processes
    
    def upsert_train_wait_device(self, query_dict, update_dict):
        device_id = self.db.trainDevice.update_one(query_dict, {
            '$set': update_dict
        }, upsert=True)
        return device_id.upserted_id
    
    def delete_train_wait_device(self, device):
        result = self.db.trainDevice.delete_one(device)

    def delete_train_wait_devices(self, device):
        result = self.db.trainDevice.delete_many(device)
    
    def get_train_wait_devices(self, dict):
        devices = self.db.trainDevice.find(dict)
        return devices
    
    def save_log_flask_api(self, log):
        log_id = self.db.logFlaskAPI.insert_one(log)
        return log_id.inserted_id
