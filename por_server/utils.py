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
from pathlib import Path

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'JPG', 'JPEG'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_dirpath(request_dict, prefix):
    for key, value in request_dict.items():
        print(key, value)
    if 'Label' in request_dict:
        save_dirpath = os.path.join(prefix, request_dict['PID'], request_dict['Category'],
                                     'photos', request_dict['Label'])
    else:
        save_dirpath = os.path.join(prefix, request_dict['PID'], 'photos', request_dict['Category'])
    save_dirpath_path = Path(save_dirpath)
    save_dirpath_path.mkdir(parents=True, exist_ok=True)
    return save_dirpath

def create_filepath(request_dict, prefix, filename):
    for key, value in request_dict.items():
        print(key, value)
    if 'Label' in request_dict:
        save_filepath = os.path.join(prefix, request_dict['PID'], request_dict['Category'],
                                     'photos', request_dict['Label'], filename)
    else:
        save_filepath = os.path.join(prefix, request_dict['PID'], 'photos', request_dict['Category'], filename)
    save_filepath_path = Path(save_filepath)
    save_filepath_path.parent.mkdir(parents=True, exist_ok=True)
    return save_filepath

def create_model_filepath(pid, category, prefix):
    save_filepath = os.path.join(prefix, pid, category)
    save_filepath_path = Path(save_filepath)
    save_filepath_path.mkdir(parents=True, exist_ok=True)
    return save_filepath
