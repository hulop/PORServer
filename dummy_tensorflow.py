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

################################################################################################
# Caution : This file is just for avoiding current tensorflow issue
# If there is no process that uses GPU, launching tensorflow in CPU mode takes several seconds.
# This issue will make CPU inference slower.
# This script is just for keeping tensorflow process that uses small GPU memory.
# Please call this script with option "python -i" to keep opening python interpreter.
# If there is a way to avoid this issue, this script should be removed.
################################################################################################

import tensorflow as tf
from utils import gpu_utils

# Find available GPU ID
gpu_id = gpu_utils.pick_gpu_lowest_memory()
print('found lowest memory gpu id : ' + str(gpu_id))

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list=str(gpu_id),
        per_process_gpu_memory_fraction=0.0001,
        allow_growth=True
    )
)
sess = tf.Session(config=config)
message = sess.run(tf.constant('Start tensorflow process just for acceralating launching CPU inference'))
print(message)
