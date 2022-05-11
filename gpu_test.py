import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)


print(tf.test.gpu_device_name())
