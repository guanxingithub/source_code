import cupy
import random

import tensorflow as tf

# gpu_to_use = 0      # Works
gpu_to_use = 1        # after all physical devices are visible, this works

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_visible_devices(gpus, "GPU")   #set physical devices visible so that the logical devices are available for tf
    #tf.config.experimental.set_visible_devices(gpus[gpu_to_use], "GPU")

# Converting from TF to CuPy with dlpack works for both devices
with tf.device(f"/GPU:{gpu_to_use}"):
    tensor = tf.random.uniform((10,))

dltensor = tf.experimental.dlpack.to_dlpack(tensor)
array1 = cupy.fromDlpack(dltensor)

# Converting from CuPy to TF with dlpack works on all specified gpu if it is available
array1 = cupy.array([random.uniform(0.0, 1.0) for i in range(10)], dtype=cupy.float32)
dltensor = array1.toDlpack()
x = tf.experimental.dlpack.from_dlpack(dltensor)
print(x)

