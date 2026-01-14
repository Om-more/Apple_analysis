import os
import tensorflow as tf

print("TF version:", tf.__version__)
print("Using legacy keras:", os.environ.get("TF_USE_LEGACY_KERAS"))
print("tf.keras module:", tf.keras)
