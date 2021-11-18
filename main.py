import tensorflow as tf
from tensorflow.keras.utils import get_file

from model import Deeplabv3

WEIGHTS_PATH_X = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
                        WEIGHTS_PATH_X,
                        cache_dir="model",
                        cache_subdir=""
                        )

model = Deeplabv3()
model.load_weights(weights_path, by_name=True)

tf.keras.utils.plot_model(
    model, to_file='./model.png', show_shapes=True, show_dtype=False,
    show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96,
    layer_range=None, show_layer_activations=False
)
