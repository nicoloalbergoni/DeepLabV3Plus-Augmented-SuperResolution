import tensorflow as tf
from model import Deeplabv3

model = Deeplabv3()

# model.summary()


tf.keras.utils.plot_model(
    model, to_file='./model.png', show_shapes=True, show_dtype=False,
    show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96,
    layer_range=None, show_layer_activations=False
)
