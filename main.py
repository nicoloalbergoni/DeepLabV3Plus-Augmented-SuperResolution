import numpy as np
import tensorflow as tf
from PIL import Image
from utils import preprocess_image, plot_prediction

from matplotlib import pyplot as plt


from model import Deeplabv3


TEST_IMG_PATH = "./test_img.jpg"
TARGET_IMG_WIDTH = 512


def main(plot_model=False):

    model = Deeplabv3(input_shape=(512, 512, 3), classes=21, OS=16, activation=None)

    image = np.array(Image.open(TEST_IMG_PATH))
    resized_image, pad_width, pad_height = preprocess_image(image, target_size_width=TARGET_IMG_WIDTH)

    result = model.predict(np.expand_dims(resized_image, 0))
    labels = np.argmax(result.squeeze(), -1)

    if pad_width > 0:
        labels = labels[:-pad_width]
    if pad_height > 0:
        labels = labels[:, :-pad_height]
    labels = np.array(Image.fromarray(labels.astype('uint8')).resize((image.shape[1], image.shape[0])))

    plot_prediction([image, labels], only_prediction=True, show_overlay=True)

    if plot_model:
        tf.keras.utils.plot_model(
            model, to_file='./model.png', show_shapes=True, show_dtype=False,
            show_layer_names=True, rankdir='TB', show_layer_activations=False)


if __name__ == '__main__':
    main(plot_model=False)
