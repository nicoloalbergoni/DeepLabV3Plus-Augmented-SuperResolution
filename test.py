import os
import numpy as np
from numpy.random.mtrand import seed
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy, MeanIoU
from utils import plot_prediction, sparse_accuracy_ignoring_last_label, sparse_crossentropy_ignoring_last_label, Jaccard
from data_scripts.pascal_voc_dataset import PascalVOC2012Dataset
from tensorflow.keras.optimizers import Adam
from model import Deeplabv3

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
PASCAL_ROOT = os.path.join(DATA_DIR, "VOC2012")
IMGS_PATH = os.path.join(PASCAL_ROOT, "JPEGImages")
SEG_PATH = os.path.join(PASCAL_ROOT,)

SEED = 123
IMG_SIZE = 512
BATCH_SIZE = 1
BUFFER_SIZE = 1000
EPOCHS = 30
AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_img_paths(filename, dataset_path):
    """
    Obtains the list of image base names that have been labelled for semantic segmentation.
    Images are stored in JPEG format, and segmentation ground truth in PNG format.

    :param filename: The dataset name, either 'train', 'val' or 'test'.
    :param dataset_path: The root path of the dataset.
    :return: The list of image base names for either the training, validation, or test set.
    """
    assert filename in ('train', 'val', 'test')
    filename = os.path.join(dataset_path, "ImageSets",
                            "Segmentation", filename + '.txt')
    return [os.path.join(dataset_path, "JPEGImages", line.rstrip() + ".jpg") for line in open(filename)]


def load_image(img_path: str) -> dict:

    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    mask_path = tf.strings.regex_replace(
        img_path, "JPEGImages", "SegmentationClassRaw")
    mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
    mask = tf.io.read_file(mask_path)

    # The masks contain a class index for each pixels
    mask = tf.image.decode_png(mask, channels=1)

    return {"image": image, "mask": mask}


@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask


@tf.function
def parse_image(datapoint: dict, isTrain=False) -> tuple:
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['mask'], (IMG_SIZE, IMG_SIZE))

    if isTrain:
        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)
    input_mask = tf.reshape(input_mask, [IMG_SIZE * IMG_SIZE, 1])

    return input_image, input_mask


def main():
    img_path_train = get_img_paths("train", PASCAL_ROOT)
    img_path_val = get_img_paths("val", PASCAL_ROOT)

    TRAIN_SIZE = len(img_path_train)
    VAL_SIZE = len(img_path_val)

    print(
        f"Found {TRAIN_SIZE} images in train and {VAL_SIZE} in val")

    train_dataset = tf.data.Dataset.from_tensor_slices(img_path_train)
    train_dataset = train_dataset.map(load_image)
    train_dataset = train_dataset.map(lambda x: parse_image(x, isTrain=True))
    train_dataset = train_dataset.shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
    train_dataset = train_dataset.repeat(EPOCHS)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(img_path_val)
    val_dataset = val_dataset.map(load_image)
    val_dataset = val_dataset.map(parse_image)
    val_dataset = val_dataset.repeat(EPOCHS)
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

    model = Deeplabv3(input_shape=(512, 512, 3),
                      classes=21, OS=16, load_weights=True, activation="softmax", reshape_outputs=False)

    losses = sparse_crossentropy_ignoring_last_label
    #losses = SparseCategoricalCrossentropy(from_logits=True)
    metrics = [Jaccard, sparse_accuracy_ignoring_last_label]
    optimizer = Adam(learning_rate=7e-4, epsilon=1e-8, decay=1e-6)

    model.compile(optimizer=optimizer, sample_weight_mode="temporal",
                  loss=losses, metrics=metrics, run_eagerly=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=3, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model_unet.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]

    TRAIN_STEPS = TRAIN_SIZE // BATCH_SIZE
    VAL_STEPS = VAL_SIZE // BATCH_SIZE

    model_history = model.fit(train_dataset, epochs=EPOCHS,
                              validation_data=val_dataset,
                              steps_per_epoch=TRAIN_STEPS,
                              validation_steps=VAL_STEPS,
                              callbacks=callbacks)


if __name__ == '__main__':
    main()
