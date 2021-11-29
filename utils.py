import cv2
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K


def preprocess_image(image, target_size_width=512, mean_subtraction_value=127.5):

    w, h, _ = image.shape
    ratio = float(target_size_width) / np.max([w, h])
    new_size = (int(ratio * h), int(ratio * w))
    resized_img = np.array(Image.fromarray(
        image.astype("uint8")).resize(new_size))

    # Normlization
    resized_img = (resized_img / mean_subtraction_value) - 1.

    # Padding to desired dimension
    pad_width = int(target_size_width - resized_img.shape[0])
    pad_height = int(target_size_width - resized_img.shape[1])

    pad_size = ((0, pad_width), (0, pad_height), (0, 0))

    resized_img = np.pad(resized_img, pad_size,
                         constant_values=0, mode="constant")

    return resized_img, pad_width, pad_height


def plot_prediction(display_list, only_prediction=True, show_overlay=True):
    plt.figure(figsize=(18, 18))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list) + 1, i + 1)
        if only_prediction and i == 1:
            plt.title(title[-1])
        else:
            plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')

    if show_overlay:
        plt.subplot(1, len(display_list) + 1, len(display_list) + 1)
        plt.title("Overlay")
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[0]))
        plt.imshow(tf.keras.preprocessing.image.array_to_img(
            display_list[-1]), alpha=0.5)
        plt.axis("off")

    plt.show()


def sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_true = K.one_hot(
        tf.cast(y_true[:, :, 0], tf.int32), nb_classes + 1)[:, :, :-1]

    return K.categorical_crossentropy(y_true, y_pred, from_logits=False)


def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    y_true = tf.cast(K.flatten(y_true), tf.int64)
    legal_labels = ~K.equal(y_true, nb_classes)
    return K.sum(tf.cast(legal_labels & K.equal(y_true, K.argmax(y_pred, axis=-1)), tf.float32)) / K.sum(tf.cast(legal_labels, tf.float32))


def Jaccard(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    pred_pixels = K.argmax(y_pred, axis=-1)
    print(pred_pixels.numpy().shape)
    for i in range(0, nb_classes):  # exclude first label (background) and last label (void)
        true_labels = K.equal(y_true[:, :, 0], i)
        pred_labels = K.equal(pred_pixels, i)
        inter = tf.cast(true_labels & pred_labels, tf.int32)
        union = tf.cast(true_labels | pred_labels, tf.int32)
        legal_batches = K.sum(tf.cast(true_labels, tf.int32), axis=1) > 0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches))))
        # iou.append(K.mean(ious[legal_batches]))

    iou = tf.stack(iou)
    legal_labels = ~tf.math.is_nan(iou)
    iou = iou[legal_labels]
    return K.mean(iou)


def _random_crop(image, label, crop_shape):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')

    if (crop_shape[0] < image.shape[1]) and (crop_shape[1] < image.shape[0]):
        x = random.randrange(image.shape[1]-crop_shape[0])
        y = random.randrange(image.shape[0]-crop_shape[1])

        return image[y:y+crop_shape[1], x:x+crop_shape[0], :], label[y:y+crop_shape[1], x:x+crop_shape[0]]
    else:
        image = cv2.resize(image, crop_shape)
        label = cv2.resize(label, crop_shape, interpolation=cv2.INTER_NEAREST)
        return image, label
