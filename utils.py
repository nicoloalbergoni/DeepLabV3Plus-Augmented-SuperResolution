import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K


def plot_prediction(display_list, only_prediction=True, show_overlay=True):
    plt.figure(figsize=(18, 18))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list) + 1, i + 1)
        if only_prediction and i == 1:
            plt.title(title[-1])
        else:
            plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(
            display_list[i]))
        plt.axis('off')

    if show_overlay:
        plt.subplot(1, len(display_list) + 1, len(display_list) + 1)
        plt.title("Overlay")
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[0]))
        plt.imshow(tf.keras.preprocessing.image.array_to_img(
            display_list[-1]), alpha=0.5)
        plt.axis("off")

    plt.show()


def plot_images(image_list, rows, columns):
    for i in range(len(image_list)):
        plt.subplot(rows, columns, i + 1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(
            image_list[i]))
        plt.axis('off')
    plt.show()


def sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    nb_classes = (y_pred.shape.as_list())[-1]
    y_true = tf.one_hot(
        tf.cast(y_true[:, :, 0], tf.int32), nb_classes + 1)[:, :, :-1]

    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)


def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    nb_classes = (y_pred.shape.as_list())[-1]
    y_pred = tf.reshape(y_pred, (-1, nb_classes))
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int64)
    # All labels not equal to the ignored one
    legal_labels = ~tf.equal(y_true, 255)
    right_labels = tf.reduce_sum(tf.cast(legal_labels & tf.equal(y_true, tf.argmax(
        y_pred, axis=-1)), tf.float32))  # Number of right and legal labels
    # Total number of legal labels
    total_labels = tf.reduce_sum(tf.cast(legal_labels, tf.float32))
    return right_labels / total_labels


def sparse_Mean_IOU(y_true, y_pred):
    # tf.print(K.shape(y_true))
    # tf.print(K.shape(y_pred))
    nb_classes = (y_pred.shape.as_list())[-1]
    iou = []
    pred_pixels = tf.argmax(y_pred, axis=-1)
    for i in range(0, nb_classes):  # exclude last label (void)
        y_true_squeeze = y_true[:, :, 0]
        true_labels = tf.equal(y_true_squeeze, i)
        pred_labels = tf.equal(pred_pixels, i)
        inter = tf.cast(true_labels & pred_labels, tf.int32)
        union = tf.cast(true_labels | pred_labels, tf.int32)
        legal_batches = tf.reduce_sum(
            tf.cast(true_labels, tf.int32), axis=1) > 0  # check if the current class is present in the image
        # tf.print(K.shape(legal_batches))
        ious = tf.reduce_sum(inter, axis=1) / tf.reduce_sum(union, axis=1)
        # tf.print(K.shape(ious))
        # returns average IoU of the same objects
        iou.append(tf.reduce_mean(
            tf.gather(ious, indices=tf.where(legal_batches))))

    iou = tf.stack(iou)
    legal_labels = ~tf.math.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return tf.reduce_mean(iou)


def single_class_IOU(y_true, y_pred, class_id):
    y_true_squeeze = tf.squeeze(y_true)
    y_pred_squeeze = tf.squeeze(y_pred)
    classes = [0, class_id]  # Only check in background and given class

    y_true_squeeze = tf.where(y_true_squeeze != class_id, 0, y_true_squeeze)

    ious = []
    for i in classes:
        true_labels = tf.equal(y_true_squeeze, i)
        pred_labels = tf.equal(y_pred_squeeze, i)
        inter = tf.cast(true_labels & pred_labels, tf.int32)
        union = tf.cast(true_labels | pred_labels, tf.int32)

        iou = tf.reduce_sum(inter) / tf.reduce_sum(union)
        ious.append(iou)

    ious = tf.stack(ious)
    legal_labels = ~tf.math.is_nan(ious)
    ious = tf.gather(ious, indices=tf.where(legal_labels))
    return tf.reduce_mean(ious)


def load_image(img_path, image_size=None, normalize=True, is_png=False, resize_method="bilinear"):
    raw_img = tf.io.read_file(img_path)

    # Defaults to jpg images
    if not is_png:
        image = tf.image.decode_jpeg(raw_img, channels=3)
    else:
        image = tf.image.decode_png(raw_img, channels=1)

    # Resize only if size is specified
    if image_size is not None:
        image = tf.image.resize(image, image_size, method=resize_method)

    image = tf.cast(image, tf.float32)

    if normalize:
        image = image / 255.0

    return image


def create_mask(pred_mask):
    # pred_mask -> [IMG_SIZE, IMG_SIZE, N_CLASS]
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = tf.expand_dims(pred_mask, axis=-1)  # add 1 dim for plotting
    return pred_mask


def get_prediction(model, input_image):

    prediction = model.predict(input_image[tf.newaxis, ...])
    mask = create_mask(prediction[0])

    return mask


def plot_image(image):
    plt.figure(figsize=(20, 20))
    plt.imshow(tf.keras.preprocessing.image.array_to_img(image))
    plt.axis('off')
    plt.show()


def plot_histogram(image):
    plt.figure(figsize=(18, 18))
    vals = image.flatten()
    b, bins, patches = plt.hist(vals, 255)
    plt.show()


def print_labels(masks):
    title = ["Standard Labels: ", "Superres Labels: "]
    for i in range(2):
        values, count = np.unique(masks[i], return_counts=True)
        print(title[i] + str(dict(zip(values, count))))


def compute_IoU(true_image, image, img_size=(512, 512), class_id=None):
    """
    Compute the IoU between the true image and the given imge.

    Args:
        true_image (Tensor): Ground Truth image HR
        image (Tensor): Given image
        img_size (tuple, optional): HR image size. Defaults to (512, 512).
        class_id (int, optional): id of the choosen class. Defaults to 8.

    Returns:
        float, float: The IoUs of the standard and superresolution images
    """
    true_image = tf.reshape(true_image, (img_size[0] * img_size[1], 1))
    image = tf.reshape(image, (img_size[0] * img_size[1], 1))

    if class_id is not None:
        iou = single_class_IOU(true_image, image, class_id=class_id)
    else:
        iou = sparse_Mean_IOU(true_image, image)

    return iou.numpy()
