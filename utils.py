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
    nb_classes = (y_pred.shape.as_list())[-1]
    iou = []
    pred_pixels = tf.argmax(y_pred, axis=-1)
    for i in range(0, nb_classes):  # exclude first label (background) and last label (void)
        y_true_squeeze = y_true[:, :, 0]
        true_labels = tf.equal(y_true_squeeze, i)
        pred_labels = tf.equal(pred_pixels, i)
        inter = tf.cast(true_labels & pred_labels, tf.int32)
        union = tf.cast(true_labels | pred_labels, tf.int32)
        legal_batches = tf.reduce_sum(
            tf.cast(true_labels, tf.int32), axis=1) > 0  # check if the current class is present in the image

        ious = tf.reduce_sum(inter, axis=1) / tf.reduce_sum(union, axis=1)
        # returns average IoU of the same objects
        iou.append(tf.reduce_mean(
            tf.gather(ious, indices=tf.where(legal_batches))))

    iou = tf.stack(iou)
    legal_labels = ~tf.math.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return tf.reduce_mean(iou)
