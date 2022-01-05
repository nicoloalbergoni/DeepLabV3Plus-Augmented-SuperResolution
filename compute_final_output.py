import os
import numpy as np
import tensorflow as tf
from superresolution import Superresolution
from utils import plot_images, plot_prediction


def load_image(img_path, image_size=(512, 512)):
    raw_img = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(raw_img, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0

    return image


def load_images(img_folder):
    images = []

    for img_name in os.listdir(img_folder):
        if ".npy" in img_name:
            continue
        raw_img = tf.io.read_file(os.path.join(img_folder, img_name))
        image = tf.image.decode_png(raw_img, channels=1)
        images.append(image)

    return images


def main():
    # augmentation parameters
    num_aug = 100
    angles = np.load("test_folder/angles.npy")
    shifts = np.load("test_folder/shifts.npy")

    # super resolution parameters
    learning_rate = 1e-2
    lambda_eng = 0.0001 * num_aug
    lambda_tv = 0.002 * num_aug
    num_iter = 500

    superresolution = Superresolution(
        angles,
        shifts,
        lambda_tv=lambda_tv,
        lambda_eng=lambda_eng,
        num_iter=num_iter,
        learning_rate=learning_rate
    )

    augmented_images = load_images("test_folder")
    augmented_images = tf.cast(augmented_images / np.max(augmented_images, axis=(1, 2, 3), keepdims=True), tf.float32)

    target_image = superresolution.compute_output(augmented_images)

    test_image_path = os.path.join(os.getcwd(), "test_img.jpg")
    image = load_image(test_image_path)

    plot_prediction([image, target_image[0]], only_prediction=True, show_overlay=True)
    # plot_images(images[:20], rows=5, columns=4)
    print("Done")


if __name__ == '__main__':
    main()
