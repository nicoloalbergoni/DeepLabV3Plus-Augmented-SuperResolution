import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from model import DeeplabV3Plus
from utils import plot_images, plot_prediction


def load_image(img_path, image_size=(512, 512)):
    raw_img = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(raw_img, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0

    return image


def augment_images(batched_images, angles, shifts):
    rotated_images = tfa.image.rotate(batched_images, angles, interpolation="bilinear")
    translated_images = tfa.image.translate(rotated_images, shifts, interpolation="bilinear")

    return translated_images


def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    # pred_mask -> [IMG_SIZE, IMG_SIZE, N_CLASS]
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = tf.expand_dims(pred_mask, axis=-1)  # add 1 dim for plotting
    return pred_mask


def save_augmented_features(model, images_array, dest_folder):
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)

    predictions = model.predict(images_array)

    for i, prediction in enumerate(predictions):
        mask = create_mask(prediction)
        tf.keras.utils.save_img(f"{dest_folder}/{i}.png", mask)

    return predictions


def main():
    num_aug = 100

    # augmentation parameters
    angle_min = -0.5  # in radians
    angle_max = 0.5
    angles = np.random.uniform(angle_min, angle_max, num_aug)
    shift_min = -30
    shift_max = 30
    shifts = np.random.uniform(shift_min, shift_max, (num_aug, 2))
    # first Grad-CAM is not augmented
    angles[0] = 0
    shifts[0] = np.array([0, 0])
    angles = angles.astype("float32")
    shifts = shifts.astype("float32")

    test_image_path = os.path.join(os.getcwd(), "test_img.jpg")
    image = load_image(test_image_path)
    batched_images = tf.tile(tf.expand_dims(image, 0), [num_aug, 1, 1, 1])  # Size [100, 512, 512, 3]

    augmented_images = augment_images(batched_images, angles, shifts)

    model = DeeplabV3Plus(
        input_shape=(512, 512, 3),
        classes=21,
        OS=16,
        last_activation=None,
        load_weights=True,
        backbone="mobilenet",
        alpha=1.).build_model(final_upsample=False)

    dest_folder = "test_folder"
    predictions = save_augmented_features(model, augmented_images, dest_folder=dest_folder)
    np.save(os.path.join(dest_folder, "angles"), angles)
    np.save(os.path.join(dest_folder, "shifts"), shifts)

    # sample_image = augmented_images[67]
    # sample_mask = create_mask(predictions[67])
    # sample_mask = tf.image.resize(sample_mask, (512, 512))
    # plot_prediction([sample_image, sample_mask], only_prediction=True, show_overlay=True)
    # plot_images(augmented_images[:20], rows=5, columns=4)

    print("Done")


if __name__ == '__main__':
    main()
