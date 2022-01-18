import os
import argparse
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from model import DeeplabV3Plus
from utils import plot_images, plot_prediction, load_image, create_mask, get_img_paths

parser = argparse.ArgumentParser()
parser.add_argument("num_aug", help="Number of augmented copies", type=int)

args = parser.parse_args()


def augment_images(batched_images, angles, shifts):

    rotated_images = tfa.image.rotate(batched_images, angles, interpolation="bilinear")
    translated_images = tfa.image.translate(rotated_images, shifts, interpolation="bilinear")

    return translated_images


def save_augmented_features(model, images_array, dest_folder):
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)

    predictions = model.predict(images_array, batch_size=2)

    for i, prediction in enumerate(predictions):
        mask = create_mask(prediction)
        tf.keras.utils.save_img(f"{dest_folder}/{i}.png", mask, scale=False)

    return predictions


def precompute_augmented_features(image_path_list, dest_root_folder, model, num_aug=100, angle_max=0.5, shift_max=30):
    for image_path in tqdm(image_path_list):
        image = load_image(image_path, image_size=(512, 512), normalize=True)
        batched_image = tf.tile(tf.expand_dims(image, axis=0), [num_aug, 1, 1, 1])  # Size [num_aug, 512, 512, 3]
        angles = np.random.uniform(-angle_max, angle_max, num_aug)
        shifts = np.random.uniform(-shift_max, shift_max, (num_aug, 2))
        # First sample is not augmented
        angles[0] = 0
        shifts[0] = np.array([0, 0])
        angles = angles.astype("float32")
        shifts = shifts.astype("float32")

        augmented_images = augment_images(batched_image, angles, shifts)

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        dest_folder = os.path.join(dest_root_folder, image_name)

        save_augmented_features(model, augmented_images, dest_folder=dest_folder)
        np.save(os.path.join(dest_folder, f"{image_name}_angles"), angles)
        np.save(os.path.join(dest_folder, f"{image_name}_shifts"), shifts)


def main():
    # augmentation parameters
    num_aug = args.num_aug

    #TODO: Set this as optional arguments
    angle_max = 0.5  # in radians
    shift_max = 30

    data_root = os.path.join(os.getcwd(), "data")
    image_list_path = os.path.join(data_root, "augmented_file_lists", "valaug.txt")
    image_folder_path = os.path.join(data_root, "VOCdevkit", "VOC2012", "JPEGImages")
    image_paths = get_img_paths(image_list_path, image_folder_path)[:20]

    dest_root_folder = os.path.join(data_root, "precomputed_features")
    if not os.path.exists(dest_root_folder):
        os.mkdir(dest_root_folder)

    model = DeeplabV3Plus(
        input_shape=(512, 512, 3),
        classes=21,
        OS=16,
        last_activation=None,
        load_weights=True,
        backbone="mobilenet",
        alpha=1.).build_model(final_upsample=False)

    precompute_augmented_features(image_paths, dest_root_folder, model, num_aug=num_aug,
                                  angle_max=angle_max, shift_max=shift_max)

    # sample_image = augmented_images[67]
    # sample_mask = create_mask(predictions[67])
    # sample_mask = tf.image.resize(sample_mask, (512, 512))
    # plot_prediction([sample_image, sample_mask], only_prediction=True, show_overlay=True)
    # plot_images(augmented_images[:20], rows=5, columns=4)

    print("Done")


if __name__ == '__main__':
    main()
