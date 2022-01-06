import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from superresolution import Superresolution
from utils import plot_images, plot_prediction, load_image


def load_images(img_folder):
    images = []

    for img_name in os.listdir(img_folder):
        if ".npy" in img_name:
            continue
        image = load_image(os.path.join(img_folder, img_name), normalize=False, is_png=True)
        images.append(image)

    return images


def get_precomputed_folders_path(root_dir, num_aug=100):
    valid_folders = []
    for path in os.listdir(root_dir):
        full_path = os.path.join(root_dir, path)
        if len(os.listdir(full_path)) == (num_aug + 2):
            valid_folders.append(full_path)
        else:
            print(f"Skipped folder named {path} as it is not valid")

    return valid_folders


def compute_save_final_output(superresolution_obj, precomputed_features_folders, output_folder):

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for folder in tqdm(precomputed_features_folders):
        augmented_images = load_images(folder)
        augmented_images = tf.cast(augmented_images / np.max(augmented_images, axis=(1, 2, 3), keepdims=True),
                                   tf.float32)

        base_name = os.path.basename(os.path.normpath(folder))
        angles = np.load(os.path.join(folder, f"{base_name}_angles.npy"))
        shifts = np.load(os.path.join(folder, f"{base_name}_shifts.npy"))
        target_image = superresolution_obj.compute_output(augmented_images, angles, shifts)

        tf.keras.utils.save_img(f"{output_folder}/{base_name}.png", target_image[0])


def main():
    # augmentation parameters
    num_aug = 50

    # super resolution parameters
    learning_rate = 1e-3
    lambda_eng = 0.0001 * num_aug
    lambda_tv = 0.002 * num_aug
    num_iter = 400

    superresolution = Superresolution(
        lambda_tv=lambda_tv,
        lambda_eng=lambda_eng,
        num_iter=num_iter,
        num_aug=num_aug,
        learning_rate=learning_rate
    )

    data_root = os.path.join(os.getcwd(), "data")
    precomputed_root_dir = os.path.join(data_root, "precomputed_features")
    output_folder = os.path.join(data_root, "final_output")

    precomputed_folders_path = get_precomputed_folders_path(precomputed_root_dir, num_aug=num_aug)
    compute_save_final_output(superresolution, precomputed_folders_path, output_folder)

    # plot_prediction([image, target_image[0]], only_prediction=True, show_overlay=True)
    # plot_images(images[:20], rows=5, columns=4)
    print("Done")


if __name__ == '__main__':
    main()
