import os
import h5py
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from superresolution_scripts.superresolution import Superresolution
from utils import load_image
from superresolution_scripts.superres_utils import min_max_normalization, \
    list_precomputed_data_paths, check_hdf5_validity, threshold_image, single_class_IOU

DATA_DIR = os.path.join(os.getcwd(), "data")
PASCAL_ROOT = os.path.join(DATA_DIR, "dataset_root", "VOCdevkit", "VOC2012")
IMGS_PATH = os.path.join(PASCAL_ROOT, "JPEGImages")

SUPERRES_ROOT = os.path.join(DATA_DIR, "superres_root")
PRECOMPUTED_OUTPUT_DIR = os.path.join(SUPERRES_ROOT, "precomputed_features")
STANDARD_OUTPUT_DIR = os.path.join(SUPERRES_ROOT, "standard_output")
SUPERRES_OUTPUT_DIR = os.path.join(SUPERRES_ROOT, "superres_output")

SEED = 1234

tf.keras.utils.set_random_seed(SEED)

IMG_SIZE = (512, 512)
NUM_AUG = 50
CLASS_ID = 8
NUM_SAMPLES = 50
MODE = "slice"


def compute_superresolution_output(precomputed_data_paths, superresolution_obj, dest_folder, mode="slice", num_aug=100,
                                   global_normalize=True, save_output=False):
    superres_masks = {}
    losses = {}

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for file_path in tqdm(precomputed_data_paths):

        file = h5py.File(f"{file_path}", "r")

        if not check_hdf5_validity(file, num_aug=num_aug):
            print(f"File: {file_path} is invalid, skipping...")
            file.close()
            continue

        filename = file.attrs["filename"]
        angles = file["angles"][:]
        shifts = file["shifts"][:]

        class_masks = file["class_masks"][:]
        class_masks = tf.stack(class_masks)

        if mode == "slice":
            max_masks = file["max_masks"][:]
            max_masks = tf.stack(max_masks)

        file.close()

        global_min, global_max = (tf.reduce_min(class_masks), tf.reduce_max(class_masks)) if global_normalize else (
            None, None)

        class_masks = tf.map_fn(
            fn=lambda image: min_max_normalization(image.numpy(), new_min=0.0, new_max=1.0, global_min=global_min,
                                                   global_max=global_max), elems=class_masks)

        target_image_class, class_loss = superresolution_obj.compute_output(class_masks, angles, shifts)
        target_image_class = (target_image_class[0]).numpy()
        # print(f"Final class loss for image {filename}: {class_loss}")

        if mode == "slice":
            global_min, global_max = (tf.reduce_min(max_masks), tf.reduce_max(max_masks)) if global_normalize else (
                None, None)

            max_masks = tf.map_fn(
                fn=lambda image: min_max_normalization(image.numpy(), new_min=0.0, new_max=1.0, global_min=global_min,
                                                       global_max=global_max), elems=max_masks)

            target_image_max, max_loss = superresolution_obj.compute_output(max_masks, angles, shifts)
            target_image_max = (target_image_max[0]).numpy()
            # print(f"Final max loss for image {filename}: {max_loss}")

        if save_output:
            tf.keras.utils.save_img(f"{dest_folder}/{filename}_class.png", target_image_class, scale=True)
            if mode == "slice":
                tf.keras.utils.save_img(f"{dest_folder}/{filename}_max.png", target_image_max, scale=True)

        superres_masks[filename] = {"class": target_image_class,
                                    "max": target_image_max} if mode == "slice" else target_image_class

    return superres_masks, losses


def evaluate_IOU(true_mask, standard_mask, superres_mask, img_size=(512, 512)):
    true_mask = tf.reshape(true_mask, (img_size[0] * img_size[1], 1))
    standard_mask = tf.reshape(standard_mask, (img_size[0] * img_size[1], 1))
    superres_mask = tf.reshape(superres_mask, (img_size[0] * img_size[1], 1))

    standard_IOU = single_class_IOU(true_mask, standard_mask, class_id=CLASS_ID)
    superres_IOU = single_class_IOU(true_mask, superres_mask, class_id=CLASS_ID)

    return standard_IOU.numpy(), superres_IOU.numpy()


def compare_results(superres_dict, image_size=(512, 512), verbose=False):
    standard_IOUs = []
    superres_IOUs = []

    for key in superres_dict:
        true_mask_path = os.path.join(PASCAL_ROOT, "SegmentationClassAug", f"{key}.png")
        true_mask = load_image(true_mask_path, image_size=image_size, normalize=False,
                               is_png=True, resize_method="nearest")

        standard_mask_path = os.path.join(STANDARD_OUTPUT_DIR, f"{key}.png")
        standard_mask = load_image(standard_mask_path, image_size=image_size, normalize=False, is_png=True,
                                   resize_method="nearest")

        superres_image = superres_dict[key]

        standard_IOU, superres_IOU = evaluate_IOU(true_mask, standard_mask, superres_image, img_size=image_size)
        standard_IOUs.append(standard_IOU)
        superres_IOUs.append(superres_IOU)

        if verbose:
            print(f"IOUs for image {key} - Standard: {str(standard_IOU)}, Superres: {str(superres_IOU)}")

    return standard_IOUs, superres_IOUs


def main():
    learning_rate = 1e-3
    lambda_eng = 0.109
    lambda_tv = 0.2294
    num_iter = 450

    superresolution = Superresolution(
        lambda_tv=lambda_tv,
        lambda_eng=lambda_eng,
        num_iter=num_iter,
        num_aug=NUM_AUG,
        learning_rate=learning_rate,
        verbose=False
    )

    path_list = list_precomputed_data_paths(PRECOMPUTED_OUTPUT_DIR)
    precomputed_data_paths = path_list if NUM_SAMPLES is None else path_list[:NUM_SAMPLES]

    superres_masks_dict, losses = compute_superresolution_output(precomputed_data_paths, superresolution, mode=MODE,
                                                                 dest_folder=SUPERRES_OUTPUT_DIR, num_aug=NUM_AUG,
                                                                 global_normalize=True, save_output=True)

    superres_masks_dict_th = {}

    for key in superres_masks_dict:
        target_dict = superres_masks_dict[key]
        if MODE == "slice":
            th_mask = threshold_image(target_dict["class"], CLASS_ID, th_mask=target_dict["max"])
        else:
            th_mask = threshold_image(target_dict["class"], CLASS_ID, th_factor=.15)

        superres_masks_dict_th[key] = th_mask

    standard_IOUs, superres_IOUs = compare_results(superres_masks_dict_th, image_size=IMG_SIZE, verbose=False)
    print(f"Standard mean IOU: {np.mean(standard_IOUs)},  Superres mean IOU: {np.mean(superres_IOUs)}")


if __name__ == '__main__':
    main()
