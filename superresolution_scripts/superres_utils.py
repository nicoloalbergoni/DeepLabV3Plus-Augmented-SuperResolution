import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from utils import load_image


def get_img_paths(image_list_path, image_folder, is_png=False, sort=True):
    """
    Given a file containing image identifiers returns the complete path to the image in the specified folder

    Args:
        image_list_path: Path to the file containing the image-names list
        image_folder: Path to the folder containing the images
        is_png: Type of the images (jpg or png)
        sort: If True sort the path list in alphabetical order (based on the basepath)

    Returns: List of full paths to the images
    """
    ext = ".jpg" if not is_png else ".png"
    paths = [os.path.join(image_folder, line.rstrip() + ext)
             for line in open(image_list_path)]

    if sort:
        paths = sorted(paths, key=lambda p: int(
            os.path.basename(p).split('.')[0]))

    return paths


def filter_image(image_path, class_id, image_size=(512, 512)):

    mask_path = image_path.replace(
        "JPEGImages", "SegmentationClassAug").replace("jpg", "png")
    mask = load_image(mask_path, image_size=image_size,
                      normalize=False, is_png=True, resize_method="nearest")
    if np.any(mask == class_id):
        image = load_image(image_path, image_size=image_size, normalize=True)
        return image
    else:
        return None


def load_images(path_list, num_images=None, filter_class_id=None, image_size=(512, 512)):

    image_dict = {}

    max_images = num_images if num_images is not None else len(path_list)

    for path in path_list:
        if len(image_dict) == max_images:
            break

        image_name = os.path.splitext(os.path.basename(path))[0]

        if filter_class_id is not None:
            image = filter_image(
                path, class_id=filter_class_id, image_size=image_size)
            if image is not None:
                image_dict[image_name] = image
        else:
            image_dict[image_name] = load_image(
                path, image_size=image_size, normalize=True)

    return image_dict


def min_max_normalization(image, new_min=0.0, new_max=255.0, global_min=None, global_max=None):
    min = image.min() if global_min is None else global_min
    max = image.max() if global_max is None else global_max

    num = (image - min) * (new_max - new_min)
    den = (max - min) if (max - min) != 0 else 1.0
    return new_min + (num / den)


def load_precomputed_images(img_folder):
    images = []
    # Sort images based on their filename which is an integer indicating the augmented copy number
    image_list = sorted([name.replace(".png", "") for name in os.listdir(
        img_folder) if ".npy" not in name], key=int)

    for img_name in image_list:
        if ".npy" in img_name:
            continue
        image = load_image(os.path.join(
            img_folder, f"{img_name}.png"), normalize=False, is_png=True)
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


def list_precomputed_data_paths(root_dir, sort=False):
    paths = []

    for path, subdirs, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith(".hdf5"):
                paths.append(os.path.join(path, filename))

    if sort:
        paths = sorted(paths, key=lambda p: int(
            os.path.basename(p).split('.')[0]))

    return paths


def check_hdf5_validity(file, num_aug=100):
    # Check if all datasets in the hdf5 file have at least num_aug images
    for keys in file:
        num = file[keys].shape[0]
        if num < num_aug:
            return False

    return True


def threshold_image(image, th_value, th_factor=.15, th_mask=None):
    """
    Perform the pixel-wise threshold of the given image.

    Optionally if th_mask is given the input image is thresholded against the value in th_mask

    Args:
        image: The mask to be thresholded
        th_value: Pixel value for the destination image
        th_factor: Percentage of the image max value used as thresholding value
        th_mask: Thresholding mask

    Returns: The thresholded image that contains either 0 or th_value

    """
    if th_mask is not None:
        th_image = tf.where(image >= th_mask, th_value, 0)
    else:
        max_value = tf.cast(tf.reduce_max(image), tf.float32) * th_factor
        th_image = tf.where(image > max_value, th_value, 0)

    return th_image.numpy()


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


def normalize_coefficients(coeff_dict):
    """
    Given a dictionary of coefficients returns a new dictionary 
    containing the normalized coefficients such that they sum up to one
    """
    normalizer = np.sum(list(coeff_dict.values()))
    new_dict = {key: (value / normalizer)
                for (key, value) in coeff_dict.items()}

    return new_dict
