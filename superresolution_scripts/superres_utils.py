import os
import h5py
import numpy as np
import tensorflow as tf
from superresolution_scripts.superresolution import Superresolution
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


def class_in_image(image_path, class_id, image_size=(512, 512)):
    mask_path = image_path.replace(
        "JPEGImages", "SegmentationClassAug").replace("jpg", "png")
    mask = load_image(mask_path, image_size=image_size,
                      normalize=False, is_png=True, resize_method="nearest")

    return np.any(mask == class_id)


def filter_images_by_class(path_list, filter_class_id, num_images=None, image_size=(512, 512)):

    max_images = num_images if num_images is not None else len(path_list)
    image_paths = []

    for path in path_list:
        if len(image_paths) == max_images:
            break

        if class_in_image(path, class_id=filter_class_id, image_size=image_size):
            image_paths.append(path)

    return image_paths


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


def normalize_coefficients(coeff_dict):
    """
    Given a dictionary of coefficients returns a new dictionary 
    containing the normalized coefficients such that they sum up to one
    """
    normalizer = np.sum(list(coeff_dict.values()))
    new_dict = {key: (value / normalizer)
                for (key, value) in coeff_dict.items()}

    return new_dict


def load_SR_data(filepath, num_aug=100, mode_slice=True, global_normalize=True):
    """
    Load and unpacks a hdf5 file that contains the data for the super-resolution problem

    Args:
        filepath (str): the full path to the hdf5 file
        num_aug (int, optional): The (minimum) length of each image array in the file. 
            Used to check the file validity. Defaults to 100.
        mode_slice (bool, optional): Toggle between slice mode and argmax mode. Defaults to True.
        global_normalize (bool, optional): Whether to normalize the images with respect to the global max/min from all image array. Defaults to True.

    Raises:
        Exception: File does not contain enough images 

    Returns:
        Alla the data stored in the file: class_masks, max_masks (if slice mode None otherwise), angles, shifts, filename.
    """
    file = h5py.File(f"{filepath}", "r")

    if not check_hdf5_validity(file, num_aug=num_aug):
        file.close()
        raise Exception(f"File: {filepath} is invalid")

    filename = file.attrs["filename"]
    angles = file["angles"][:num_aug]
    shifts = file["shifts"][:num_aug]

    class_masks = file["class_masks"][:num_aug]
    class_masks = tf.stack(class_masks)

    global_min, global_max = (tf.reduce_min(class_masks), tf.reduce_max(class_masks)) if global_normalize else (
        None, None)

    class_masks = tf.map_fn(
        fn=lambda image: min_max_normalization(image.numpy(), new_min=0.0, new_max=1.0, global_min=global_min,
                                               global_max=global_max), elems=class_masks)

    max_masks = None

    if mode_slice:
        max_masks = file["max_masks"][:num_aug]
        max_masks = tf.stack(max_masks)

        global_min, global_max = (tf.reduce_min(max_masks), tf.reduce_max(max_masks)) if global_normalize else (
            None, None)

        max_masks = tf.map_fn(
            fn=lambda image: min_max_normalization(image.numpy(), new_min=0.0, new_max=1.0, global_min=global_min,
                                                   global_max=global_max), elems=max_masks)

    file.close()

    return class_masks, max_masks, angles, shifts, filename


def compute_SR(superresolution_obj: Superresolution, class_masks, angles, shifts, filename, dest_folder,
               SR_type="aug", max_masks=None, save_output=False, class_id=8):
    """
    Computes the SR problem.

    Args:
        superresolution_obj (Superresolution): The Superresolution class object
        class_masks (Tensor): The array of class (LR) images
        angles (ndarray): The arry of angles used in the augmentaion process
        shifts (ndarray): The arry of shifts used in the augmentaion process
        filename (str): The filename asscociated with the image
        SR_type (str): One of 'aug', 'mean', 'max'. Defines the type of SR
        max_masks (Tensor, optional): Used in slice mode. It's the array of the max images. Defaults to None.
        save_output (bool, optional): Store the intermediate class/max HR images. Defaults to False.
        class_id (int, optional): class id of the selected class. Defaults to 8.
        dest_folder (Path, optional): Path to store the final target image.

    Returns:
        ndarray: The final HR image
    """

    assert(SR_type in ["aug", "mean", "max"],
           "SR_type must be either 'aug', 'mean' or 'max'")

    out_folder = os.path.join(dest_folder, f"{SR_type}_SR")

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if SR_type == "aug":
        SR_function = superresolution_obj.augmented_superresolution
    elif SR_type == "mean":
        SR_function = superresolution_obj.mean_superresolution
    elif SR_type == "max":
        SR_function = superresolution_obj.max_superresolution

    target_image_class, _ = SR_function(class_masks, angles, shifts)

    if max_masks is not None:
        target_image_max, _ = SR_function(max_masks, angles, shifts)
        th_mask = threshold_image(
            target_image_class, class_id, th_mask=target_image_max)

    else:
        th_mask = threshold_image(target_image_class, class_id, th_factor=.15)

    if save_output:
        tf.keras.utils.save_img(
            f"{out_folder}/{filename}_class.png", target_image_class, scale=True)
        if max_masks is not None:
            tf.keras.utils.save_img(
                f"{out_folder}/{filename}_max.png", target_image_max, scale=True)

    tf.keras.utils.save_img(
        f"{out_folder}/{filename}_{SR_type}_SR.png", th_mask, scale=True)

    return th_mask
