import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from utils import load_image


def get_img_paths(image_list_path, image_folder, is_png=False):
    """
    Given a file containing image identifiers returns the complete path to the image in the specified folder

    Args:
        image_list_path: Path to the file containing the image-names list
        image_folder: Path to the folder containing the images
        is_png: Type of the images (jpg or png)

    Returns: List of full paths to the images
    """
    ext = ".jpg" if not is_png else ".png"
    return [os.path.join(image_folder, line.rstrip() + ext) for line in open(image_list_path)]


def filter_by_class(img_paths, class_id, image_size=(512, 512)):
    """
    Given a list of image paths, return the images that contain the given class id in the respective mask

    Args:
        img_paths: List of image paths to check
        class_id: Class id used for filering
        image_size: Size of the image used to load and resize the image

    Returns: A dictionary whose keys are the image filename and values are the actual images

    """
    images_dict = {}
    for img_path in img_paths:
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = img_path.replace("JPEGImages", "SegmentationClassAug").replace("jpg", "png")
        mask = load_image(mask_path, image_size=image_size, normalize=False, is_png=True, resize_method="nearest")
        if np.any(mask == class_id):
            image = load_image(img_path, image_size=image_size, normalize=True)
            images_dict[image_name] = image

    return images_dict


def min_max_normalization(image, new_min=0.0, new_max=255.0, global_min=None, global_max=None):
    min = image.min() if global_min is None else global_min
    max = image.max() if global_max is None else global_max

    num = (image - min) * (new_max - new_min)
    den = (max - min) if (max - min) != 0 else 1.0
    return new_min + (num / den)


def load_images(img_folder):
    images = []
    # Sort images based on their filename which is an integer indicating the augmented copy number
    image_list = sorted([name.replace(".png", "") for name in os.listdir(img_folder) if ".npy" not in name], key=int)

    for img_name in image_list:
        if ".npy" in img_name:
            continue
        image = load_image(os.path.join(img_folder, f"{img_name}.png"), normalize=False, is_png=True)
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


def list_precomputed_data_paths(root_dir):
    paths = []

    for path, subdirs, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith(".hdf5"):
                paths.append(os.path.join(path, filename))

    return paths


def check_hdf5_validity(file, num_aug=100):
    # Check if all datasets in the file have the right cardinality
    for keys in file:
        num = file[keys].shape[0]
        if num != num_aug:
            return False

    return True
