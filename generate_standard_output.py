import os
import h5py
import argparse
import gc
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from model import DeeplabV3Plus
import tensorflow_addons as tfa
from utils import load_image, get_prediction, create_mask
from superresolution_scripts.superres_utils import get_img_paths, filter_images_by_class, min_max_normalization

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_samples", help="Number of samples taken from the dataset", action="store", type=int, default=None)

parser.add_argument(
    "--class_id", help="class_id to binarize the image", action="store", type=int, default=None, choices=range(21))

parser.add_argument("--backbone", help="Either mobilenet or xception, specifies the type of backbone to use", action="store",
                    type=str, choices=["mobilenet", "xception"], default="xception")

parser.add_argument("--use_validation",
                    help="Create data from validation set", action="store_true")

args = parser.parse_args()

SEED = 1234

np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (512, 512)
BATCH_SIZE = 16
CLASS_ID = args.class_id
NUM_SAMPLES = args.num_samples
MODEL_BACKBONE = args.backbone
USE_VALIDATION = args.use_validation

DATA_DIR = os.path.join(os.getcwd(), "data")
PASCAL_ROOT = os.path.join(DATA_DIR, "dataset_root", "VOCdevkit", "VOC2012")
IMGS_PATH = os.path.join(PASCAL_ROOT, "JPEGImages")

SUPERRES_ROOT = os.path.join(DATA_DIR, "superres_root")
STANDARD_OUTPUT_ROOT = os.path.join(SUPERRES_ROOT, "standard_output")
STANDARD_OUTPUT_DIR = os.path.join(
    STANDARD_OUTPUT_ROOT, f"{MODEL_BACKBONE}_{CLASS_ID}{'_validation' if USE_VALIDATION else ''}")


def compute_standard_output(images_paths, model, dest_folder, filter_class_id=None, image_size=(512, 512)):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for image_path in tqdm(images_paths):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(dest_folder, f"{image_name}.png")

        image = load_image(image_path, image_size=image_size, normalize=True)
        standard_mask = get_prediction(model, image)
        if filter_class_id is not None:
            standard_mask = tf.where(standard_mask == filter_class_id, standard_mask,
                                     0)  # Set to 0 all predictions different from the given class
        tf.keras.utils.save_img(save_path, standard_mask, scale=False)


def main():
    image_list_path = os.path.join(DATA_DIR, "augmented_file_lists",
                                   f"{'valaug' if USE_VALIDATION else 'trainaug'}.txt")
    image_paths = get_img_paths(
        image_list_path, IMGS_PATH, is_png=False, sort=True)

    if CLASS_ID is not None:
        image_paths = filter_images_by_class(
            image_paths, filter_class_id=CLASS_ID, image_size=IMG_SIZE)

    image_paths_partial = image_paths[:NUM_SAMPLES]
    print(
        f"Valid images: {len(image_paths_partial)} (Initial: {len(image_paths)})")

    model = DeeplabV3Plus(
        input_shape=(512, 512, 3),
        classes=21,
        OS=16,
        last_activation=None,
        load_weights=True,
        backbone=MODEL_BACKBONE,
        alpha=1.).build_model(final_upsample=True)

    # Compute standard (classicl upsample) masks
    print("Computing standard output images...")
    compute_standard_output(image_paths_partial, model, dest_folder=STANDARD_OUTPUT_DIR,
                            filter_class_id=CLASS_ID)


if __name__ == '__main__':
    main()
