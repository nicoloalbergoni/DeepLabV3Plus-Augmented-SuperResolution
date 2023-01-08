import os
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from model import DeeplabV3Plus
from superresolution_scripts.superres_utils import get_img_paths, filter_images_by_class
from superresolution_scripts.augmentation_utils import compute_augmented_feature_maps

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

parser = argparse.ArgumentParser()
parser.add_argument("--num_aug", help="Number of augmented copies created for each image",
                    action="store", type=int, default=100)
parser.add_argument(
    "--num_samples", help="Number of samples taken from the dataset", action="store", type=int, default=500)

parser.add_argument(
    "--mode", help="Whether to operate in slicing, slicing variation or argmax mode", action="store", type=str, choices=["slice_max", "slice", "argmax"], default="argmax")

parser.add_argument("--angle_max", help="Max angle value (in radians) used for rotations", action="store",
                    type=float, default=0.3)

parser.add_argument("--shift_max", help="Max shift value used for traslations", action="store",
                    type=int, default=30)

parser.add_argument("--backbone", help="Either mobilenet or xception, specifies the type of backbone to use", action="store",
                    type=str, choices=["mobilenet", "xception"], default="xception")

parser.add_argument("--use_validation",
                    help="Create data from validation set", action="store_true")

parser.add_argument(
    "--class_id", help="class_id for image filtering", action="store", type=int, default=8, choices=range(21), required=True)


args = parser.parse_args()


SEED = 1234

np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (512, 512)
BATCH_SIZE = 16
NUM_AUG = args.num_aug
CLASS_ID = args.class_id
NUM_SAMPLES = args.num_samples
ANGLE_MAX = args.angle_max
SHIFT_MAX = args.shift_max
MODE = args.mode
MODEL_BACKBONE = args.backbone
USE_VALIDATION = args.use_validation

DATA_DIR = os.path.join(os.getcwd(), "data")
PASCAL_ROOT = os.path.join(DATA_DIR, "dataset_root", "VOCdevkit", "VOC2012")
IMGS_PATH = os.path.join(PASCAL_ROOT, "JPEGImages")

SUPERRES_ROOT = os.path.join(DATA_DIR, "superres_root")
AUGMENTED_COPIES_ROOT = os.path.join(SUPERRES_ROOT, "augmented_copies")
AUGMENTED_COPIES_OUTPUT_DIR = os.path.join(AUGMENTED_COPIES_ROOT,
                                           f"{MODEL_BACKBONE}_{MODE}_{CLASS_ID}_{NUM_AUG}{'_validation' if USE_VALIDATION else ''}")


def main():
    image_list_path = os.path.join(DATA_DIR, "augmented_file_lists",
                                   f"{'valaug' if USE_VALIDATION else 'trainaug'}.txt")
    image_paths = get_img_paths(
        image_list_path, IMGS_PATH, is_png=False, sort=True)
    images_paths_filtered = filter_images_by_class(
        image_paths, filter_class_id=CLASS_ID, num_images=NUM_SAMPLES, image_size=IMG_SIZE)

    print(
        f"Valid images: {len(images_paths_filtered)} (Initial: {len(image_paths)})")

    model = DeeplabV3Plus(
        input_shape=(512, 512, 3),
        classes=21,
        OS=16,
        last_activation=None,
        load_weights=True,
        backbone=MODEL_BACKBONE,
    ).build_model(final_upsample=False)

    print("Generating augmented copies...")
    for image_path in tqdm(images_paths_filtered):
        compute_augmented_feature_maps(image_path, model, mode=MODE, filter_class_id=CLASS_ID, num_aug=NUM_AUG,
                                       angle_max=ANGLE_MAX, shift_max=SHIFT_MAX,
                                       image_size=IMG_SIZE, batch_size=BATCH_SIZE, dest_folder=AUGMENTED_COPIES_OUTPUT_DIR)


if __name__ == '__main__':
    main()
