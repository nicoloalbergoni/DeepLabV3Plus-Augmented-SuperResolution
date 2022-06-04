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
from superresolution_scripts.superres_utils import get_img_paths, filter_images_by_class

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--num_aug", help="Number of augmented copies created for each image",
                    action="store", type=int, default=100)
parser.add_argument(
    "--num_samples", help="Number of samples taken from the dataset", action="store", type=int, default=500)

parser.add_argument(
    "--mode_slice", help="Whether to operate in slicing or argmax mode", action="store_true")

parser.add_argument("--angle_max", help="Max angle value (in radians) used for rotations", action="store",
                    type=float, default=0.3)

parser.add_argument("--shift_max", help="Max shift value used for traslations", action="store",
                    type=int, default=30)

parser.add_argument("--backbone", help="Either mobilenet or xception, specifies the type of backbone to use", action="store",
                    type=str, choices=["mobilenet", "xception"], default="xception")

args = parser.parse_args()


SEED = 1234

np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (512, 512)
BATCH_SIZE = 16
NUM_AUG = args.num_aug
CLASS_ID = 8
NUM_SAMPLES = args.num_samples
ANGLE_MAX = args.angle_max
SHIFT_MAX = args.shift_max
MODE_SLICE = args.mode_slice
MODEL_BACKBONE = args.backbone
USE_VALIDATION = False

DATA_DIR = os.path.join(os.getcwd(), "data")
PASCAL_ROOT = os.path.join(DATA_DIR, "dataset_root", "VOCdevkit", "VOC2012")
IMGS_PATH = os.path.join(PASCAL_ROOT, "JPEGImages")

SUPERRES_ROOT = os.path.join(DATA_DIR, "superres_root")
AUGMENTED_COPIES_ROOT = os.path.join(SUPERRES_ROOT, "augmented_copies")
AUGMENTED_COPIES_OUTPUT_DIR = os.path.join(AUGMENTED_COPIES_ROOT,
                                           f"{MODEL_BACKBONE}_{'slice' if MODE_SLICE else 'argmax'}_{NUM_AUG}{'_validation' if USE_VALIDATION else ''}")
STANDARD_OUTPUT_ROOT = os.path.join(SUPERRES_ROOT, "standard_output")
STANDARD_OUTPUT_DIR = os.path.join(
    STANDARD_OUTPUT_ROOT, f"{MODEL_BACKBONE}{'_validation' if USE_VALIDATION else ''}")


def compute_standard_output(images_paths, model, dest_folder, filter_class_id=None, image_size=(512, 512), overwrite=False):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for image_path in tqdm(images_paths):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(dest_folder, f"{image_name}.png")

        if not overwrite and os.path.exists(save_path):
            continue

        image = load_image(image_path, image_size=image_size, normalize=True)
        standard_mask = get_prediction(model, image)
        if filter_class_id is not None:
            standard_mask = tf.where(standard_mask == filter_class_id, standard_mask,
                                     0)  # Set to 0 all predictions different from the given class
        tf.keras.utils.save_img(save_path, standard_mask, scale=False)


def create_augmented_copies(image, num_aug, angle_max, shift_max):
    batched_images = tf.tile(tf.expand_dims(image, axis=0), [
                             num_aug, 1, 1, 1])  # Size [num_aug, 512, 512, 3]
    angles = np.random.uniform(-angle_max, angle_max, num_aug)
    shifts = np.random.uniform(-shift_max, shift_max, (num_aug, 2))
    # First sample is not augmented
    angles[0] = 0
    shifts[0] = np.array([0, 0])
    angles = angles.astype("float32")
    shifts = shifts.astype("float32")

    rotated_images = tfa.image.rotate(
        batched_images, angles, interpolation="bilinear")
    translated_images = tfa.image.translate(
        rotated_images, shifts, interpolation="bilinear")

    return translated_images, angles, shifts


def create_augmented_copies_chunked(image, num_aug, angle_max, shift_max, chunk_size=100):
    if (num_aug % chunk_size) != 0:
        raise Exception("Num aug must be a multiple of 50")

    num_chunks = num_aug // chunk_size

    angles = np.random.uniform(-angle_max, angle_max, num_aug)
    shifts = np.random.uniform(-shift_max, shift_max, (num_aug, 2))
    angles[0] = 0
    shifts[0] = np.array([0, 0])
    angles = angles.astype("float32")
    shifts = shifts.astype("float32")

    angles_chunks = np.split(angles, num_chunks)
    shifts_chunks = np.split(shifts, num_chunks)

    augmented_chunks = []

    for i in range(num_chunks):
        images_chunk = tf.tile(tf.expand_dims(
            image, axis=0), [chunk_size, 1, 1, 1])
        rotated_chunk = tfa.image.rotate(
            images_chunk, angles_chunks[i], interpolation="bilinear")
        translated_chunk = tfa.image.translate(
            rotated_chunk, shifts_chunks[i], interpolation="bilinear")
        augmented_chunks.append(translated_chunk.numpy())

    augmented_copies = np.concatenate(augmented_chunks, axis=0)

    return augmented_copies, angles, shifts


def compute_augmented_features(images_paths, model, dest_folder, filter_class_id, mode_slice=True, num_aug=100,
                               angle_max=0.5, shift_max=30, save_output=False, relu_output=False, image_size=(512, 512)):

    for image_path in tqdm(images_paths):
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # Load image
        image = load_image(image_path, image_size=image_size, normalize=True)
        # Create augmented copies
        augmented_copies, angles, shifts = create_augmented_copies(image, num_aug=num_aug, angle_max=angle_max,
                                                                   shift_max=shift_max)

        # Create destination folder
        output_folder = os.path.join(dest_folder, image_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        class_masks = []
        max_masks = []

        predictions = model.predict(augmented_copies, batch_size=BATCH_SIZE)

        # Used to clear memory as it appears that there is a memory leak with something related to model.predict
        _ = gc.collect()

        for i, prediction in enumerate(predictions):

            if mode_slice:
                # Get the slice corresponding to the class id
                class_mask = tf.gather(
                    prediction, filter_class_id, axis=-1)[..., tf.newaxis]

                # Get all the other slices and compute the max pixel-wise
                gather_indexes = np.delete(
                    np.arange(0, tf.shape(prediction)[-1], step=1), filter_class_id)
                max_mask = tf.reduce_max(
                    tf.gather(prediction, gather_indexes, axis=-1), axis=-1)[..., tf.newaxis]

                # ReLU is only needed when working with slices
                if relu_output:
                    class_mask = (tf.nn.relu(class_mask)).numpy()
                    max_mask = (tf.nn.relu(max_mask)).numpy()

                max_masks.append(max_mask)

            else:
                class_mask = create_mask(prediction)
                # Set to 0 all predictions different from the given class
                class_mask = tf.where(
                    class_mask == filter_class_id, class_mask, 0)
                # Necessary for super-resolution operations
                class_mask = tf.cast(class_mask, tf.float32)
                class_mask = class_mask.numpy()

            class_masks.append(class_mask)

            if save_output:
                tf.keras.utils.save_img(
                    f"{output_folder}/{i}_class.png", class_mask, scale=True)
                if mode_slice:
                    tf.keras.utils.save_img(
                        f"{output_folder}/{i}_max.png", max_mask, scale=True)

        file = h5py.File(f"{output_folder}/{image_name}.hdf5", "w")
        file.create_dataset("class_masks", data=class_masks)

        if mode_slice:
            file.create_dataset("max_masks", data=max_masks)

        file.create_dataset("angles", data=angles)
        file.create_dataset("shifts", data=shifts)
        file.attrs["filename"] = image_name
        file.attrs["mode"] = "slice" if mode_slice else "argmax"
        file.attrs["angle_max"] = angle_max
        file.attrs["shift_max"] = shift_max
        file.attrs["backbone"] = MODEL_BACKBONE

        file.close()


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
        alpha=1.).build_model(final_upsample=True)

    # Compute standard (classicl upsample) masks
    print("Computing standard output images...")
    compute_standard_output(images_paths_filtered, model, dest_folder=STANDARD_OUTPUT_DIR,
                            filter_class_id=CLASS_ID, overwrite=False)

    model = DeeplabV3Plus(
        input_shape=(512, 512, 3),
        classes=21,
        OS=16,
        last_activation=None,
        load_weights=True,
        backbone=MODEL_BACKBONE,
        alpha=1.).build_model(final_upsample=False)

    print("Generating augmented copies...")
    compute_augmented_features(images_paths_filtered, model, mode_slice=MODE_SLICE,
                               dest_folder=AUGMENTED_COPIES_OUTPUT_DIR, filter_class_id=CLASS_ID,
                               num_aug=NUM_AUG, angle_max=ANGLE_MAX, shift_max=SHIFT_MAX,
                               save_output=True, relu_output=False, image_size=IMG_SIZE)


if __name__ == '__main__':
    main()
