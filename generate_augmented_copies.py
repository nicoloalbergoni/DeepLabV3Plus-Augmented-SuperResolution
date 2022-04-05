import os
import h5py
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from model import DeeplabV3Plus
import tensorflow_addons as tfa
from utils import load_image, get_prediction, create_mask
from superresolution_scripts.superres_utils import get_img_paths, load_images

SEED = 1234

np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (512, 512)
BATCH_SIZE = 8
NUM_AUG = 100
CLASS_ID = 8
NUM_SAMPLES = None
MODE = "slice"
USE_VALIDATION = False

DATA_DIR = os.path.join(os.getcwd(), "data")
PASCAL_ROOT = os.path.join(DATA_DIR, "dataset_root", "VOCdevkit", "VOC2012")
IMGS_PATH = os.path.join(PASCAL_ROOT, "JPEGImages")

SUPERRES_ROOT = os.path.join(DATA_DIR, "superres_root")
PRECOMPUTED_OUTPUT_DIR = os.path.join(SUPERRES_ROOT, f"precomputed_features{'_validation' if USE_VALIDATION else ''}")
STANDARD_OUTPUT_DIR = os.path.join(SUPERRES_ROOT, f"standard_output{'_validation' if USE_VALIDATION else ''}")


def compute_standard_output(image_dict, model, dest_folder, filter_class_id=None):
    standard_masks = {}
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for key in tqdm(image_dict):
        standard_mask = get_prediction(model, image_dict[key])
        if filter_class_id is not None:
            standard_mask = tf.where(standard_mask == filter_class_id, standard_mask,
                                     0)  # Set to 0 all predictions different from the given class
        tf.keras.utils.save_img(f"{dest_folder}/{key}.png", standard_mask, scale=False)
        standard_masks[key] = standard_mask

    return standard_masks


def create_augmented_copies(image, num_aug, angle_max, shift_max):
    batched_images = tf.tile(tf.expand_dims(image, axis=0), [num_aug, 1, 1, 1])  # Size [num_aug, 512, 512, 3]
    angles = np.random.uniform(-angle_max, angle_max, num_aug)
    shifts = np.random.uniform(-shift_max, shift_max, (num_aug, 2))
    # First sample is not augmented
    angles[0] = 0
    shifts[0] = np.array([0, 0])
    angles = angles.astype("float32")
    shifts = shifts.astype("float32")

    rotated_images = tfa.image.rotate(batched_images, angles, interpolation="bilinear")
    translated_images = tfa.image.translate(rotated_images, shifts, interpolation="bilinear")

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
        images_chunk = tf.tile(tf.expand_dims(image, axis=0), [chunk_size, 1, 1, 1])
        rotated_chunk = tfa.image.rotate(images_chunk, angles_chunks[i], interpolation="bilinear")
        translated_chunk = tfa.image.translate(rotated_chunk, shifts_chunks[i], interpolation="bilinear")
        augmented_chunks.append(translated_chunk.numpy())

    augmented_copies = np.concatenate(augmented_chunks, axis=0)

    return augmented_copies, angles, shifts


def compute_augmented_features(image_filenames, model, dest_folder, filter_class_id, mode="slice", num_aug=100,
                               angle_max=0.5, shift_max=30, save_output=False, relu_output=False):
    augmented_features = {}

    for filename in tqdm(image_filenames):

        # Load image
        image_path = os.path.join(IMGS_PATH, f"{filename}.jpg")
        image = load_image(image_path, image_size=IMG_SIZE, normalize=True)

        # Create augmented copies
        augmented_copies, angles, shifts = create_augmented_copies(image, num_aug=num_aug, angle_max=angle_max,
                                                                   shift_max=shift_max)

        # Create destination folder
        output_folder = os.path.join(dest_folder, filename)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        class_masks = []
        max_masks = []

        predictions = model.predict(augmented_copies, batch_size=BATCH_SIZE)

        for i, prediction in enumerate(predictions):

            if mode == "slice":
                class_slice = prediction[:, :, filter_class_id]
                class_mask = class_slice[..., np.newaxis]

                no_class_prediction = np.delete(prediction, filter_class_id, axis=-1)
                max_mask = no_class_prediction.max(axis=-1)
                max_mask = max_mask[..., np.newaxis]

                # ReLU is only needed when working with slices
                if relu_output:
                    class_mask = (tf.nn.relu(class_mask)).numpy()
                    max_mask = (tf.nn.relu(max_mask)).numpy()

                max_masks.append(max_mask)

            elif mode == "argmax":
                class_mask = create_mask(prediction)
                # Set to 0 all predictions different from the given class
                class_mask = tf.where(class_mask == filter_class_id, class_mask, 0)
                class_mask = tf.cast(class_mask, tf.float32)  # Necessary for super-resolution operations
                class_mask = class_mask.numpy()

            class_masks.append(class_mask)

            if save_output:
                tf.keras.utils.save_img(f"{output_folder}/{i}_class.png", class_mask, scale=True)
                if mode == "slice":
                    tf.keras.utils.save_img(f"{output_folder}/{i}_max.png", max_mask, scale=True)

        file = h5py.File(f"{output_folder}/{filename}.hdf5", "w")
        file.create_dataset("class_masks", data=class_masks)

        if mode == "slice":
            file.create_dataset("max_masks", data=max_masks)

        file.create_dataset("angles", data=angles)
        file.create_dataset("shifts", data=shifts)
        file.attrs["filename"] = filename
        file.attrs["mode"] = mode

        file.close()

        augmented_features[filename] = {"class": class_masks, "max": max_masks}

    return augmented_features


def main():
    image_list_path = os.path.join(DATA_DIR, "augmented_file_lists",
                                   f"{'valaug' if USE_VALIDATION else 'trainaug'}.txt")
    image_paths = get_img_paths(image_list_path, IMGS_PATH, is_png=False, sort=True)
    images_dict = load_images(image_paths, num_images=NUM_SAMPLES, filter_class_id=CLASS_ID, image_size=IMG_SIZE)

    print(f"Valid images: {len(images_dict)} (Initial:  {len(image_paths)})")

    model_no_upsample = DeeplabV3Plus(
        input_shape=(512, 512, 3),
        classes=21,
        OS=16,
        last_activation=None,
        load_weights=True,
        backbone="mobilenet",
        alpha=1.).build_model(final_upsample=False)

    model_standard = DeeplabV3Plus(
        input_shape=(512, 512, 3),
        classes=21,
        OS=16,
        last_activation=None,
        load_weights=True,
        backbone="mobilenet",
        alpha=1.).build_model(final_upsample=True)

    # Compute standard (classicl upsample) masks
    print("Computing standard output images...")
    compute_standard_output(images_dict, model_standard, dest_folder=STANDARD_OUTPUT_DIR,
                            filter_class_id=CLASS_ID)

    angle_max = 0.5  # in radians
    shift_max = 30

    print("Generating augmented copies...")
    compute_augmented_features(images_dict, model_no_upsample, mode=MODE,
                               dest_folder=PRECOMPUTED_OUTPUT_DIR, filter_class_id=CLASS_ID,
                               num_aug=NUM_AUG, angle_max=angle_max, shift_max=shift_max,
                               save_output=False, relu_output=False)


if __name__ == '__main__':
    main()
