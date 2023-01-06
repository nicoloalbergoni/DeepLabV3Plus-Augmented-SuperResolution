import os
import gc
import h5py
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_addons as tfa
from utils import load_image, create_mask
from superresolution_scripts.superres_utils import min_max_normalization


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


def compute_augmented_features(images_paths, model, dest_folder, filter_class_id, mode="slice", num_aug=100,
                               angle_max=0.5, shift_max=30, save_output=False, image_size=(512, 512), batch_size=16):

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

        predictions = model.predict(augmented_copies, batch_size=batch_size)
        # Used to clear memory as it appears that there is a memory leak with something related to model.predict
        _ = gc.collect()

        for i, prediction in enumerate(predictions):

            if mode == "slice_max":
                # Get the slice corresponding to the class id
                class_mask = tf.gather(
                    prediction, filter_class_id, axis=-1)[..., tf.newaxis]

                # Get all the other slices and compute the max pixel-wise
                gather_indexes = np.delete(
                    np.arange(0, tf.shape(prediction)[-1], step=1), filter_class_id)
                max_mask = tf.reduce_max(
                    tf.gather(prediction, gather_indexes, axis=-1), axis=-1)[..., tf.newaxis]

                max_masks.append(max_mask)

            elif mode == "slice":
                # Get the slice corresponding to the class id
                class_mask = tf.gather(
                    prediction, filter_class_id, axis=-1)[..., tf.newaxis]

                global_max = tf.reduce_max(prediction)
                global_min = tf.reduce_min(prediction)

                class_mask = min_max_normalization(class_mask.numpy(), new_min=0.0, new_max=1.0, global_min=global_min,
                                                   global_max=global_max)

            else:
                class_mask = create_mask(prediction)
                # Set to 0 all predictions different from the given class
                class_mask = tf.where(
                    class_mask == filter_class_id, class_mask, 0)
                # Necessary for super-resolution operations
                class_mask = tf.cast(class_mask, tf.float32)
                class_mask = class_mask.numpy()

            class_masks.append(class_mask)

            if save_output and (i % 10) == 0:
                tf.keras.utils.save_img(
                    f"{output_folder}/{i}_class.png", class_mask, scale=True)
                if mode == "slice_max":
                    tf.keras.utils.save_img(
                        f"{output_folder}/{i}_max.png", max_mask, scale=True)

        file = h5py.File(f"{output_folder}/{image_name}.hdf5", "w")
        file.create_dataset("class_masks", data=class_masks)

        if mode == "slice_max":
            file.create_dataset("max_masks", data=max_masks)

        file.create_dataset("angles", data=angles)
        file.create_dataset("shifts", data=shifts)
        file.attrs["filename"] = image_name
        file.attrs["mode"] = mode
        file.attrs["angle_max"] = angle_max
        file.attrs["shift_max"] = shift_max

        file.close()
