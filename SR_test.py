import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from utils import load_image, plot_images
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio
from superresolution_scripts.superresolution import Superresolution
from superresolution_scripts.superres_utils import min_max_normalization, normalize_coefficients

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SEED = 1234

np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (512, 512)
COPIES_SIZE = (64, 64)
NUM_AUG = 100
ANGLE_MAX = 0.5
SHIFT_MAX = 30

DATA_DIR = os.path.join(os.getcwd(), "data")
IMAGE_FOLDER = os.path.join(DATA_DIR, "test_images")


def generate_augmented_copies(original_image, angle_max, shift_max, num_aug=100, copies_size=(64, 64)):
    downscaled_image = tf.image.resize(
        original_image, size=copies_size)[tf.newaxis, :]
    batched_images = tf.tile(downscaled_image, [num_aug, 1, 1, 1])
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


def solve_superresolution(superresolution_obj, augmented_copies, angles, shifts):

    global_min = tf.reduce_min(augmented_copies)
    global_max = tf.reduce_max(augmented_copies)

    # augmented_copies = tf.map_fn(fn=lambda image: min_max_normalization(image.numpy(
    # ), new_min=0.0, new_max=1.0, global_min=global_min, global_max=global_max), elems=augmented_copies)

    target_image, loss = superresolution_obj.compute_output(
        augmented_copies, angles, shifts)

    return target_image, loss


def main():

    coeff_dict = {
        "lambda_tv": 0.5,
        "lambda_L2": 0.5,
        "lambda_L1": 0.05,
    }

    normalize_coefficients(coeff_dict)

    superres_params = {
        "lambda_df": 1.0,
        **coeff_dict,
        "num_iter": 100,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "num_aug": NUM_AUG,
        "copy_dropout": 0.0,
        "use_BTV": False
    }

    # superres_params["lambda_tv"], superres_params["lambda_L2"], superres_params["lambda_L1"] = normalize_coefficients(
    #     [0.6, 0.55, 0.085])

    optimizer_config = {
        "lr_scheduler": True,
        "momentum": 0.2,
        "nesterov": True,
        "decay_rate": 0.5,
        "decay_steps": 50,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-7,
        "amsgrad": True,
        "initial_accumulator_value": 0.1,
    }

    superres_obj = Superresolution(
        **superres_params, optimizer_params=optimizer_config)
    image_path = os.path.join(IMAGE_FOLDER, "test_image.png")
    original_image = load_image(image_path, normalize=True, is_png=True)
    augmented_copies, angles, shifts = generate_augmented_copies(
        original_image, angle_max=ANGLE_MAX, shift_max=SHIFT_MAX, num_aug=NUM_AUG)

    target_image, loss = solve_superresolution(
        superres_obj, augmented_copies, angles, shifts)

    target_image = target_image[0].numpy()
    original_image = original_image.numpy()

    tf.keras.utils.save_img(
        f"{IMAGE_FOLDER}/test_image_SR.png", target_image, scale=True)

    # mse = tf.keras.metrics.mean_squared_error(original_image, target_image)
    mse = mean_squared_error(original_image, target_image)
    psnr = peak_signal_noise_ratio(original_image, target_image, data_range=1)
    ssm = structural_similarity(
        original_image, target_image, multichannel=True)

    print(f"MSE: {mse}, PSNR: {psnr}, SSM: {ssm}")
    plot_images([original_image, target_image], rows=1, columns=2)


if __name__ == '__main__':
    main()
