import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import tensorflow_addons as tfa
from utils import load_image, plot_images
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio
from superresolution_scripts.superresolution import Superresolution
from superresolution_scripts.optimizer import Optimizer
from superresolution_scripts.superres_utils import min_max_normalization, normalize_coefficients

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
SR_OUTPUT_FOLDER = os.path.join(IMAGE_FOLDER, "SR_Output")


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


def test_aug_params(original_image, superres_obj: Superresolution, optimizer_obj: Optimizer, dest_folder):

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    angle_max_range = np.arange(0.0, 2.1, step=0.1)
    # shift_max_range = np.arange(0, 60, step=10)
    shift_max_range = np.full((1), 0)
    angle_shift_permutations = [(a, s)
                                for a in angle_max_range for s in shift_max_range]
    data_list = []

    for i, (angle_max, shift_max) in tqdm(enumerate(angle_shift_permutations)):
        augmented_copies, angles, shifts = generate_augmented_copies(
            original_image, angle_max=angle_max, shift_max=0, num_aug=NUM_AUG)

        target_image, loss = superres_obj.augmented_superresolution(optimizer_obj,
                                                                    augmented_copies, angles, shifts)

        # original_image = original_image.numpy()

        tf.keras.utils.save_img(
            f"{dest_folder}/test_image_{i}_SR.png", target_image, scale=True)

        # mse = tf.keras.metrics.mean_squared_error(original_image, target_image)
        mse = mean_squared_error(original_image, target_image)
        psnr = peak_signal_noise_ratio(
            original_image, target_image, data_range=1)
        ssm = structural_similarity(
            original_image, target_image, multichannel=True)

        data_list.append(
            {"Angle_max": angle_max, "Shift Max": shift_max, "MSE": mse, "PSNR": psnr, "SSM": ssm, "Loss": loss.numpy()})

    return pd.DataFrame(data_list)


def main():

    coeff_dict = {
        "lambda_tv": 9.0,
        "lambda_L2": 0.01,
        "lambda_L1": 2,
    }

    coeff_dict = normalize_coefficients(coeff_dict)

    superres_params = {
        "lambda_df": 1.0,
        "num_iter": 300,
        "num_aug": NUM_AUG,
        "use_BTV": False,
        "copy_dropout": 0.0
    }

    optimizer_config = {
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "epsilon": 1e-7,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "amsgrad": True,
        "initial_accumulator_value": 0.1,
        "momentum": 0.2,
        "nesterov": True,
        "lr_scheduler": True,
        "decay_rate": 0.5,
        "decay_steps": 50
    }

    optimizer = Optimizer(**optimizer_config)

    superres_obj = Superresolution(**superres_params, **coeff_dict)
    image_path = os.path.join(IMAGE_FOLDER, "test_image.png")
    original_image = load_image(image_path, normalize=True, is_png=True)
    original_image = original_image.numpy()

    df = test_aug_params(original_image, superres_obj,
                         optimizer, SR_OUTPUT_FOLDER)
    print(df)
    print(df.loc[df['PSNR'].idxmax()])


if __name__ == '__main__':
    main()
