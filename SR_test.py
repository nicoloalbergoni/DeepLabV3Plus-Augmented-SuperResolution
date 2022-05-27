import os
import wandb
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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SEED = 1234

np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (512, 512)
COPIES_SIZE = (64, 64)
NUM_AUG = 100
ANGLE_MAX = 0.3
SHIFT_MAX = 20

DATA_DIR = os.path.join(os.getcwd(), "data")
TEST_ROOT = os.path.join(DATA_DIR, "test_root")
IMAGE_DIR = os.path.join(TEST_ROOT, "normal_images")
SR_OUTPUT_FOLDER = os.path.join(TEST_ROOT, "SR_Output")


def get_images_paths(root_folder, is_png=True):
    paths = []

    for path in os.listdir(root_folder):
        if path.endswith(".png" if is_png else ".jpg"):
            paths.append(os.path.join(root_folder, path))

    return paths


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
        batched_images, angles, interpolation="bilinear", fill_mode="constant")
    translated_images = tfa.image.translate(
        rotated_images, shifts, interpolation="bilinear", fill_mode="constant")

    return translated_images, angles, shifts


def test_aug_params(original_image, superres_obj: Superresolution, dest_folder):

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    angle_max_range = np.arange(0.0, 3.2, step=0.1)
    shift_max_range = np.arange(0, 60, step=5)
    # shift_max_range = np.full((1), 0)
    angle_shift_permutations = [(a, s)
                                for a in angle_max_range for s in shift_max_range]
    data_list = []

    for i, (angle_max, shift_max) in tqdm(enumerate(angle_shift_permutations)):

        # shift_max = 0
        augmented_copies, angles, shifts = generate_augmented_copies(
            original_image, angle_max=angle_max, shift_max=shift_max, num_aug=NUM_AUG)

        target_augmented_SR, loss = superres_obj.augmented_superresolution(
            augmented_copies, angles, shifts)

        target_max_SR, _ = superres_obj.max_superresolution(
            augmented_copies, angles, shifts)
        target_mean_SR, _ = superres_obj.mean_superresolution(
            augmented_copies, angles, shifts)

        # original_image = original_image.numpy()

        tf.keras.utils.save_img(
            f"{dest_folder}/test_image_{i}_SR.png", target_augmented_SR, scale=True)

        # mse = tf.keras.metrics.mean_squared_error(original_image, target_image)
        mse_augmented_SR = mean_squared_error(
            original_image, target_augmented_SR)
        psnr_augmented_SR = peak_signal_noise_ratio(
            original_image, target_augmented_SR, data_range=1)

        mse_max_SR = mean_squared_error(original_image, target_max_SR)
        psnr_max_SR = peak_signal_noise_ratio(
            original_image, target_max_SR, data_range=1)

        mse_mean_SR = mean_squared_error(original_image, target_mean_SR)
        psnr_mean_SR = peak_signal_noise_ratio(
            original_image, target_mean_SR, data_range=1)
        # ssm = structural_similarity(
        #     original_image, target_augmented_SR, multichannel=True)

        metrics = {
            "Angle_max": angle_max,
            "Shift Max": shift_max,
            "PSNR Aug": psnr_augmented_SR,
            "PSNR Max": psnr_max_SR,
            "PSNR Mean": psnr_mean_SR,
            "MSE Aug": mse_augmented_SR,
            "MSE Max": mse_max_SR,
            "MSE Mean": mse_mean_SR,
        }

        data_list.append(metrics)

    return pd.DataFrame(data_list)


def compute_SR(original_image, superres_obj: Superresolution, dest_folder, angle_max=0.3, shift_max=30, save_copies=False):

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    augmented_copies, angles, shifts = generate_augmented_copies(
        original_image, angle_max=angle_max, shift_max=shift_max, num_aug=NUM_AUG)

    target_augmented_SR, _ = superres_obj.augmented_superresolution(
        augmented_copies, angles, shifts)

    target_augmented_SR = tf.clip_by_value(
        target_augmented_SR, 0.0, 255.0).numpy()

    if save_copies:
        augmented_copies = augmented_copies.numpy()
        copies_folder = os.path.join(dest_folder, "augmented_copies")
        os.makedirs(copies_folder, exist_ok=True)

        for i, copy in enumerate(augmented_copies):
            tf.keras.utils.save_img(
                f"{copies_folder}/{i}.png", copy, scale=True)

    mse_augmented_SR = mean_squared_error(
        original_image, target_augmented_SR)
    psnr_augmented_SR = peak_signal_noise_ratio(
        original_image, target_augmented_SR, data_range=255)

    return target_augmented_SR, psnr_augmented_SR, mse_augmented_SR


def main():

    hyperparameters_default = {
        "lambda_df": 1.0,
        "lambda_tv": 0.9,
        "lambda_L2": 0.05,
        "lambda_L1": 0.03,
        "num_iter": 300,
        "num_aug": NUM_AUG,
        "use_BTV": True,
        "copy_dropout": 0.0,
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "lr_scheduler": True,
        "decay_rate": 0.5,
        "decay_steps": 50,
        "angle_max": 0.4,
        "shift_max": 0
    }

    wandb_dir = os.path.join(DATA_DIR, "wandb_logs")
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)

    wandb.init(config=hyperparameters_default, dir=wandb_dir)

    config = wandb.config

    coeff_dict = {
        "lambda_tv": config.lambda_tv,
        "lambda_L2": config.lambda_L2,
        "lambda_L1": config.lambda_L1,
    }

    coeff_dict = normalize_coefficients(coeff_dict)

    optimizer_obj = Optimizer(optimizer=config.optimizer, learning_rate=config.learning_rate,
                              lr_scheduler=config.lr_scheduler, decay_steps=config.decay_steps, decay_rate=config.decay_rate)

    superresolution_obj = Superresolution(lambda_df=config.lambda_df, **coeff_dict, num_iter=config.num_iter,
                                          num_aug=config.num_aug, optimizer=optimizer_obj, use_BTV=config.use_BTV, copy_dropout=config.copy_dropout)

    images_paths = get_images_paths(IMAGE_DIR, is_png=True)
    psnr_aug_list = []
    mse_aug_list = []

    for image_path in tqdm(images_paths):
        original_image = load_image(image_path, normalize=False, is_png=True)
        original_image = original_image.numpy()

        # df = test_aug_params(original_image, superresolution_obj, SR_OUTPUT_FOLDER)
        # print(df)
        # print(df.loc[df[['PSNR Aug', "PSNR Max", "PSNR Mean"]].idxmax()])

        target_augmented_SR, psnr_augmented_SR, mse_augmented_SR = compute_SR(
            original_image, superresolution_obj, SR_OUTPUT_FOLDER, angle_max=config.angle_max, shift_max=config.shift_max, save_copies=False)

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        tf.keras.utils.save_img(
            f"{SR_OUTPUT_FOLDER}/{image_name}_SR.png", target_augmented_SR, scale=False)

        psnr_aug_list.append(psnr_augmented_SR)
        mse_aug_list.append(mse_augmented_SR)

    avg_psnr_aug = np.mean(psnr_aug_list)
    avg_mse_aug = np.mean(mse_aug_list)

    wandb.log({"Avg_PSNR_Aug": avg_psnr_aug, "Avg_MSE_Aug": avg_mse_aug})

    print(f"Avg. PSNR Aug: {avg_psnr_aug}, Avg. MSE Aug: {avg_mse_aug}")


if __name__ == '__main__':
    main()
