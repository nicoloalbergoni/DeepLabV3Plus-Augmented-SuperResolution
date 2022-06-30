import os
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from superresolution_scripts.superresolution import Superresolution
from superresolution_scripts.optimizer import Optimizer
from utils import load_image, compute_IoU
from superresolution_scripts.superres_utils import list_precomputed_data_paths, load_SR_data, normalize_coefficients, threshold_image

SEED = 1234

np.random.seed(SEED)
tf.random.set_seed(SEED)

# tf.config.run_functions_eagerly(True)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IMG_SIZE = (512, 512)
FEATURE_SIZE = (128, 128)
NUM_AUG = 100
CLASS_ID = 8
NUM_SAMPLES = 300

MODE = "argmax"
MODEL_BACKBONE = "xception"
USE_VALIDATION = False
SAVE_SLICE_OUTPUT = False

DATA_DIR = os.path.join(os.getcwd(), "data")
PASCAL_ROOT = os.path.join(DATA_DIR, "dataset_root", "VOCdevkit", "VOC2012")
IMGS_PATH = os.path.join(PASCAL_ROOT, "JPEGImages")

SUPERRES_ROOT = os.path.join(DATA_DIR, "superres_root")
AUGMENTED_COPIES_ROOT = os.path.join(SUPERRES_ROOT, "augmented_copies")
PRECOMPUTED_OUTPUT_DIR = os.path.join(
    AUGMENTED_COPIES_ROOT, f"{MODEL_BACKBONE}_{MODE}_{NUM_AUG}{'_validation' if USE_VALIDATION else ''}")
STANDARD_OUTPUT_ROOT = os.path.join(SUPERRES_ROOT, "standard_output")
STANDARD_OUTPUT_DIR = os.path.join(
    STANDARD_OUTPUT_ROOT, f"{MODEL_BACKBONE}{'_validation' if USE_VALIDATION else ''}")
SUPERRES_OUTPUT_DIR = os.path.join(
    SUPERRES_ROOT, f"superres_output{'_validation' if USE_VALIDATION else ''}")

OUTPUT_FOLDER = os.path.join(DATA_DIR, "threshold_test")


def main():
    hyperparamters_default = {
        "lambda_df": 1.0,
        "lambda_tv": 0.876,
        "lambda_L2": 0.0039,
        "lambda_L1": 0.0,
        "num_iter": 300,
        "num_aug": NUM_AUG,
        "num_samples": NUM_SAMPLES,
        "copy_dropout": 0.1,
        "use_BTV": False,
        "optimizer": "adam",
        "learning_rate": 1e-2,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-7,
        "amsgrad": False,
        "initial_accumulator_value": 0.1,
        "nesterov": True,
        "momentum": 0.2,
        "lr_scheduler": True,
        "decay_steps": 90,
        "decay_rate": 0.6,
    }

    wandb_dir = os.path.join(DATA_DIR, "wandb_logs")
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    wandb.init(config=hyperparamters_default, dir=wandb_dir)

    config = wandb.config

    coeff_dict = {
        "lambda_tv": config.lambda_tv,
        "lambda_L2": config.lambda_L2,
        "lambda_L1": config.lambda_L1,
    }

    coeff_dict = normalize_coefficients(coeff_dict)
    print(coeff_dict)

    optimizer_obj = Optimizer(optimizer=config.optimizer, learning_rate=config.learning_rate, epsilon=config.epsilon, beta_1=config.beta_1, beta_2=config.beta_2,
                              amsgrad=config.amsgrad, initial_accumulator_value=config.initial_accumulator_value, momentum=config.momentum, nesterov=config.nesterov,
                              lr_scheduler=config.lr_scheduler, decay_steps=config.decay_steps, decay_rate=config.decay_rate)

    superresolution_obj = Superresolution(lambda_df=config.lambda_df, **coeff_dict, num_iter=config.num_iter,
                                          num_aug=config.num_aug, optimizer=optimizer_obj, use_BTV=config.use_BTV, copy_dropout=config.copy_dropout, feature_size=FEATURE_SIZE)

    path_list = list_precomputed_data_paths(PRECOMPUTED_OUTPUT_DIR, sort=True)
    precomputed_data_paths = path_list if config.num_samples is None else path_list[
        :config.num_samples]

    th_values = [round(v, 2) for v in np.arange(0.1, 0.95, step=0.05)]
    ious_th = np.empty((len(th_values), NUM_SAMPLES))
    standard_ious = []

    for i, filepath in tqdm(enumerate(precomputed_data_paths)):

        try:
            class_masks, _, angles, shifts, filename = load_SR_data(
                filepath, num_aug=NUM_AUG, global_normalize=True)
        except Exception:
            print(f"File: {filepath} is invalid, skipping...")
            continue

        ground_truth_path = os.path.join(
            PASCAL_ROOT, "SegmentationClassAug", f"{filename}.png")
        ground_truth = load_image(ground_truth_path, image_size=IMG_SIZE, normalize=False,
                                  is_png=True, resize_method="nearest")

        standard_mask_path = os.path.join(
            STANDARD_OUTPUT_DIR, f"{filename}.png")
        standard_mask = load_image(standard_mask_path, image_size=IMG_SIZE, normalize=False, is_png=True,
                                   resize_method="nearest")

        standard_ious.append(compute_IoU(
            ground_truth, standard_mask, img_size=IMG_SIZE, class_id=CLASS_ID))

        target_augmented_SR, _ = superresolution_obj.augmented_superresolution(
            class_masks, angles, shifts)

        tf.keras.utils.save_img(
            f"{OUTPUT_FOLDER}/{filename}.png", target_augmented_SR, scale=True)

        for k, value in enumerate(th_values):
            th_mask = threshold_image(
                target_augmented_SR, CLASS_ID, th_factor=value)
            augmented_SR_iou = compute_IoU(
                ground_truth, th_mask, img_size=IMG_SIZE, class_id=CLASS_ID)

            ious_th[k, i] = augmented_SR_iou
            # tf.keras.utils.save_img(
            #     f"{OUTPUT_FOLDER}/{filenames[z]}_th_{value}.png", th_mask, scale=True)

    data_list = []
    for v in range(len(th_values)):

        data_list.append({
            "Th Value": th_values[v],
            "IoU": np.mean(ious_th[v])
        })

    df = pd.DataFrame(data_list)
    print(df)
    print(f"Best record: {df.iloc[df['IoU'].idxmax()]}")
    print(f"Standard IoU: {np.mean(standard_ious)}")
    print("Done")


if __name__ == '__main__':
    main()
