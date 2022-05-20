import os
import h5py
import wandb
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from superresolution_scripts.superresolution import Superresolution
from superresolution_scripts.optimizer import Optimizer
from utils import load_image
from superresolution_scripts.superres_utils import compare_results, \
    list_precomputed_data_paths, load_SR_data, compute_augmented_SR, single_class_IOU, normalize_coefficients

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SEED = 1234

np.random.seed(SEED)
tf.random.set_seed(SEED)

# tf.config.run_functions_eagerly(True)

IMG_SIZE = (512, 512)
NUM_AUG = 100
CLASS_ID = 8
NUM_SAMPLES = 100
MODE_SLICE = True
USE_VALIDATION = False

DATA_DIR = os.path.join(os.getcwd(), "data")
PASCAL_ROOT = os.path.join(DATA_DIR, "dataset_root", "VOCdevkit", "VOC2012")
IMGS_PATH = os.path.join(PASCAL_ROOT, "JPEGImages")

SUPERRES_ROOT = os.path.join(DATA_DIR, "superres_root")
PRECOMPUTED_OUTPUT_DIR = os.path.join(
    SUPERRES_ROOT, f"precomputed_features_{'slice' if MODE_SLICE else 'argmax'}{'_validation' if USE_VALIDATION else ''}")
STANDARD_OUTPUT_DIR = os.path.join(
    SUPERRES_ROOT, f"standard_output{'_validation' if USE_VALIDATION else ''}")
SUPERRES_OUTPUT_DIR = os.path.join(
    SUPERRES_ROOT, f"superres_output{'_validation' if USE_VALIDATION else ''}")


def main():
    hyperparamters_default = {
        "lambda_df": 0.46,
        "lambda_tv": 4.75,
        "lambda_L2": 0.11,
        "lambda_L1": 0.0,
        "num_iter": 450,
        "num_aug": NUM_AUG,
        "num_samples": NUM_SAMPLES,
        "use_BTV": True,
        "copy_dropout": 0.5,
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-7,
        "amsgrad": False,
        "initial_accumulator_value": 0.1,
        "momentum": 0.6,
        "nesterov": False,
        "lr_scheduler": False,
        "decay_steps": 50,
        "decay_rate": 0.4,
    }

    wandb_dir = os.path.join(DATA_DIR, "wandb_logs")
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)

    wandb.init(config=hyperparamters_default, dir=wandb_dir)

    config = wandb.config

    coeff_dict = {
        "lambda_tv": config.lambda_tv,
        "lambda_L2": config.lambda_L2,
        "lambda_L1": config.lambda_L1,
    }

    coeff_dict = normalize_coefficients(coeff_dict)

    optimizer_obj = Optimizer(optimizer=config.optimizer, learning_rate=config.learning_rate, epsilon=config.epsilon, beta_1=config.beta_1, beta_2=config.beta_2,
                              amsgrad=config.amsgrad, initial_accumulator_value=config.initial_accumulator_value, momentum=config.momentum, nesterov=config.nesterov,
                              lr_scheduler=config.lr_scheduler, decay_steps=config.decay_steps, decay_rate=config.decay_rate)

    superresolution_obj = Superresolution(lambda_df=config.lambda_df, **coeff_dict, num_iter=config.num_iter,
                                          num_aug=config.num_aug, use_BTV=config.use_BTV, copy_dropout=config.copy_dropout)

    path_list = list_precomputed_data_paths(PRECOMPUTED_OUTPUT_DIR, sort=True)
    precomputed_data_paths = path_list if config.num_samples is None else path_list[
        :config.num_samples]

    standard_IOUs = []
    superres_IOUs = []

    for filepath in tqdm(precomputed_data_paths):

        try:
            class_masks, max_masks, angles, shifts, filename = load_SR_data(
                filepath, num_aug=NUM_AUG, mode_slice=MODE_SLICE, global_normalize=True)
        except Exception:
            print(f"File: {filepath} is invalid, skipping...")
            continue

        final_image = compute_augmented_SR(superresolution_obj, optimizer_obj, class_masks, angles, shifts, filename, max_masks=max_masks,
                                           save_output=False, class_id=CLASS_ID, dest_folder=SUPERRES_OUTPUT_DIR)

        true_mask_path = os.path.join(
            PASCAL_ROOT, "SegmentationClassAug", f"{filename}.png")
        true_mask = load_image(true_mask_path, image_size=IMG_SIZE, normalize=False,
                               is_png=True, resize_method="nearest")

        standard_mask_path = os.path.join(
            STANDARD_OUTPUT_DIR, f"{filename}.png")
        standard_mask = load_image(standard_mask_path, image_size=IMG_SIZE, normalize=False, is_png=True,
                                   resize_method="nearest")

        standard_IOU, superres_IOU = compare_results(true_mask,
                                                     standard_mask, final_image, img_size=IMG_SIZE, class_id=CLASS_ID)

        standard_IOUs.append(standard_IOU)
        superres_IOUs.append(superres_IOU)
    print(
        f"Standard mean IOU: {np.mean(standard_IOUs)},  Superres mean IOU: {np.mean(superres_IOUs)}")

    wandb.log({"mean_superres_iou": np.mean(superres_IOUs),
               "mean_standard_iou": np.mean(standard_IOUs)})


if __name__ == '__main__':
    main()
