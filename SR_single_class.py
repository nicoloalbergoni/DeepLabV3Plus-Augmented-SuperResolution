"""
Computes the Augmented Super-Resolution procedure for the PASCAL VOC dataset.
Works on precomputed augmented copies generated with the generate_augmented_copies.py script.
"""
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from utils import load_image, compute_IoU
from superresolution_scripts.optimizer import Optimizer
from superresolution_scripts.superresolution import Superresolution
from superresolution_scripts.superres_utils import list_precomputed_data_paths, load_SR_data, compute_SR

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

SEED = 1234

np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (512, 512)
FEATURE_SIZE = (128, 128)
NUM_AUG = 10
CLASS_ID = 8
NUM_SAMPLES = 500
MODE = "argmax"
MODEL_BACKBONE = "xception"
USE_VALIDATION = True
SAVE_SLICE_OUTPUT = False
SAVE_FINAL_SR_OUTPUT = False
TH_FACTOR = 0.65

DATA_DIR = os.path.join(os.getcwd(), "data")
PASCAL_ROOT = os.path.join(DATA_DIR, "dataset_root", "VOCdevkit", "VOC2012")
IMGS_PATH = os.path.join(PASCAL_ROOT, "JPEGImages")

SUPERRES_ROOT = os.path.join(DATA_DIR, "superres_root")
AUGMENTED_COPIES_ROOT = os.path.join(SUPERRES_ROOT, "augmented_copies")
PRECOMPUTED_OUTPUT_DIR = os.path.join(
    AUGMENTED_COPIES_ROOT, f"{MODEL_BACKBONE}_{MODE}_{CLASS_ID}_{NUM_AUG}{'_validation' if USE_VALIDATION else ''}")
STANDARD_OUTPUT_ROOT = os.path.join(SUPERRES_ROOT, "standard_output")
STANDARD_OUTPUT_DIR = os.path.join(
    STANDARD_OUTPUT_ROOT, f"{MODEL_BACKBONE}_{CLASS_ID}{'_validation' if USE_VALIDATION else ''}")
SUPERRES_OUTPUT_DIR = os.path.join(
    SUPERRES_ROOT, f"superres_output{'_validation' if USE_VALIDATION else ''}")


def main():
    hyperparameters = {
        "lambda_df": 1,
        "lambda_tv": 0.3,
        "lambda_L2": 0.7,
        "lambda_L1": 0.0,
        "num_iter": 300,
        "num_aug": NUM_AUG,
        "num_samples": NUM_SAMPLES,
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "amsgrad": True,
        "lr_scheduler": True,
        "decay_steps": 60,
        "decay_rate": 0.3,
    }

    optimizer_obj = Optimizer(optimizer=hyperparameters["optimizer"], learning_rate=hyperparameters["learning_rate"], amsgrad=hyperparameters["amsgrad"],
                              lr_scheduler=hyperparameters["lr_scheduler"], decay_steps=hyperparameters["decay_steps"], decay_rate=hyperparameters["decay_rate"])

    superresolution_obj = Superresolution(lambda_df=hyperparameters["lambda_df"], lambda_tv=hyperparameters["lambda_tv"], lambda_L2=hyperparameters["lambda_L2"], lambda_L1=hyperparameters["lambda_L1"],
                                          num_iter=hyperparameters["num_iter"], num_aug=hyperparameters["num_aug"], optimizer=optimizer_obj, feature_size=FEATURE_SIZE)

    path_list = list_precomputed_data_paths(PRECOMPUTED_OUTPUT_DIR, sort=True)
    precomputed_data_paths = path_list if hyperparameters["num_samples"] is None else path_list[
        :hyperparameters["num_samples"]]

    standard_ious_single = []
    standard_ious_multiple = []
    aug_SR_ious_single = []
    aug_SR_ious_multiple = []
    max_SR_ious = []
    mean_SR_ious = []

    for filepath in tqdm(precomputed_data_paths):

        try:
            class_masks, max_masks, angles, shifts, filename = load_SR_data(
                filepath, num_aug=NUM_AUG, global_normalize=True)
        except Exception:
            print(f"File: {filepath} is invalid, skipping...")
            continue

        true_mask_path = os.path.join(
            PASCAL_ROOT, "SegmentationClassAug", f"{filename}.png")
        true_mask = load_image(true_mask_path, image_size=IMG_SIZE, normalize=False,
                               is_png=True, resize_method="nearest")

        standard_mask_path = os.path.join(
            STANDARD_OUTPUT_DIR, f"{filename}.png")
        standard_mask = load_image(standard_mask_path, image_size=IMG_SIZE, normalize=False, is_png=True,
                                   resize_method="nearest")

        target_augmented_SR = compute_SR(superresolution_obj, class_masks, angles, shifts, filename, max_masks=max_masks, SR_type="aug", save_final_output=SAVE_FINAL_SR_OUTPUT,
                                         save_intermediate_output=SAVE_SLICE_OUTPUT, class_id=CLASS_ID, dest_folder=SUPERRES_OUTPUT_DIR, th_factor=TH_FACTOR)
        target_max_SR = compute_SR(superresolution_obj, class_masks, angles, shifts, filename, max_masks=max_masks, SR_type="max", save_final_output=SAVE_FINAL_SR_OUTPUT,
                                   save_intermediate_output=SAVE_SLICE_OUTPUT, class_id=CLASS_ID, dest_folder=SUPERRES_OUTPUT_DIR, th_factor=TH_FACTOR)
        target_mean_SR = compute_SR(superresolution_obj, class_masks, angles, shifts, filename, max_masks=max_masks, SR_type="mean", save_final_output=SAVE_FINAL_SR_OUTPUT,
                                    save_intermediate_output=SAVE_SLICE_OUTPUT, class_id=CLASS_ID, dest_folder=SUPERRES_OUTPUT_DIR, th_factor=TH_FACTOR)

        standard_iou_single = compute_IoU(
            true_mask, standard_mask, img_size=IMG_SIZE, class_id=CLASS_ID)
        standard_iou_multiple = compute_IoU(
            true_mask, standard_mask, img_size=IMG_SIZE, class_id=CLASS_ID, include_bg=True)
        aug_SR_iou_single = compute_IoU(
            true_mask, target_augmented_SR, img_size=IMG_SIZE, class_id=CLASS_ID)
        aug_SR_iou_multiple = compute_IoU(
            true_mask, target_augmented_SR, img_size=IMG_SIZE, class_id=CLASS_ID, include_bg=True)
        max_SR_iou = compute_IoU(
            true_mask, target_max_SR, img_size=IMG_SIZE, class_id=CLASS_ID)
        mean_SR_iou = compute_IoU(
            true_mask, target_mean_SR, img_size=IMG_SIZE, class_id=CLASS_ID)

        standard_ious_single.append(standard_iou_single)
        standard_ious_multiple.append(standard_iou_multiple)
        aug_SR_ious_single.append(aug_SR_iou_single)
        aug_SR_ious_multiple.append(aug_SR_iou_multiple)
        max_SR_ious.append(max_SR_iou)
        mean_SR_ious.append(mean_SR_iou)

    avg_standard_iou_single = np.mean(standard_ious_single)
    avg_standard_iou_multiple = np.mean(standard_ious_multiple)
    avg_aug_SR_iou_single = np.mean(aug_SR_ious_single)
    avg_aug_SR_iou_multiple = np.mean(aug_SR_ious_multiple)
    avg_max_SR_iou = np.mean(max_SR_ious)
    avg_mean_SR_iou = np.mean(mean_SR_ious)

    print(
        f"Avg. Standard IoUs (No bg): {avg_standard_iou_single},  Avg. Augmented SR IoUs (No bg): {avg_aug_SR_iou_single}")
    print(
        f"Avg. Standard IoUs (with bg): {avg_standard_iou_multiple},  Avg. Augmented SR IoUs (with bg): {avg_aug_SR_iou_multiple}")
    print(
        f"Avg. Max SR IoUs: {avg_max_SR_iou}, Avg. Mean SR IoUs: {avg_mean_SR_iou}")


if __name__ == '__main__':
    main()
