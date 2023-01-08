import os
import numpy as np
import tensorflow as tf
from model import DeeplabV3Plus
from utils import load_image, compute_IoU, plot_prediction
from superresolution_scripts.optimizer import Optimizer
from superresolution_scripts.superresolution import Superresolution
from superresolution_scripts.augmentation_utils import compute_augmented_feature_maps
from superresolution_scripts.superres_utils import compute_SR

# Env variables for tensorflow
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Set Seed
SEED = 1234
np.random.seed(SEED)
tf.random.set_seed(SEED)

# General Parameters
IMG_SIZE = (512, 512)
FEATURE_SIZE = (128, 128)
BATCH_SIZE = 16
CLASS_ID = 8
MODE = "argmax"
SAVE_IMAGES = False
MODEL_BACKBONE = "xception"

# Augmentation Parameters
NUM_AUG = 100
ANGLE_MAX = 0.15
SHIFT_MAX = 80

# Optimizer Parameters
OPTIMIZER = "adam"
LEARNING_RATE = 1e-3
AMSGRAD = True
LR_SCHEDULER = True
DECAY_STEPS = 60
DECAY_RATE = 0.3

# Super-Resolution Parameters
LAMBDA_DF = 1.0
LAMBDA_TV = 0.3
LAMBDA_L2 = 0.7
LAMBDA_L1 = 0.0
NUM_ITER = 300
TH_FACTOR = 0.2

# Paths
TEST_IMAGES_DIR = os.path.join(os.getcwd(), "test_images")
IMG_PATH = os.path.join(TEST_IMAGES_DIR, "test_cat.jpg")
GT_PATH = os.path.join(TEST_IMAGES_DIR, "test_cat_gt.png")
SR_OUTPUT_DIR = os.path.join(TEST_IMAGES_DIR, "SR_output")


def main():
    model = DeeplabV3Plus(
        input_shape=(512, 512, 3),
        classes=21,
        OS=16,
        last_activation=None,
        load_weights=True,
        backbone=MODEL_BACKBONE,
    ).build_model(final_upsample=False)

    optimizer_obj = Optimizer(optimizer=OPTIMIZER, learning_rate=LEARNING_RATE, amsgrad=AMSGRAD,
                              lr_scheduler=LR_SCHEDULER, decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE)

    superresolution_obj = Superresolution(lambda_df=LAMBDA_DF, lambda_tv=LAMBDA_TV, lambda_L2=LAMBDA_L2, lambda_L1=LAMBDA_L1,
                                          num_iter=NUM_ITER, num_aug=NUM_AUG, optimizer=optimizer_obj, feature_size=FEATURE_SIZE)

    class_masks, max_masks, angles, shifts, filename = compute_augmented_feature_maps(
        IMG_PATH, model, filter_class_id=CLASS_ID, mode=MODE, num_aug=NUM_AUG, angle_max=ANGLE_MAX,
        shift_max=SHIFT_MAX, image_size=IMG_SIZE, batch_size=BATCH_SIZE)

    target_augmented_SR = compute_SR(superresolution_obj, class_masks, angles, shifts, filename, max_masks=max_masks, SR_type="aug", save_final_output=True,
                                     class_id=CLASS_ID, dest_folder=SR_OUTPUT_DIR, th_factor=TH_FACTOR)
    target_max_SR = compute_SR(superresolution_obj, class_masks, angles, shifts, filename, max_masks=max_masks, SR_type="max", save_final_output=True,
                               class_id=CLASS_ID, dest_folder=SR_OUTPUT_DIR, th_factor=TH_FACTOR)
    target_mean_SR = compute_SR(superresolution_obj, class_masks, angles, shifts, filename, max_masks=max_masks, SR_type="mean", save_final_output=True,
                                class_id=CLASS_ID, dest_folder=SR_OUTPUT_DIR, th_factor=TH_FACTOR)

    input_image = load_image(IMG_PATH, image_size=IMG_SIZE,
                             normalize=False, is_png=False, resize_method="nearest")
    gt_mask = load_image(GT_PATH, image_size=IMG_SIZE,
                         normalize=False, is_png=True, resize_method="nearest")

    aug_SR_iou_single = compute_IoU(
        gt_mask, target_augmented_SR, img_size=IMG_SIZE, class_id=CLASS_ID)
    max_SR_iou = compute_IoU(
        gt_mask, target_max_SR, img_size=IMG_SIZE, class_id=CLASS_ID)
    mean_SR_iou = compute_IoU(
        gt_mask, target_mean_SR, img_size=IMG_SIZE, class_id=CLASS_ID)

    print(
        f"Aug. SR ({MODE} OPM) IoU: {aug_SR_iou_single}, Max SR IoU: {max_SR_iou}, Mean SR IoU: {mean_SR_iou}")

    plot_prediction([input_image, gt_mask, target_augmented_SR],
                    only_prediction=False, show_overlay=True)


if __name__ == '__main__':
    main()
