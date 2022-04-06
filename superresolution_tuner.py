import os
import wandb
import h5py
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from superresolution_scripts.superresolution import Superresolution
from utils import load_image
from superresolution_scripts.superres_utils import min_max_normalization, list_precomputed_data_paths, \
    check_hdf5_validity, threshold_image, single_class_IOU
import keras_tuner as kt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SEED = 1234

np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (512, 512)
NUM_AUG = 100
CLASS_ID = 8
NUM_SAMPLES = 100
MODE = "slice"
USE_VALIDATION = False

DATA_DIR = os.path.join(os.getcwd(), "data")
PASCAL_ROOT = os.path.join(DATA_DIR, "dataset_root", "VOCdevkit", "VOC2012")
IMGS_PATH = os.path.join(PASCAL_ROOT, "JPEGImages")

SUPERRES_ROOT = os.path.join(DATA_DIR, "superres_root")
PRECOMPUTED_OUTPUT_DIR = os.path.join(SUPERRES_ROOT, f"precomputed_features{'_validation' if USE_VALIDATION else ''}")
STANDARD_OUTPUT_DIR = os.path.join(SUPERRES_ROOT, f"standard_output{'_validation' if USE_VALIDATION else ''}")
SUPERRES_OUTPUT_DIR = os.path.join(SUPERRES_ROOT, f"superres_output{'_validation' if USE_VALIDATION else ''}")


def evaluate_IOU(true_mask, standard_mask, superres_mask, class_id, img_size=(512, 512)):
    true_mask = tf.reshape(true_mask, (img_size[0] * img_size[1], 1))
    standard_mask = tf.reshape(standard_mask, (img_size[0] * img_size[1], 1))
    superres_mask = tf.reshape(superres_mask, (img_size[0] * img_size[1], 1))

    standard_IOU = single_class_IOU(true_mask, standard_mask, class_id=class_id)
    superres_IOU = single_class_IOU(true_mask, superres_mask, class_id=class_id)

    return standard_IOU.numpy(), superres_IOU.numpy()


def compute_superresolution_output(precomputed_data_paths, superres_args, dest_folder, mode="slice", num_aug=100,
                                   global_normalize=True, save_output=False):
    class_losses = []
    max_losses = []
    standard_ious = []
    superres_ious = []

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for file_path in tqdm(precomputed_data_paths):

        file = h5py.File(f"{file_path}", "r")

        if not check_hdf5_validity(file, num_aug=num_aug):
            print(f"File: {file_path} is invalid, skipping...")
            file.close()
            continue

        filename = file.attrs["filename"]
        angles = file["angles"][:]
        shifts = file["shifts"][:]

        class_masks = file["class_masks"][:]
        class_masks = tf.stack(class_masks)

        if mode == "slice":
            max_masks = file["max_masks"][:]
            max_masks = tf.stack(max_masks)

        file.close()

        superresolution_obj = Superresolution(
            **superres_args,
            num_aug=NUM_AUG,
            verbose=False
        )

        global_min, global_max = (tf.reduce_min(class_masks), tf.reduce_max(class_masks)) if global_normalize else (
            None, None)

        class_masks = tf.map_fn(
            fn=lambda image: min_max_normalization(image.numpy(), new_min=0.0, new_max=1.0, global_min=global_min,
                                                   global_max=global_max), elems=class_masks)

        target_image_class, class_loss = superresolution_obj.compute_output(class_masks, angles, shifts)
        target_image_class = (target_image_class[0]).numpy()
        # print(f"Final class loss for image {filename}: {class_loss}")

        class_losses.append(class_loss)

        if mode == "slice":
            global_min, global_max = (tf.reduce_min(max_masks), tf.reduce_max(max_masks)) if global_normalize else (
                None, None)

            max_masks = tf.map_fn(
                fn=lambda image: min_max_normalization(image.numpy(), new_min=0.0, new_max=1.0, global_min=global_min,
                                                       global_max=global_max), elems=max_masks)

            target_image_max, max_loss = superresolution_obj.compute_output(max_masks, angles, shifts)
            target_image_max = (target_image_max[0]).numpy()
            # print(f"Final max loss for image {filename}: {max_loss}")

            max_losses.append(max_loss)

        th_mask = threshold_image(target_image_class, CLASS_ID, th_mask=None if mode == "argmax" else target_image_max,
                                  th_factor=.15)

        true_mask_path = os.path.join(PASCAL_ROOT, "SegmentationClassAug", f"{filename}.png")
        standard_mask_path = os.path.join(STANDARD_OUTPUT_DIR, f"{filename}.png")

        true_mask = load_image(true_mask_path, image_size=IMG_SIZE, normalize=False,
                               is_png=True, resize_method="nearest")
        standard_mask = load_image(standard_mask_path, image_size=IMG_SIZE, normalize=False, is_png=True,
                                   resize_method="nearest")

        standard_iou, superres_iou = evaluate_IOU(true_mask, standard_mask, th_mask, class_id=CLASS_ID,
                                                  img_size=IMG_SIZE)
        standard_ious.append(standard_iou)
        superres_ious.append(superres_iou)

        if save_output:
            tf.keras.utils.save_img(f"{dest_folder}/{filename}_th_{mode}.png", th_mask, scale=True)

    mean_standard_iou = np.mean(standard_ious)
    mean_superres_iou = np.mean(superres_ious)
    mean_class_loss = np.mean(class_losses)
    mean_max_loss = np.mean(max_losses)

    return mean_standard_iou, mean_superres_iou, mean_class_loss, mean_max_loss


class SuperresTuner(kt.RandomSearch):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        data_path = list_precomputed_data_paths(PRECOMPUTED_OUTPUT_DIR)
        self.precomputed_data_paths = data_path if NUM_SAMPLES is None else data_path[:NUM_SAMPLES]

    def run_trial(self, trial, **kwargs):
        hp = trial.hyperparameters

        superres_args = {
            "lambda_df": hp.Float("lambda_df", min_value=0.001, max_value=1.0, sampling="log"),
            "lambda_tv": hp.Float("lambda_tv", min_value=0.001, max_value=1.0, sampling="log"),
            "lambda_eng": hp.Float("lambda_eng", min_value=0.001, max_value=1.0, sampling="log"),
            "lambda_L1": 0.0,
            # "lambda_eng": 0.5,
            # "num_iter": hp.Int("num_iter", min_value=450, max_value=1000, step=50),
            "num_iter": 450,
            "learning_rate": 1e-3,
            # "optimizer": hp.Choice("optimizer", ["adam", "adadelta", "adagrad"]),
            "optimizer": "adam",
            "L1_reg": False,
            "df_lp_norm": 2.0
        }

        global_normalize = True

        wandb_dir = os.path.join(DATA_DIR, "wandb_logs")
        if not os.path.exists(wandb_dir):
            os.makedirs(wandb_dir)

        run = wandb.init(project="Added DF coefficient", entity="albergoni-nicolo", dir=wandb_dir,
                         config=hp.values)

        wandb.config.num_aug = NUM_AUG
        wandb.config.num_sample = NUM_SAMPLES
        wandb.config.update(superres_args)

        mean_standard_iou, mean_superres_iou, mean_class_loss, mean_max_loss = compute_superresolution_output(
            self.precomputed_data_paths,
            superres_args,
            dest_folder=SUPERRES_OUTPUT_DIR,
            mode=MODE, num_aug=NUM_AUG,
            global_normalize=global_normalize,
            save_output=False)

        run.log({"mean_superres_iou": mean_superres_iou,
                 "mean_standard_iou": mean_standard_iou,
                 "mean_class_loss": mean_class_loss,
                 "mean_max_loss": mean_max_loss})

        run.finish()

        return 1.0 - mean_superres_iou


def main():
    tuner = SuperresTuner(
        # No hypermodel or objective specified.
        max_trials=30,
        overwrite=True,
        directory=DATA_DIR,
        project_name="tuner_trials",
        seed=SEED
    )

    tuner.search()


if __name__ == '__main__':
    main()
