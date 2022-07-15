import os
import wandb
import random
import gc
import csv
import itertools
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from matplotlib import pyplot as plt
import tensorflow_addons as tfa
from model import DeeplabV3Plus
from superresolution_scripts.superres_utils import get_img_paths
from utils import load_image, compute_IoU, create_mask

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# tf.config.run_functions_eagerly(True)

IMG_SIZE = (512, 512)
CLASS_ID = 8
NUM_SAMPLES = 350
MODEL_BACKBONE = "xception"
USE_VALIDATION = False
BATCH_SIZE = 16

DATA_DIR = os.path.join(os.getcwd(), "data")
PASCAL_ROOT = os.path.join(DATA_DIR, "dataset_root", "VOCdevkit", "VOC2012")
IMGS_PATH = os.path.join(PASCAL_ROOT, "JPEGImages")


DEST_FOLDER = os.path.join(DATA_DIR, "robustness_check")
IMAGE_OUTPUT_FOLDER = os.path.join(DEST_FOLDER, "image_output")


def augment_images(images, angle, shift_x, shift_y, interpolation="bilinear"):
    rotated_images = tfa.image.rotate(
        images, angle, interpolation=interpolation, fill_mode="constant")
    translated_images = tfa.image.translate(
        rotated_images, [shift_x, shift_y], interpolation=interpolation, fill_mode="constant")

    return translated_images


def save_plot(input_image, prediction, ground_truth, plot_title, save_path):
    plt.figure()

    plt.subplot(1, 2, 1)
    plt.title(plot_title)
    plt.imshow(tf.keras.preprocessing.image.array_to_img(input_image))
    plt.imshow(tf.keras.preprocessing.image.array_to_img(
        prediction), alpha=0.5, cmap="jet")

    plt.subplot(1, 2, 2)
    plt.title("Ground Truth")
    plt.imshow(tf.keras.preprocessing.image.array_to_img(ground_truth))

    plt.axis("off")
    # plt.colorbar()
    plt.figtext(0.99, 0.01, f"Labels: {str(np.unique(prediction))}, GT: {str(np.unique(ground_truth))}",
                horizontalalignment='right')
    plt.savefig(save_path)
    plt.close()


def main():

    if not os.path.exists(IMAGE_OUTPUT_FOLDER):
        os.makedirs(IMAGE_OUTPUT_FOLDER)

    image_list_path = os.path.join(DATA_DIR, "augmented_file_lists",
                                   f"{'valaug' if USE_VALIDATION else 'trainaug'}.txt")
    image_paths = get_img_paths(
        image_list_path, IMGS_PATH, is_png=False, sort=False)
    image_paths = random.sample(image_paths, NUM_SAMPLES)
    gt_paths = [path.replace("JPEGImages", "SegmentationClassAug").replace(
        ".jpg", ".png") for path in image_paths]

    images = tf.stack([load_image(path, image_size=IMG_SIZE, normalize=True)
                       for path in image_paths])

    gt_images = tf.stack([load_image(path, image_size=IMG_SIZE, normalize=False, is_png=True, resize_method="nearest")
                          for path in gt_paths])

    model = DeeplabV3Plus(
        input_shape=(512, 512, 3),
        classes=21,
        OS=16,
        last_activation=None,
        load_weights=True,
        backbone=MODEL_BACKBONE,
        reshape_outputs=False,
        alpha=1.).build_model()

    # angle_values = [round(angle, 2) for angle in np.linspace(0.0, 3.14, num=9)]
    # shift_x_values = np.linspace(0, 80, num=9, dtype=int)
    # shift_y_values = np.linspace(0, 80, num=9, dtype=int)

    angle_values = [round(angle, 2)
                    for angle in np.arange(-1.0, 1.05, step=0.05)]
    shift_x_values = np.linspace(0, 80, num=9, dtype=int)
    shift_y_values = np.linspace(0, 80, num=9, dtype=int)

    all_combinations = list(itertools.product(
        angle_values, shift_x_values, shift_y_values))

    csv_header = ["Angle", "Shift_X", "Shift_Y", "IoU"]
    csv_path = f"{DEST_FOLDER}/robustness_small_{NUM_SAMPLES}.csv"
    f = open(csv_path, "w")
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    writer.writerow(csv_header)

    for i, (angle, shift_x, shift_y) in tqdm(enumerate(all_combinations)):

        # wandb.init(project="Robustness check (350 samples, N=9)",
        #            entity="albergoni-nicolo")

        aug_images = augment_images(images, angle, shift_x, shift_y)
        aug_gt = augment_images(
            gt_images, angle, shift_x, shift_y, interpolation="nearest")

        predictions = model.predict(aug_images, batch_size=BATCH_SIZE)
        _ = gc.collect()

        ious = []

        for k, pred in enumerate(predictions):
            image_name = os.path.splitext(os.path.basename(image_paths[k]))[0]
            save_path = os.path.join(IMAGE_OUTPUT_FOLDER, f"{image_name}.png")
            iou = round(compute_IoU(aug_gt[k], pred), 3)

            # plot_title = f"mIoU: {iou}, Angle: {angle}, Shift X: {shift_x}, Shift Y: {shift_y}"
            # save_plot(aug_images[k], create_mask(
            #     pred), aug_gt[k], plot_title, save_path)

            ious.append(iou)

        avg_mean_iou = round(np.mean(ious), 3)

        print(
            f"Angle: {angle}, Shift X: {shift_x}, Shift Y: {shift_y}, IoU: {avg_mean_iou}")
        writer.writerow([angle, shift_x, shift_y, avg_mean_iou])
        f.flush()
        # wandb.log({
        #     "Angle": angle,
        #     "Shift_X": shift_x,
        #     "Shift_Y": shift_y,
        #     "IoU": avg_mean_iou
        # })

        # wandb.finish(quiet=True)

    f.close()
    print("Done")


if __name__ == '__main__':
    main()
