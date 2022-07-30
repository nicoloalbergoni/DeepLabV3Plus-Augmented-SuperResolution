import os
import gc
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from model import DeeplabV3Plus
from superresolution_scripts.superres_utils import get_img_paths, filter_images_by_class, load_image, threshold_image
from utils import create_mask, compute_IoU

SEED = 1234

np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IMG_SIZE = (512, 512)
BATCH_SIZE = 16
NUM_REPETITION = 150
CLASS_ID = 8
NUM_SAMPLES = 500
MODEL_BACKBONE = "xception"
USE_VALIDATION = False

DATA_DIR = os.path.join(os.getcwd(), "data")
PASCAL_ROOT = os.path.join(DATA_DIR, "dataset_root", "VOCdevkit", "VOC2012")
IMGS_PATH = os.path.join(PASCAL_ROOT, "JPEGImages")

SUPERRES_ROOT = os.path.join(DATA_DIR, "superres_root")
STANDARD_OUTPUT_ROOT = os.path.join(SUPERRES_ROOT, "standard_output")
STANDARD_OUTPUT_DIR = os.path.join(
    STANDARD_OUTPUT_ROOT, f"{MODEL_BACKBONE}{'_validation' if USE_VALIDATION else ''}")

DROPOUT_ROOT = os.path.join(DATA_DIR, "dropout_root")
DROPOUT_OUTPUT = os.path.join(DROPOUT_ROOT, "output")


def modify_model(model, dropout_factor=.4):
    layers = model.layers
    do = tf.keras.layers.Dropout(dropout_factor)(
        layers[402].output, training=True)

    x = do
    for i in range(403, len(layers)):
        x = layers[i](x)

    result_model = tf.keras.Model(inputs=layers[0].input, outputs=x)

    return result_model


def dropout_sampling(image_paths, model, num_repetition, image_size=(512, 512), save_output=False):

    if not os.path.exists(DROPOUT_OUTPUT):
        os.makedirs(DROPOUT_OUTPUT)

    standard_ious_single = []
    mask_ious_single = []
    for image_path in tqdm(image_paths):
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # Load image
        image = load_image(image_path, image_size=image_size, normalize=True)
        image_batch = tf.repeat(tf.expand_dims(
            image, axis=0), num_repetition, axis=0)

        predictions = model.predict(image_batch, batch_size=BATCH_SIZE)
        _ = gc.collect()

        predicted_masks = tf.map_fn(
            fn=lambda x: create_mask(x), elems=predictions, fn_output_signature=tf.int64)

        mask_aggregated = tf.reduce_max(predicted_masks, axis=0)

        if save_output:
            tf.keras.utils.save_img(os.path.join(
                DROPOUT_OUTPUT, f"{image_name}.png"), mask_aggregated, scale=False)

        true_mask_path = os.path.join(
            PASCAL_ROOT, "SegmentationClassAug", f"{image_name}.png")
        true_mask = load_image(true_mask_path, image_size=IMG_SIZE, normalize=False,
                               is_png=True, resize_method="nearest")

        standard_mask_path = os.path.join(
            STANDARD_OUTPUT_DIR, f"{image_name}.png")
        standard_mask = load_image(standard_mask_path, image_size=IMG_SIZE, normalize=False, is_png=True,
                                   resize_method="nearest")

        standard_iou_single = compute_IoU(
            true_mask, standard_mask, img_size=IMG_SIZE, class_id=CLASS_ID)
        mask_iou_single = compute_IoU(
            true_mask, mask_aggregated, img_size=IMG_SIZE, class_id=CLASS_ID)

        standard_ious_single.append(standard_iou_single)
        mask_ious_single.append(mask_iou_single)

    avg_standard_iou_single = np.mean(standard_ious_single)
    avg_mask_iou_single = np.mean(mask_ious_single)

    print(
        f"Mask Single: {avg_mask_iou_single}, Standard Single: {avg_standard_iou_single}")

    return avg_standard_iou_single, avg_mask_iou_single


def main():
    image_list_path = os.path.join(DATA_DIR, "augmented_file_lists",
                                   f"{'valaug' if USE_VALIDATION else 'trainaug'}.txt")
    image_paths = get_img_paths(
        image_list_path, IMGS_PATH, is_png=False, sort=True)
    images_paths_filtered = filter_images_by_class(
        image_paths, filter_class_id=CLASS_ID, num_images=NUM_SAMPLES, image_size=IMG_SIZE)

    print(
        f"Valid images: {len(images_paths_filtered)} (Initial: {len(image_paths)})")

    model = DeeplabV3Plus(
        input_shape=(512, 512, 3),
        classes=21,
        OS=16,
        last_activation=None,
        load_weights=True,
        backbone="xception",
        alpha=1.,
        reshape_outputs=False).build_model()

    data_list = []
    dropout_values = np.arange(0.05, 0.95, step=0.05)

    for d_value in dropout_values:
        modified_model = modify_model(model, dropout_factor=round(d_value, 2))
        avg_standard_iou_single, avg_mask_iou_single = dropout_sampling(images_paths_filtered, modified_model,
                                                                        NUM_REPETITION, save_output=False)

        data_list.append({
            "D_value": d_value,
            "sIoU": avg_standard_iou_single,
            "dIoU": avg_mask_iou_single
        })

    df = pd.DataFrame(data_list)
    df.to_csv(os.path.join(DROPOUT_ROOT,
              f"{NUM_REPETITION}rep_{NUM_SAMPLES}_samples.csv"))
    print("Done")


if __name__ == '__main__':
    main()
