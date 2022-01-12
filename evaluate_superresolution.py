import os
import argparse
import tensorflow as tf
from model import DeeplabV3Plus
from utils import load_image, create_mask

parser = argparse.ArgumentParser()
parser.add_argument("--save_standard", help="Save standard network output for comparison",
                    action="store_true")

args = parser.parse_args()


def get_img_paths(image_list_path, image_folder):
    return [os.path.join(image_folder, line.rstrip() + ".jpg") for line in open(image_list_path)]


def Mean_IOU(y_true, y_pred):
    nb_classes = 21  # TODO: set this as a parameter
    ious = []
    for i in range(0, nb_classes):  # exclude last label (void)
        y_true_squeeze = tf.squeeze(y_true)
        y_pred_squeeze = tf.squeeze(y_pred)
        true_labels = tf.equal(y_true_squeeze, i)
        pred_labels = tf.equal(y_pred_squeeze, i)
        inter = tf.cast(true_labels & pred_labels, tf.int32)
        union = tf.cast(true_labels | pred_labels, tf.int32)

        iou = tf.reduce_sum(inter) / tf.reduce_sum(union)
        # returns average IoU of the same objects
        ious.append(iou)

    ious = tf.stack(ious)
    legal_labels = ~tf.math.is_nan(ious)
    ious = tf.gather(ious, indices=tf.where(legal_labels))
    return tf.reduce_mean(ious)


def evaluate_IOU(true_mask, standard_mask, superres_mask, img_size=(512, 512)):
    true_mask = tf.reshape(true_mask, (img_size[0] * img_size[1], 1))
    standard_mask = tf.reshape(standard_mask, (img_size[0] * img_size[1], 1))
    superres_mask = tf.reshape(superres_mask, (img_size[0] * img_size[1], 1))

    standard_IOU = Mean_IOU(true_mask, standard_mask)
    superres_IOU = Mean_IOU(true_mask, superres_mask)

    return standard_IOU, superres_IOU


def get_prediction(model, input_image):

    prediction = model.predict(input_image[tf.newaxis, ...])
    mask = create_mask(prediction[0])

    return mask


def compare_results(image_paths, model, superres_image_folder, standard_out_folder, image_size=(512, 512)):
    for image_path in image_paths:
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        true_mask_path = image_path.replace("JPEGImages", "SegmentationClassAug").replace(".jpg", ".png")
        true_mask = load_image(true_mask_path, image_size=image_size, normalize=False,
                               is_png=True, resize_method="nearest")

        input_image = load_image(image_path, image_size=image_size, normalize=True)
        standard_mask = get_prediction(model, input_image)
        if args.save_standard:
            if not os.path.exists(standard_out_folder):
                os.mkdir(standard_out_folder)
            tf.keras.utils.save_img(f"{standard_out_folder}/{image_name}.png", standard_mask, scale=False)

        superres_image_path = os.path.join(superres_image_folder, f"{image_name}.png")
        superres_image = load_image(superres_image_path, normalize=False, is_png=True)

        standard_IOU, superres_IOU = evaluate_IOU(true_mask, standard_mask, superres_image, img_size=image_size)
        print(f"IOUs for image {image_name} - Standard: {str(standard_IOU.numpy())}, Superres: {str(superres_IOU.numpy())}")


def main():
    image_size = (512, 512)
    data_root = os.path.join(os.getcwd(), "data")
    image_list_path = os.path.join(data_root, "augmented_file_lists", "valaug.txt")
    image_folder_path = os.path.join(data_root, "VOCdevkit", "VOC2012", "JPEGImages")
    superres_image_folder = os.path.join(data_root, "superres_output")
    standard_out_folder = os.path.join(data_root, "standard_output")

    image_paths = get_img_paths(image_list_path, image_folder_path)[:20]

    model = DeeplabV3Plus(
        input_shape=(512, 512, 3),
        classes=21,
        OS=16,
        last_activation=None,
        load_weights=True,
        backbone="mobilenet",
        alpha=1.).build_model()

    compare_results(image_paths, model, superres_image_folder, standard_out_folder, image_size=image_size)


if __name__ == '__main__':
    main()
