import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def preprocess_image(image, target_size_width=512, mean_subtraction_value=127.5):

    w, h, _ = image.shape
    ratio = float(target_size_width) / np.max([w, h])
    new_size = (int(ratio * h), int(ratio * w))
    resized_img = np.array(Image.fromarray(image.astype("uint8")).resize(new_size))

    # Normlization
    resized_img = (resized_img / mean_subtraction_value) - 1.

    # Padding to desired dimension
    pad_width = int(target_size_width - resized_img.shape[0])
    pad_height = int(target_size_width - resized_img.shape[1])

    pad_size = ((0, pad_width), (0, pad_height), (0, 0))

    resized_img = np.pad(resized_img, pad_size, constant_values=0, mode="constant")

    return resized_img, pad_width, pad_height


def plot_prediction(display_list, only_prediction=True, show_overlay=True):
    plt.figure(figsize=(18, 18))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list) + 1, i + 1)
        if only_prediction and i == 1:
            plt.title(title[-1])
        else:
            plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')

    if show_overlay:
        plt.subplot(1, len(display_list) + 1, len(display_list) + 1)
        plt.title("Overlay")
        plt.imshow(display_list[0])
        plt.imshow(display_list[-1], alpha=0.5)
        plt.axis("off")

    plt.show()
