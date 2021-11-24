import os
from matplotlib import pyplot as plt
from data_scripts.pascal_voc_dataset import PascalVOC2012Dataset

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
PASCAL_ROOT = os.path.join(DATA_DIR, "VOCdevkit", "VOC2012")
TF_RECORD_DIR = os.path.join(DATA_DIR, "TFRecords")

AUGMENTATION_PARAMS = {'saturation_range': (-20, 20), 'value_range': (-20, 20),
                       'brightness_range': None, 'contrast_range': None, 'blur_params': None,
                       'flip_lr': True, 'rotation_range': (-10, 10), 'shift_range': (32, 32),
                       'zoom_range': (0.5, 2.0), 'ignore_label': 21}

dataset_obj = PascalVOC2012Dataset(augmentation_params=AUGMENTATION_PARAMS)

dataset = dataset_obj.load_dataset(True, TF_RECORD_DIR, 16)


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()


for images, masks in dataset.take(2):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image, sample_mask])
