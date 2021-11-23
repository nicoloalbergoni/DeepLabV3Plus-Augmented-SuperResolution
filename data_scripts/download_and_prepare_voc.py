import os
from data_utils import *
from remove_gt_colormap import remove_gt_colormap
from build_voc_2012_data import build_tfrecords
from pascal_voc_dataset import PascalVOC2012Dataset

BASE_DIR = os.getcwd()
DATASET_URL = "https://data.deepai.org/PascalVOC2012.zip"
DATASET_DIR = os.path.join(BASE_DIR, "data")
PASCAL_ROOT = os.path.join(DATASET_DIR, "VOC2012")

filepath = download_dataset(DATASET_URL, DATASET_DIR)

extract_zip_file(filepath, DATASET_DIR)


# SEG_FOLDER = os.path.join(PASCAL_ROOT, "SegmentationClass")
# SEMANTIC_SEG_FOLDER = os.path.join(PASCAL_ROOT, "SegmentationClassRaw")

# remove_gt_colormap(SEG_FOLDER, SEMANTIC_SEG_FOLDER)

TF_RECORDS_DIR = os.path.join(DATASET_DIR, "tfrecord")

# IMAGE_FOLDER = os.path.join(PASCAL_ROOT, "JPEGImages")
# LIST_FOLDER = os.path.join(PASCAL_ROOT, "ImageSets", "Segmentation")

# build_tfrecords(LIST_FOLDER, IMAGE_FOLDER, SEMANTIC_SEG_FOLDER, TF_RECORDS_DIR)


def main(data_dir, tf_records_dir):
    """
    Export the PASCAL VOC segmentation dataset in 2 ways:
        1. Converts ground truth segmentation classes to sparse labels.
        2. Export the dataset to TFRecords, one for the training set and another one for the validation set.
    """
    dataset = PascalVOC2012Dataset(data_dir, augmentation_params=None)
    train_basenames = dataset.get_basenames('train')
    print('Found', len(train_basenames), 'training samples')

    val_basenames = dataset.get_basenames('val')
    print('Found', len(val_basenames), 'validation samples')

    # Encode and save sparse ground truth segmentation image labels
    # print('Exporting training set sparse labels...')
    # dataset.export_sparse_encoding('train', data_dir)
    # print('Exporting validation set sparse labels...')
    # dataset.export_sparse_encoding('val', data_dir)

    # Export train and validation datasets to TFRecords
    dataset.export_tfrecord('train', tf_records_dir,
                            'segmentation_train.tfrecords')
    dataset.export_tfrecord('val', tf_records_dir,
                            'segmentation_val.tfrecords')
    print('Finished exporting')


if __name__ == '__main__':
    main(PASCAL_ROOT, DATASET_DIR)
