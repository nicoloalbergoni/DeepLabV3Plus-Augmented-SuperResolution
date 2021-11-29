import os
from data_scripts.data_utils import download_dataset, extract_file
from data_scripts.remove_gt_colormap import remove_gt_colormap
from data_scripts.pascal_voc_dataset import PascalVOC2012Dataset


def main():

    BASE_DIR = os.getcwd()
    #DATASET_URL = "https://data.deepai.org/PascalVOC2012.zip"
    DATASET_URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    DATA_DIR = os.path.join(BASE_DIR, "data")
    #PASCAL_ROOT = os.path.join(DATA_DIR, "VOC2012")
    PASCAL_ROOT = os.path.join(DATA_DIR, "VOCdevkit", "VOC2012")
    TF_RECORDS_DIR = os.path.join(DATA_DIR, "TFRecords")

    filepath = download_dataset(DATASET_URL, DATA_DIR)

    extract_file(filepath, DATA_DIR, is_extracted=PASCAL_ROOT)

    SEG_FOLDER = os.path.join(PASCAL_ROOT, "SegmentationClass")
    SEMANTIC_SEG_FOLDER = os.path.join(PASCAL_ROOT, "SegmentationClassRaw")
    remove_gt_colormap(SEG_FOLDER, SEMANTIC_SEG_FOLDER)

    dataset = PascalVOC2012Dataset(augmentation_params=None)
    train_basenames = dataset.get_basenames('train', PASCAL_ROOT)
    print('Found', len(train_basenames), 'training samples')

    val_basenames = dataset.get_basenames('val', PASCAL_ROOT)
    print('Found', len(val_basenames), 'validation samples')

    # Export train and validation datasets to TFRecords
    dataset.export_tfrecord('train', PASCAL_ROOT, TF_RECORDS_DIR,
                            'segmentation_train.tfrecords')
    dataset.export_tfrecord('val', PASCAL_ROOT, TF_RECORDS_DIR,
                            'segmentation_val.tfrecords')
    print('Finished exporting')


if __name__ == '__main__':
    main()
