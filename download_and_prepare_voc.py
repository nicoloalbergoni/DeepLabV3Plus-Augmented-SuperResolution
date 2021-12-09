import os
import argparse
from data_scripts.data_utils import download_dataset, extract_file
from data_scripts.remove_gt_colormap import remove_gt_colormap
from data_scripts.pascal_voc_dataset import PascalVOC2012Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--generate_tf_records", help="Optionally generate tfrecord files for the dataset",
                    action="store_true")
parser.add_argument(
    "--remove_cmap", help="Remove colormap from masks, used in PASCAL VOC", action="store_true")

parser.add_argument(
    "--use_mirror", help="Download the dataser from a mirror site", action="store_true")

parser.add_argument("--pascal_root", help="Root directory of the PASCAL VOC dataset", nargs='?',
                    type=str, default="./data/VOCdevkit/VOC2012", const="./data/VOCdevkit/VOC2012")

args = parser.parse_args()


def main():

    BASE_DIR = os.getcwd()
    #DATASET_URL = "https://data.deepai.org/PascalVOC2012.zip"

    if args.use_mirror:
        DATASET_URL = "http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar"
    else:
        DATASET_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"

    DATA_DIR = os.path.join(BASE_DIR, "data")
    #PASCAL_ROOT = os.path.join(DATA_DIR, "VOCdevkit", "VOC2012")
    PASCAL_ROOT = os.path.normpath(args.pascal_root)

    filepath = download_dataset(DATASET_URL, DATA_DIR)

    extract_file(filepath, DATA_DIR, is_extracted=PASCAL_ROOT)

    if args.remove_cmap:
        SEG_FOLDER = os.path.join(PASCAL_ROOT, "SegmentationClass")
        SEMANTIC_SEG_FOLDER = os.path.join(PASCAL_ROOT, "SegmentationClassRaw")
        remove_gt_colormap(SEG_FOLDER, SEMANTIC_SEG_FOLDER)

    if args.generate_tf_records:
        TF_RECORDS_DIR = os.path.join(DATA_DIR, "TFRecords")
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
