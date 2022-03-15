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
                    type=str, default="./data/dataset_root/VOCdevkit/VOC2012", const="./data/dataset_root/VOCdevkit/VOC2012")

parser.add_argument(
    "--download_berkley", help="Download the augmented dataset provided by Berkley", action="store_true")

args = parser.parse_args()


def main():
    if args.use_mirror:
        DATASET_URL = "http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar"
    else:
        DATASET_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"

    DATA_DIR = os.path.join(os.getcwd(), "data")
    DATASET_ROOT = os.path.join(DATA_DIR, "dataset_root")
    PASCAL_ROOT = os.path.normpath(args.pascal_root)

    filepath = download_dataset(DATASET_URL, dest_folder=DATASET_ROOT)
    extract_file(filepath, dest_folder=DATASET_ROOT, is_extracted=PASCAL_ROOT)

    if args.download_berkley:
        BERKLEY_URL = "https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=1"
        filepath = download_dataset(BERKLEY_URL, DATASET_ROOT)
        extract_file(filepath, dest_folder=PASCAL_ROOT, is_extracted=os.path.join(PASCAL_ROOT, "SegmentationClassAug"))

    if args.remove_cmap:
        SEG_FOLDER = os.path.join(PASCAL_ROOT, "SegmentationClass")
        SEMANTIC_SEG_FOLDER = os.path.join(PASCAL_ROOT, "SegmentationClassRaw")
        remove_gt_colormap(SEG_FOLDER, output_dir=SEMANTIC_SEG_FOLDER)

    if args.generate_tf_records:
        TF_RECORDS_DIR = os.path.join(DATASET_ROOT, "TFRecords")
        dataset = PascalVOC2012Dataset(augmentation_params=None)
        train_basenames = dataset.get_basenames('train', PASCAL_ROOT)
        print('Found', len(train_basenames), 'training samples')

        val_basenames = dataset.get_basenames('val', PASCAL_ROOT)
        print('Found', len(val_basenames), 'validation samples')

        # Export train and validation datasets to TFRecords
        dataset.export_tfrecord('train', dataset_path=PASCAL_ROOT, tf_record_dest_dir=TF_RECORDS_DIR,
                                tfrecord_filename='segmentation_train.tfrecords')
        dataset.export_tfrecord('val', dataset_path=PASCAL_ROOT, tf_record_dest_dir=TF_RECORDS_DIR,
                                tfrecord_filename='segmentation_val.tfrecords')
        print('Finished exporting')


if __name__ == '__main__':
    main()
