import os
from data_utils import *
from remove_gt_colormap import remove_gt_colormap
from build_voc_2012_data import build_tfrecords

BASE_DIR = os.getcwd()
DATASET_URL = "https://data.deepai.org/PascalVOC2012.zip"
DATASET_DIR = os.path.join(BASE_DIR, "data")

filepath = download_dataset(DATASET_URL, DATASET_DIR)

extract_zip_file(filepath, DATASET_DIR)

PASCAL_ROOT = os.path.join(DATASET_DIR, "VOC2012")
SEG_FOLDER = os.path.join(PASCAL_ROOT, "SegmentationClass")
SEMANTIC_SEG_FOLDER = os.path.join(PASCAL_ROOT, "SegmentationClassRaw")

remove_gt_colormap(SEG_FOLDER, SEMANTIC_SEG_FOLDER)

TF_RECORDS_DIR = os.path.join(DATASET_DIR, "tfrecord")

IMAGE_FOLDER = os.path.join(PASCAL_ROOT, "JPEGImages")
LIST_FOLDER = os.path.join(PASCAL_ROOT, "ImageSets", "Segmentation")

build_tfrecords(LIST_FOLDER, IMAGE_FOLDER, SEMANTIC_SEG_FOLDER, TF_RECORDS_DIR)
