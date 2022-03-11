"""
This script is used to generate the files that contain the lists of image filenames (both for training and validation)
of the augmented version of Pascal VOC 2012 (i.e. with the data provided by Berkley University)
"""

import os


def get_pascal_img_list(pascal_root):

    filename_train = os.path.join(pascal_root, "ImageSets", "Segmentation", "train.txt")
    filename_val = os.path.join(pascal_root, "ImageSets", "Segmentation", "val.txt")

    train_list = [line.rstrip() for line in open(filename_train)]
    val_list = [line.rstrip() for line in open(filename_val)]

    return set(train_list), set(val_list)


def get_berkley_img_list(berkley_root):

    filename_train = os.path.join(berkley_root, "train.txt")
    filename_val = os.path.join(berkley_root, "val.txt")

    train_list = [line.rstrip() for line in open(filename_train)]
    val_list = [line.rstrip() for line in open(filename_val)]

    return set(train_list), set(val_list)


def write_list_to_file(file_path, data_list):
    with open(file_path, "w") as file:
        file.write('\n'.join(data_list))


def main():
    DATA_ROOT = os.path.join(os.getcwd(), "data")
    PASCAL_ROOT = os.path.join(DATA_ROOT, "VOCdevkit", "VOC2012")
    BERKLEY_ROOT = os.path.join(DATA_ROOT, "berkley_file_lists")
    OUTPUT_FOLDER = os.path.join(DATA_ROOT, "augmented_file_lists")

    pascal_train, pascal_val = get_pascal_img_list(PASCAL_ROOT)
    berkley_train, berkley_val = get_berkley_img_list(BERKLEY_ROOT)

    full_pascal = pascal_train | pascal_val
    full_berkley = berkley_train | berkley_val

    everything = full_pascal | full_berkley

    validation = pascal_val

    train = everything - validation

    print(f"Train images: {len(train)}, Validation images: {len(validation)}")

    write_list_to_file(os.path.join(OUTPUT_FOLDER, "trainaug.txt"), train)
    write_list_to_file(os.path.join(OUTPUT_FOLDER, "valaug.txt"), validation)


if __name__ == '__main__':
    main()
