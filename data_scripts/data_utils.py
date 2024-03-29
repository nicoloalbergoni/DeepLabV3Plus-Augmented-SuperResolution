import os
import sys
import zipfile
import tarfile
import urllib.request


def shorten(s, subs):
    i = s.index(subs)
    return s[:i+len(subs)]


def download_dataset(dataset_url, dest_folder):
    """
    Donwloand the PASCAL VOC 2012 dataset from the given URL

    Args:
        dataset_url (str): ULR of the dataset
        dest_folder (str): Destination folder in which the dataset is downloaded

    Returns:
        str: Full path to the downloaded dataset
    """
    # TODO: handle other extension
    extension = ".zip" if ".zip" in dataset_url else ".tar"
    filename = shorten(dataset_url.split("/")[-1], extension)
    full_dest_path = os.path.join(dest_folder, filename)

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    if os.path.exists(full_dest_path):
        print(
            f"File {filename} already in destination folder, skipping download")
        return full_dest_path

    def _progress(count, block_size, total_size):
        sys.stdout.write('\rDownloading %s %.1f%%' % (
            filename, 100.0 * count * block_size / total_size))
        sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(
        dataset_url, full_dest_path, _progress)

    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    return filepath


def extract_file(filepath, dest_folder, is_extracted="./data/dataset_root/VOCdevkit"):
    """
    Extracts the dataset from the zip/tar file

    Args:
        filepath (str): Path of the dataset
        dest_folder (str): Destination path for the extracted contents
        is_extracted (str, optional): Optional path to check if the dataset has already been extracted. Defaults to "./data/dataset_root/VOCdevkit".

    Raises:
        ValueError: if the files is neither zip nor tar
    """

    # TODO: Find a better way to understand when to skip unzip even for different datasets
    if os.path.exists(is_extracted):
        print("File already extracted")
        return

    file_type = filepath.split(".")[-1]

    if file_type == "zip":
        print("Extracting zip file...")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)
        print("Finished extraction")

    elif file_type == "tar":
        print('Extracting tarball...')
        tarfile.open(filepath, 'r').extractall(dest_folder)
        print('Finished extracting')
    else:
        raise ValueError("The specified file is not a zip or a tar file")
