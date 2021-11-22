import os
import sys
import zipfile
import urllib.request


def download_dataset(dataset_url, dest_folder):
    zip_filename = dataset_url.split("/")[-1]
    full_dest_path = os.path.join(dest_folder, zip_filename)

    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)

    if os.path.exists(full_dest_path):
        print("File already present in destination folder, skipping download")
        return full_dest_path

    def _progress(count, block_size, total_size):
        sys.stdout.write('\rDownloading %s %.1f%%' % (
            zip_filename, 100.0 * count * block_size / total_size))
        sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(
        dataset_url, full_dest_path, _progress)

    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', zip_filename, statinfo.st_size, 'bytes.')

    return filepath


def extract_zip_file(filepath, dest_folder):

    # TODO: Find a better way to understand when to skip unzip even for different datasets
    if os.path.exists(os.path.join(dest_folder, "VOC2012")):
        print("VOC dataset already extracted")
        return

    print("Extracting zip file...")
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)

    print("Finished extraction")
