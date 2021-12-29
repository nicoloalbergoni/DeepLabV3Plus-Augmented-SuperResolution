import os
import sys
import zipfile
import tarfile
import urllib.request


def shorten(s, subs):
    i = s.index(subs)
    return s[:i+len(subs)]


def download_dataset(dataset_url, dest_folder):
    #TODO: handle other extension
    extension = ".zip" if ".zip" in dataset_url else ".tar"
    filename = shorten(dataset_url.split("/")[-1], extension)
    full_dest_path = os.path.join(dest_folder, filename)

    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)

    if os.path.exists(full_dest_path):
        print(f"File {filename} already in destination folder, skipping download")
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


def extract_file(filepath, dest_folder, is_extracted="./data/VOC2101"):

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
