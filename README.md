# Augmented Super-Resolution on DeepLabV3+ model

This repository contains the code of the **Augmented Super-Resolution (ASR)** framework which refers to the master thesis project titled "*Augmented Super-Resolution: a novel framework for semantic segmentation using Test Time Augmentation*". The idea and the Super-Resolution implementation is based on [Augmented Grad-CAM](https://github.com/diegocarrera89/AugmentedGradCAM) while the Tensorflow 2 implementation of the DeepLabV3+ model is based on the code provided by [bonlime](https://github.com/bonlime/keras-deeplab-v3-plus).

## Note about this version of Augmented Super-Resolution
In the current implementation of Augmented Super-Reesolution, the framework works only in the single class case due to a limitation which is discussed in details in the thesis. Hence when a class id is specified we are referring to the selected class id for the current configuration.

## Setup

Clone the repository and create a virtual-environment.

```bash
git clone https://github.com/nicoloalbergoni/DeepLabV3Plus-Augmented-SuperResolution
cd DeepLabV3Plus-Augmented-SuperResolution
virtualenv -p python3 venv
```

Activate the environment and install the required libraries contained in `configs/requirements.txt`.

```bash
source venv/bin/activate
pip install -r configs/requirements.txt
```

## Execution on the PASCAL VOC 2012 dataset

### Download and prepare the data

To correctly set-up the PASCAL VOC 2012 dataset (augmented with the data provided by the Berkley University) execute the `download_and_prepare_voc.py` script. 
```bash
python download_and_prepare_voc.py --remove_cmap --download_berkley
```
This will download, extract and prepare the dataset for all the following operations. The dataset can be found under the `data/dataset_root` folder.

### Precompute the augmented feature maps and the standard output

To precompute the augmented features maps from the dataset's images, run the `generate_augmented_copies.py` script like in the following example
```bash
python generate_augmented_copies.py --num_aug 100 --num_samples 500 --mode argmax --angle_max 0.15 --shift_max 80 --class_id 8
```
This will take 1000 random samples of class 8 from the dataset, and for each image it will compute 100 augmented feature maps by applying random rotation/translation using as range ± 0.15 radians and ± 80 pixels respectively. In this example the augmented feature maps are computed using the argmax OPM, at the end all the data required for Super-Reesolution are saved in a hdf5 file under `data/superres_root/augmented_copies`.

Moreover, by running the `generate_standard_output.py` script you can compute the output of the standard DeepLabV3+ model (i.e. without modifications) in order to have data that can be used for comparisons against Augmented Super-Resolution. 

### Compute the Augmented Super-Resolution procedure

Finally to solve the Super-Resolution problem starting from the precomputed augmented feature maps, run the `SR_single_class.py` script as
```bash
python SR_single_class.py
```
The script outputs the final segmentation results under the `data/superres_root/superres_output` folder. 
Note that in this case the parameters that control the behaviour of this script can be directly modified inside the code.

## Example test script
You can also execute the `test_SR.py` script to test the full behaviour of the Augmented Super-Resolution framework in a single run on a sample image provided in the `test_image` folder.