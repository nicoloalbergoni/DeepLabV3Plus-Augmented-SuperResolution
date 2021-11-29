import os
import random
import numpy as np
import cv2
from sklearn.utils import class_weight
from utils import _random_crop
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import Sequence

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


class SegmentationGenerator(Sequence):

    def __init__(self, root_folder='data/VOC2012', mode='train', n_classes=21, batch_size=1, resize_shape=None,
                 seed=7, crop_shape=None, horizontal_flip=True, blur=0, vertical_flip=0, brightness=0.1, rotation=5.0, zoom=0.1, do_ahisteq=True):

        self.root_folder = root_folder
        self.blur = blur
        self.histeq = do_ahisteq
        self.mode = mode
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.brightness = brightness
        self.rotation = rotation
        self.zoom = zoom

        self.image_path_list = sorted(self._get_img_paths())

        np.random.seed(seed)

        # Preallocate memory
        if self.crop_shape:
            self.X = np.zeros(
                (batch_size, crop_shape[1], crop_shape[0], 3), dtype='float32')
            self.SW = np.zeros(
                (batch_size, crop_shape[1] * crop_shape[0]), dtype='float32')
            self.Y = np.zeros(
                (batch_size, crop_shape[1] * crop_shape[0], 1), dtype='float32')
            self.F = np.zeros(
                (batch_size, crop_shape[1] * crop_shape[0], 1), dtype='float32')
            self.F_SW = np.zeros(
                (batch_size, crop_shape[1] * crop_shape[0]), dtype='float32')
        elif self.resize_shape:
            self.X = np.zeros(
                (batch_size, resize_shape[1], resize_shape[0], 3), dtype='float32')
            self.SW = np.zeros(
                (batch_size, resize_shape[1] * resize_shape[0]), dtype='float32')
            self.Y = np.zeros(
                (batch_size, resize_shape[1] * resize_shape[0], 1), dtype='float32')
            self.F = np.zeros(
                (batch_size, resize_shape[1] * resize_shape[0], 1), dtype='float32')
            self.F_SW = np.zeros(
                (batch_size, resize_shape[1] * resize_shape[0]), dtype='float32')
        else:
            raise Exception('No image dimensions specified!')

    def _get_img_paths(self):
        """
        Obtains the list of image base names that have been labelled for semantic segmentation.
        Images are stored in JPEG format, and segmentation ground truth in PNG format.

        :param filename: The dataset name, either 'train', 'val' or 'test'.
        :param dataset_path: The root path of the dataset.
        :return: The list of image base names for either the training, validation, or test set.
        """
        filename = os.path.join(self.root_folder, "ImageSets", "Segmentation",
                                "train.txt" if self.mode == "train" else "val.txt")
        return [os.path.join(self.root_folder, "JPEGImages", line.rstrip() + ".jpg") for line in open(filename)]

    def __len__(self):
        return len(self.image_path_list) // self.batch_size

    def __getitem__(self, i):

        for n, image_path in enumerate(self.image_path_list[i * self.batch_size:(i + 1) * self.batch_size]):

            image = cv2.imread(image_path, 1)
            label_path = image_path.replace(
                "JPEGImages", "SegmentationClass").replace("jpg", "png")
            label = cv2.imread(label_path, 0)
            labels = np.unique(label)

            if self.blur and random.randint(0, 1):
                image = cv2.GaussianBlur(image, (self.blur, self.blur), 0)

            if self.resize_shape and not self.crop_shape:
                image = cv2.resize(image, self.resize_shape)
                label = cv2.resize(label, self.resize_shape,
                                   interpolation=cv2.INTER_NEAREST)

            if self.crop_shape:
                image, label = _random_crop(image, label, self.crop_shape)

            # Do augmentation
            if self.horizontal_flip and random.randint(0, 1):
                image = cv2.flip(image, 1)
                label = cv2.flip(label, 1)
            if self.vertical_flip and random.randint(0, 1):
                image = cv2.flip(image, 0)
                label = cv2.flip(label, 0)
            if self.brightness:
                factor = 1.0 + random.gauss(mu=0.0, sigma=self.brightness)
                if random.randint(0, 1):
                    factor = 1.0/factor
                table = np.array(
                    [((i / 255.0) ** factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
                image = cv2.LUT(image, table)
            if self.rotation:
                angle = random.gauss(mu=0.0, sigma=self.rotation)
            else:
                angle = 0.0
            if self.zoom:
                scale = random.gauss(mu=1.0, sigma=self.zoom)
            else:
                scale = 1.0
            if self.rotation or self.zoom:
                M = cv2.getRotationMatrix2D(
                    (image.shape[1]//2, image.shape[0]//2), angle, scale)
                image = cv2.warpAffine(
                    image, M, (image.shape[1], image.shape[0]))
                label = cv2.warpAffine(
                    label, M, (label.shape[1], label.shape[0]))

            if self.histeq:  # and convert to RGB
                img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
                image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)  # to BGR

            label = label.astype('int32')
            for j in np.setxor1d(np.unique(label), labels):
                label[label == j] = self.n_classes

            y = label.flatten()
            y[y > (self.n_classes - 1)] = self.n_classes

            self.X[n] = image
            self.Y[n] = np.expand_dims(y, -1)

        return self.X, self.Y

        #     # get all pixels that aren't background
        #     self.F[n] = (self.Y[n] != 0).astype('float32')
        #     # get all pixels (bg and foregroud) that aren't void
        #     valid_pixels = self.F[n][self.Y[n] != self.n_classes]
        #     u_classes = np.unique(valid_pixels)
        #     class_weights = compute_class_weight(
        #         class_weight='balanced', classes=u_classes, y=valid_pixels)
        #     class_weights = {class_id: w for class_id,
        #                      w in zip(u_classes, class_weights)}
        #     if len(class_weights) == 1:  # no bg\no fg
        #         if 1 in u_classes:
        #             class_weights[0] = 0.
        #         else:
        #             class_weights[1] = 0.
        #     elif not len(class_weights):
        #         class_weights[0] = 0.
        #         class_weights[1] = 0.

        #     sw_valid = np.ones(y.shape)
        #     # background weights
        #     np.putmask(sw_valid, self.Y[n] == 0, class_weights[0])
        #     # foreground wegihts
        #     np.putmask(sw_valid, self.F[n], class_weights[1])
        #     np.putmask(sw_valid, self.Y[n] == self.n_classes, 0)
        #     self.F_SW[n] = sw_valid
        #     self.X[n] = image

        #     # Create adaptive pixels weights
        #     filt_y = y[y != self.n_classes]
        #     u_classes = np.unique(filt_y)
        #     if len(u_classes):
        #         class_weights = compute_class_weight(
        #             class_weight='balanced', classes=u_classes, y=filt_y)
        #         class_weights = {class_id: w for class_id,
        #                          w in zip(u_classes, class_weights)}
        #     class_weights[self.n_classes] = 0.
        #     for yy in u_classes:
        #         np.putmask(self.SW[n], y == yy, class_weights[yy])

        #     np.putmask(self.SW[n], y == self.n_classes, 0)

        # sample_dict = {'pred_mask': self.SW}
        # return self.X, self.Y, sample_dict

    def on_epoch_end(self):
        # Shuffle dataset for next epoch
        c = list(self.image_path_list)
        random.shuffle(c)
        self.image_path_list = c
