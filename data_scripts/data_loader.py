import tensorflow as tf
import random
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataLoader(object):
    """A TensorFlow Dataset API based loader for semantic segmentation problems."""

    def __init__(self, root_folder, image_size=(512, 512), mode="train", reshape_masks=False,
                 channels=(3, 3), crop_percent=None, seed=None, use_berkley=False,
                 augment=True, compose=False, one_hot_encoding=False, palette=None):
        """
        Initializes the data loader object
        Args:
            image_paths: List of paths of train images.
            mask_paths: List of paths of train masks (segmentation masks)
            image_size: Tuple, the final height, width of the loaded images.
            channels: Tuple of ints, first element is number of channels in images,
                      second is the number of channels in the mask image (needed to
                      correctly read the images into tensorflow and apply augmentations)
            crop_percent: Float in the range 0-1, defining percentage of image 
                          to randomly crop.
            palette: A list of RGB pixel values in the mask. If specified, the mask
                     will be one hot encoded along the channel dimension.
            seed: An int, if not specified, chosen randomly. Used as the seed for 
                  the RNG in the data pipeline.
        """
        self.root_folder = root_folder
        self.reshape_masks = reshape_masks
        self.mode = mode
        self.use_berkley = use_berkley
        self.palette = palette
        self.image_size = image_size
        self.augment = augment
        self.compose = compose
        self.one_hot_encoding = one_hot_encoding
        self.channels = channels

        self.image_paths = self._get_img_paths()

        if crop_percent is not None:
            if 0.0 < crop_percent <= 1.0:
                self.crop_percent = tf.constant(crop_percent, tf.float32)
            elif 0 < crop_percent <= 100:
                self.crop_percent = tf.constant(
                    crop_percent / 100., tf.float32)
            else:
                raise ValueError("Invalid value entered for crop size. Please use an \
                                  integer between 0 and 100, or a float between 0 and 1.0")
        else:
            self.crop_percent = None

        if seed is None:
            self.seed = random.randint(0, 1000)
        else:
            self.seed = seed

    def _get_img_paths(self):
        """
        Obtains the list of image base names that have been labelled for semantic segmentation.
        Images are stored in JPEG format, and segmentation ground truth in PNG format.

        :param filename: The dataset name, either 'train', 'val' or 'test'.
        :param dataset_path: The root path of the dataset.
        :return: The list of image base names for either the training, validation, or test set.
        """
        if self.use_berkley:
            file_list_path = os.path.join(os.getcwd(), "data", "augmented_file_lists",
                                "trainaug.txt" if self.mode == "train" else "valaug.txt")
        else:
            file_list_path = os.path.join(self.root_folder, "ImageSets", "Segmentation",
                                    "train.txt" if self.mode == "train" else "val.txt")

        return [os.path.join(self.root_folder, "JPEGImages", line.rstrip() + ".jpg") for line in open(file_list_path)]

    def _corrupt_brightness(self, image, mask):
        """
        Radnomly applies a random brightness change.
        """
        cond_brightness = tf.cast(tf.random.uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_brightness, lambda: tf.image.random_brightness(
            image, 0.1), lambda: tf.identity(image))
        return image, mask

    def _corrupt_contrast(self, image, mask):
        """
        Randomly applies a random contrast change.
        """
        cond_contrast = tf.cast(tf.random.uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_contrast, lambda: tf.image.random_contrast(
            image, 0.1, 0.8), lambda: tf.identity(image))
        return image, mask

    def _corrupt_saturation(self, image, mask):
        """
        Randomly applies a random saturation change.
        """
        cond_saturation = tf.cast(tf.random.uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_saturation, lambda: tf.image.random_saturation(
            image, 0.1, 0.8), lambda: tf.identity(image))
        return image, mask

    def _crop_random(self, image, mask):
        """
        Randomly crops image and mask in accord.
        """
        cond_crop_image = tf.cast(tf.random.uniform(
            [], maxval=2, dtype=tf.int32, seed=self.seed), tf.bool)

        shape = tf.cast(tf.shape(image), tf.float32)
        h = tf.cast(shape[0] * self.crop_percent, tf.int32)
        w = tf.cast(shape[1] * self.crop_percent, tf.int32)

        comb_tensor = tf.concat([image, mask], axis=2)

        comb_tensor = tf.cond(cond_crop_image, lambda: tf.image.random_crop(
            comb_tensor, [h, w, self.channels[0] + self.channels[1]], seed=self.seed), lambda: tf.identity(comb_tensor))
        image, mask = tf.split(
            comb_tensor, [self.channels[0], self.channels[1]], axis=2)

        return image, mask

    def _flip_left_right(self, image, mask):
        """
        Randomly flips image and mask left or right in accord.
        """
        comb_tensor = tf.concat([image, mask], axis=2)
        comb_tensor = tf.image.random_flip_left_right(
            comb_tensor, seed=self.seed)
        image, mask = tf.split(
            comb_tensor, [self.channels[0], self.channels[1]], axis=2)

        return image, mask

    def _resize_data(self, image, mask):
        """
        Resizes images to specified size.
        """
        image = tf.image.resize(image, self.image_size)
        mask = tf.image.resize(mask, self.image_size, method="nearest")

        if self.reshape_masks:
            mask = tf.reshape(
                mask, (self.image_size[0] * self.image_size[1], self.channels[-1]))

        return image, mask

    def _normalize(self, image, mask):
        image = tf.cast(image, tf.float32) / 255.0

        return image, mask

    def _parse_data(self, image_path):
        """
        Reads image and mask files depending on
        specified exxtension.
        """
        image_content = tf.io.read_file(image_path)
        mask_path = tf.strings.regex_replace(
            image_path, "JPEGImages", "SegmentationClassRaw" if not self.use_berkley else "SegmentationClassAug")
        mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
        mask_content = tf.io.read_file(mask_path)

        image = tf.image.decode_jpeg(image_content, channels=self.channels[0])
        mask = tf.image.decode_jpeg(mask_content, channels=self.channels[1])

        return image, mask

    def _one_hot_encode(self, image, mask):
        """
        Converts mask to a one-hot encoding specified by the semantic map.
        """
        one_hot_map = []
        for colour in self.palette:
            class_map = tf.reduce_all(tf.equal(mask, colour), axis=-1)
            one_hot_map.append(class_map)
        one_hot_map = tf.stack(one_hot_map, axis=-1)
        one_hot_map = tf.cast(one_hot_map, tf.float32)

        return image, one_hot_map

    @tf.function
    def _map_function(self, images_path):
        image, mask = self._parse_data(images_path)

        def _augmentation_func(image_f, mask_f):
            if self.augment:
                if self.compose:
                    image_f, mask_f = self._corrupt_brightness(image_f, mask_f)
                    image_f, mask_f = self._corrupt_contrast(image_f, mask_f)
                    image_f, mask_f = self._corrupt_saturation(image_f, mask_f)
                    image_f, mask_f = self._crop_random(image_f, mask_f)
                    image_f, mask_f = self._flip_left_right(image_f, mask_f)
                else:
                    options = [self._corrupt_brightness,
                               self._corrupt_contrast,
                               self._corrupt_saturation,
                               self._crop_random,
                               self._flip_left_right]
                    augment_func = random.choice(options)
                    image_f, mask_f = augment_func(image_f, mask_f)

            if self.one_hot_encoding:
                if self.palette is None:
                    raise ValueError('No Palette for one-hot encoding specified in the data loader! \
                                      please specify one when initializing the loader.')
                image_f, mask_f = self._one_hot_encode(image_f, mask_f)

            image_f, mask_f = self._resize_data(image_f, mask_f)
            image_f, mask_f = self._normalize(image_f, mask_f)

            return image_f, mask_f
        return tf.py_function(_augmentation_func, [image, mask], [tf.float32, tf.uint8])

    def data_batch(self, batch_size, shuffle=False):
        """
        Reads data, normalizes it, shuffles it, then batches it, returns a
        the next element in dataset op and the dataset initializer op.
        Inputs:
            batch_size: Number of images/masks in each batch returned.
            augment: Boolean, whether to augment data or not.
            shuffle: Boolean, whether to shuffle data in buffer or not.
            one_hot_encode: Boolean, whether to one hot encode the mask image or not.
                            Encoding will done according to the palette specified when
                            initializing the object.
        Returns:
            data: A tf dataset object.
        """

        # Create dataset out of the 2 files:
        data = tf.data.Dataset.from_tensor_slices(self.image_paths)

        if shuffle:
            data = data.shuffle(len(self.image_paths))

        # Parse images and labels
        data = data.map(self._map_function, num_parallel_calls=AUTOTUNE)

        data = data.batch(batch_size).prefetch(2)

        return data
