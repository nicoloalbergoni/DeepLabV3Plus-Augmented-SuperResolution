import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from superresolution_scripts.optimizer import Optimizer


@tf.function
def bilateral_tv(target_image, alpha=0.6, shift_factor=2):
    h_shifts = np.arange(-shift_factor, shift_factor + 1, step=1)
    v_shifts = np.arange(0, shift_factor + 1, step=1)
    pairs = [[h, v] for h in h_shifts for v in v_shifts]
    target_batched = tf.tile(target_image, [len(pairs), 1, 1, 1])

    shifted_batch = tfa.image.translate(target_batched, pairs)
    difference_batched = tf.math.subtract(target_batched, shifted_batch)
    l1_batched = tf.map_fn(fn=lambda image: tf.norm(
        image, ord=1), elems=difference_batched)
    alpha_batched = tf.pow(alpha, tf.cast(
        tf.reduce_sum(tf.abs(pairs), axis=1), tf.float32))
    final_btv_batched = tf.multiply(alpha_batched, l1_batched)

    return tf.reduce_sum(final_btv_batched)


class Superresolution:
    def __init__(self, lambda_df, lambda_tv, lambda_L2, lambda_L1, num_iter=200, num_aug=100, optimizer: Optimizer = None,
                 feature_size=(64, 64), output_size=(512, 512),  use_BTV=False, verbose=False, copy_dropout=0.0):

        self.lambda_df = lambda_df
        self.lambda_tv = lambda_tv
        self.lambda_L2 = lambda_L2
        self.lambda_L1 = lambda_L1

        self.num_iter = num_iter
        self.num_aug = num_aug
        self.optimizer = optimizer
        self.feature_size = feature_size
        self.output_size = output_size
        self.use_BTV = use_BTV
        self.verbose = verbose
        self.copy_dropout = copy_dropout

    @tf.function
    def loss_function(self, target_image, augmented_samples, angles, shifts, n_drop=0):

        if n_drop != 0:
            bool_mask = np.full(self.num_aug, fill_value=True)
            bool_mask[:n_drop] = False
            np.random.shuffle(bool_mask)
            augmented_samples = tf.boolean_mask(augmented_samples, bool_mask)
            angles = tf.boolean_mask(angles, bool_mask)
            shifts = tf.boolean_mask(shifts, bool_mask)

        # Augmentation operators

        # Use to make dimensions consistent with augmented_samples size
        # as in case of dropout size is different from num_aug
        target_batch_size = tf.shape(augmented_samples)[0]
        target_batched = tf.tile(target_image, [target_batch_size, 1, 1, 1])
        target_rot = tfa.image.rotate(
            target_batched, angles, interpolation="bilinear")
        target_aug = tfa.image.translate(
            target_rot, shifts, interpolation="bilinear")

        # Downsampling operator
        D_operator = tf.image.resize(
            target_aug, self.feature_size, name="downsampling")

        # Data fidelity term
        df = tf.reduce_sum(tf.math.squared_difference(
            D_operator, augmented_samples), name="data_fidelity")
        # df = tf.reduce_sum(tf.abs(tf.subtract(D_operator, augmented_samples))) # L1 norm version
        # df = tf.reduce_sum(
        #     tf.math.square(tf.norm(tf.subtract(D_operator, augmented_samples), ord=self.df_lp_norm))) # Lp norm squared

        # TV Term
        if self.use_BTV:
            tv = bilateral_tv(target_image)
        else:
            target_gradients = tf.image.image_gradients(target_image)
            tv = tf.reduce_sum(
                tf.add(tf.abs(target_gradients[0]), tf.abs(target_gradients[1])))

        L2_norm = tf.reduce_sum(tf.square(target_image))

        df_lambda = tf.scalar_mul(self.lambda_df, df)
        tv_lambda = tf.scalar_mul(self.lambda_tv, tv)
        L2_lambda = tf.scalar_mul(self.lambda_L2, L2_norm)

        # Loss definition
        partial_loss = tf.add(df_lambda, tv_lambda)
        loss = tf.add(partial_loss, L2_lambda)

        if self.lambda_L1 > 0.0:
            L1_term = tf.reduce_sum(tf.abs(target_image))
            L1_lambda = tf.scalar_mul(self.lambda_L1, L1_term)
            loss = tf.add(loss, L1_lambda)

        return loss

    def augmented_superresolution(self, augmented_copies, angles, shifts):

        if self.optimizer is None:
            raise Exception(
                "You must provide an instance of the Optimizer class to compute the augmented SR")

        # Variable for the target output image
        # target_image = tf.Variable(tf.zeros([1, self.output_size[0], self.output_size[1], 1]), name="Target_Image")

        # Initilizing the variabile with the first non augmented copy
        initial_value = tf.image.resize(
            augmented_copies[0], self.output_size)[tf.newaxis, :]
        target_image = tf.Variable(initial_value, name="Target_Image")

        trainable_vars = [target_image]

        n_drop = int(self.num_aug * self.copy_dropout)

        for i in range(self.num_iter):
            if self.optimizer.lr_scheduler:
                self.optimizer.lr_decay(i)

            # print(self.optimizer.optimizer.learning_rate)

            with tf.GradientTape() as tape:
                loss = self.loss_function(
                    target_image, augmented_copies, angles, shifts, n_drop=n_drop)

                if self.verbose and (i % 10 == 0 or i == self.num_iter - 1):
                    print(f"{i + 1}/{self.num_iter} -- loss = {loss}")

            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.optimizer.apply_gradients(
                zip(gradients, trainable_vars))

        return target_image[0].numpy(), loss

    def max_superresolution(self, augmented_copies, angles, shifts):
        augmented_copies_upsampled = tf.image.resize(
            augmented_copies, size=self.output_size)
        augmented_copies_translated = tfa.image.translate(augmented_copies_upsampled,
                                                          -shifts,
                                                          interpolation="BILINEAR")
        augmented_copies_rotated = tfa.image.rotate(augmented_copies_translated,
                                                    -angles,
                                                    interpolation="BILINEAR")

        return tf.reduce_max(augmented_copies_rotated, axis=0).numpy(), None

    def mean_superresolution(self, augmented_copies, angles, shifts):
        augmented_copies_upsampled = tf.image.resize(
            augmented_copies, size=self.output_size)
        augmented_copies_translated = tfa.image.translate(augmented_copies_upsampled,
                                                          -shifts,
                                                          interpolation="BILINEAR")
        augmented_copies_rotated = tfa.image.rotate(augmented_copies_translated,
                                                    -angles,
                                                    interpolation="BILINEAR")

        return tf.reduce_mean(augmented_copies_rotated, axis=0).numpy(), None
