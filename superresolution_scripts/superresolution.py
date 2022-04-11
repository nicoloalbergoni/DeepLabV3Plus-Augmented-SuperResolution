import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


class Superresolution:
    def __init__(self, lambda_df, lambda_tv, lambda_L2, lambda_L1=0.0, num_iter=200, learning_rate=1e-3,
                 optimizer="adam", feature_size=(64, 64), output_size=(512, 512), num_aug=100,
                 verbose=False, df_lp_norm=2.0, lr_scheduler=False):

        self.lambda_df, self.lambda_tv, self.lambda_L2, self.lambda_L1 = Superresolution.__normalize_coefficients(
            lambda_df, lambda_tv,
            lambda_L2, lambda_L1)
        self.num_iter = num_iter
        self.num_aug = num_aug
        self.output_size = output_size
        self.feature_size = feature_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.optimizer = optimizer
        self.df_lp_norm = df_lp_norm
        self.lr_scheduler = lr_scheduler

    @staticmethod
    def __normalize_coefficients(lambda_df, lambda_tv, lambda_L2, lambda_L1):
        coeff_list = [lambda_df, lambda_tv, lambda_L2, lambda_L1]
        normalized_coeff = np.array(coeff_list / np.sum(coeff_list))
        return tuple(normalized_coeff)

    @tf.function
    def loss_function(self, target_image, augmented_samples, angles, shifts):
        # Augmentation operators
        target_batched = tf.tile(target_image, [self.num_aug, 1, 1, 1])
        target_rot = tfa.image.rotate(target_batched, angles, interpolation="bilinear")
        target_aug = tfa.image.translate(target_rot, shifts, interpolation="bilinear")

        # Downsampling operator
        D_operator = tf.expand_dims(tf.image.resize(target_aug, self.feature_size, name="downsampling"), 0)
        # Image gradients
        target_gradients = tf.image.image_gradients(target_image)

        # Loss terms
        # df = tf.reduce_sum(tf.math.squared_difference(D_operator, augmented_samples), name="data_fidelity")

        df = tf.reduce_sum(tf.math.square(tf.norm(tf.subtract(D_operator, augmented_samples), ord=self.df_lp_norm)))

        tv = tf.reduce_sum(tf.add(tf.abs(target_gradients[0]), tf.abs(target_gradients[1])))
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

    def compute_output(self, augmented_samples, angles, shifts):

        if self.optimizer == "adadelta":
            optimizer = tf.optimizers.Adadelta(learning_rate=self.learning_rate)
        elif self.optimizer == "adagrad":
            optimizer = tf.optimizers.Adagrad(learning_rate=self.learning_rate)
        elif self.optimizer == "sgd":
            optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate, momentum=.9, nesterov=True)
        else:
            optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

        if self.lr_scheduler:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                self.learning_rate, decay_steps=300, decay_rate=0.9
            )

        # Variable for the target output image
        target_image = tf.Variable(tf.zeros([1, self.output_size[0], self.output_size[1], 1]), name="Target_Image")
        trainable_vars = [target_image]

        for i in range(self.num_iter):
            # optimizer.minimize(lambda: self.loss_function(target_image, augmented_samples), var_list=[target_image])
            if self.lr_scheduler:
                lr = lr_schedule(i)
                optimizer.learning_rate = lr

            with tf.GradientTape() as tape:
                loss = self.loss_function(target_image, augmented_samples, angles, shifts)

                if self.verbose and (i % 10 == 0 or i == self.num_iter - 1):
                    print(f"{i + 1}/{self.num_iter} -- loss = {loss}")

            gradients = tape.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(gradients, trainable_vars))

        return target_image, loss
