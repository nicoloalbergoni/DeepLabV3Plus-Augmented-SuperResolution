import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


@tf.function
def bilateral_tv(target_image, alpha=0.6, shift_factor=3):
    h_shifts = np.arange(-shift_factor, shift_factor + 1, step=1)
    v_shifts = np.arange(0, shift_factor + 1, step=1)
    pairs = [[h, v] for h in h_shifts for v in v_shifts]
    target_batched = tf.tile(target_image, [len(pairs), 1, 1, 1])

    shifted_batch = tfa.image.translate(target_batched, pairs)
    difference_batched = tf.math.subtract(target_batched, shifted_batch)
    l1_batched = tf.map_fn(fn=lambda image: tf.norm(image, ord=1), elems=difference_batched)
    alpha_batched = tf.pow(alpha, tf.cast(tf.reduce_sum(tf.abs(pairs), axis=1), tf.float32))
    final_btv_batched = tf.multiply(alpha_batched, l1_batched)

    return tf.reduce_sum(final_btv_batched)


class Superresolution:
    def __init__(self, lambda_df, lambda_tv, lambda_L2, lambda_L1=0.0, num_iter=200, learning_rate=1e-3,
                 optimizer="adam", feature_size=(64, 64), output_size=(512, 512), num_aug=100, use_BTV=False,
                 verbose=False, df_lp_norm=2.0, lr_scheduler=False, optimizer_params=None, copy_dropout=0.0):

        self.lambda_df, self.lambda_tv, self.lambda_L2, self.lambda_L1 = Superresolution.__normalize_coefficients(
            0.0, lambda_tv,
            lambda_L2, lambda_L1)
        self.lambda_df = 1.0

        # self.lambda_df = lambda_df
        # self.lambda_tv = lambda_tv
        # self.lambda_L2 = lambda_L2
        # self.lambda_L1 = lambda_L1

        self.num_iter = num_iter
        self.num_aug = num_aug
        self.output_size = output_size
        self.feature_size = feature_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.optimizer = optimizer
        self.df_lp_norm = df_lp_norm
        self.lr_scheduler = lr_scheduler
        self.optimizer_params = optimizer_params
        self.copy_dropout = copy_dropout
        self.use_BTV = use_BTV

    @staticmethod
    def __normalize_coefficients(lambda_df, lambda_tv, lambda_L2, lambda_L1):
        coeff_list = [lambda_df, lambda_tv, lambda_L2, lambda_L1]
        normalized_coeff = np.array(coeff_list / np.sum(coeff_list))

        return tuple(normalized_coeff)

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
        target_rot = tfa.image.rotate(target_batched, angles, interpolation="bilinear")
        target_aug = tfa.image.translate(target_rot, shifts, interpolation="bilinear")

        # Downsampling operator
        D_operator = tf.image.resize(target_aug, self.feature_size, name="downsampling")

        # Data fidelity term
        df = tf.reduce_sum(tf.math.squared_difference(D_operator, augmented_samples), name="data_fidelity")
        # df = tf.reduce_sum(
        #     tf.math.square(tf.norm(tf.subtract(D_operator, augmented_samples), ord=self.df_lp_norm)))  # Lp norm squared

        # df = tf.reduce_sum(tf.abs(tf.subtract(D_operator, augmented_samples)))

        # TV Term
        if self.use_BTV:
            tv = bilateral_tv(target_image)
        else:
            target_gradients = tf.image.image_gradients(target_image)
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
            optimizer = tf.optimizers.Adagrad(learning_rate=self.learning_rate,
                                              initial_accumulator_value=self.optimizer_params[
                                                  "initial_accumulator_value"],
                                              epsilon=self.optimizer_params["epsilon"])
        elif self.optimizer == "adamax":
            optimizer = tf.keras.optimizers.Adamax(learning_rate=self.learning_rate,
                                                   epsilon=self.optimizer_params["epsilon"],
                                                   beta_1=self.optimizer_params["beta_1"],
                                                   beta_2=self.optimizer_params["beta_2"])
        elif self.optimizer == "sgd":
            optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.optimizer_params["momentum"],
                                          nesterov=self.optimizer_params["nesterov"])
        else:
            optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate,
                                           epsilon=self.optimizer_params["epsilon"],
                                           beta_1=self.optimizer_params["beta_1"],
                                           beta_2=self.optimizer_params["beta_2"],
                                           amsgrad=self.optimizer_params["amsgrad"])

        if self.lr_scheduler:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                self.learning_rate, decay_steps=self.optimizer_params["decay_steps"],
                decay_rate=self.optimizer_params["decay_rate"]
            )

        # Variable for the target output image
        target_image = tf.Variable(tf.zeros([1, self.output_size[0], self.output_size[1], 1]), name="Target_Image")
        trainable_vars = [target_image]

        n_drop = int(self.num_aug * self.copy_dropout)

        for i in range(self.num_iter):
            # optimizer.minimize(lambda: self.loss_function(target_image, augmented_samples), var_list=[target_image])
            if self.lr_scheduler:
                lr = lr_schedule(i)
                optimizer.learning_rate = lr

            with tf.GradientTape() as tape:
                loss = self.loss_function(target_image, augmented_samples, angles, shifts, n_drop=n_drop)

                if self.verbose and (i % 10 == 0 or i == self.num_iter - 1):
                    print(f"{i + 1}/{self.num_iter} -- loss = {loss}")

            gradients = tape.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(gradients, trainable_vars))

        return target_image, loss
