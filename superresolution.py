import tensorflow as tf
import tensorflow_addons as tfa


class Superresolution:
    def __init__(self, lambda_tv, lambda_eng, num_iter=200, learning_rate=1e-3,
                 feature_size=(64, 64), output_size=(512, 512), num_aug=100):
        self.num_iter = num_iter
        self.lambda_eng = lambda_eng
        self.lambda_tv = lambda_tv
        self.num_aug = num_aug
        self.output_size = output_size
        self.feature_size = feature_size
        self.learning_rate = learning_rate

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
        df = tf.reduce_sum(tf.math.squared_difference(D_operator, augmented_samples), name="data_fidelity")
        tv = tf.reduce_sum(tf.add(tf.abs(target_gradients[0]), tf.abs(target_gradients[1])))
        norm = tf.reduce_sum(tf.square(target_image))

        tv_lambda = tf.scalar_mul(self.lambda_tv, tv)
        norm_mu = tf.scalar_mul(self.lambda_eng, norm)

        # Loss definition
        partial_loss = tf.add(df, tv_lambda)
        loss = tf.add(partial_loss, norm_mu)
        return loss

    def compute_output(self, augmented_samples, angles, shifts):

        optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

        # Variable for the target output image
        target_image = tf.Variable(tf.zeros([1, self.output_size[0], self.output_size[1], 1]), name="Target_Image")
        trainable_vars = [target_image]

        # Optimizer
        for i in range(self.num_iter):
            # optimizer.minimize(lambda: self.loss_function(target_image, augmented_samples), var_list=[target_image])
            with tf.GradientTape() as tape:
                loss = self.loss_function(target_image, augmented_samples, angles, shifts)
                print(f"{i + 1}/{self.num_iter} -- loss = {loss}")

            gradients = tape.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(gradients, trainable_vars))

        return target_image
