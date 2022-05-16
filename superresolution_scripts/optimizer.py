import tensorflow as tf


class Optimizer:
    def __init__(self, optimizer="adam", learning_rate=1e-3,
                 epsilon=1e-7, beta_1=.9, beta_2=.999, amsgrad=False,
                 initial_accumulator_value=.1, momentum=.0, nesterov=False,
                 lr_scheduler=False, decay_steps=.5, decay_rate=100) -> None:

        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.amsgrad = amsgrad
        self.initial_accumulator_value = initial_accumulator_value
        self.momentum = momentum
        self.nesterov = nesterov
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        if optimizer == "adadelta":
            self.optimizer = tf.optimizers.Adadelta(
                learning_rate=self.learning_rate)
        elif optimizer == "adagrad":
            self.optimizer = tf.optimizers.Adagrad(learning_rate=self.learning_rate,
                                                   initial_accumulator_value=self.initial_accumulator_value,
                                                   epsilon=self.epsilon)
        elif optimizer == "adamax":
            self.optimizer = tf.keras.optimizers.Adamax(learning_rate=self.learning_rate,
                                                        epsilon=self.epsilon,
                                                        beta_1=self.beta_1,
                                                        beta_2=self.beta_2)
        elif optimizer == "sgd":
            self.optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum,
                                               nesterov=self.nesterov)
        else:
            self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate,
                                                epsilon=self.epsilon,
                                                beta_1=self.beta_1,
                                                beta_2=self.beta_2,
                                                amsgrad=self.amsgrad)

        if lr_scheduler:
            self.lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
                self.learning_rate, decay_steps=self.decay_steps,
                decay_rate=self.decay_rate)
        else:
            self.lr_scheduler = False

    def lr_decay(self, iteration):
        lr = self.lr_scheduler(iteration)
        self.optimizer.learning_rate = lr
