import os
import tensorflow as tf
from utils import plot_prediction, sparse_accuracy_ignoring_last_label, sparse_crossentropy_ignoring_last_label, Jaccard
from data_scripts.pascal_voc_dataset import PascalVOC2012Dataset
from tensorflow.keras.optimizers import Adam
from model import Deeplabv3


BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
PASCAL_ROOT = os.path.join(DATA_DIR, "VOCdevkit", "VOC2012")
TF_RECORD_DIR = os.path.join(DATA_DIR, "TFRecords")

AUGMENTATION_PARAMS = {'saturation_range': (-20, 20), 'value_range': (-20, 20),
                       'brightness_range': None, 'contrast_range': None, 'blur_params': None,
                       'flip_lr': True, 'rotation_range': (-10, 10), 'shift_range': (32, 32),
                       'zoom_range': (0.5, 2.0), 'ignore_label': 21}

dataset_obj = PascalVOC2012Dataset(augmentation_params=AUGMENTATION_PARAMS)

train_dataset = dataset_obj.load_dataset(
    TF_RECORD_DIR, is_training=True, batch_size=1)
val_dataset = dataset_obj.load_dataset(
    TF_RECORD_DIR, is_training=False, batch_size=1)

for images, masks in train_dataset.take(2):
    sample_image, sample_mask = images[0], masks[0]
    #plot_prediction([sample_image, sample_mask])


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # clear_output(wait=True)
        plot_prediction([sample_image, sample_mask])
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))


model = Deeplabv3(input_shape=(512, 512, 3),
                  classes=21, OS=16, load_weights=True)


losses = sparse_crossentropy_ignoring_last_label
metrics = [Jaccard, sparse_accuracy_ignoring_last_label]

model.compile(optimizer=Adam(learning_rate=7e-4, epsilon=1e-8, decay=1e-6), sample_weight_mode="temporal",
              loss=losses, metrics=metrics, run_eagerly=True)


EPOCHS = 30

callbacks = [
    DisplayCallback(),
    tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=3, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model_unet.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          validation_data=val_dataset,
                          callbacks=callbacks)
