import tensorflow as tf
from model import DeeplabV3Plus


def main(plot_model=False):

    model = DeeplabV3Plus(
        input_shape=(512, 512, 3),
        classes=21,
        OS=16,
        last_activation=None,
        load_weights=True,
        backbone="mobilenet",
        alpha=1.,
        reshape_outputs=False).build_model(only_DCNN_output=False, only_ASPP_output=False)

    if plot_model:
        tf.keras.utils.plot_model(
            model, to_file=f'./{model.name}.png', show_shapes=True, show_dtype=False,
            show_layer_names=True, rankdir='TB', show_layer_activations=False)


if __name__ == '__main__':
    main(plot_model=True)
