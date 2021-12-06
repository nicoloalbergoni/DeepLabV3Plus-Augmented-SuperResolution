import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.utils import get_file
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, ZeroPadding2D, Input, DepthwiseConv2D, Add, GlobalAveragePooling2D, Concatenate, Activation, Reshape
from tensorflow.keras.utils import get_source_inputs

WEIGHTS_PATH_X = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"


class DeeplabV3Plus():
    def __init__(self, weights='pascal_voc', input_tensor=None, input_shape=(512, 512, 3), classes=21, OS=16,
                 last_activation=None, load_weights=True, reshape_outputs=False):

        if not (weights in {'pascal_voc', None}):
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization) or `pascal_voc`'
                             '(pre-trained on PASCAL VOC)')

        if not (last_activation in {"softmax", "sigmoid", None}):
            raise ValueError(
                "The last_activation parameter must be either None, softmax or sigmoid")

        self.weights = weights
        self.input_shape = input_shape
        self.classes = classes
        self.last_activation = last_activation
        self.load_weights = load_weights
        self.reshape_outputs = reshape_outputs

        if OS == 8:
            self.entry_block3_stride = 1
            self.middle_block_rate = 2  # ! Not mentioned in paper, but required
            self.exit_block_rates = (2, 4)
            self.atrous_rates = (12, 24, 36)
        else:
            self.entry_block3_stride = 2
            self.middle_block_rate = 1
            self.exit_block_rates = (1, 2)
            self.atrous_rates = (6, 12, 18)

        if input_tensor is None:
            self.img_input = Input(shape=input_shape)
        else:
            self.img_input = input_tensor

        self.input_tensor = input_tensor

    def build_model(self, only_DCNN_output=False):
        x, skip = self.EntryFlowBlock(self.img_input)
        x = self.MiddleFlowBlocks(x, block_number=16)
        x = self.ExitFlowBlock(x)
        x = self.AtrousSpatialPyramidPooling(x)

        x = self.Decoder(x, skip)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if self.input_tensor is not None:
            inputs = get_source_inputs(self.input_tensor)
        else:
            inputs = self.img_input

        if self.reshape_outputs:
            x = Reshape(
                (self.input_shape[0] * self.input_shape[1], self.classes))(x)

        if self.last_activation in {'softmax', 'sigmoid'}:
            x = Activation(self.last_activation)(x)

        model = Model(inputs, x, name='deeplabv3plus')

        if self.load_weights:
            if not os.path.exists("model"):
                os.mkdir("model")

            weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_X,
                                    cache_dir="model",
                                    cache_subdir=""
                                    )

            model.load_weights(weights_path, by_name=True)

        return model

    def EntryFlowBlock(self, img_input):
        x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), name='entry_flow_conv1_1',
                   use_bias=False, padding='same')(img_input)
        x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
        x = ReLU()(x)

        x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), name='entry_flow_conv1_2',
                   use_bias=False, padding='same')(x)
        x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
        x = ReLU()(x)

        x = self._Xception_block(x, [128, 128, 128], "entry_flow_block1", skip_connection_type="conv",
                                 last_stride=2, depth_activation=False, return_skip=False)

        x, skip = self._Xception_block(x, [256, 256, 256], "entry_flow_block2", skip_connection_type="conv",
                                       last_stride=2, depth_activation=False, return_skip=True)

        x = self._Xception_block(x, [728, 728, 728], "entry_flow_block3", skip_connection_type="conv",
                                 last_stride=self.entry_block3_stride, depth_activation=False, return_skip=False)

        return x, skip

    def MiddleFlowBlocks(self, x, block_number=16):
        for i in range(block_number):
            x = self._Xception_block(x, [728, 728, 728], f"middle_flow_unit_{i + 1}", skip_connection_type="sum",
                                     last_stride=1, rate=self.middle_block_rate, depth_activation=False, return_skip=False)

        return x

    def ExitFlowBlock(self, inputs):
        x = self._Xception_block(inputs, [728, 1024, 1024], "exit_flow_block1", skip_connection_type="conv",
                                 last_stride=1, rate=self.exit_block_rates[0], depth_activation=False, return_skip=False)

        x = self._Xception_block(x, [1536, 1536, 2048], "exit_flow_block2", skip_connection_type=None,
                                 last_stride=1, rate=self.exit_block_rates[1], depth_activation=True, return_skip=False)

        return x

    def AtrousSpatialPyramidPooling(self, inputs):
        input_features_shape = tf.keras.backend.int_shape(inputs)

        # Image Level Features Block
        image_pooling = GlobalAveragePooling2D(keepdims=True)(
            inputs)  # Output shape (BS, 1, 1, channels)
        image_pooling = Conv2D(256, (1, 1), padding='same',
                               use_bias=False, name='image_pooling')(image_pooling)
        image_pooling = BatchNormalization(
            name='image_pooling_BN', epsilon=1e-5)(image_pooling)
        image_pooling = ReLU()(image_pooling)
        image_pooling = tf.keras.layers.Resizing(
            *input_features_shape[1:3], interpolation="bilinear")(image_pooling)

        # 1x1 Conv Block
        b0 = Conv2D(256, (1, 1), padding='same',
                    use_bias=False, name="aspp0")(inputs)
        b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
        b0 = ReLU()(b0)

        # 3x3 Conv, Rate 6/12
        b1 = self._SepConv_BN(inputs, 256, "aspp1", stride=1, kernel_size=3,
                              rate=self.atrous_rates[0], depth_activation=True)
        # 3x3 Conv, Rate 12/24
        b2 = self._SepConv_BN(inputs, 256, "aspp2", stride=1, kernel_size=3,
                              rate=self.atrous_rates[1], depth_activation=True)
        # 3x3 Conv, Rate 18/36
        b3 = self._SepConv_BN(inputs, 256, "aspp3", stride=1, kernel_size=3,
                              rate=self.atrous_rates[2], depth_activation=True)

        output = Concatenate()([image_pooling, b0, b1, b2, b3])
        output = Conv2D(256, (1, 1), padding='same', use_bias=False,
                        name="concat_projection")(output)
        output = BatchNormalization(
            name="concat_projection_BN", epsilon=1e-5)(output)
        output = ReLU()(output)

        return output

    def Decoder(self, inputs, skip):
        # For input size of 512x512 skip_size is 128x128 as it corresponds to a x4 upsample of the encoder output feature
        # which for OS 16 is 32x32
        skip_size = tf.keras.backend.int_shape(skip)

        x = tf.keras.layers.Resizing(
            *skip_size[1:3], interpolation="bilinear")(inputs)

        decoder_skip = Conv2D(48, (1, 1), padding="same",
                              use_bias=False, name='feature_projection0')(skip)
        decoder_skip = BatchNormalization(
            name='feature_projection0_BN', epsilon=1e-5)(decoder_skip)
        decoder_skip = ReLU()(decoder_skip)

        x = Concatenate()([x, decoder_skip])
        x = self._SepConv_BN(x, 256, 'decoder_conv0',
                             depth_activation=True, epsilon=1e-5)
        x = self._SepConv_BN(x, 256, 'decoder_conv1',
                             depth_activation=True, epsilon=1e-5)

        # Final Convolution for class prediction and upsampling
        if self.classes == 21 and self.weights == 'pascal_voc':
            last_layer_name = 'logits_semantic'
        else:
            last_layer_name = 'custom_logits_semantic'

        x = Conv2D(self.classes, (1, 1), padding='same',
                   name=last_layer_name)(x)
        x = tf.keras.layers.Resizing(
            *self.input_shape[0:2], interpolation="bilinear")(x)

        return x

    def _Xception_block(self, inputs, filter_list, prefix, skip_connection_type, last_stride,
                        rate=1, depth_activation=False, return_skip=False):
        """ Basic building block of modified Xception network
            Args:
                inputs: input tensor
                filter_list: number of filters in each SepConv layer. len(depth_list) == 3
                prefix: prefix before name
                skip_connection_type: one of {'conv','sum', None}
                last_stride: stride at last depthwise conv
                rate: atrous rate for depthwise convolution
                depth_activation: flag to use activation between depthwise & pointwise convs
                return_skip: flag to return additional tensor after 2 SepConvs for decoder
        """
        residual = inputs
        for i in range(3):
            residual = self._SepConv_BN(residual,
                                        filter_list[i],
                                        prefix + f'_separable_conv{i + 1}',
                                        stride=last_stride if i == 2 else 1,
                                        rate=rate,
                                        depth_activation=depth_activation)
            if i == 1:
                skip = residual

        if skip_connection_type == 'conv':
            shortcut = self._conv2d_same(inputs, filter_list[-1], prefix + '_shortcut',
                                         kernel_size=1,
                                         stride=last_stride)
            shortcut = BatchNormalization(
                name=prefix + '_shortcut_BN')(shortcut)
            outputs = Add()([residual, shortcut])

        elif skip_connection_type == 'sum':
            outputs = Add()([residual, inputs])

        elif skip_connection_type is None:
            outputs = residual

        if return_skip:
            return outputs, skip
        else:
            return outputs

    def _SepConv_BN(self, x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
        """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
        """
        if stride == 1:
            depth_padding = 'same'
        else:
            kernel_size_effective = kernel_size + \
                (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = ZeroPadding2D((pad_beg, pad_end))(x)
            depth_padding = 'valid'

        if not depth_activation:
            x = ReLU()(x)

        x = DepthwiseConv2D(kernel_size=(kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                            padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
        x = BatchNormalization(
            name=prefix + '_depthwise_BN', epsilon=epsilon)(x)

        if depth_activation:
            x = ReLU()(x)

        x = Conv2D(filters, (1, 1), padding='same',
                   use_bias=False, name=prefix + '_pointwise')(x)
        x = BatchNormalization(
            name=prefix + '_pointwise_BN', epsilon=epsilon)(x)

        if depth_activation:
            x = ReLU()(x)

        return x

    def _conv2d_same(self, x, filters, prefix, stride=1, kernel_size=3, rate=1):
        """Implements right 'same' padding for even kernel sizes
            Without this there is a 1 pixel drift when stride = 2
            Args:
                x: input tensor
                filters: num of filters in pointwise convolution
                prefix: prefix before name
                stride: stride at depthwise conv
                kernel_size: kernel size for depthwise convolution
                rate: atrous rate for depthwise convolution
        """
        if stride == 1:
            return Conv2D(filters,
                          (kernel_size, kernel_size),
                          strides=(stride, stride),
                          padding='same', use_bias=False,
                          dilation_rate=(rate, rate),
                          name=prefix)(x)
        else:
            kernel_size_effective = kernel_size + \
                (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = ZeroPadding2D((pad_beg, pad_end))(x)
            return Conv2D(filters,
                          (kernel_size, kernel_size),
                          strides=(stride, stride),
                          padding='valid', use_bias=False,
                          dilation_rate=(rate, rate),
                          name=prefix)(x)
