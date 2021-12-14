import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.utils import get_file
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, ZeroPadding2D, Input, DepthwiseConv2D, Add, \
    GlobalAveragePooling2D, Concatenate, Activation, Reshape
from tensorflow.keras.utils import get_source_inputs

WEIGHTS_PATH_XCEPTION = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_MOBILE = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"


class DeeplabV3Plus:
    def __init__(self, weights='pascal_voc', input_tensor=None, input_shape=(512, 512, 3), classes=21, OS=16,
                 last_activation=None, load_weights=True, reshape_outputs=False, backbone="xception", alpha=1.):

        if not (weights in {'pascal_voc', None}):
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization) or `pascal_voc`'
                             '(pre-trained on PASCAL VOC)')

        if not (last_activation in {"softmax", "sigmoid", None}):
            raise ValueError(
                "The last_activation parameter must be either None, softmax or sigmoid")

        if not (backbone in {"xception", "mobilenet"}):
            raise ValueError("Backbone must be either xception or mobilenet")

        self.weights = weights
        self.input_shape = input_shape
        self.classes = classes
        self.last_activation = last_activation
        self.load_weights = load_weights
        self.reshape_outputs = reshape_outputs
        self.backbone = backbone
        self.alpha = alpha

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

    def build_model(self, only_DCNN_output=False, only_ASPP_output=False, first_upsample_size=(128, 128)):

        if only_DCNN_output is True and only_ASPP_output is True:
            raise ValueError("Both only_DCNN_output and only_ASPP_output cannot be True at \
                            the same time")

        if self.backbone == "xception":
            entry_flow_output, skip = self.EntryFlowBlock(self.img_input)
            middle_flow_output = self.MiddleFlowBlocks(
                entry_flow_output, block_number=16)
            encoder_output = self.ExitFlowBlock(middle_flow_output)

        else:
            #TODO: Handle skip definition in case of mobilenet backbone
            skip = None
            entry_block_output = self.EntryBlockMobile(self.img_input)
            encoder_output = self.MobileNet_Backbone_Encoder(entry_block_output)

        ASPP_output = self.AtrousSpatialPyramidPooling(encoder_output)

        model_name_prefix = f"DLV3Plus-{self.backbone}"

        if only_DCNN_output:
            final_output = self.Decoder_only_DCNN(
                encoder_output, first_upsample_size)
            model_name = model_name_prefix + "-Only_DCNN_Output"
        elif only_ASPP_output:
            final_output = self.Decoder_only_ASPP(
                ASPP_output, first_upsample_size)
            model_name = model_name_prefix + "-Only_ASPP_Output"
        else:
            final_output = self.Decoder(ASPP_output, skip)
            model_name = model_name_prefix

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if self.input_tensor is not None:
            inputs = get_source_inputs(self.input_tensor)
        else:
            inputs = self.img_input

        if self.reshape_outputs:
            final_output = Reshape(
                (self.input_shape[0] * self.input_shape[1], self.classes))(final_output)

        if self.last_activation in {'softmax', 'sigmoid'}:
            final_output = Activation(self.last_activation)(final_output)

        model = Model(inputs, final_output, name=model_name)

        if self.load_weights:
            if not os.path.exists("model"):
                os.mkdir("model")

            if self.backbone == "xception":
                weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
                                        WEIGHTS_PATH_XCEPTION,
                                        cache_dir="model",
                                        cache_subdir="")

            else:
                weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
                                        WEIGHTS_PATH_MOBILE,
                                        cache_dir="model",
                                        cache_subdir="")

            model.load_weights(weights_path, by_name=True, skip_mismatch=True)

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

        x = DeeplabV3Plus._Xception_block(x, [128, 128, 128], "entry_flow_block1", skip_connection_type="conv",
                                          last_stride=2, depth_activation=False, return_skip=False)

        x, skip = DeeplabV3Plus._Xception_block(x, [256, 256, 256], "entry_flow_block2", skip_connection_type="conv",
                                                last_stride=2, depth_activation=False, return_skip=True)

        x = DeeplabV3Plus._Xception_block(x, [728, 728, 728], "entry_flow_block3", skip_connection_type="conv",
                                          last_stride=self.entry_block3_stride, depth_activation=False,
                                          return_skip=False)

        return x, skip

    def MiddleFlowBlocks(self, x, block_number=16):
        for i in range(block_number):
            x = DeeplabV3Plus._Xception_block(x, [728, 728, 728], f"middle_flow_unit_{i + 1}",
                                              skip_connection_type="sum",
                                              last_stride=1, rate=self.middle_block_rate, depth_activation=False,
                                              return_skip=False)

        return x

    def ExitFlowBlock(self, inputs):
        x = DeeplabV3Plus._Xception_block(inputs, [728, 1024, 1024], "exit_flow_block1", skip_connection_type="conv",
                                          last_stride=1, rate=self.exit_block_rates[0], depth_activation=False,
                                          return_skip=False)

        x = DeeplabV3Plus._Xception_block(x, [1536, 1536, 2048], "exit_flow_block2", skip_connection_type=None,
                                          last_stride=1, rate=self.exit_block_rates[1], depth_activation=True,
                                          return_skip=False)

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

        if self.backbone == "xception":
            # 3x3 Conv, Rate 6/12
            b1 = DeeplabV3Plus._SepConv_BN(inputs, 256, "aspp1", stride=1, kernel_size=3,
                                           rate=self.atrous_rates[0], depth_activation=True)
            # 3x3 Conv, Rate 12/24
            b2 = DeeplabV3Plus._SepConv_BN(inputs, 256, "aspp2", stride=1, kernel_size=3,
                                           rate=self.atrous_rates[1], depth_activation=True)
            # 3x3 Conv, Rate 18/36
            b3 = DeeplabV3Plus._SepConv_BN(inputs, 256, "aspp3", stride=1, kernel_size=3,
                                           rate=self.atrous_rates[2], depth_activation=True)

            output = Concatenate()([image_pooling, b0, b1, b2, b3])
        else:
            output = Concatenate()([image_pooling, b0])

        output = Conv2D(256, (1, 1), padding='same', use_bias=False,
                        name="concat_projection")(output)
        output = BatchNormalization(
            name="concat_projection_BN", epsilon=1e-5)(output)
        output = ReLU()(output)

        return output

    def Decoder(self, x, skip):

        if self.backbone == "xception":
            # For input size of 512x512 skip_size is 128x128 as it corresponds to a x4 upsample of the encoder output feature
            # which for OS 16 is 32x32
            skip_size = tf.keras.backend.int_shape(skip)
            # Upsample ASPP output
            x = tf.keras.layers.Resizing(
                *skip_size[1:3], interpolation="bilinear")(x)

            # Reduce low-level features depth dimensionality
            decoder_skip = Conv2D(48, (1, 1), padding="same",
                                  use_bias=False, name='feature_projection0')(skip)
            decoder_skip = BatchNormalization(
                name='feature_projection0_BN', epsilon=1e-5)(decoder_skip)
            decoder_skip = ReLU()(decoder_skip)

            # Concatenate low-level features with ASPP ouput
            x = Concatenate()([x, decoder_skip])
            x = DeeplabV3Plus._SepConv_BN(x, 256, 'decoder_conv0',
                                          depth_activation=True, epsilon=1e-5)
            x = DeeplabV3Plus._SepConv_BN(x, 256, 'decoder_conv1',
                                          depth_activation=True, epsilon=1e-5)

        # Final Convolution for class prediction
        if self.classes == 21 and self.weights == 'pascal_voc':
            last_layer_name = 'logits_semantic'
        else:
            last_layer_name = 'custom_logits_semantic'

        x = Conv2D(self.classes, (1, 1), padding='same',
                   name=last_layer_name)(x)

        # Bilinear upsample to input shape
        x = tf.keras.layers.Resizing(
            *self.input_shape[0:2], interpolation="bilinear")(x)

        return x

    def Decoder_only_DCNN(self, inputs, first_upsample_size):

        # Reduce depth dimensionality of low level features
        x = Conv2D(48, (1, 1), padding="same",
                   use_bias=False, name='feature_projection0')(inputs)
        x = BatchNormalization(
            name='feature_projection0_BN', epsilon=1e-5)(x)
        x = ReLU()(x)

        # First bilinear upsample
        x = tf.keras.layers.Resizing(
            *first_upsample_size, interpolation="bilinear")(x)

        # Refine output
        x = DeeplabV3Plus._SepConv_BN(x, 256, 'decoder_conv0',
                                      depth_activation=True, epsilon=1e-5)
        x = DeeplabV3Plus._SepConv_BN(x, 256, 'decoder_conv1',
                                      depth_activation=True, epsilon=1e-5)

        # Final Convolution for class prediction
        if self.classes == 21 and self.weights == 'pascal_voc':
            last_layer_name = 'logits_semantic'
        else:
            last_layer_name = 'custom_logits_semantic'

        x = Conv2D(self.classes, (1, 1), padding='same',
                   name=last_layer_name)(x)

        # Bilinear upsample to input shape
        x = tf.keras.layers.Resizing(
            *self.input_shape[0:2], interpolation="bilinear")(x)

        return x

    def Decoder_only_ASPP(self, inputs, first_upsample_size):

        # Upsample ASPP output
        x = tf.keras.layers.Resizing(
            *first_upsample_size, interpolation="bilinear")(inputs)

        # Refine output
        x = DeeplabV3Plus._SepConv_BN(x, 256, 'decoder_conv0',
                                      depth_activation=True, epsilon=1e-5)
        x = DeeplabV3Plus._SepConv_BN(x, 256, 'decoder_conv1',
                                      depth_activation=True, epsilon=1e-5)

        # Final Convolution for class prediction
        if self.classes == 21 and self.weights == 'pascal_voc':
            last_layer_name = 'logits_semantic'
        else:
            last_layer_name = 'custom_logits_semantic'

        x = Conv2D(self.classes, (1, 1), padding='same',
                   name=last_layer_name)(x)

        # Bilinear upsample to input shape
        x = tf.keras.layers.Resizing(
            *self.input_shape[0:2], interpolation="bilinear")(x)

        return x

    def EntryBlockMobile(self, inputs):
        first_block_filters = DeeplabV3Plus._make_divisible(32 * self.alpha, 8)
        pointwise_conv_filters = int(16 * self.alpha)
        pointwise_filters = DeeplabV3Plus._make_divisible(pointwise_conv_filters, 8)
        prefix = "expanded_conv_"

        # First concolution
        x = Conv2D(first_block_filters,
                   kernel_size=3,
                   strides=(2, 2), padding='same', use_bias=False,
                   name='Conv' if self.input_shape[2] == 3 else 'Conv_')(inputs)
        x = BatchNormalization(
            epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
        x = Activation(tf.nn.relu6, name='Conv_Relu6')(x)

        # Depthwise
        x = DepthwiseConv2D(kernel_size=3, strides=1, activation=None,
                            use_bias=False, padding='same', name=prefix + 'depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'depthwise_BN')(x)
        x = Activation(tf.nn.relu6, name=prefix + 'depthwise_relu')(x)

        # Project
        x = Conv2D(pointwise_filters,
                   kernel_size=1, padding='same', use_bias=False, activation=None,
                   name=prefix + 'project')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)
        return x

    def MobileNet_Backbone_Encoder(self, inputs):
        x = DeeplabV3Plus._inverted_res_block(inputs, filters=24, alpha=self.alpha, stride=2,
                                              expansion_factor=6, block_id=1, skip_connection=False)
        x = DeeplabV3Plus._inverted_res_block(x, filters=24, alpha=self.alpha, stride=1,
                                              expansion_factor=6, block_id=2, skip_connection=True)

        x = DeeplabV3Plus._inverted_res_block(x, filters=32, alpha=self.alpha, stride=2,
                                              expansion_factor=6, block_id=3, skip_connection=False)
        x = DeeplabV3Plus._inverted_res_block(x, filters=32, alpha=self.alpha, stride=1,
                                              expansion_factor=6, block_id=4, skip_connection=True)
        x = DeeplabV3Plus._inverted_res_block(x, filters=32, alpha=self.alpha, stride=1,
                                              expansion_factor=6, block_id=5, skip_connection=True)

        # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
        x = DeeplabV3Plus._inverted_res_block(x, filters=64, alpha=self.alpha, stride=1,  # 1!
                                              expansion_factor=6, block_id=6, skip_connection=False)
        x = DeeplabV3Plus._inverted_res_block(x, filters=64, alpha=self.alpha, stride=1, rate=2,
                                              expansion_factor=6, block_id=7, skip_connection=True)
        x = DeeplabV3Plus._inverted_res_block(x, filters=64, alpha=self.alpha, stride=1, rate=2,
                                              expansion_factor=6, block_id=8, skip_connection=True)
        x = DeeplabV3Plus._inverted_res_block(x, filters=64, alpha=self.alpha, stride=1, rate=2,
                                              expansion_factor=6, block_id=9, skip_connection=True)

        x = DeeplabV3Plus._inverted_res_block(x, filters=96, alpha=self.alpha, stride=1, rate=2,
                                              expansion_factor=6, block_id=10, skip_connection=False)
        x = DeeplabV3Plus._inverted_res_block(x, filters=96, alpha=self.alpha, stride=1, rate=2,
                                              expansion_factor=6, block_id=11, skip_connection=True)
        x = DeeplabV3Plus._inverted_res_block(x, filters=96, alpha=self.alpha, stride=1, rate=2,
                                              expansion_factor=6, block_id=12, skip_connection=True)

        x = DeeplabV3Plus._inverted_res_block(x, filters=160, alpha=self.alpha, stride=1, rate=2,  # 1!
                                              expansion_factor=6, block_id=13, skip_connection=False)
        x = DeeplabV3Plus._inverted_res_block(x, filters=160, alpha=self.alpha, stride=1, rate=4,
                                              expansion_factor=6, block_id=14, skip_connection=True)
        x = DeeplabV3Plus._inverted_res_block(x, filters=160, alpha=self.alpha, stride=1, rate=4,
                                              expansion_factor=6, block_id=15, skip_connection=True)

        x = DeeplabV3Plus._inverted_res_block(x, filters=320, alpha=self.alpha, stride=1, rate=4,
                                              expansion_factor=6, block_id=16, skip_connection=False)

        return x

    @staticmethod
    def _Xception_block(inputs, filter_list, prefix, skip_connection_type, last_stride,
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
            residual = DeeplabV3Plus._SepConv_BN(residual,
                                                 filter_list[i],
                                                 prefix + f'_separable_conv{i + 1}',
                                                 stride=last_stride if i == 2 else 1,
                                                 rate=rate,
                                                 depth_activation=depth_activation)
            if i == 1:
                skip = residual

        if skip_connection_type == 'conv':
            shortcut = DeeplabV3Plus._conv2d_same(inputs, filter_list[-1], prefix + '_shortcut',
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

    @staticmethod
    def _inverted_res_block(inputs, expansion_factor, stride, alpha, filters, block_id, skip_connection, rate=1):
        in_channels = inputs.shape[-1]
        pointwise_conv_filters = int(filters * alpha)
        pointwise_filters = DeeplabV3Plus._make_divisible(pointwise_conv_filters, 8)
        prefix = f"expanded_conv_{block_id}_"

        # Expand
        x = Conv2D(expansion_factor * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand')(inputs)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        x = Activation(tf.nn.relu6, name=prefix + 'expand_relu')(x)

        # Depthwise
        x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                            use_bias=False, padding='same', dilation_rate=(rate, rate),
                            name=prefix + 'depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'depthwise_BN')(x)

        x = Activation(tf.nn.relu6, name=prefix + 'depthwise_relu')(x)

        # Project
        x = Conv2D(pointwise_filters,
                   kernel_size=1, padding='same', use_bias=False, activation=None,
                   name=prefix + 'project')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'project_BN')(x)

        if skip_connection:
            return Add(name=prefix + 'add')([inputs, x])

        return x

    @staticmethod
    def _SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
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

        x = DepthwiseConv2D(kernel_size=(kernel_size, kernel_size), strides=(stride, stride),
                            dilation_rate=(rate, rate),
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

    @staticmethod
    def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
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

    @staticmethod
    def _make_divisible(value, divisor, min_value=None):
        """
        Returns a value that is divisible bi divisor

        Used to ensure that all layers have a number of channels divisible by divisor
        """
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(value + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * value:
            new_v += divisor
        return new_v
