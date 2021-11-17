import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, ZeroPadding2D, Input, DepthwiseConv2D, Add, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.utils import get_source_inputs


def Deeplabv3(weights='pascal_voc', input_tensor=None, input_shape=(512, 512, 3), classes=21, OS=16, activation=None):

    if not (weights in {'pascal_voc', None}):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `pascal_voc`'
                         '(pre-trained on PASCAL VOC)')

    if OS == 8:
        entry_block3_stride = 1
        middle_block_rate = 2  # ! Not mentioned in paper, but required
        exit_block_rates = (2, 4)
        atrous_rates = (12, 24, 36)
    else:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        atrous_rates = (6, 12, 18)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    x = EntryFlowBlock(img_input, entry_block3_stride)
    x = MiddleFlowBlock(x, middle_block_rate, block_number=16)
    x = ExitFlowBlock(x, exit_block_rates)
    x = AtrousSpatialPyramidPooling(x, atrous_rates)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    if activation in {'softmax', 'sigmoid'}:
        x = tf.keras.layers.Activation(activation)(x)

    return Model(inputs, x, name='deeplabv3plus')


def EntryFlowBlock(img_input, entry_block3_stride):
    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), name='entry_flow_conv1_1',
               use_bias=False, padding='same')(img_input)
    x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
    x = ReLU()(x)

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), name='entry_flow_conv1_2',
               use_bias=False, padding='same')(x)
    x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
    x = ReLU()(x)

    x = Xception_block(x, [128, 128, 128], "entry_flow_block1", skip_connection_type="conv",
                       last_stride=2, depth_activation=False, return_skip=False)

    x = Xception_block(x, [256, 256, 256], "entry_flow_block2", skip_connection_type="conv",
                       last_stride=2, depth_activation=False, return_skip=False)

    x = Xception_block(x, [728, 728, 728], "entry_flow_block3", skip_connection_type="conv",
                       last_stride=entry_block3_stride, depth_activation=False, return_skip=False)

    return x


def MiddleFlowBlock(inputs, middle_block_rate, block_number=16):
    x = inputs
    for i in range(block_number):
        x = Xception_block(x, [728, 728, 728], f"middle_flow_unit_{i + 1}", skip_connection_type="sum",
                           last_stride=1, rate=middle_block_rate, depth_activation=False, return_skip=False)

    return x


def ExitFlowBlock(inputs, exit_block_rates):
    x = Xception_block(inputs, [728, 1024, 1024], "exit_flow_block1", skip_connection_type="conv",
                       last_stride=2, rate=exit_block_rates[0], depth_activation=False, return_skip=False)

    x = Xception_block(x, [1536, 1536, 2048], "exit_flow_block2", skip_connection_type=None,
                       last_stride=1, rate=exit_block_rates[1], depth_activation=True, return_skip=False)

    return x


def AtrousSpatialPyramidPooling(inputs, atrous_rates):
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
    b1 = SepConv_BN(inputs, 256, "aspp1", stride=1, kernel_size=3,
                    rate=atrous_rates[0], depth_activation=True)
    # 3x3 Conv, Rate 12/24
    b2 = SepConv_BN(inputs, 256, "aspp2", stride=1, kernel_size=3,
                    rate=atrous_rates[1], depth_activation=True)
    # 3x3 Conv, Rate 18/36
    b3 = SepConv_BN(inputs, 256, "aspp3", stride=1, kernel_size=3,
                    rate=atrous_rates[2], depth_activation=True)

    output = Concatenate()([b0, b1, b2, b3, image_pooling])
    output = Conv2D(256, (1, 1), padding='same', use_bias=False,
                    name="concat_projection")(output)
    output = BatchNormalization(
        name="concat_projection_BN", epsilon=1e-5)(output)
    output = ReLU()(output)

    return output


def Xception_block(inputs, filter_list, prefix, skip_connection_type, last_stride,
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
        residual = SepConv_BN(residual,
                              filter_list[i],
                              prefix + f'_separable_conv{i + 1}',
                              stride=last_stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual

    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, filter_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=last_stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = Add()([residual, shortcut])

    elif skip_connection_type == 'sum':
        outputs = Add()([residual, inputs])

    elif skip_connection_type == None:
        outputs = residual

    if return_skip:
        return outputs, skip
    else:
        return outputs


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
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
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = ReLU()(x)

    x = DepthwiseConv2D(kernel_size=(kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)

    if depth_activation:
        x = ReLU()(x)

    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)

    if depth_activation:
        x = ReLU()(x)

    return x


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
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
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
