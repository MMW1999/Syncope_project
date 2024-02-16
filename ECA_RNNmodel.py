import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Flatten, Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Flatten, Dense, MultiHeadAttention, Layer, GlobalAveragePooling1D, multiply)

import numpy as np
import math


class ResidualUnit(object):
    def __init__(self, n_samples_out, n_filters_out, kernel_initializer='he_normal',
             dropout_keep_prob=0.8, kernel_size=16, preactivation=True,
             postactivation_bn=False, activation_function='relu'):
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = 1 - dropout_keep_prob
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function

    def _skip_connection(self, y, downsample, n_filters_in):
        """Implement skip connection."""

        # Deal with downsampling
        if downsample > 1:
            y = MaxPooling1D(downsample, strides=downsample, padding='same')(y)
        elif downsample == 1:
            y = y
        else:
            raise ValueError("Number of samples should always decrease.")
        # Deal with n_filters dimension increase
        if n_filters_in != self.n_filters_out:
            # This is one of the two alternatives presented in ResNet paper
            # Other option is to just fill the matrix with zeros.
            y = Conv1D(self.n_filters_out, 1, padding='same',
                       use_bias=False, kernel_initializer=self.kernel_initializer)(y)
        return y

    def _batch_norm_plus_activation(self, x):
        if self.postactivation_bn:
            x = Activation(self.activation_function)(x)
            x = BatchNormalization(center=False, scale=False)(x)
        else:
            x = BatchNormalization()(x)
            x = Activation(self.activation_function)(x)
        return x

    def __call__(self, inputs):
        """Residual unit."""
        x, y = inputs
        n_samples_in = y.shape[1]
        downsample = n_samples_in // self.n_samples_out
        n_filters_in = y.shape[2]
        y = self._skip_connection(y, downsample, n_filters_in)

        # 1st layer
        x = Conv1D(filters=self.n_filters_out, kernel_size=self.kernel_size, padding='same',
                   use_bias=False, kernel_initializer=self.kernel_initializer)(x)
        x = self._batch_norm_plus_activation(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)

        # 2nd layer
        x = Conv1D(filters=self.n_filters_out, kernel_size=self.kernel_size, strides=downsample,
                   padding='same', use_bias=False,
                   kernel_initializer=self.kernel_initializer)(x)
        if self.preactivation:
            x = Add()([x, y])  # Sum skip connection and main connection
            y = x
            x = self._batch_norm_plus_activation(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
        else:
            x = BatchNormalization()(x)
            x = Add()([x, y])  # Sum skip connection and main connection
            x = Activation(self.activation_function)(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
            y = x
        return [x, y]
    
def EfficientChannelAttention(input_layer, kernel_size=3, adaptive=False, name=None):
	"""ECA-Net: Efficient Channel Attention Block
     source tensorflow implementation; https://github.com/AndrzejMiskow/FER-with-Attention-and-Objective-Activation-Functions/blob/main/attention_modules.py, 
    Args:
      input_layer: input tensor
      kernel_size: integer, default: 3, size of the kernel for the convolution
      adaptive: bool, default false , set kernel size depending on the number of input channels
      name: string, block label
    Returns:
      Output A tensor for the ECA-Net attention block
    """
	if adaptive:
		b = 1
		gamma = 2
		channels = input_layer.shape[-1]
		kernel_size = int(abs((math.log2(channels) + b / gamma)))
		if (kernel_size % 2) == 0:
			kernel_size = kernel_size + 1
		else:
			kernel_size = kernel_size

	squeeze = GlobalAveragePooling1D(name=name + "_Squeeze_GlobalPooling")(input_layer)
	squeeze = tf.expand_dims(squeeze, axis=1)
	excitation = Conv1D(filters=1,
							   kernel_size=kernel_size,
							   padding='same',
                               #kernel_initializer='relu', 
							   use_bias=False,
							   name=name + "_Excitation_Conv_1D")(squeeze)

	excitation = tf.expand_dims(tf.transpose(excitation, [0, 2, 1]), 2)
	excitation = tf.math.sigmoid(excitation)

	output = multiply([input_layer, excitation])

	return output



def model(n_classes, kernel_size=16, dropout_keep_prob=0.8,
              learning_rate=0.001, kernel_initializer='he_normal',
              activation_function='relu', last_layer='softmax',
              loss_function='categorical_crossentropy', 
              metrics= [tf.keras.metrics.BinaryAccuracy(
                    name='accuracy', dtype=None, threshold=0.5),tf.keras.metrics.Recall(name='Recall'),
                    tf.keras.metrics.Precision(name='Precision'), tf.keras.metrics.AUC(
                    num_thresholds=200,
                    curve="ROC",
                    summation_method="interpolation",
                    name="AUC",
                    dtype=None,
                    thresholds=None,
                    multi_label=True,
                    label_weights=None,
                )]):    #sigmoid/sofmax

    signal = Input(shape=(5000,12), dtype=np.float64, name='signal')

    
    x = Conv1D(64, kernel_size, padding='same', use_bias=False,
               kernel_initializer=kernel_initializer)(signal)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    #x = EfficientChannelAttention(x, kernel_size=3, adaptive=True, name='ECA_module')

    x, y = ResidualUnit(1024, 128, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, x])
    #x = EfficientChannelAttention(x, kernel_size=3, adaptive=True, name='ECA_module')
    
    x, y = ResidualUnit(256, 196, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])
    #x = EfficientChannelAttention(x, kernel_size=3, adaptive=True, name='ECA_module')

    x, y = ResidualUnit(64, 256, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])
    #x = EfficientChannelAttention(x, kernel_size=3, adaptive=True, name='ECA_module')
                        
    x, _ = ResidualUnit(16, 320, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])
    
    x = EfficientChannelAttention(x, kernel_size=3, adaptive=True, name='ECA_module')
    #x = Add()([attention_output, x])
    
    x = Flatten()(x)

    #x = Dense(units=5000, activation='relu')(x)
    diagn = Dense(n_classes, activation=last_layer, kernel_initializer=kernel_initializer)(x)
    model = Model(signal, diagn, name='ResNet_Ribeiro')
    

    model.compile(optimizer=Adam(learning_rate), loss=loss_function, metrics=metrics)
    
    return model



    """Residual unit block (unidimensional).
    Parameters
    ----------
    n_samples_out: int
        Number of output samples.
    n_filters_out: int
        Number of output filters.
    kernel_initializer: str, optional
        Initializer for the weights matrices. See Keras initializers. By default it uses
        'he_normal'.
    dropout_keep_prob: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. Default is 17.
    preactivation: bool, optional
        When preactivation is true use full preactivation architecture proposed
        in [1]. Otherwise, use architecture proposed in the original ResNet
        paper [2]. By default it is true.
    postactivation_bn: bool, optional
        Defines if you use batch normalization before or after the activation layer (there
        seems to be some advantages in some cases:
        https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md).
        If true, the batch normalization is used before the activation
        function, otherwise the activation comes first, as it is usually done.
        By default it is false.
    activation_function: string, optional
        Keras activation function to be used. By default 'relu'.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027 [cs], Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """