import tensorflow as tf
from tensorflow import keras

class RevIN(keras.layers.Layer):
    """Reversible Instance Normalization for Accurate Time-Series Forecasting
       against Distribution Shift, ICLR2022.

    Parameters
    ----------
    eps: float, a value added for numerical stability, default 1e-5.
    affine: bool, if True(default), RevIN has learnable affine parameters.
    """
    def __init__(self, eps=1e-5, affine=True, **kwargs):
        super(RevIN, self).__init__(**kwargs)
        self.eps = eps
        self.affine = affine

    def build(self, input_shape):
        self.affine_weight = self.add_weight(name='affine_weight',
                                 shape=(1, input_shape[-1]),
                                 initializer='ones',
                                 trainable=self.affine)

        self.affine_bias = self.add_weight(name='affine_bias',
                                 shape=(1, input_shape[-1]),
                                 initializer='zeros',
                                 trainable=self.affine)
        super(RevIN, self).build(input_shape)

    def call(self, inputs, mode):
        if mode == 'norm':
            self._get_statistics(inputs)
            x = self._normalize(inputs)
        elif mode == 'denorm':
            x = self._denormalize(inputs)
        else:
            raise NotImplementedError('Only modes norm and denorm are supported.')
        return x

    def _get_statistics(self, x):
        self.mean = tf.stop_gradient(tf.reduce_mean(x, axis=1, keepdims=True))
        self.stdev = tf.stop_gradient(tf.math.sqrt(tf.math.reduce_variance(x, axis=1, keepdims=True)) + self.eps)

    def _normalize(self, x):
        xs = x - self.mean
        xs = xs / self.stdev
        
        if self.affine:
            xs = xs * self.affine_weight
            xs = xs + self.affine_bias
        return xs

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        
        x = x * self.stdev
        x = x + self.mean
        return x
    