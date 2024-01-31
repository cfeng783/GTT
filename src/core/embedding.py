import tensorflow as tf
from tensorflow import keras
import math
from ..utils.tf_utils import shape_list

  
class TSPositionalEncoding(keras.layers.Layer):
    def __init__(self, num_positions: int, embed_dim: int):
        super(TSPositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.

        position = tf.range(0, num_positions,dtype=tf.float32)
        position = tf.expand_dims(position, axis=1)
        
        div_term = tf.exp(tf.range(0, embed_dim, 2,dtype=tf.float32) * -(math.log(10000.0) / embed_dim))
        # print('pe',pe.shape)
        
        sin_t = tf.sin(position * div_term)
        cos_t = tf.cos(position * div_term)
        
        ps = []
        for i in range(sin_t.shape[1]):
            ps.append(sin_t[:,i])
            ps.append(cos_t[:,i])
        pe = tf.stack(ps,axis=1)
        pe = tf.expand_dims(pe, axis=0)
        self.pe = tf.stop_gradient( pe )

    def call(self, x):
        return self.pe[:, :x.shape[1]]

class TSConvPatchEmbedding(keras.layers.Layer):
    def __init__(self, patch_size, embed_dim, block_size=1024):
        super(TSConvPatchEmbedding, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(embed_dim, kernel_size=patch_size, strides=patch_size, padding="valid", name="conv1")
        self.patch_size = patch_size
        self.block_size = block_size
        self.n_embd = embed_dim
        
    def call(self, x):
        x = tf.expand_dims(x,-1)
        T = shape_list(x)[1]
        
        if T < self.patch_size:
            raise ValueError(
                f"input length must be at least {self.patch_size}!"
            )
        
        if self.block_size > T: #padding in the front
            # length = ((T // self.patch_size) + 1) * self.patch_size
            x = tf.pad(x,[[0,0],[self.block_size-T,0],[0,0]])
        
        x = self.conv1(x)
        return x
    
