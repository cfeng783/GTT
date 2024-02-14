import tensorflow as tf
from ..utils.tf_utils import shape_list
from .embedding import TSConvPatchEmbedding,TSPositionalEncoding


def get_max_embd_length(config):
    max_embed_length = config.block_size//config.patch_size
    return max_embed_length
    
class TSAttention(tf.keras.layers.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_mask_len : int,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.max_mask_len = max_mask_len
        
        mask = tf.linalg.band_part(tf.ones([max_mask_len, max_mask_len]), -1, 0)  
        mask = tf.reshape(mask, [1, 1, max_mask_len, max_mask_len])
        self.mask = tf.stop_gradient(mask)

        self.k_proj = tf.keras.layers.Dense(embed_dim, use_bias=False, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="v_proj")
        self.q_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="q_proj")
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias, name="out_proj")

    def _shape(self, tensor: tf.Tensor, seq_len: int):
        return tf.transpose(tf.reshape(tensor, (-1, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

    def call(
        self,
        hidden_states,
        enable_masking = True
    ):
        """Input shape: Batch x Time x Channel"""

        _, tgt_len, embed_dim = shape_list(hidden_states)
        src_len = tgt_len
        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), tgt_len)
        value_states = self._shape(self.v_proj(hidden_states), tgt_len)
            
        key_states = tf.reshape(key_states, (-1, src_len, self.head_dim))
        value_states = tf.reshape(value_states, (-1, src_len, self.head_dim))
        query_states = tf.reshape(self._shape(query_states, tgt_len), (-1, tgt_len, self.head_dim))
        
        key_states = tf.cast(key_states,tf.float32)
        query_states = tf.cast(query_states,tf.float32)
        attn_weights = tf.matmul(query_states, key_states, transpose_b=True)

        if enable_masking:
            attn_weights = tf.reshape(attn_weights, (-1, self.num_heads, tgt_len, src_len))
            attn_weights = tf.cast(attn_weights,tf.float32)
            attn_weights = tf.where(self.mask[:,:,:tgt_len,:src_len] == 0, float('-inf'), attn_weights)  
            attn_weights = tf.reshape(attn_weights, (-1, tgt_len, src_len))

        attn_weights = tf.nn.softmax(attn_weights, axis=-1) 

        attn_probs = self.dropout(attn_weights)
        attn_output = tf.matmul(attn_probs, value_states)

        attn_output = tf.transpose(
            tf.reshape(attn_output, (-1, self.num_heads, tgt_len, self.head_dim)), (0, 2, 1, 3)
        )
        attn_output = tf.reshape(attn_output, (-1, tgt_len, embed_dim))

        attn_output = self.out_proj(attn_output)

        return attn_output


class TSEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = config.n_embd
        
        self.self_attn = TSAttention(
            self.embed_dim, config.encoder_attention_heads, 
            max_mask_len = get_max_embd_length(config),
            dropout=config.attention_dropout,name='self_attn'
        )
 
        self.temporal_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="temporal_attn_layer_norm")
        self.channel_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="channel_attn_layer_norm")
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.activation_dropout = tf.keras.layers.Dropout(config.activation_dropout)
        self.fc1 = tf.keras.layers.Dense(config.encoder_ffn_dim, name="fc1")
        self.fc2 = tf.keras.layers.Dense(self.embed_dim, name="fc2")
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")

    def call(
        self, hidden_states, 
        patch_num,
        channel_num
    ):
        ### cross time attention
        residual = hidden_states
        hidden_states = self.temporal_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states,enable_masking=False)
        
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        ### cross channel attention
        residual = hidden_states
        hidden_states = tf.reshape(hidden_states,[-1, channel_num, patch_num, self.embed_dim])
        hidden_states = tf.transpose(hidden_states,perm=[0, 2, 1, 3])
        hidden_states = tf.reshape(hidden_states,[-1, channel_num, self.embed_dim])
        
        hidden_states = self.channel_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states,enable_masking=False)
        hidden_states = self.dropout(hidden_states)

        hidden_states = tf.reshape(hidden_states,[-1, patch_num, channel_num, self.embed_dim])
        hidden_states = tf.transpose(hidden_states,perm=[0, 2, 1, 3])
        hidden_states = tf.reshape(hidden_states,[-1, patch_num, self.embed_dim])
        hidden_states = residual + hidden_states
        
        ### mlp
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = tf.keras.activations.gelu(self.fc1(hidden_states))
        hidden_states = self.activation_dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class TSEncoder(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.layerdrop = config.encoder_layerdrop
        self.embed_dim = config.n_embd
        
        self.max_embed_length = get_max_embd_length(config)
        self.position_embedding = TSPositionalEncoding(self.max_embed_length,embed_dim=config.n_embd)
        
        self.patch_embedding = TSConvPatchEmbedding(config.patch_size, config.n_embd, config.block_size)
        self.dropout = tf.keras.layers.Dropout(config.embedd_pdrop)
        self.encoder_layers = [TSEncoderLayer(config, name=f"layers.{i}") for i in range(config.encoder_layers)]
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="layer_norm")

    def call(self, x, patch_num, channel_num):

        hidden_states = self.patch_embedding(x) 
        
        if patch_num <= self.max_embed_length:
            hidden_states = tf.cast(hidden_states, tf.float32) + self.position_embedding(hidden_states)
        else:
            raise ValueError(
                f"patch_num must be less than 16!"
            )
        
        hidden_states = self.dropout(hidden_states)
        
        for encoder_layer in self.encoder_layers:
            hidden_states = encoder_layer(
                hidden_states,patch_num,channel_num
            )


        hidden_states = self.layer_norm(hidden_states)
        return hidden_states