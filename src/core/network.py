import tensorflow as tf
from tensorflow import keras
from .encoder import TSEncoder
from .revin import RevIN
    
class GTTNet(keras.Model):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = config.patch_size
        self.block_size = config.block_size
        self.target_dim = config.target_dim
        self.covariate_dim = config.covariate_dim
        self.timefeat_dim = config.timefeat_dim
        self.n_embd = config.n_embd
        self.forecast_mode = config.forecast_mode
        self.enable_revin = config.enable_revin
        self.revin_time = config.revin_time
        self.pred_len = config.pred_len
        
        if self.enable_revin:
            self.revin = RevIN(affine=config.affine, dtype=tf.float32)
        
        self.encoder = TSEncoder(config)
        
        if self.pred_len is None:
            self.mu_head = keras.layers.Dense(self.patch_size,activation='linear', name='mu_head', dtype=tf.float32)
        else:
            self.mu_head = keras.layers.Dense(self.pred_len,activation='linear', name='mu_head', dtype=tf.float32)
        
    def call(self, inputs):
        if self.enable_revin:
            # instance normalization
            if self.timefeat_dim > 0 and self.revin_time==False:
                x_val = inputs[:,:,:self.target_dim+self.covariate_dim]
                x_time = inputs[:,:,-self.timefeat_dim:]
                x_val = self.revin(x_val,mode='norm')
                x_enc = tf.concat([x_val,x_time],axis=-1)
            else:
                x_enc = self.revin(inputs,mode='norm')
        else:
            x_enc = inputs

        _,T,C = x_enc.shape
        patch_num = self.block_size//self.patch_size
        
        #channel independence
        x_enc = tf.transpose(x_enc,perm=[0, 2, 1])
        x_enc = tf.reshape(x_enc,[-1, T])
        
        x_dec = self.encoder(x_enc, patch_num, C)
        x_dec = x_dec[:,-1,:] ##B*C,n_embd
        
        
        outputs_mu = self.mu_head(x_dec) ##B*C,patch_size
        if self.pred_len is None:
            outputs_mu = tf.reshape(outputs_mu, [-1, C, self.patch_size]) ##B,C,patch_size
        else:
            outputs_mu = tf.reshape(outputs_mu, [-1, C, self.pred_len]) ##B,C,pred_len
        outputs_mu = tf.transpose(outputs_mu,perm=[0, 2, 1])##B,patch_size,C
        if self.revin_time == False:
            outputs_mu = outputs_mu[:,:,:self.target_dim+self.covariate_dim]
        if self.enable_revin:
            outputs_mu = self.revin(outputs_mu,mode='denorm')
        outputs_mu = outputs_mu[:, :, :self.target_dim]##B,patch_size,C
        return outputs_mu
    
    @classmethod
    def build_raw_model(cls, mc):
        model = cls(mc)
        input_dim = mc.target_dim+mc.covariate_dim+mc.timefeat_dim
        model.build(input_shape=(4,mc.block_size,input_dim)) 
        # model.summary(expand_nested=True,show_trainable=True)
        return model
    
    @classmethod
    def from_pretrained(cls, pretrained_model, mc):
        sd_hf = pretrained_model.estimator.get_weight_paths()
        config_hf = pretrained_model.configs
        
        mc.block_size = config_hf.block_size
        mc.patch_size = config_hf.patch_size
        mc.forecast_mode = config_hf.forecast_mode  
        mc.embedd_pdrop = config_hf.embedd_pdrop
        mc.dropout  = config_hf.dropout
        mc.activation_dropout  = config_hf.activation_dropout
        mc.attention_dropout = config_hf.attention_dropout
        mc.n_embd  = config_hf.n_embd
        mc.encoder_layers  = config_hf.encoder_layers
        mc.encoder_attention_heads  = config_hf.encoder_attention_heads
        mc.encoder_layerdrop  = config_hf.encoder_layerdrop
        mc.encoder_ffn_dim  = config_hf.encoder_ffn_dim
            
        model = cls(mc)
        input_dim = mc.target_dim+mc.covariate_dim+mc.timefeat_dim
        model.build(input_shape=(4,mc.block_size,input_dim)) 
        # print('mc',mc)
        sd = model.get_weight_paths()
        # for key in sd:
        #     print(key,sd[key].shape)
              
        # filer out unnecessary keys and layers deeper than n_layer  
        drop_keys = ['revin']
        if mc.pred_len is not None:
            drop_keys.append('mu_head')
        # print('drop_keys',drop_keys)
        
        
        sd_hf = {k: v for k, v in sd_hf.items() if not any((k_ in k) for k_ in drop_keys)} 
        # for key in sd_hf:
        #     print(key,sd_hf[key].shape) 
        for k in sd_hf:  
            # vanilla copy over the other parameters
            local_key = k
            # print('assign',k,'to',local_key,'shape',sd_hf[k].shape,sd[local_key].shape)
            assert sd_hf[k].shape == sd[local_key].shape  
            sd[local_key].assign(sd_hf[k])
        
        model.encoder.trainable=False
        # model.summary(expand_nested=True,show_trainable=True)
        # tf.keras.utils.plot_model(model,show_shapes=True)
        return model
    
    