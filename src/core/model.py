from tensorflow import keras
from ..utils.data_handler import DataHandler
import numpy as np
from .network import GTTNet
import pickle,os
from ..utils.signals import sigtype,ContinuousSignal,CategoricalSignal
from ..utils.data_util import DataUtil
import tensorflow as tf
from dataclasses import dataclass, asdict
from ..utils.tf_utils import unbiased_mae_loss
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    block_size: int = None
    patch_size: int = None 
    
    target_dim: int = None
    covariate_dim : int = None
    timefeat_dim : int = None
    
    embedd_pdrop: float = 0.1
    dropout : float = 0.0 
    activation_dropout : float = 0.0
    attention_dropout : float = 0.0
    n_embd : int = 384
    encoder_layers : int = 4
    encoder_attention_heads : int = 6
    encoder_layerdrop : float = 0.0
    encoder_ffn_dim : int = 1536
    
    ##the following shall only be adapted in the fine-tune stage
    enable_revin : bool = False
    affine : bool = False
    revin_time: bool = True
    forecast_mode : str = 'point' ##support 'point' for point estimation, 'gaussian' for gaussian probabilistic
    pred_len : int = None

class GTT:
    '''
    GTT model
    
    Parameters
    ----------
    signals : list
        the list of signals the model is dealing with.
    configs: None
    '''
    def __init__(self, signals=None, configs=None):       
        if signals is not None:
            self.du = DataUtil(signals, scaling_method='standard')
            self.configs = configs
            self._meta_init()
    
    def _meta_init(self):
        self.covariates = []
        self.timefeats = []
        self.targets = []
        for signal in self.du.signals:
            if signal.stype == sigtype.target:
                self.targets.append(signal.name)
            elif signal.stype == sigtype.covariate:
                if isinstance(signal, ContinuousSignal):
                    self.covariates.append(signal.name)
                if isinstance(signal, CategoricalSignal):
                    self.covariates.extend(signal.get_onehot_feature_names())
            elif signal.stype == sigtype.timefeat:
                if isinstance(signal, ContinuousSignal):
                    self.timefeats.append(signal.name)
                if isinstance(signal, CategoricalSignal):
                    self.timefeats.extend(signal.get_onehot_feature_names())
        
        self.configs.target_dim = len(self.targets)
        self.configs.covariate_dim = len(self.covariates)
        self.configs.timefeat_dim = len(self.timefeats)
        
        logger.debug(f'covariates: {self.covariates}')
        logger.debug(f'timefeats: {self.timefeats}')
        logger.debug(f'targets: {self.targets}')
        if self.configs.pred_len is None:
            self._data_handler = DataHandler(self.configs.block_size, self.configs.patch_size, 
                                                self.targets, self.covariates, self.timefeats, self.du)
        else:
            self._data_handler = DataHandler(self.configs.block_size, self.configs.pred_len, 
                                                self.targets, self.covariates, self.timefeats, self.du)
        
    
    @property
    def data_handler(self):
        """
        get the data handler
        """
        return self._data_handler   
        
    def train(self, train_df, val_df, cp, pm=None, optimizer=None,
              batch_size=256,epochs=10,distribute=False,mixed_precision=False,verbose=0):
        """
        train the model
        
        Parameters
        ----------
        train_df : dataframe
            the training data
        val_df : dataframe
            the validation data
        cp : string
            checkpoint path
        pm : WhisperFM
            a pretrained Whisper model for time series data, if set to None, then will load model from huggingface
        optimizer : string or optimizer
            the optimizer for gradient descent, if set to None, adamW will be used
        batch_size : int, default is 256
            the batch size
        epochs : int, default is 10
            the maximum epochs to train the model
        distribute : bool
            whether to use distributed training
        verbose : int, default is 0
            0 indicates silent, higher values indicate more messages will be printed
        
        Returns
        -------
        dict
            the training log
        """
        keras.backend.clear_session()
        
        if mixed_precision:
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
        
        train_df = self.du.normalize_and_encode(train_df)
        val_df = self.du.normalize_and_encode(val_df)
        
        if not os.path.exists(cp):
            os.makedirs(cp)
        
        if distribute:
            physical_devices = tf.config.list_physical_devices('GPU')
            gpus = ['/gpu:'+str(i) for i in range(len(physical_devices))]
            strategy = tf.distribute.MirroredStrategy(devices=gpus,cross_device_ops=tf.distribute.ReductionToOneDevice())
        else:  # Use the Default Strategy
            strategy = tf.distribute.get_strategy()
        
        with strategy.scope():   
            if optimizer is None:
                optimizer = tf.keras.optimizers.Adam()
            
            if pm is None:
                model = GTTNet.build_raw_model(self.configs)
            else:
                model = GTTNet.from_pretrained(pm, self.configs)
            model.compile(optimizer=optimizer, loss='mae')
            checkpoint_path = os.path.join(cp,'GTT.ckpt')
            cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_best_only=True, save_weights_only=True)
            es_callback = keras.callbacks.EarlyStopping(patience=3)                        
            
        if distribute:
            global_batch_size = strategy.num_replicas_in_sync*batch_size
            train_ds = self._data_handler.make_dataset(train_df.values, global_batch_size).repeat(epochs+1).prefetch(8)
            val_ds = self._data_handler.make_dataset(val_df.values, global_batch_size).repeat(epochs+1).prefetch(4)
            train_ds = strategy.experimental_distribute_dataset(train_ds)
            val_ds = strategy.experimental_distribute_dataset(val_ds)

            hist = model.fit(train_ds, epochs=epochs, validation_data = val_ds, callbacks=[cp_callback,es_callback], 
                             steps_per_epoch=len(train_df)//global_batch_size, validation_steps=len(val_df)//global_batch_size, verbose=verbose)
        else:
            train_ds = self._data_handler.make_dataset(train_df.values, batch_size)
            val_ds = self._data_handler.make_dataset(val_df.values, batch_size)
            hist = model.fit(train_ds, epochs=epochs, validation_data = val_ds, callbacks=[cp_callback,es_callback], verbose=verbose)
            
        model.load_weights(checkpoint_path)
        self.estimator = model
        return hist
    
    def predict_ft(self,df,input_len,pred_len,batch_size=8):
        df = self.du.normalize_and_encode(df)
        x, c, y = self._data_handler.extract_data4inference(df.values, input_len, pred_len, stride=1)
        # print(x.shape,t.shape)
        c_inputs = c[:,:input_len,:]
        inputs = np.concatenate((x,c_inputs),axis=-1)
        x_next = self.estimator.predict(inputs,batch_size=batch_size,verbose=0)
            
        y_pred = x_next[:,-pred_len:,:]
        y_true = y
        return y_pred,y_true
    
    def predict(self,df,input_len,pred_len,batch_size=8):
        """
        Predict the targets in the forecast range given input samples
        
        Parameters
        ----------
        df : dataframe
            the data 
            
        Returns
        -------
        ndarray 
            y_pred
        ndarray
            y_pred
        """
        df = self.du.normalize_and_encode(df)
        x, c, y = self._data_handler.extract_data4inference(df.values, input_len, pred_len, stride=1)
        # print(x.shape,t.shape)
        
        if self.configs.covariate_dim+self.configs.timefeat_dim > 0:
            c_inputs = c[:,:input_len,:]
            for i in range(0,pred_len,self.configs.patch_size):
                print(i,'/',pred_len)
                
                if x.shape[1] > self.configs.block_size:
                    x = x[:,-self.configs.block_size:,:]
                
                if c_inputs.shape[1] > self.configs.block_size:
                    c_inputs = c_inputs[:,-self.configs.block_size:,:]
                
                inputs = np.concatenate((x,c_inputs),axis=-1)
                x_next = self.estimator.predict(inputs,batch_size=batch_size,verbose=0)

                if i+self.configs.patch_size > pred_len:
                    throw = i+self.configs.patch_size-pred_len
                    x = np.concatenate((x, x_next[:,-self.configs.patch_size:-throw,:]),axis=1)
                else:
                    c_inputs = c[:,:input_len+i+self.configs.patch_size,:]
                    x = np.concatenate((x, x_next[:,-self.configs.patch_size:,:]),axis=1)
        else:
            for i in range(0,pred_len,self.configs.patch_size):
                print(i,'/',pred_len)
                
                if x.shape[1] > self.configs.block_size:
                    x = x[:,-self.configs.block_size:,:]
                
                x_next = self.estimator.predict(x,batch_size=batch_size,verbose=0)

                if i+self.configs.patch_size > pred_len:
                    throw = i+self.configs.patch_size-pred_len
                    x = np.concatenate((x, x_next[:,-self.configs.patch_size:-throw,:]),axis=1)
                else:
                    x = np.concatenate((x, x_next[:,-self.configs.patch_size:,:]),axis=1)
            
            # print(x_inputs.shape,x_next.shape)
        y_pred = x[:,-pred_len:,:]
        y_true = y
        return y_pred,y_true
        
    def save_model(self,model_path=None,hist=None):
        """
        save the model to files
        
        Parameters
        ----------
        model_path : string, default is None
            the target folder whether the model files are saved.
            If None, a tempt folder is created
        """
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        self.estimator.save_weights(os.path.join(model_path,'GTT.h5'))
        pickle.dump(asdict(self.configs),open(os.path.join(model_path,'configs.pkl'),'wb'))
        self.du.save2jsonfile(os.path.join(model_path,'data_util.json'))
        if hist is not None:
            pickle.dump(hist.history,open(os.path.join(model_path,'history.pkl'),'wb'))
    
    def load_model(self, model_path=None, pm=None):
        """
        load the model from files
        
        Parameters
        ----------
        model_path : string, default is None
            the target folder whether the model files are located
            If None, load models from the tempt folder
        """
        configs = pickle.load(open(os.path.join(model_path,'configs.pkl'),'rb'))
        self.configs = ModelConfig(**configs)
        self.du = DataUtil(filename=os.path.join(model_path,'data_util.json'))
        self._meta_init()
        if pm is None:
            model = GTTNet.build_raw_model(self.configs)
        else:
            model = GTTNet.from_pretrained(pm, self.configs)
        model.load_weights(model_path+'/GTT.h5')
        model.compile(optimizer='adam', loss='mae', run_eagerly=True)
        self.estimator = model
        return self
    
    @classmethod
    def from_tsfoundation(cls, signals, foundation_path, cp=None):
        """
        load the model from files
        
        Parameters
        ----------
        model_path : string, default is None
            the target folder whether the model files are located
            If None, load models from the tempt folder
        """
        model = cls()
        configs = pickle.load(open(os.path.join(foundation_path,'configs.pkl'),'rb'))
        model.configs = ModelConfig(**configs)
        model.du = DataUtil(signals, scaling_method='standard')
        model._meta_init()        
        pm = TSFoundation.load_model(model_path=foundation_path,cp=cp)
        # pm.save_model(foundation_path)
        model.configs.enable_revin = True
        model.configs.affine = False
        model.configs.revin_time = True
        model.estimator = GTTNet.from_pretrained(pm, model.configs)
        model.estimator.compile(optimizer='adam', loss='mae', run_eagerly=True)
        return model


SHUFFLE_BUFFER = 1024*1024*2
    
class TSFoundation:
    '''
    A time series model ready for end-task usages
    
    Parameters
    ----------
    configs: None
    '''
    def __init__(self, configs=None):       
        self.configs = configs
       
        
    def train(self, train_ds, val_ds, train_datasize,val_datasize, cp, 
              optimizer=None,batch_size=256,epochs=10,
              distribute=True,mixed_precision=True,
              verbose=0):
        """
        train the model
        
        Parameters
        ----------
        train_df : dataframe
            the training data
        val_df : dataframe
            the validation data
        cp : string
            checkpoint path
        optimizer : string or optimizer, default is 'adam'
            the optimizer for gradient descent
        batch_size : int, default is 256
            the batch size
        epochs : int, default is 10
            the maximum epochs to train the model
        verbose : int, default is 0
            0 indicates silent, higher values indicate more messages will be printed
        
        Returns
        -------
        dict
            the training log
        """
        keras.backend.clear_session()
        
        if mixed_precision:
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
        
        if not os.path.exists(cp):
            os.makedirs(cp)
        
        if distribute:
            physical_devices = tf.config.list_physical_devices('GPU')
            gpus = ['/gpu:'+str(i) for i in range(len(physical_devices))]
            strategy = tf.distribute.MirroredStrategy(devices=gpus)#,cross_device_ops=tf.distribute.ReductionToOneDevice())
        else:  # Use the Default Strategy
            strategy = tf.distribute.get_strategy()
        
        with strategy.scope():   
            if optimizer is None:
                lrs = tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=1e-3,
                    decay_steps=20*train_datasize//(batch_size*4),
                    alpha=0.1,
                    warmup_steps=2048
                )
                
                optimizer = tf.keras.optimizers.AdamW(
                    learning_rate=lrs,
                    beta_1=0.9,
                    beta_2=0.98,
                    epsilon=1e-06,
                    clipnorm=1.0
                )
            
                     
            self.estimator = GTTNet.build_raw_model(self.configs)
            
            pickle.dump(asdict(self.configs),open(os.path.join(cp,'configs.pkl'),'wb'))
            
            lossfn = unbiased_mae_loss()
            self.estimator.compile(optimizer=optimizer, loss=lossfn)
            
            checkpoint_path = os.path.join(cp, "cp-{epoch:04d}.ckpt")
            os.path.dirname(checkpoint_path)
            cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_freq='epoch', save_weights_only=True)
            es_callback = keras.callbacks.EarlyStopping(patience=3)                        
            
        if distribute:
            global_batch_size = strategy.num_replicas_in_sync*batch_size
            train_ds = train_ds.shuffle(buffer_size=SHUFFLE_BUFFER).batch(global_batch_size,num_parallel_calls=tf.data.AUTOTUNE,drop_remainder=True).repeat().prefetch(8)
            val_ds = val_ds.batch(global_batch_size,num_parallel_calls=tf.data.AUTOTUNE,drop_remainder=True).repeat().prefetch(8)
            
            train_ds = strategy.experimental_distribute_dataset(train_ds)
            val_ds = strategy.experimental_distribute_dataset(val_ds)
            
            hist = self.estimator.fit(train_ds, epochs=epochs, validation_data = val_ds, callbacks=[cp_callback,es_callback], 
                             steps_per_epoch=train_datasize//global_batch_size, validation_steps=val_datasize//global_batch_size, verbose=verbose)
        else:
            train_ds = train_ds.batch(batch_size,drop_remainder=True).repeat().prefetch(8)
            val_ds = val_ds.batch(batch_size,drop_remainder=True).repeat().prefetch(4)
            
            hist = self.estimator.fit(train_ds, epochs=epochs, validation_data = val_ds, callbacks=[cp_callback,es_callback],
                             steps_per_epoch=train_datasize//batch_size, validation_steps=val_datasize//batch_size, verbose=verbose)
            
        return hist
    
    def save_model(self,model_path=None,hist=None):
        """
        save the model to files
        
        Parameters
        ----------
        model_path : string, default is None
            the target folder whether the model files are saved.
            If None, a tempt folder is created
        """
        self.estimator.save_weights(os.path.join(model_path,'tsfoundation.h5'))
        pickle.dump(asdict(self.configs),open(os.path.join(model_path,'configs.pkl'),'wb'))
        if hist is not None:
            pickle.dump(hist.history,open(os.path.join(model_path,'history.pkl'),'wb'))
    
    @classmethod
    def load_model(cls, model_path,cp=None):
        """
        load the model from files
        
        Parameters
        ----------
        model_path : string, default is None
            the target folder whether the model files are located
            If None, load models from the tempt folder
        """
        model = cls()
        configs = pickle.load(open(os.path.join(model_path,'configs.pkl'),'rb'))
        model.configs = ModelConfig(**configs)
        model.estimator = GTTNet.build_raw_model(model.configs)
        if cp is None:
            model.estimator.load_weights(model_path+'/tsfoundation.h5')
        else:
            model.estimator.load_weights(model_path+'/'+cp)
        return model