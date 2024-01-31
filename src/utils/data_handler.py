import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class DataHandler():
    '''
    Data Generator for the inference phase
    
    Parameters
    ----------
    block_size : int
        the context length
    patch_size : int
        the patch size
    targets : list of strings
        the target variables
    covariates : list of strings
        the covariate variables
    timefeats : list of strings
        the time-related features
    du : DataUtil
        data utility
    '''
    def __init__(self, block_size, patch_size, targets, covariates, timefeats, du):
        self.block_size = block_size
        self.patch_size = patch_size
        self.du = du
        # Work out the window parameters.
        
        self.column_indices = {name: i for i, name in enumerate(du.column_names())}
        self.targets = targets
        self.covariates = covariates
        self.timefeats = timefeats
        
        self.target_indices = [self.column_indices[name] for name in self.targets]
        self.covariate_indices = [self.column_indices[name] for name in self.covariates]
        self.timefeat_indices = [self.column_indices[name] for name in self.timefeats]
        
        self.input_indices = self.target_indices+self.covariate_indices+self.timefeat_indices
        
    def _split_window_train(self, features):
        input_slice = slice(0, self.block_size)
        train_output_slice = slice(self.block_size, self.block_size+self.patch_size)

        
        inputs = features[:, input_slice, :]
        outputs = features[:, train_output_slice, :]
        
        inputs = tf.stack([inputs[:, :, idx] for idx in self.input_indices],axis=-1)
        outputs = tf.stack([outputs[:, :, idx] for idx in self.target_indices],axis=-1)
        
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.block_size, len(self.input_indices)])
        outputs.set_shape([None, self.patch_size, len(self.target_indices)])

        return inputs, outputs
    
    def _split_window_inference(self, features):
        target_input_slice = slice(0, self.input_len)
        covarite_input_slice = slice(0, self.input_len+self.pred_len)
        target_output_slice = slice(self.input_len, self.input_len+self.pred_len)
        
        target_inputs = features[:, target_input_slice, :]
        target_outputs = features[:, target_output_slice, :]
        covarite_inputs = features[:, covarite_input_slice, :]
        
        
        target_inputs = tf.stack([target_inputs[:, :, idx] for idx in self.target_indices],axis=-1)
        target_inputs.set_shape([None, self.input_len, len(self.target_indices)])
        
        if len(self.covariate_indices+self.timefeat_indices) > 0:
            covarite_inputs = tf.stack([covarite_inputs[:, :, idx] for idx in self.covariate_indices+self.timefeat_indices],axis=-1)
            covarite_inputs.set_shape([None, self.input_len+self.pred_len, len(self.covariate_indices+self.timefeat_indices)])
        
        target_outputs = tf.stack([target_outputs[:, :, idx] for idx in self.target_indices],axis=-1)
        target_outputs.set_shape([None, self.pred_len, len(self.target_indices)])
        
        return target_inputs, covarite_inputs, target_outputs
    
    def make_dataset(self, data, batch_size=256):
        """
        make a tensorflow dataset object for model training and validation
        
        Parameters
        ----------
        data : ndarray or list of ndarray
            the numpy array from where the samples are extracted
        batch_size : int
            the batch_size
        
        Returns
        -------
        Dataset 
            a tf.data.Dataset object
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
              data=data,
              targets=None,
              sequence_length=self.block_size+self.patch_size,
              sequence_stride=1,
              shuffle=True,
              batch_size=batch_size,)
        
        ds = ds.map(self._split_window_train)
        return ds
    
    def extract_data4inference(self, data, input_len, pred_len, stride=1):
        """
        extract input samples as well as output samples from raw data
        
        Parameters
        ----------
        data : ndarray
            the numpy array from where the samples are extracted
        stride : int
            period between successive extracted sequences
        
        
        Returns
        -------
        ndarray 
            the taget inputs, matrix of shape = [n_samples, block_size, n_dim]
        ndarray 
            the covariate inputs, matrix of shape = [n_samples, block_size, n_dim]
        ndarray 
            the timefeats inputs, shape = [n_samples, block_size, n_dim]
        """
        self.input_len = input_len
        self.pred_len = pred_len
        
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
              data=data,
              targets=None,
              sequence_length=self.input_len+self.pred_len,
              sequence_stride=stride,
              shuffle=False,
              batch_size=1,)
        ds = ds.map(self._split_window_inference)
        
        x_list, y_list, z_list = [], [], []
        for x, y, z in ds.as_numpy_iterator():
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
        x, y, z = np.concatenate(x_list), np.concatenate(y_list), np.concatenate(z_list)
        return x, y, z
    
    
    def denormalize(self,obs_x,forecast_x):
        """
        denormalize target variables
    
        Parameters
        ----------
        obs_x : ndarray
            the ground truth data, matrix of shape = [n_samples, n_features]
        forecast_x : ndarray
            the predicted data, matrix of shape = [n_samples, n_features]
    
        Returns
        -------
        ndarray 
            the denormalized ground truth data
        ndarray 
            the denormalized predicted data
        """
    
        obs_x = self.du.denormalize(obs_x,self.targets)
        forecast_x = self.du.denormalize(forecast_x,self.targets)
        return obs_x,forecast_x
    
    def plotRes(self,obs_x,forecast_x,start_pos=None,prefix=''):
        """
        plot prediction compared with inputs
    
        Parameters
        ----------
        obs_x : ndarray
            the ground truth data, matrix of shape = [n_samples, n_features]
        forecast_x : ndarray
            the predicted data, matrix of shape = [n_samples, n_features]
        """
        timeline = np.linspace(0,len(obs_x),len(obs_x)) 
        colors = ['r','g','b','c','m','y']
    
        plt.figure(1)
        k = 0
        for target_var in self.targets:
            plt.subplot(len(self.targets),1,1+k)
            plt.plot(timeline[-len(forecast_x):], forecast_x[:,k], colors[k%6]+"-",label='prediction')
            plt.plot(timeline,obs_x[:,k],"k-",label='ground truth')
            if start_pos is not None:
                plt.axvline(x = start_pos, ls='--',color="y")
            
            plt.legend(loc='best', prop={'size': 6})
            plt.title(prefix+target_var)
            k+=1
        plt.show()
    
    def plotResWithCI(self,obs_x,forecast_x,forecast_std,start_pos=None,prefix=''):
        """
        plot prediction compared with inputs
    
        Parameters
        ----------
        obs_x : ndarray
            the ground truth data, matrix of shape = [n_samples, n_features]
        forecast_x : ndarray
            the predicted data, matrix of shape = [n_samples, n_features]
        """
        timeline = np.linspace(0,len(obs_x),len(obs_x)) 
        colors = ['r','g','b','c','m','y']
    
        plt.figure(1)
        k = 0
        for target_var in self.targets:
            plt.subplot(len(self.targets),1,1+k)
            plt.plot(timeline[-len(forecast_x):], forecast_x[:,k], colors[k%6]+"-",label='prediction')
            plt.fill_between(timeline, forecast_x[:,k]-forecast_std[:,k], forecast_x[:,k]+forecast_std[:,k], color='blue', alpha=0.1)
            plt.plot(timeline,obs_x[:,k],"k-",label='ground truth')
            if start_pos is not None:
                plt.axvline(x = start_pos, ls='--',color="y")
            
            plt.legend(loc='best', prop={'size': 6})
            plt.title(prefix+target_var)
            k+=1
        plt.show()
