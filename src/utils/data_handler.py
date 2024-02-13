import tensorflow as tf
import numpy as np

class DataHandler:
    '''
    Data Handler
    
    Parameters
    ----------
    block_size : int
        the length range
    pred_len : int
        the forecast range
    targets : list of strings
        the target variables
    covariates : list of strings
        the covariate variables
    timefeats : list of strings
        the time-related features
    du : DataUtil
        data utility
    '''
    def __init__(self, block_size, pred_len, targets, covariates, timefeats, du):
        self.du = du
        self.column_indices = {name: i for i, name in enumerate(du.column_names())}
        self.targets = targets
        self.covariates = covariates
        self.timefeats = timefeats
        
        self.target_indices = [self.column_indices[name] for name in self.targets]
        self.covariate_indices = [self.column_indices[name] for name in self.covariates]
        self.timefeat_indices = [self.column_indices[name] for name in self.timefeats]
        
        self.input_indices = self.target_indices+self.covariate_indices+self.timefeat_indices
        
        # Work out the window parameters.
        self.block_size = block_size
        self.pred_len = pred_len
        
    def _split_window_train(self, features):
        input_slice = slice(0, self.block_size)
        train_output_slice = slice(self.block_size, self.block_size+self.pred_len)

        
        inputs = features[:, input_slice, :]
        outputs = features[:, train_output_slice, :]
        
        inputs = tf.stack([inputs[:, :, idx] for idx in self.input_indices],axis=-1)
        outputs = tf.stack([outputs[:, :, idx] for idx in self.target_indices],axis=-1)
        
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.block_size, len(self.input_indices)])
        outputs.set_shape([None, self.pred_len, len(self.target_indices)])

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
        data : ndarray
            the numpy array from where the samples are extracted
        batch_size : int
            the batch_size
        
        Returns
        -------
        Dataset 
            a tf.data.Dataset object
        """
        data = np.array(data, dtype=np.float32)
        if len(data) < 1088:
            pre_dummy = np.zeros(shape=(1024-64,data.shape[1]))
            data = np.vstack((pre_dummy,data))
        
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
              data=data,
              targets=None,
              sequence_length=self.block_size+self.pred_len,
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
    
