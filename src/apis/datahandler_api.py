import numpy as np
import matplotlib.pyplot as plt
from ..utils.signals import sigtype,ContinuousSignal
from ..utils.data_util import DataUtil
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class DataHandler():
    def __init__(self, df, targets, covariates, timefeats, df_time, pred_start, pred_len):
        self.data_du = DataUtil([ContinuousSignal(name, sigtype.target, mean_value=df[name].mean(), std_value=df[name].std()) 
                                 for name in targets+covariates], scaling_method='standard')
        self.df = self.data_du.normalize_and_encode(df)
        if pred_start > 0:
            self.x_hist = self.df.loc[:len(df)-(pred_start+1),targets+covariates].values
            self.x_futr = self.df.loc[-pred_start:,targets+covariates].values
        else:
            self.x_hist = self.df.loc[:,targets+covariates].values
            self.x_futr = None
        
        if df_time is not None:
            self.t_hist = df_time.loc[:len(df)-pred_start-1,timefeats].values
            # self.x = np.concatenate([self.x,t_hist],axis=-1)
            if len(df_time) > len(df)-pred_start:
                self.t_futr = df_time.loc[len(df)-pred_start:,timefeats].values
            else:
                self.t_futr = None
        else:
            self.t_hist = None
            self.t_futr = None
        
        self.targets = targets
        self.timefeats = timefeats
        self.covariates = covariates
        self.pred_start = pred_start
        self.pred_len = pred_len
    
    def _split_window_train(self, features):
        input_slice = slice(0, self.block_size)
        output_slice = slice(self.block_size, self.block_size+self.patch_size)
       
        inputs = features[:, input_slice, :]
        inputs.set_shape([None, self.block_size, len(self.targets+self.covariates+self.timefeats)])
        
        outputs = features[:, output_slice, :len(self.targets+self.covariates)]
        outputs.set_shape([None, self.patch_size, len(self.targets+self.covariates)]) 
        return inputs, outputs
    
    def get_data_for_finetune(self, batch_size, block_size=1024,patch_size=64):
        if self.t_hist is None:
            data = np.copy(self.x_hist)
        else:
            data = np.concatenate([self.x_hist,self.t_hist],axis=-1)
        # post_dummpy = np.zeros(shape=(block_size-patch_size,data.shape[1]))
        # data = np.vstack((data,post_dummpy))
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
              data=data,
              targets=None,
              sequence_length=block_size+patch_size,
              sequence_stride=1,
              shuffle=True,
              batch_size=batch_size,)
        
        self.block_size = block_size
        self.patch_size = patch_size
        
        ds = ds.map(self._split_window_train)
        
        x_list, y_list = [], []
        for x, y in ds.as_numpy_iterator():
            x_list.append(x)
            y_list.append(y)
        x, y = np.concatenate(x_list), np.concatenate(y_list)
        return x, y
        
    def get_data_for_inference(self,block_size):
        if self.t_hist is None:
            x = np.copy(self.x_hist)
        else:
            x = np.concatenate([self.x_hist,self.t_hist],axis=-1)
        
        data = x[-block_size:, :]
        data = np.expand_dims(data,0)
        return data
    
    def get_t_future(self):
        if self.t_futr is None:
            return None
        return np.expand_dims(self.t_futr, 0)
    
    def get_x_future(self):
        if self.x_futr is None:
            return None
        return np.expand_dims(self.x_futr, 0)
    
    def denormalize(self,forecast_x):
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
    
        forecast_x = self.data_du.denormalize(forecast_x,self.targets)
        return forecast_x
    
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
