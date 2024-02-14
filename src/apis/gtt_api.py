import numpy as np
from src.core.network import GTTNet
from src.core.model import ModelConfig,TSFoundation
import pickle,os
from .datetime_api import process_datetime
from .datahandler_api import DataHandler
from pandas.api.types import is_numeric_dtype
import logging

logger = logging.getLogger(__name__)

class GTTAPI():
    '''
    A GTT API for end-task usages
    '''
    def __init__(self, df, targets, covariates, timefeat, pred_len, pred_start):
        self.df = df.copy()
        self.targets = [target for target in targets]
        self.covariates = [covariate for covariate in covariates if is_numeric_dtype(df[covariate])]
        if timefeat is not None and len(timefeat) > 0:
            self.timefeats, self.df_time, self.datetimes = process_datetime(df,timefeat,pred_len,pred_start)
        else:
            self.timefeats = []
            self.df_time = None
            self.datetimes = None
            
        self.data_handler = DataHandler(self.df, self.targets, self.covariates, self.timefeats, self.df_time, pred_start,pred_len)
        self.pred_start = pred_start
        self.pred_len = pred_len
        
        
    def predict(self, modelpath):
        pred_len = self.pred_len
        self.load_pretrained(modelpath)
        
        x = self.data_handler.get_data_for_inference(block_size=self.configs.block_size)
        logger.debug(f'x shape: {x.shape}')
        
        t_futr = self.data_handler.get_t_future()
        x_futr = self.data_handler.get_x_future()
        for i in range(0,pred_len,self.configs.patch_size):
            logger.debug(f"{i}/{pred_len}")
            
            if x.shape[1] > self.configs.block_size:
                x = x[:,-self.configs.block_size:,:]
            
            x_next = self.estimator.predict(x,verbose=0)
            logger.debug(f'x_next shape: {x_next.shape}')
            if i+self.configs.patch_size > pred_len:
                throw = i+self.configs.patch_size-pred_len
                logger.debug(f'throw: {throw}')
                x_next = x_next[:,:-throw,:]
                logger.debug(f'x_next shape: {x_next.shape}')
                if len(self.covariates)>0 and x_futr is not None and len(x_futr) >= i+self.configs.patch_size-throw:
                    'assign covariates values if given'
                    x_next[:,:,len(self.targets):] = x_futr[:,i:i+self.configs.patch_size-throw,len(self.targets):]
                if t_futr is not None:
                    t_next = t_futr[:,i:i+self.configs.patch_size-throw,:]
                    logger.debug(f't_next shape: {t_next.shape}, x_next shape: {x_next.shape}')
                    x_next = np.concatenate((x_next,t_next),axis=-1)
                    
                x = np.concatenate((x, x_next),axis=1)
            else:
                if len(self.covariates)>0 and x_futr is not None and len(x_futr) >= i+self.configs.patch_size:
                    'assign covariates values if given'
                    x_next[:,:,len(self.targets):] = x_futr[:,i:i+self.configs.patch_size,len(self.targets):]
                if t_futr is not None:
                    t_next = t_futr[:,i:i+self.configs.patch_size,:]
                    x_next = np.concatenate((x_next,t_next),axis=-1)
                x = np.concatenate((x, x_next),axis=1)
            
        return x[0,-pred_len:,:len(self.targets)]
    
    def load_pretrained(self, modelpath):
        self.configs = pickle.load(open(os.path.join(modelpath,'configs.pkl'),'rb'))
        self.configs = ModelConfig(**self.configs)
        pm = TSFoundation.load_model(model_path=modelpath)
        
        self.configs.target_dim = len(self.targets+self.covariates)
        self.configs.covariate_dim = 0
        self.configs.timefeat_dim = len(self.timefeats)


        self.configs.enable_revin = True
        self.configs.affine = False
        self.configs.revin_time = True
        self.estimator = GTTNet.from_pretrained(pm,self.configs)
        self.estimator.compile(optimizer='adam', loss='mae', run_eagerly=True)
        return self
    