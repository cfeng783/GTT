import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "../../")
import logging
from scipy.stats import wasserstein_distance
from src.apis.gtt_api import GTTAPI
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def forecast(df, targets, covariates, timefeat, pred_len, pred_start, modelpath, autodiff=False):
    """
    zeroshot forecast api
    
    Parameters
    ----------
    df : dataframe
        the historical data
    targets: list of strings
        target variables, prediction targets
    covariates: list of strings
        covariates
    timefeat: string
        datetime variable, timefeat can also be empty, i.e., timefeat=''
    pred_len : int
        prediction length, the number of future time steps to forecast
    pred_start: int
        prediction start position, it's N to last step in the historical data 
    modelpath : string
        the location of GTT model files 
    autodiff : bool
        whether do 1st-order differencing before prediction, sometimes set autodiff=True will boost prediction accuracy
    
        
    Returns
    -------
    dict 
        the forecast results
    """
    try:
        df = df.ffill()
        if autodiff == False:
            targets2use = targets
        else:
            targets2use = []
            for target in targets:
                df[target+'_d'] = df[target].diff().bfill()
                x = df[target+'_d'].values[-1024:]
                x = (x-np.mean(x))/(np.std(x)+1e-5)
            
                y = df[target].values[-1024:]
                y = (y-np.mean(y))/(np.std(y)+1e-5)
            
                dx = wasserstein_distance(x[:len(x)//2],x[len(x)//2:])
                dy = wasserstein_distance(y[:len(y)//2],y[len(y)//2:])
                
                logger.debug(f'd(y-x):{dy-dx}')
                if dy-dx > 1.0:
                    targets2use.append(target+'_d')
                else:
                    targets2use.append(target)
        
        tsf = GTTAPI(df, targets2use, covariates, timefeat, pred_len, pred_start)
        x_pred = tsf.predict(modelpath)
        x_pred = tsf.data_handler.denormalize(x_pred)
        
        res = {}
        res['ret'] = 'success'
        
        if tsf.datetimes is not None:
            res['xlabel'] = timefeat
            res['xdata'] = tsf.datetimes.tolist()
        else:
            res['xlabel'] = 'index'
            res['xdata'] = list(range(len(df)+pred_len-pred_start))
        
        res['forecast_start'] = len(df)-pred_start
        
        cov_list = []
        for covariate in covariates:
            cov_dict = {}
            cov_dict['name'] = covariate
            cov_dict['values'] = df[covariate].values.tolist()
            cov_list.append(cov_dict)
        res['covariates'] = cov_list
        
        tar_list = []
        for i in range(len(targets)):
            tar_dict = {}
            tar_dict['name'] = targets[i]
            tar_dict['values'] = df[targets[i]].values.tolist()
            if targets[i] in targets2use:
                tar_dict['preds'] = x_pred[:,i].tolist()
            else:
                initv = df.loc[len(df)-pred_start-1, targets[i]]
                diffs = x_pred[:,i].tolist()
                preds = []
                for k in range(len(diffs)):
                    initv += diffs[k]
                    preds.append(initv)
                tar_dict['preds'] = preds
                
            tar_list.append(tar_dict)
        res['targets'] = tar_list
        
            
    except Exception as e:
        logger.exception(e)
        res = {}
        res['ret'] = 'fail'
        res['message'] = str(e)
    
    return res
            
def plot_res(res):
    colors = ['r','g','b','c','m','y']
    split_pos = res['forecast_start']
    plt.figure(1)
    k = 0
    for target_var in res['targets']:
        plt.subplot(len(res['targets']),1,1+k)
        plt.plot(res['xdata'][:len(target_var['values'])], target_var['values'], "k-",label=target_var['name'])
        plt.plot(res['xdata'][split_pos:], target_var['preds'], colors[k%6]+"-",label='Forecast')
        plt.axvline(x = res['xdata'][split_pos], ls='--',color="y")
        
        plt.legend(loc='best', prop={'size': 6})
        k+=1
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('../../datasets/Air_Passengers.csv')
    
    targets = ['#Passengers']
    covariates = []
    timefeat = 'Date'
    pred_len = 64
    pred_start = 24
    modelchoice = 'small'
    autodiff = False
    modelpath = f'../../checkpoints/GTT-{modelchoice}'
    res = forecast(df, targets, covariates, timefeat, pred_len, pred_start, modelpath, autodiff=autodiff)
    if res['ret'] == 'success':
        plot_res(res)
    else:
        print(res)
    
    