import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def process_datetime(df_ori, date_col, pred_len, pred_start):
    df = df_ori.copy()
    hist_len = len(df)
    df['time'] = pd.to_datetime(df[date_col],format='mixed')
    df = df.set_index('time')
    freq = df.index.inferred_freq
    
    if pred_len > pred_start:
        if freq is None:
            return [], None, None
        else:
            idxrange = pd.date_range(start=df.index[0], periods=hist_len+pred_len-pred_start, freq=freq)
            df = df.reindex(idxrange)
    
    df = df.reset_index(names=['time'])
    # logger.debug(df.head())
    
    if freq is None:
        df['month_of_year'] = df['time'].dt.month
        df['sin_month'] = np.sin(2*np.pi*df.month_of_year/ (df['month_of_year'].max()+0.0001) )
        df['cos_month'] = np.cos(2*np.pi*df.month_of_year/ (df['month_of_year'].max()+0.0001) )
        
        df['time_of_day'] = df['time'].dt.hour*3600+df['time'].dt.minute*60 + df['time'].dt.second
        df['sin_time'] = np.sin(2*np.pi*df.time_of_day/ (df['time_of_day'].max()+0.0001) )
        df['cos_time'] = np.cos(2*np.pi*df.time_of_day/ (df['time_of_day'].max()+0.0001) )

        df['sin_week'] = 0
        df['cos_week'] = 0
        
    elif 'M' in freq or 'Q' in freq or 'A' in freq or 'Y' in freq or 'W' in freq or '30D' in freq or '90D' in freq or '365D' in freq:
        df['month_of_year'] = df['time'].dt.month
        df['sin_month'] = np.sin(2*np.pi*df.month_of_year/ (df['month_of_year'].max()+0.0001) )
        df['cos_month'] = np.cos(2*np.pi*df.month_of_year/ (df['month_of_year'].max()+0.0001) )
        
        df['sin_time'] = 0
        df['cos_time'] = 0
        df['sin_week'] = 0
        df['cos_week'] = 0
    else:
        df['month_of_year'] = df['time'].dt.month
        df['sin_month'] = np.sin(2*np.pi*df.month_of_year/ (df['month_of_year'].max()+0.0001) )
        df['cos_month'] = np.cos(2*np.pi*df.month_of_year/ (df['month_of_year'].max()+0.0001) )
        
        df['time_of_day'] = df['time'].dt.hour*3600+df['time'].dt.minute*60 + df['time'].dt.second
        df['sin_time'] = np.sin(2*np.pi*df.time_of_day/ (df['time_of_day'].max()+0.0001) )
        df['cos_time'] = np.cos(2*np.pi*df.time_of_day/ (df['time_of_day'].max()+0.0001) )
    
        df['day_of_week'] = df['time'].dt.weekday+1
        df['sin_week'] = np.sin( 2*np.pi*df.day_of_week/ (df['day_of_week'].max()+0.0001) )
        df['cos_week'] = np.cos( 2*np.pi*df.day_of_week/ (df['day_of_week'].max()+0.0001) )

    timefeats = ['sin_time','cos_time','sin_week','cos_week','sin_month','cos_month']
    
    return timefeats, df.loc[:,timefeats], df.loc[:,'time'].astype(str).values
    