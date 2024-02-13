import pandas as pd
import numpy as np
import zipfile
from .signals import ContinuousSignal,sigtype

def process_datetime(df, date_col):
    df['time'] = pd.to_datetime(df[date_col],format='mixed')
    df = df.set_index('time')
    freq = df.index.inferred_freq
    print('freq:',freq)
    df = df.reset_index()
    
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
    
    signals = []
    for tfeat in ['sin_time','cos_time','sin_week','cos_week','sin_month','cos_month']:
        df[[tfeat]] = df[[tfeat]].ffill()
        if df[tfeat].min() == df[tfeat].max():
            df[tfeat] = 0
        signals.append( ContinuousSignal(tfeat, sigtype.timefeat, 
                                        min_value=df[tfeat].min(), max_value=df[tfeat].max(),
                                        mean_value=df[tfeat].mean(), std_value=df[tfeat].std()) )
    # df = df.drop(columns=['time', 'month_of_year','time_of_day','day_of_week'])
    return df, signals

def load_ett_data(fp='../datasets/ETT/',name='ETTm1.csv', context_len=1024, uni=False):
    df = pd.read_csv(fp+name)
    df,signals = process_datetime(df, date_col='date')  
    
    if 'm' in name:
        border1s = [0, 34465 - context_len, 34465 + 11521 - context_len]
        border2s = [34465, 34465 + 11521, 34465+11521+11521]
    else:
        border1s = [0, 8545 - context_len, 8545 + 2881 - context_len]
        border2s = [8545, 8545+2881, 8545 + 2881 + 2881]
    
    train_df = df.loc[:border2s[0],:].reset_index(drop=True)
    val_df = df.loc[border1s[1]:border2s[1],:].reset_index(drop=True)
    test_df = df.loc[border1s[2]:border2s[2],:].reset_index(drop=True)
    # train_df,test_df = train_val_split(df,val_ratio=0.2)
    covariates = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
    target = 'OT'
    
    cov_sigtype = sigtype.covariate if uni else sigtype.target
    for name in covariates:
        if df[name].min() != df[name].max():
            signals.append( ContinuousSignal(name, cov_sigtype,
                                        min_value=train_df[name].min(), max_value=train_df[name].max(),
                                        mean_value=train_df[name].mean(), std_value=train_df[name].std()) )

    signals.append( ContinuousSignal(target, sigtype.target,
                                        min_value=train_df[target].min(), max_value=train_df[target].max(),
                                        mean_value=train_df[target].mean(), std_value=train_df[target].std()) )

    return train_df,val_df,test_df,signals


def load_electricity_data(fp='../datasets/Electricity/', name='electricity.csv', context_len=1024, uni=False):

    df = pd.read_csv(fp+name)
    df,signals = process_datetime(df, date_col='date')  

    
    border1s = [0,  18317 - context_len, 18317+2633 - context_len]
    border2s = [18317, 18317+2633, 18317+2633+5261]
    
    train_df = df.loc[:border2s[0],:].reset_index(drop=True)
    val_df = df.loc[border1s[1]:border2s[1],:].reset_index(drop=True)
    test_df = df.loc[border1s[2]:,:].reset_index(drop=True)
    
   
    covariates = [str(i) for i in range(320)]    
    target = 'OT'
    cov_sigtype = sigtype.covariate if uni else sigtype.target
    for name in covariates:
        if train_df[name].min() != train_df[name].max():
            signals.append( ContinuousSignal(name, cov_sigtype,
                                        min_value=train_df[name].min(), max_value=train_df[name].max(),
                                        mean_value=train_df[name].mean(), std_value=train_df[name].std()) )

    signals.append( ContinuousSignal(target, sigtype.target,
                                        min_value=train_df[target].min(), max_value=train_df[target].max(),
                                        mean_value=train_df[target].mean(), std_value=train_df[target].std()) )
    

    return train_df,val_df,test_df,signals

def load_traffic_data(fp='../datasets/Traffic/', context_len=1024, uni=False):
    z_tr = zipfile.ZipFile(fp+'/traffic.zip', "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    df=pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()
    
    df,signals = process_datetime(df, date_col='date')  

    border1s = [0,  12185 - context_len,  12185+1757 - context_len]
    border2s = [12185, 12185+1757, 12185+1757+3509]
    
    train_df = df.loc[:border2s[0],:].reset_index(drop=True)
    val_df = df.loc[border1s[1]:border2s[1],:].reset_index(drop=True)
    test_df = df.loc[border1s[2]:,:].reset_index(drop=True)
    
    covariates = [str(i) for i in range(861)]
    cov_sigtype = sigtype.covariate if uni else sigtype.target
    for name in covariates:
        if train_df[name].min() != train_df[name].max():
            signals.append( ContinuousSignal(name, cov_sigtype,
                                        min_value=train_df[name].min(), max_value=train_df[name].max(),
                                        mean_value=train_df[name].mean(), std_value=train_df[name].std()) )
    
    target = 'OT'
    signals.append(ContinuousSignal(target, sigtype.target,
                                    min_value=np.float64(train_df[target].min()), max_value=np.float64(train_df[target].max()),
                                    mean_value=np.float64(train_df[target].mean()), std_value=np.float64(train_df[target].std())))

    return train_df, val_df, test_df, signals

def load_weather_data(fp='../datasets/Weather/', name='weather.csv', context_len=1024, uni=False):
    df = pd.read_csv(fp+name)
    df,signals = process_datetime(df, date_col='date')  

    border1s = [0,  36792 - context_len, 36792+5271 - context_len]
    border2s = [36792, 36792+5271, 36792+5271+10540]
    
    train_df = df.loc[:border2s[0],:].reset_index(drop=True)
    val_df = df.loc[border1s[1]:border2s[1],:].reset_index(drop=True)
    test_df = df.loc[border1s[2]:,:].reset_index(drop=True)
    
    covariates = ['p', 'T', 'Tpot', 'Tdew', 'rh', 'Vpmax', 'Vpact', 'Vpdef', 'sh', 'H2OC', 'rho', 'wv', 'max. wv',
                  'wd', 'rain', 'raining', 'SWDR', 'PAR', 'max. PAR', 'Tlog']
    target = 'OT'
    cov_sigtype = sigtype.covariate if uni else sigtype.target
    for name in covariates:
        if train_df[name].min() != train_df[name].max():
            signals.append(ContinuousSignal(name, cov_sigtype,
                                            min_value=np.float64(train_df[name].min()), max_value=np.float64(train_df[name].max()),
                                            mean_value=np.float64(train_df[name].mean()), std_value=np.float64(train_df[name].std())))

    signals.append(ContinuousSignal(target, sigtype.target,
                                    min_value=np.float64(train_df[target].min()), max_value=np.float64(train_df[target].max()),
                                    mean_value=np.float64(train_df[target].mean()), std_value=np.float64(train_df[target].std())))

    return train_df, val_df, test_df, signals

def load_illness_data(fp='../datasets/ILL/', name='national_illness.csv', context_len=128, uni=False):
    df = pd.read_csv(fp+name)
    df,signals = process_datetime(df, date_col='date')
    
    border1s = [0,  617 - context_len, 617+74 - context_len]
    border2s = [617, 617+74, 617+74+170]
    
    train_df = df.loc[:border2s[0],:].reset_index(drop=True)
    val_df = df.loc[border1s[1]:border2s[1],:].reset_index(drop=True)
    test_df = df.loc[border1s[2]:,:].reset_index(drop=True)
    
    covariates = ['% WEIGHTED ILI', '%UNWEIGHTED ILI', 'AGE 0-4', 'AGE 5-24', 'ILITOTAL', 'NUM. OF PROVIDERS']
    target = 'OT'
    cov_sigtype = sigtype.covariate if uni else sigtype.target
    for name in covariates:
        if train_df[name].min() != train_df[name].max():
            signals.append(ContinuousSignal(name, cov_sigtype,
                                            min_value=np.float64(train_df[name].min()), max_value=np.float64(train_df[name].max()),
                                            mean_value=np.float64(train_df[name].mean()), std_value=np.float64(train_df[name].std())))

    signals.append(ContinuousSignal(target, sigtype.target,
                                    min_value=np.float64(train_df[target].min()), max_value=np.float64(train_df[target].max()),
                                    mean_value=np.float64(train_df[target].mean()), std_value=np.float64(train_df[target].std())))

    return train_df, val_df, test_df, signals