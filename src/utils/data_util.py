from .signals import CategoricalSignal,ContinuousSignal,sigtype
import json
import warnings
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
class DataUtil():
    '''
    Utility class for data normalization, denormalization and onehot encoding
    
    Parameters
    ----------
    signals : list, default is None
        the list of signals, must be specified when filename is None
    scaling method : {None,"min_max","standard"}, default is None 
    
        scaler=None indicates no normalization; 
        
        scaler="min_max" indicates using MinMax scaling;
        
        scaler="standard" indicates using standard scaling;
    filename : string, default is None
        the path of json file to load DataUtil, must be specified when signals is None
    '''
    def __init__(self, signals=None, scaling_method=None, filename=None):
        if filename is None:
            self.signals = signals
            self.scaling_method = scaling_method
        else:
            self._load_jsonfile(filename)
        
        self.signal_map = {signal.name: signal for signal in self.signals}
        self.eps = 1e-6
        
    def column_names(self):
        """
        get df column names given signals
        
        Parameters
        ----------
        signals : list
            the list of signals
            
        Returns
        -------
        list of strings
            the dataframe column names
             
        """
        cols = []
        for signal in self.signals:
            if isinstance(signal, CategoricalSignal):
                cols.extend(signal.get_onehot_feature_names())
            if isinstance(signal, ContinuousSignal):
                cols.append(signal.name)
        return cols
    
    def normalize_and_encode(self, df):
        """
        Normalize and onehot encode the signals in the dataset
        
        Parameters
        ----------
        df : DataFrame
            The dataset
            
        Returns
        -------
        Dataframe 
            the modified dataframe
             
        """
        df = df.copy()
        #'onehot encoding and normalisation'
        for signal in self.signals:
            if isinstance(signal, CategoricalSignal):
                for value in signal.values:
                    new_entry = signal.get_feature_name(value)
                    df[new_entry] = 0
                    df.loc[df[signal.name]==value,new_entry] = 1
            if isinstance(signal, ContinuousSignal) and signal.stype != sigtype.timefeat:
                df[signal.name] = df[signal.name].astype(float)
                if self.scaling_method == 'min_max':
                    if signal.max_value is None or signal.min_value is None:
                        msg  = 'please specify min max values for signal' + signal.name
                        warnings.warn(msg)
                    if signal.max_value != signal.min_value:
                        df[signal.name]= df[signal.name].apply(lambda x:float(x-signal.min_value)/float(signal.max_value-signal.min_value))
                    else:
                        msg  = signal.name + ' has no variation, consider remove this signal!'
                        warnings.warn(msg)
                elif self.scaling_method == 'standard':
                    if signal.mean_value is None or signal.std_value is None:
                        msg  = 'please specify mean and std values for signal' + signal.name
                        warnings.warn(msg)
                    
                    df[signal.name]=df[signal.name].apply(lambda x:float(x-signal.mean_value)/float(signal.std_value+self.eps))
                    if signal.std_value == 0:
                        msg  = signal.name + ' has no variation, consider remove this signal!'
                        warnings.warn(msg)
        df = df[self.column_names()]
        return df
    
    
    def onehot_encode(self, df):
        """
        onehot encode the signals in the dataset
        
        Parameters
        ----------
        df : DataFrame
            The dataset
            
        Returns
        -------
        Dataframe 
            the modified dataframe
        """
        df = df.copy()
        #'onehot encoding'
        cols = []
        for signal in self.signals:
            if isinstance(signal, CategoricalSignal):
                for value in signal.values:
                    new_entry = signal.get_feature_name(value)
                    df[new_entry] = 0
                    df.loc[df[signal.name]==value,new_entry] = 1
                    cols.append(new_entry)
            if isinstance(signal, ContinuousSignal):
                cols.append(signal.name)
        df = df[cols]
        return df
    
    def denormalize(self,data,cols):
        """
        Denormalize the data according to its column names
        
        Parameters
        ----------
        data : ndarray
            The data, must be matrix of shape = [n_samples, n_cols]
        cols : list of strings
            The column names, each column name must be the name of a signal
        
        Returns
        -------
        ndarray 
            The denormalized data
             
        """
        for i in range(len(cols)):
            signal = self.signal_map[cols[i]]
            if self.scaling_method == 'min_max':
                if signal.min_value != signal.max_value:
                    data[:,i] = (signal.max_value-signal.min_value)*data[:,i]+signal.min_value
            elif self.scaling_method == 'standard':
                data[:,i] = data[:,i]*float(signal.std_value+self.eps)+signal.mean_value
        return data
    
    def save2jsonfile(self,filename):
        """
        save to a Json file
        
        Parameters
        ----------
        filename : string
            the path of json file to save DataUtil
             
        """
        ret = {'scaling_method':self.scaling_method}
        ret['signals'] = []
        for signal in self.signals:
            ret['signals'].append(signal.to_dict())
        with open(filename, 'w') as outfile:
            json.dump(ret,outfile,cls=NpEncoder)
    
    def _load_jsonfile(self,filename):
        with open(filename, 'r') as inputfile:
            data = inputfile.read()
        configJson = json.loads(data)
        self.scaling_method = configJson['scaling_method']
        self.signals = []
        for sd in configJson['signals']:
            if sd['stype'] == sigtype.target.value:
                stype = sigtype.target
            elif sd['stype'] == sigtype.covariate.value:
                stype = sigtype.covariate
            else:
                stype = sigtype.timefeat
            
            if sd['class'] == 'continuous':
                signal = ContinuousSignal(sd['name'],stype,
                                          min_value=sd['min_value'],max_value=sd['max_value'],
                                          mean_value=sd['mean_value'],std_value=sd['std_value'])
            elif sd['class'] == 'categorical':
                signal = CategoricalSignal(sd['name'],stype, values=sd['values'])
            self.signals.append(signal)
        
        
        
        
        