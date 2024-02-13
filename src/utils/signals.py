from enum import Enum

class sigtype(Enum):
    target = 101
    covariate = 102 ##covariates must be available in the prediction horizon, otherwise, use target instead of covariate
    timefeat = 103

class BaseSignal(object):
    '''
    The base signal class
    
    Parameters
    ----------
    name : string
        the name of the signal
    stype : string
        the type of the signal, can be sigtype.target, sigtype.covariate or sigtype.timefeat       
    '''

    def __init__(self, name, stype):
        self.name = name
        self.stype = stype
    
    def to_dict(self):
        return {'name':self.name, 'stype':self.stype.value}   


class ContinuousSignal(BaseSignal):
    
    '''
    The class for signals which take continuous values
    
    Parameters
    ----------
    name : string
        the name of the signal
    stype : string
        the type of the signal, can be sigtype.target, sigtype.covariate or sigtype.timefeat    
    min_value : float, default is None
        the minimal value for the signal
    max_value : float, default is None
        the maximal value for the signal
    mean_value : float, default is None
        the mean for the signal value distribution
    std_value : float, default is None
        the std for the signal value distribution 
    '''


    def __init__(self, name, stype, min_value=None, max_value=None, mean_value=None, std_value=None):
        '''
        Constructor
        '''
        super().__init__(name, stype)
        self.min_value = min_value
        self.max_value = max_value
        self.mean_value = mean_value
        self.std_value = std_value
    
    def to_dict(self):
        pdict = super().to_dict()
        cdict = {'class':'continuous','min_value':self.min_value,'max_value':self.max_value,'mean_value':self.mean_value,'std_value':self.std_value}
        return {**pdict,**cdict}


class CategoricalSignal(BaseSignal):
    '''
    The class for signals which take discrete values
    
    Parameters
    ----------
    name : string
        the name of the signal
    stype : string
        the type of the signal, can be sigtype.target, sigtype.covariate or sigtype.timefeat    
    values : list
        the list of possible values for the signal
    
    '''


    def __init__(self, name, stype, values):
        '''
        Constructor
        '''
        super().__init__(name, stype)
        self.values = values
    
    def get_onehot_feature_names(self):
        """
        Get the one-hot encoding feature names for the possible values of the signal
        
        Returns
        -------
        list 
            the list of one-hot encoding feature names 
        """
        name_list = []
        for value in self.values:
            name_list.append(self.name+'='+str(value))
        # if len(self.values) > 2:
        #     for value in self.values:
        #         name_list.append(self.name+'='+str(value))
        # else:
        #     name_list.append(self.name+'='+str(self.values[0]))
        return name_list
    
    def get_feature_name(self,value):
        """
        Get the one-hot encoding feature name for a possible value of the signal
        
        Parameters
        ----------
        value : object
            A possible value of the signal
        
        Returns
        -------
        string 
            the one-hot encoding feature name of the given value
        """
        return self.name+'='+str(value)
    
    def to_dict(self):
        pdict = super().to_dict()
        cdict = {'class':'categorical','values':self.values}
        return {**pdict,**cdict}
        