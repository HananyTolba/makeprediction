from abc import ABC, abstractmethod
import numpy as np



class IMetrics(ABC):
    @abstractmethod
    def residue():
        pass
    
    
class Metrics(IMetrics):
    @classmethod
    def residue(cls,Y_true:np.ndarray, Y_pred:np.ndarray):
        Y_true = Y_true.ravel()
        Y_pred = Y_pred.ravel()
        return np.subtract(Y_true, Y_pred)
    @classmethod
    def mbe(cls,Y_true:np.ndarray, Y_pred:np.ndarray):
        return cls.residue(Y_true, Y_pred).mean()
    @classmethod
    def mse(cls,Y_true:np.ndarray, Y_pred:np.ndarray):
        return np.square(cls.residue(Y_true, Y_pred)).mean()

    @classmethod
    def mae(cls,Y_true, Y_pred):
        return np.abs(cls.residue(Y_true, Y_pred)).mean()

    @classmethod
    def r2(cls,Y_true, Y_pred):
        Y_true = Y_true.ravel()
        Y_pred = Y_pred.ravel()
        return np.corrcoef(Y_true, Y_pred)[0, 1]**2
    @classmethod
    def rmse(cls,Y_true, Y_pred):
        return np.sqrt(cls.mse(Y_true, Y_pred))
    
    @classmethod
    def nrmse(cls,Y_true, Y_pred):
        return cls.rmse(Y_true, Y_pred) / Y_true.mean()
            
 

