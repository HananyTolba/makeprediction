from abc import ABC, abstractmethod
from operator import mod
from makeprediction.gpts import IGaussianProcessTimeSerie
import os 
import joblib
import numpy as np
import datetime
from makeprediction.exceptions import LoadingError


import pandas as pd

class IModelStore(ABC):
    '''This is a abstract class for Gaussian Process Time Serie model saving and loading.'''

    @abstractmethod
    def load():
        pass
    @abstractmethod
    def save():
        pass

class Loader():
    pass


class Store(IModelStore):
    '''This is a class for Gaussian Process Time Serie model saving and loading.'''
    @classmethod
    def load(cls,directory: str):
        filepath = os.path.join(os.getcwd(), directory)
        files = os.listdir(filepath)
        if files:
            files = [f for f in files if not f.startswith(".")]
            files_num = list(map(int, files))
            if files_num:
                filepath = os.path.join(filepath, str(files[np.argmax(files_num)]))
                filepath = os.path.join(filepath, os.listdir(filepath)[0])
                return joblib.load(filepath)
            else:
                raise LoadingError(f'The directory "{filepath}" does not contain a valid model.')
        else:
            raise LoadingError(f'The directory "{filepath}" is empty.')
            


    @classmethod
    def save(cls,model:IGaussianProcessTimeSerie, dirname=None, if_exists = False):
        new_directory = str(int(datetime.datetime.now().timestamp()))
        print(type(model._xtrain))
        if isinstance(model._xtrain, pd.DatetimeIndex):
            print(f'time zone is {model._xtrain.tz}')
        if dirname is None:
            dirname = "gpts_model"
        if if_exists and os.listdir(dirname)!=[] and os.path.isdir(dirname):
            print(f'Model already exists in:\n ==> {os.path.abspath(dirname)}.')
            return
        # if os.listdir(dirname) and os.path.isdir(dirname):
        #     print('dir model exist!')
        #     return

        path = os.path.join(dirname, new_directory)
        try:
            os.makedirs(path, exist_ok=True)
            # print(f"Directory {new_directory} Created Successfully." )
            joblib.dump(model, os.path.join(path, 'gpts.joblib'))
            print(f'Model saved successfully on:\n ==> {os.path.abspath(path)}.')
        except OSError:
            print(f"Directory {new_directory} Creation Failed.")
            print(f'Model not saved.')
    @classmethod
    def remove(cls,directory: str):
        filepath = os.path.join(os.getcwd(), directory)
        os.remove(filepath)




