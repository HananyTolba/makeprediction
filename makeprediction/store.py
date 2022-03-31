from abc import ABC, abstractmethod
from operator import mod
import os 
import joblib
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from makeprediction.exceptions import LoadingError
from makeprediction.gpts import IGaussianProcessTimeSerie



class IModelStore(ABC):
    '''This is a abstract class for Gaussian Process Time Serie model saving and loading.'''

    @abstractmethod
    def load():
        pass
    @abstractmethod
    def save():
        pass


class Store(IModelStore):
    '''This is a class for Gaussian Process Time Serie model saving and loading.'''
    @classmethod
    def load_v0(cls,directory: str):
        '''Load a model from str path to directory. Old version.'''
        filepath = os.path.join(os.getcwd(), directory)
        files = os.listdir(filepath)
        print(f"model files: {files}")
        if files:
            files = [f for f in files if not f.startswith(".")]
            files_num = list(map(int, files))
            if files_num:
                filepath = os.path.join(filepath, str(files[np.argmax(files_num)]))
                filepath = os.path.join(filepath, os.listdir(filepath)[0])
                print(f'{filepath} => model path')
                return joblib.load(filepath)
            else:
                raise LoadingError(f'The directory "{filepath}" does not contain a valid model.')
        else:
            raise LoadingError(f'The directory "{filepath}" is empty.')
            
    @classmethod
    def load(cls,directory: str):
        '''Load a model from str path to directory.'''
        files = Path(directory).iterdir()
        files = filter(lambda f:f.name.isnumeric(), files)
        files = sorted(files, key=os.path.getmtime, reverse = True)
        if files:
            names = list(map(lambda x:pd.Timestamp(int(x.name),unit='s'), files))
            dirname, date = files[0], names[0]
            print(f'=> {dirname}: model has been found (saved at {date})')
            path = os.path.join(dirname.as_posix(),'gpts.joblib')
            if os.path.isfile(path):
                return joblib.load(path)
            raise ValueError(f'The directory "{dirname}" does not contain a valid model.') 
        else:
            raise ValueError(f'The directory "{directory}" does not contain a valid model.') 

            
            

    @classmethod
    def save(cls,model:IGaussianProcessTimeSerie, dirname=None, if_exists = False):
        '''Save a model on directory'''
        new_directory = str(int(datetime.datetime.now().timestamp()))
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


    
