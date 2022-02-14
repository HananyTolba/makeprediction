#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Union
import enum
from abc import ABC
import datetime
import re


class Time:

    @staticmethod
    def convert2date(x):
        '''convert an input sequence to pandas.DatetimeIndex object.'''
        if isinstance(x,(pd.Index, np.ndarray,pd.DatetimeIndex,list,tuple,pd.Timestamp)):
            return pd.to_datetime(x)
        elif isinstance(x,(pd.Series,)):
            return pd.to_datetime(x.values)
        elif isinstance(x,datetime.datetime):
            return x
        else:
            raise TypeError(f'The "{x}" is not a valid format.')

    @classmethod
    def date2num(cls,x: Union[np.ndarray, pd.DatetimeIndex])-> np.ndarray:
        '''convert a  sequence pf pandas.DatetimeIndex object to numeric sequence values. 
        If a numeric sequence then the same value is returned.'''
        if isinstance(x,np.ndarray) and np.issubdtype(x.dtype, np.number):
            return x
        elif isinstance(x,(float,int)):
            return x
        x = cls.convert2date(x)
        if isinstance(x,pd.Timestamp):
            return x.timestamp() 
        elif isinstance(x,datetime.datetime):
            return x.timestamp() 
        elif pd.isna(x).any():
            raise ValueError("Cannot convert NaN/NaT values to integer")
        #return x.map(pd.Timestamp.timestamp)
        return x.view("int64")/ 10**9 # / 10**9 / 3600 / 24


class Size(enum.Enum):
    LARGE_SIZE = 600
    SMALL_SIZE = 300





class LargeKernelNames(enum.Enum):
    RBF = 'RBF'
    Matern32 = 'Matern32'
    Matern12 = 'Matern12'
    Matern52 = 'Matern52'
    Matern = 'Matern'
    Linear = 'Linear'
    @classmethod
    def is_valid(cls,value):
        if re.match("^Matern\([0-9]/[0-9]\)$", value) is None:
            return hasattr(cls,value)
        return True

        
class SmallKernelNames(enum.Enum):
    Polynomial = 'Polynomial'
    Constant = 'Constant'
    Periodic = 'Periodic'
    White = 'White'
    
    @classmethod
    def is_valid(cls,value):
        if re.match("^Polynomial\([0-9]+\)$", value) is None:
            return hasattr(cls,value)
        else:
            return True




class IProgressBar(ABC):
    pass

class ProgressBar(IProgressBar,tqdm):
    pass
        

class Name(enum.Enum):
    RBF =  3
    Linear =  1
    Polynomial =  0
    Periodic =  2
    Matern = 4
    @classmethod
    def priority_order(cls,l):
        out = []
        for x in l:
            if x in Name.__members__ :
                out.append(getattr(Name,x).value )
            elif 'Matern' in x:
                out.append(cls.Matern.value )
            elif 'Polynomial' in x:
                out.append(cls.Polynomial.value )
        return out


def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()


class IsPeriodic(enum.Enum):
    NonPeriodic = enum.auto()
    Periodic = enum.auto()


class IsMaternOrRBF(enum.Enum):
    Matern = enum.auto()
    RBF = enum.auto()

class IsLinear(enum.Enum):
    Linear = enum.auto()
    NonLinear = enum.auto()
