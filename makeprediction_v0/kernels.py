#!/usr/bin/env python
# coding: utf-8


from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Generator
import pandas as pd
import collections
import matplotlib.pyplot as plt
from makeprediction.exceptions import NotKernelError
from makeprediction.tools import Name


class IKernel(ABC):
    @abstractmethod
    def count():
        pass
    @abstractmethod
    def label():
        pass
    def __eq__(self, other, verbose=False):
        try:
            assert self.label() == other.label()
            assert self.hyperparameters == other.hyperparameters
        except AssertionError:
            return False
        return True

    # def decompose(self):
    #     return self

    def is_pure_sum(self):
        if isinstance(self,Sum) and len(list(self.iterkernels())) == len(self.label().split(' + ')):
            return True
        return False

    def __str__(self) -> str:       
        return f"{self.label()}: hyperparameters = {self.hyperparameters}."
    def __repr__(self):
        return f"Instance of '{self.label()}'"
    @property
    def hyperparameters(self) -> dict:
        ''' return the hyperparameters of kernel.'''
        parms = dict()
        for cle, valeur in self.__dict__.items():
            parms[cle.lstrip('_')] = valeur
        return parms

    
    @hyperparameters.setter
    def hyperparameters(self, dic: dict):
        ''' modify the hyperparameters of kernel.'''

        for cle in self.__dict__.keys():
            setattr(self, cle, dic[cle.lstrip('_')])

    
    def square_root_matrix(self, x: "np.ndarray") -> "np.ndarray":
        K = self.count(x)
        if isinstance(self,(White,Constant)):
            Q = np.sqrt(K)
        else:
            np.fill_diagonal(K, K.diagonal() + 1e-10)

            try:
                Q = np.linalg.cholesky(K)
            except BaseException:
                U, s, VT = np.linalg.svd(K)
                Q = U @ np.diag(np.sqrt(s))
        return Q
    
    
    @staticmethod
    def convert2date(x):
        '''convert a input sequence to pandas.DatetimeIndex object.'''
        if isinstance(x,(pd.Index, np.ndarray,pd.DatetimeIndex,list,tuple)):
            return pd.to_datetime(x)
        elif isinstance(x,(pd.Series,)):
            return pd.to_datetime(x.values)
        else:
            raise TypeError(f'The "{x}" is not a valid format.')
            
    def date2num(self,x: Union[np.ndarray, pd.DatetimeIndex])-> np.ndarray:
        '''convert a  sequence pf pandas.DatetimeIndex object to numeric sequence values. 
        If a numeric sequence then the same value is returned.'''
        if isinstance(x,np.ndarray) and np.issubdtype(x.dtype, np.number):
            return x
        x = self.convert2date(x)
        if pd.isna(x).any():
            raise ValueError("Cannot convert NaN/NaT values to integer")
        return x.view("int64")/ 10**9 / 3600 / 24
            
    def simulate(
            self,
            dt: Union[np.ndarray, pd.DatetimeIndex] = None,
            seed=None) -> np.ndarray:
        '''
         This method allows the simulation of a Gaussian process (1d) on a domain x.
        '''
        x = self.date2num(dt)
        m = x.size
        Q = self.square_root_matrix(x)
        if seed is None:
            iid = np.random.randn(m)
        else:
            seed = seed
            iid = np.random.randn(m)
        y = Q @ iid
        return y
    
    def __add__(self,other):
        return Sum(self,other)


    def __radd__(self, x):
        if x == 0:
            return self 
        return Sum( self, Constant(x))

    def is_kernel(self):
        if isinstance(self,IKernel):
            return True
        return False

    
     

    # def __radd__(self, x):
    #     if isinstance(self, IKernel):
    #         if isinstance(x, IKernel):
    #             return Sum(self,x)
    #         elif isinstance(x, (float,int)):
    #             if x == 0:
    #                 return self
    #             return Sum(Constant(x), self)
    #     else:
    #         raise NotKernelError
    
    def __mul__(self,other):
        return Prod(self,other)


class White(IKernel):
    """
    The White kernel: this kernel produces 'white noise'. The kernel equation is
        k(x_n, x_m) = δ(n, m) σ²
    where:
    δ(.,.) is the Kronecker delta,
    σ²  is the variance parameter.
    """
    def __init__(self,variance:float = 0):
        self._variance = variance 
        
    def count(self, x: np.ndarray, y: np.ndarray = None):
        x = x.reshape(x.size, 1)
        if y is None:
            y = x
        kernel = np.eye(x.size,y.size)
        return self._variance * kernel
    
    def label(self):
        return 'White'
    
    def plot(self,resolution:int =300):
        fig = plt.figure()
        x = np.linspace(-3,3,resolution)
        y = self.count(x,np.zeros(1))
        x, y = x.ravel(), y.ravel()
        return plt.plot(x,y)
        
class Constant(IKernel):
    """@hyperparameters.setter
    def hyperparameters(self, dic: dict):
        ''' modify the hyperparameters of kernel.'''

        for cle in self.__dict__.keys():
            setattr(self, cle, dic[cle.lstrip('_')])

    The constant kernel.
    """
    def __init__(self,constant:float = 0):
        self._constant = constant 
        
    def count(self, x: np.ndarray, y: np.ndarray = None):
        x = x.reshape(x.size, 1)
        if y is None:
            y = x
        kernel = np.full((x.size,y.size),fill_value = self._constant)
        return kernel
    
    def label(self):
        return 'Constant'
    
    def plot(self,resolution:int =300):
        fig = plt.figure()
        x = np.linspace(-3,3,resolution)
        y = self.count(x,np.zeros(1))
        x, y = x.ravel(), y.ravel()
        return plt.plot(x,y)
    
    
class Stationary(IKernel):    
    def radial_dist(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        '''
        Count the radial distance.
        '''
        x = x.ravel()
        if y is None:
            y = x
        y = y.ravel()
        r = np.abs(x - y.reshape(-1, 1))
        return r 
    
    def plot(self,resolution:int =300):
        fig = plt.figure()
        x = np.linspace(-3,3,resolution)
        y = self.count(x,np.zeros(1))
        x, y = x.ravel(), y.ravel()
        return plt.plot(x,y)

class NotStationary(IKernel):     
    def plot(self,resolution:int = 300):
        fig = plt.figure()
        x = np.linspace(-3,3,resolution)
        y = self.count(x,np.ones(1))
        x, y = x.ravel(), y.ravel()
        return plt.plot(x,y)
        
    

    
class Linear(NotStationary):
    def __init__(self, variance:float = 1):
        self._variance = variance

    def count(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        if y is None:
            y = x
        x = x.reshape(x.size, 1)
        y = y.reshape(y.size, 1)
        kernel = (x @ y.T).T
        return self._variance * kernel
    def label(self):
        return 'Linear'
    
    
    
class Polynomial(NotStationary):

    def __init__(self, offset:float = 0,  variance:float = 1.,  degree:int = 2):
        self._variance = variance
        self._offset = offset
        self._degree =  degree
    
    def count(self, x: np.ndarray, y: np.ndarray = None):
        if y is None:
            y = x
        x = x.reshape(x.size, 1)
        y = y.reshape(y.size, 1)
        kernel = ((x @ y.T + self._offset**2)**self._degree).T
        return self._variance * kernel

    def label(self):
        return f'Polynomial({self._degree})'

class RBF(Stationary):
    def __init__(self, length_scale:float = 1, variance:float = 1):
        self._length_scale = length_scale
        self._variance = variance
    def count(self, x, y=None):
        """Squared Exponential covariance function or RBF with isotropic distance measure."""
        kernel = np.exp(-self.radial_dist(x, y)**2 / (2 * self._length_scale**2))
        return self._variance * kernel
    def label(self):
        return 'RBF'
    
    
class Matern(Stationary):
    def __init__(self, length_scale:float = 1, nu:float =.5, variance:float = 1):
        self._length_scale = length_scale
        self._variance = variance
        self._nu = nu
    @property
    def nu(self):
        return self._nu
    @nu.setter
    def nu(self,new_nu):
        self._nu = new_nu
 

    
    def count(self, x: np.ndarray, y: np.ndarray = None):
        """Matern covariance function with nu = 1/2 and isotropic distance measure. """
        r = self.radial_dist(x, y)
        if self._nu == 0.5:
            kernel = np.exp(-r / self._length_scale)
        elif self._nu == 1.5:
            kernel = (1 + np.sqrt(3) * r) * \
            np.exp(-np.sqrt(3) * r / self._length_scale)
        elif self._nu == 2.5:
            kernel = (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * \
            np.exp(-np.sqrt(5) * r / self._length_scale)    
        return self._variance * kernel

    def label(self):
        if self._nu == .5:
            name = f"Matern(1/2)"
        elif self._nu == 1.5:
            name = f"Matern(3/2)"
        elif self._nu == 2.5:
            name = f"Matern(5/2)"
        else:
            name = f"Matern {self._nu}"
        return name
    
class Cosine(Stationary):
    def __init__(self,length_scale:float = 1, variance:float = 1):
        self._length_scale = length_scale
        self._variance = variance
    def count(self, x: np.ndarray, y: np.ndarray = None):
        """Stationary covariance function for a sinusoid."""
        r = self.radial_dist(x, y)

        kernel = np.cos(np.pi * r / self._length_scale)
        return self._variance * kernel
    def label(self):
        return 'Cosine'
    
class Periodic(Stationary):
    def __init__(self,length_scale:float = 1, variance:float = 1, period:float = 1):
        self._length_scale = length_scale
        self._variance = variance
        self._period = period

    def count(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Stationary covariance function for a sinusoid."""
        r = self.radial_dist(x, y)
        kernel = np.exp(-2 * np.sin(np.pi * r / self._period) ** 2 / (self._length_scale**2))
        return self._variance * kernel
    def label(self):
        return 'Periodic'
    
class Combine(IKernel):
    def __init__(self, kernel_1: IKernel,  kernel_2: IKernel):
        self._kernel_1 = kernel_1
        self._kernel_2 = kernel_2
    def iterkernels(self) -> Generator:
        for new in self.__dict__.values():
            if isinstance(new,Combine):
                yield from new.iterkernels()
            else:
                yield new
        
    def decompose(self)->list:
        # names = [x.label() for x in self.iterkernels()]
        # kernels_list = [x for _,x in sorted(zip(Name.priority_order(names),list(self.iterkernels())))]
        return list(self.iterkernels())


    def decompose_sorted(self) -> list:
        names = [x.label() for x in self.iterkernels()]
        kernels_list = [x for _,x in sorted(zip(Name.priority_order(names),list(self.iterkernels())))]
        return kernels_list
    @property
    def hyperparameters(self)-> dict:  
        hypms = map(lambda x: x.hyperparameters,self.iterkernels())
        init_names = map(lambda x: x.label(),self.iterkernels())
        lnames = list(init_names)
        for key, value in dict(collections.Counter(lnames)).items():
            count=0
            for i,k in enumerate(lnames):
                if k == key:
                    new_key = f'{k}_{count+1}'
                    lnames[i] = new_key
                    count += 1
        return dict(zip(lnames,hypms))
    
    @hyperparameters.setter
    def hyperparameters(self, D:list)->None:
        for kernel, d in zip(self.iterkernels(),D):
            kernel.hyperparameters = d

                
    def plot(self,resolution:int =300):
        if any(map(lambda k:isinstance(k,NotStationary),kernel.iterkernels())):
            fig = plt.figure()
            x = np.linspace(-3,3,resolution)
            y = self.count(x,np.ones(1))
            x, y = x.ravel(), y.ravel()
            return plt.plot(x,y)
        
        fig = plt.figure()
        x = np.linspace(-3,3,resolution)
        y = self.count(x,np.zeros(1))
        x, y = x.ravel(), y.ravel()
        return plt.plot(x,y)
        
    
class Sum(Combine):
    """
    Represents sum of a pair of kernels
    """
    def count(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        return self._kernel_1.count(x, y) + self._kernel_2.count(x, y)
    def label(self):        
        return f'{self._kernel_1.label()} + {self._kernel_2.label()}'
    

class Prod(Combine):
    """
    Represents product of a pair of kernels
    """
    def count(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        return self._kernel_1.count(x, y) * self._kernel_2.count(x, y)
    def label(self):
        if isinstance(self._kernel_1,Sum) and isinstance(self._kernel_2,Sum):
            return f'({self._kernel_1.label()}) x ({self._kernel_2.label()})'
        elif isinstance(self._kernel_1,Sum):
            return f'({self._kernel_1.label()}) x {self._kernel_2.label()}'
        elif isinstance(self._kernel_2,Sum):
            return f'{self._kernel_1.label()} x ({self._kernel_2.label()})'
        return f'{self._kernel_1.label()} x {self._kernel_2.label()}'


if '__main__' == __name__: 
    pattern1  = RBF() + RBF()
    pattern2 = sum(2*[RBF()])
    for ker in pattern1.iterkernels():
        ker.hyperparameters = {'length_scale':np.random.rand(1)[0],'variance':np.random.rand(1)[0]}
    for ker in pattern2.iterkernels():
        ker.hyperparameters = {'length_scale':np.random.rand(1)[0],'variance':np.random.rand(1)[0]}
    print(pattern1.hyperparameters)
    print(pattern2.hyperparameters)
  
    
    