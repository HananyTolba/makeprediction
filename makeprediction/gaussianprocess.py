#!/usr/bin/env python
# coding: utf-8

from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Generator
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json
from numpy.linalg import  lstsq
from scipy import interpolate
from concurrent.futures import ThreadPoolExecutor
from requests.packages.urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from collections import Counter
from scipy.signal import resample



from makeprediction.exceptions import NotKernelError
from makeprediction.tools import Time, Size, LargeKernelNames, SmallKernelNames, ProgressBar
from makeprediction.tools import IsPeriodic, softmax, IsLinear
from makeprediction.kernels import Periodic, RBF, Matern, Linear, Polynomial, Constant, Cosine, Sum, Prod, IKernel
from makeprediction.invtools import fast_pd_inverse as pdinv
from makeprediction.api import API
from makeprediction.invtools import inv_col_add_update, inv_col_pop_update




class IGaussianProcessRegressor(ABC):
    @abstractmethod
    def fit():
        pass
    @abstractmethod
    def predict():
        pass
    @abstractmethod
    def update():
        pass
    def __repr__(self):
        return f"Instance of '{self.__class__.__name__}'"

    


class GaussianProcessRegressor(IGaussianProcessRegressor):
    
    def __init__(self, xtrain:np.ndarray=None,ytrain:np.ndarray=None, kernel:IKernel=None,
                 sigma_n:float =.01,invK=None,
                 a=None,b=None):
        '''
        Constructor of the Gaussian process regression class:<
        It has five attributes:
        - _kernel: an instance of a kernels class (RBF,Matern32,...)
        - _model: is a pretrained tensorflow model
        - _sigma_n: is the standard deviation of the gaussian white noise.
        '''
        self._xtrain = xtrain
        self._ytrain = ytrain
        self._kernel = kernel
        self._sigma_n = sigma_n
        self._invK = invK
        self._a = a
        self._b = b
        
    def __str__(self) -> str:       
        message_print = f"Gaussian Process Regressor with kernel: {self._kernel.label()}"
        return message_print

    
    def __eq__(self, other, verbose=False):
        for k in self.__dict__:
            try:
                if isinstance(self.__dict__[k], np.ndarray):
                    assert np.allclose(self.__dict__[k], other.__dict__[k], 1e-7)
                elif k == '_kernel':
                    assert self.__dict__[k].label() == other.__dict__[k].label()
                    assert self.hyperparameters == other.hyperparameters
                else:
                    assert np.array_equal(self.__dict__[k], other.__dict__[k])

            except AssertionError:
                if verbose:
                    print(
                        f"Models do not have  the same value of the attribute '{k}'.")
                return False
        return True
    
    @property    
    def data(self):
        return dict(zip(['X','y'],[self._xtrain, self._ytrain]))
    @data.setter
    def data(self,args:list):
        if isinstance(args,list):
            self._xtrain = args[0]
            self._ytrain = args[1]
        elif isinstance(args,dict):
            try:
                self._xtrain = args['X']
                self._ytrain = args['y']
            except KeyError:
                l = list(filter(lambda w:w not in ('X','y'),args.keys()))
                print(f'The all keys:"{l}" of inputs dict must be "X" or "y".')
     
    @property
    def kernel(self):
        return self._kernel
    @kernel.setter
    def kernel(self,new_kernel):
        self._kernel = new_kernel
    
    @property            
    def hyperparameters(self):
        return self.kernel.hyperparameters
    
    @hyperparameters.setter
    def hyperparameters(self, new_hyperparameters: dict) -> None:
        self.kernel.hyperparameters = new_hyperparameters
        xtrainTransform, a, b = self.x_transform()
        K_noise = self._kernel.count(xtrainTransform)
        np.fill_diagonal(K_noise, K_noise.diagonal() + self._sigma_n**2)
        invK_noise = pdinv(K_noise)
        self._invK = invK_noise
        
        
    @property
    def std_noise(self):
        return self._sigma_n

    @std_noise.setter
    def std_noise(self, sigma_n):
        self._sigma_n = sigma_n

        
    def set_data_from_pandas(self, args):
        if isinstance(args, pd.DataFrame):
            if args.shape[1] >= 2:
                print('args.shape[1] >= 2:')
                data = args.iloc[:, 0].values, args.iloc[:, 1].values
                self.data = list(data)
                return

            else:
                print('ELSE args.shape[1] >= 2:')

                data = args.index, args.iloc[:, 0].values
                self.data = list(data)
                return
        elif isinstance(args, pd.Series):
                data = args.index, args.values
                self.data = list(data)
                return
        else:
            raise ValueError("Invalid args, list, tuple, dict or dataframe.")


   
    
    
    
    def sorte(self):
        args = self.data
        if isinstance(args['X'],np.ndarray) and np.issubdtype(args['X'].dtype, np.number):
            ind = np.argsort(args['X'], axis=0)
            args['X'] = args['X'][ind]
            args['y'] = args['y'][ind]
            self.data = args
        else:
            args['X'] = Time.convert2date(args['X'])        
            dataframe = pd.DataFrame.from_dict(args)
            dataframe.set_index('X',inplace = True)
            dataframe.sort_index(inplace = True)
            dataframe.reset_index(inplace = True)
            self.data = dataframe.to_dict('list')
    def plot(self):
        plt.figure(figsize = (15,5))
        plt.plot(model.data['X'],model.data['y'],'b')
    
    def to_numpy(self):
        args = self.data
        if not isinstance(args['y'],np.ndarray):
            args['y'] = np.array(args['y'])
        if not isinstance(args['X'],np.ndarray):
            args['X'] = np.array(args['X'])
            self.data = args
     
    def x_transform(self):
        '''
        This function transforms any line or segment [a, b] to segment [-3, 3] and
        then returns the parameters of the associated model.
        '''
        kernel = self._kernel
        name = kernel.label()
        size = self._xtrain.size
        if LargeKernelNames.is_valid(name) or SmallKernelNames.is_valid(name):
            if LargeKernelNames.is_valid(name):
                X = np.linspace(-3, 3, size)
            else:
                X = np.linspace(-1, 1, size)
            Y = Time.date2num(self._xtrain)
            fit = np.polyfit(Y, X, 1)
            res = fit[0] * Y + fit[1]
            return res, fit[1], fit[0]
        else:
            raise NotKernelError
            
            
        
       
    def get_hyperparameters(self):
        '''Return hyperparameters of Gaussian Process Regressor if instance is not a periodic one.'''
        y = self.preprocessing().reshape(1, -1)
        data = {"inputs": y.tolist()}
        name = self._kernel.label() 
        if not isinstance(self.kernel, Periodic):
            url = API.get_url(name)
            try:
                r = requests.post(url, data=json.dumps(data), timeout=5)
                res.append(np.array(r.json()["outputs"][0]))
            except BaseException:
                r = requests.post(url, data=json.dumps(data), timeout=5)
                res = np.array(r.json()["outputs"][0])    
            return res 

    @staticmethod
    def softmax(x):
        e_x = np.exp(x)
        return e_x / e_x.sum()

    def periodic_predict(self):
        class_names_periodic = ['NonPeriodic', 'Periodic']
        url = API.get_url('periodic_kernel_predict')
        y = self._ytrain
        f = interpolate.interp1d(np.linspace(-1, 1, y.size), y)
        xnew = np.linspace(-1, 1, Size.SMALL_SIZE.value)
        ynew = f(xnew)
        y_sd = (ynew - ynew.mean()) / ynew.std()
        data = {"inputs": y_sd.reshape(1, -1).tolist()}
        response = requests.post(url, data=json.dumps(data))
        values = np.array(response.json()["outputs"][0])
        pred_test = np.argmax(values)
        decision = class_names_periodic[pred_test]
        return decision, self.softmax(values)

    def is_periodic(self):
        class_names_periodic = [dec.name for dec in IsPeriodic]
        url = API.url_is_periodic()
        y = self._ytrain
        f = interpolate.interp1d(np.linspace(-1, 1, y.size), y)
        xnew = np.linspace(-1, 1, Size.SMALL_SIZE.value)
        ynew = f(xnew)
        y_sd = (ynew - ynew.mean()) / ynew.std()
        data = {"inputs": y_sd.reshape(1, -1).tolist()}
        response = requests.post(url, data=json.dumps(data))
        values = np.array(response.json()["outputs"][0])
        pred_test = np.argmax(values)
        decision = class_names_periodic[pred_test]
        return decision, softmax(values)

    def linear_predict(self):
        class_names = ['Linear', 'NonLinear']
        # url = kernel2url('linear_kernel_predict')
        url = API.get_url('linear_kernel_predict')

        y = self._ytrain
        y_resampled = resample(y, Size.SMALL_SIZE.value)
        y_sd = (y_resampled - y_resampled.mean()) / y_resampled.std()
        # print(y_sd.shape)
        data = {"inputs": y_sd.reshape(1, -1).tolist()}
        response = requests.post(url, data=json.dumps(data))
        # print(response)

        values = np.array(response.json()["outputs"][0])
        pred_test = np.argmax(values)
        decision = class_names[pred_test]
        return decision, self.softmax(values)

    def is_linear(self):
        class_names = [x.name for x in IsLinear]
        url = API.url_is_linear()
        y = self._ytrain
        y_resampled = resample(y, Size.SMALL_SIZE.value)
        y_sd = (y_resampled - y_resampled.mean()) / y_resampled.std()
        data = {"inputs": y_sd.reshape(1, -1).tolist()}
        response = requests.post(url, data=json.dumps(data))
        values = np.array(response.json()["outputs"][0])
        pred_test = np.argmax(values)
        decision = class_names[pred_test]
        return decision, self.softmax(values)
    
    
    def preprocessing(self):
        kernel = self._kernel
        name = kernel.label()
        if LargeKernelNames.is_valid(name) or SmallKernelNames.is_valid(name):
            if LargeKernelNames.is_valid(name):
                x_interp = np.linspace(-3, 3, Size.LARGE_SIZE.value)
            else:
                x_interp = np.linspace(-1, 1, Size.SMALL_SIZE.value)
            data = self.data
            x_train, y_train = data['X'],data['y']
            xtrain_transform, a, b = self.x_transform()
            y_interp = np.interp(x_interp, xtrain_transform, y_train)
            y_interp = (y_interp - y_interp.mean())/y_interp.std()
            return y_interp
        else:
            raise NotKernelError
    
    
    def fit(
            self,
            method=None,
            size=None,
            train_size=None,
            rayon=.001,
            granularity=99):
        '''
        This method allows the estimation of the hyperparameters of the GPR model.
        '''
        name = self.kernel.label()
        if any((SmallKernelNames.is_valid(name), LargeKernelNames.is_valid(name))):
            train = self.data
            meany, stdy = train['y'].mean(), train['y'].std()
            parmsfit_by_sampling = self.get_hyperparameters()
            if isinstance(self.kernel,Linear):
                    self.std_noise = parmsfit_by_sampling[0] * stdy
                    self.hyperparameters = dict(variance = stdy**2)
            elif isinstance(self.kernel,Polynomial):
                self.hyperparameters = dict(degree=3, offset = parmsfit_by_sampling[0], variance = stdy**2)
                self.std_noise = parmsfit_by_sampling[1] * stdy
            elif isinstance(self.kernel, RBF):
                self.kernel.hyperparameters = dict(length_scale = parmsfit_by_sampling[0],variance = stdy**2)
                self.std_noise = parmsfit_by_sampling[1] * stdy
            elif isinstance(self.kernel, Matern):
                self.kernel.hyperparameters = dict(nu = self.kernel.nu, length_scale = parmsfit_by_sampling[0],variance = stdy**2)
                self.std_noise = parmsfit_by_sampling[1] * stdy            
            elif isinstance(self.kernel, Periodic):
                if (method is None):
                    hyp_dict_list = self.get_periodic_hyperparameters()
                    periods = [w['period'] for w in hyp_dict_list]
                    periods_lh = [self.get_lh(period)[1] for period in periods]
                    lh_min = np.argmin(periods_lh)
                    p_est = periods[lh_min]
                    hyp_dict = hyp_dict_list[lh_min]
                    results = self.period_generator(
                        centre = p_est,
                        size=size,
                        train_size=train_size,
                        rayon=rayon,
                        granularity=granularity)

                    results = np.array(list(results))
                    optimal_period = results[np.argmin(results[:, 1]), 0]
                    hyp_dict['period'] = optimal_period
                    self._kernel.hyperparameters = hyp_dict
            
            xtrainTransform, a, b = self.x_transform()
            self._a = a
            self._b = b
            K_noise = self.kernel.count(xtrainTransform)
            np.fill_diagonal(K_noise, K_noise.diagonal() + self._sigma_n**2)
            invK_noise = pdinv(K_noise)
            self._invK = invK_noise
            
  

        else:
            raise NotKernelError('Unknown kernel.')

    
    def predict(self, xtest=None):
        '''
        This method allows long term prediction over an xtest position vector (time or space) via GPR model.
        It will be called when the "predict" method is used. It doesn't need to have updates of new data at regular horizon i.e. (ytest not necessary).
        '''

        train = self.data
        xtrain, ytrain = train['X'], train['y']
        if xtest is None:
            xtest = xtrain
            
        xtrain = Time.date2num(xtrain)
        xtest = Time.date2num(xtest)



        meany, stdy = ytrain.mean(), ytrain.std()

        ytrain = (ytrain - meany) / stdy
        xtrain_transform = self._b * xtrain + self._a
        xtest_transform = self._b * xtest + self._a
        invK_noise = self._invK
        Kstar = self._kernel.count(xtest_transform, xtrain_transform)
        y_pred_test = Kstar.T @ invK_noise @ ytrain

        ypred = (stdy * y_pred_test + meany)
        std_stars = self._kernel.count(xtest_transform,xtest_transform).T
        std_pred_test = std_stars - Kstar.T @ invK_noise @ Kstar
        std_pred_test = np.sqrt(np.abs(std_pred_test.diagonal()))

        return ypred, std_pred_test
    
    
   
                
    def log_likelihood(self, theta, noise=.1, size=None, train_size=None):
        

        x, y = self._xtrain, self._ytrain
        x = Time.date2num(x)
        if size is None:
            size = int(.25 * len(x))
            x_inter = np.linspace(x[0], x[-1], size)
            y_resample = np.interp(x_inter, x, y)
        else:
            size = int(size)
            x_inter = np.linspace(x[0], x[-1], size)
            y_resample = np.interp(x_inter, x, y)
            # x_inter = x
            # y_resample = y
        X_train, Y_train = np.linspace(-1, 1, size), y_resample

        def ls(a, b):
            return lstsq(a, b, rcond=-1)[0]

        if self._kernel.label() == 'Periodic':
            d = {'variance': 1, 'length_scale': 1, 'period': theta}
        else:
            d = {'variance': 1, 'length_scale': theta}

        kernel = self.kernel
        kernel.hyperparameters = d

        K = kernel.count(X_train) + noise**2 * np.eye(X_train.size)

        L = np.linalg.cholesky(K)
        return np.sum(np.log(np.diagonal(L))) +             0.5 * Y_train.dot(ls(L.T, ls(L, Y_train))) +             0.5 * len(X_train) * np.log(2 * np.pi)


    def get_lh(self, v, size=None):
        try:
            lh = self.log_likelihood(v, size=size)
        except Exception:
            lh = np.inf
        return v, lh

    def period_generator(
            self,
            centre,
            size=None,
            train_size=None,
            rayon=.001,
            granularity=99):

        params = np.linspace(centre - rayon, centre + rayon, granularity)
        params = np.hstack([params, centre])
        params = np.sort(params)

        for period in ProgressBar(params,  desc="==> Training",ascii=False, ncols=75):
            yield self.get_lh(period, size = size)

            
    @staticmethod                
    def most_frequent(List):
        '''return most frequent element of list.'''
        occurence_count = Counter(List)
        return occurence_count.most_common(1)[0][0]

            
            
    @staticmethod        
    def fetch(session, url, data):
        with session.post(url, data=json.dumps(data)) as response:
            result = np.array(response.json()["outputs"][0])
            return result
        
    @staticmethod        
    def requests_retry_session(
        retries=3,
        backoff_factor=0.3,
        status_forcelist=(500, 502, 504),
        session=None):
        session = session or requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            method_whitelist=frozenset(['GET', 'POST']),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def get_periodic_hyperparameters(self):
        '''Return hyperparameters of Gaussian Process Regressor if instance is periodic one.'''

        (URL, URL_LS, URL_IID) = API.get_url(self.kernel.label())
        x, y = self._xtrain, self._ytrain
        x = Time.date2num(x)
        std_y = y.std()
        y = (y - y.mean()) / std_y
        min_p = 50 / y.size
        p = np.linspace(min_p, 1, 100)
        mm = y.size
        int_p = mm * p
        int_p = int_p.astype(int)
        int_p = np.unique(int_p)
        y_list = [y[-s:] for s in int_p]
        data_list = []
        for _ in y_list:
            f = interpolate.interp1d(np.linspace(-1, 1, len(_)), _)
            xnew = np.linspace(-1, 1, Size.SMALL_SIZE.value)
            ynew = f(xnew)
            ynew = (ynew - ynew.mean()) / ynew.std()
            data = {"inputs": ynew.reshape(1, -1).tolist()}
            data_list.append(data)

        tt = len(data_list)
        with requests.Session() as session:
            std_noise = self.fetch(session, URL_IID, data_list[-1])
        self._sigma_n = std_noise[0] * std_y
        result = []
        with ThreadPoolExecutor(max_workers=50) as executor:
            result += executor.map(self.fetch,
                                   [self.requests_retry_session()] * tt,
                                   [URL] * tt,
                                   data_list)
            executor.shutdown(wait=True)
        result = np.array(result).ravel()
        result = result * np.array(int_p) / mm
        sd_result = (result - result.mean()) / result.std()
        p_estimate_ = sd_result[np.argmin(np.abs(np.diff(sd_result)))]
        p_estimate = result[sd_result == p_estimate_]
        sd_result_round = np.round(sd_result, 2)
        p_freq_round = self.most_frequent(sd_result_round)
        p_freq_set = result[sd_result_round == p_freq_round]
        p_freq = p_freq_set.mean()

        z_ls = y[:int(2 * p_freq * y.size)]
        z_ls_resized = resample(z_ls, Size.SMALL_SIZE.value)
        z_ls_resized = (z_ls_resized - z_ls_resized.mean()) / z_ls_resized.std()
        data_ls = {"inputs": z_ls_resized.reshape(1, -1).tolist()}

        with requests.Session() as session:
            length_scale = self.fetch(session, URL_LS, data_ls)
        length_scale = length_scale[0]

        y_resized = resample(y, Size.SMALL_SIZE.value)
        y_resized = (y_resized - y_resized.mean()) / y_resized.std()
        data_resample = {"inputs": y_resized.reshape(1, -1).tolist()}

        with requests.Session() as session:
            period_normal = self.fetch(session, URL, data_resample)
        period_normal = period_normal[0]

        hyp_dict2 = dict(zip(["length_scale", "period"], [length_scale, p_freq]))
        hyp_dict2["variance"] = std_y**2
        hyp_dict3 = dict(zip(["length_scale", "period"],
                             [length_scale, p_estimate[0]]))
        hyp_dict3["variance"] = std_y**2
        hyp_dict4 = hyp_dict2.copy()
        hyp_dict4['period'] = period_normal
        return hyp_dict2, hyp_dict3
    
    def single_update(self, x_update, y_update, method='sliding'):
        condition1 = isinstance(self.kernel,(Periodic,Linear,Constant,Polynomial))
        condition2 = x_update in self._xtrain
        condition3 = self._ytrain[np.where(self._xtrain == x_update)] == y_update
        if condition1 or condition2:
            return None
        else:
            x_train, y_train = self._xtrain, self._ytrain
            x_train_num = Time.date2num(x_train)
            xt0 = x_update
            yt = y_update
            xt = Time.date2num(xt0)
            meany, stdy = y_train.mean(), y_train.std()
            y_train = (y_train - meany) / stdy
            yt = (yt - meany) / stdy
            xtest_transform = self._b * xt + self._a
            xtrain_transform = self._b * x_train_num + self._a

            invK = self._invK
            x = self._kernel.count(xtest_transform, xtrain_transform)
            r = self._kernel.count(xtest_transform, xtest_transform)

            invK = inv_col_add_update(invK, x, r + self._sigma_n**2)

            if method == 'sliding':
                if isinstance(self._xtrain, (pd.DatetimeIndex)):
                    self._xtrain = self._xtrain[1:].insert(
                        len(self._xtrain[1:]), x_update)
                elif isinstance(self._xtrain, pd.Series):
                    self._xtrain = self._xtrain[1:].append(
                        pd.Series(x_update), ignore_index=True)
                else:
                    self._xtrain = np.hstack((self._xtrain[1:], x_update))
                self._ytrain = np.hstack((self._ytrain[1:], y_update))
                self._invK = inv_col_pop_update(invK, 0)

            elif method == 'concat':
                if isinstance(self._xtrain, (pd.DatetimeIndex)):
                    self._xtrain = self._xtrain[1:].insert(
                        len(self._xtrain), x_update)
                elif isinstance(self._xtrain, pd.Series):
                    self._xtrain = self._xtrain.append(
                        pd.Series(x_update), ignore_index=True)
                else:
                    self._xtrain = np.hstack((self._xtrain, x_update))

                self._ytrain = np.hstack((self._ytrain, y_update))
                self._invK = invK
            else:
                raise ValueError(f"Not defined '{method}' method")

   
    def update(self, x_update, y_update, method="sliding"):
        if (isinstance(x_update, pd.Series) & isinstance(y_update, pd.Series)):
            data = [{'x_update': x_update.iloc[u], 'y_update':y_update.iloc[u]}
                    for u in range(len(y_update))]
        elif (isinstance(x_update, pd.Series) & isinstance(y_update, np.ndarray)):
            data = [{'x_update': x_update.iloc[u], 'y_update':y_update[u]}
                    for u in range(len(y_update))]
        elif (isinstance(y_update, pd.Series) & isinstance(x_update, np.ndarray)):
            data = [{'x_update': x_update[u], 'y_update':y_update.iloc[u]}
                    for u in range(len(y_update))]
        elif (isinstance(x_update, np.ndarray) & isinstance(y_update, np.ndarray)):
            data = [{'x_update': x_update[u], 'y_update':y_update[u]}
                    for u in range(len(y_update))]
        elif (isinstance(x_update, pd.DatetimeIndex) & isinstance(y_update, np.ndarray)):
            data = [{'x_update': x_update[u], 'y_update':y_update[u]}
                    for u in range(len(y_update))]
        elif (isinstance(x_update, pd.DatetimeIndex) & isinstance(y_update, pd.Series)):
            data = [{'x_update': x_update[u], 'y_update':y_update.iloc[u]}
                    for u in range(len(y_update))]
        else:
            data = [{'x_update': x_update, 'y_update': y_update}]

        for d in data:
            self.single_update(**d, method=method)


