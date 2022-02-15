from copy import deepcopy
import datetime
import numpy as np
import pandas as pd
from makeprediction.gaussianprocess import GaussianProcessRegressor
from makeprediction.kernels import *
from makeprediction.tools import IsPeriodic, IsLinear, IsMaternOrRBF
from makeprediction.api import API
from makeprediction.scores import ModelScore
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from scipy.signal import resample
from makeprediction.tools import Size, softmax
import requests
import json
from makeprediction.tools import  ProgressBar
from makeprediction.exceptions import NotKernelError


class IGaussianProcessTimeSerie(ABC):
    '''
    This class implements a Python interface abstract Gaussian process time serie model.
    '''
    @abstractmethod
    def fit():
        pass
    @abstractmethod
    def predict():
        pass
    @abstractmethod
    def update():
        pass




class GaussianProcessTimeSerie(IGaussianProcessTimeSerie):
    '''
    This class implements a Time Serie Gaussian Process Regressor model.
    '''

    def __init__(
            self,
            xtrain=None,
            ytrain=None,
            kernel=None,
            yfit=None,
            std_yfit=None,
            gprs=None,
            components=None,
            xtest=None,
            ypred=None,
            std_ypred=None,
            # score=None
            ):
        self._xtrain = xtrain
        self._ytrain = ytrain
        self._kernel = kernel
        self._gprs = gprs  
        self.components = components
        self._yfit = yfit
        self._std_yfit = std_yfit
        self._xtest = xtest
        self._ypred = ypred
        self._std_ypred = std_ypred
        # self._score = score

    def __repr__(self):
        return "Instance of '{}'".format(self.__class__.__name__)

    def __str__(self):
        if self.kernel:
            message_print = f"GaussianProcessTimeSerie model with kernel: {self.kernel.label()}."
            return message_print
        return f"Empty GaussianProcessTimeSerie model."



    def __eq__(self, other, verbose=False):

        
        for k in self.__dict__.keys():
            try:
                if isinstance(self.__dict__[k], np.ndarray):
                    assert np.allclose(self.__dict__[k], other.__dict__[k], 1e-7)
                elif k == 'components':
                    for elm in range(len(self.__dict__[k])):
                        assert np.allclose(self.__dict__[k][elm], other.__dict__[k][elm], 1e-7)
                elif k == '_modelList':
                    if isinstance(self.__dict__[k], list):
                        for elem in range(len(self.__dict__[k])):
                            assert self.__dict__[k][elem] == other.__dict__[k][elem]
                    else:
                        assert self.__dict__[k] == other.__dict__[k]

                    # [self.__dict__[k][l] == other.__dict__[k][l] ]
                # elif k == '_score':
                #     assert np.allclose(list(self.__dict__[k]['train_errors'].values()), list(other.__dict__[k]['train_errors'].values()))
                elif k == '_kernel':
                    assert self.__dict__[k].label() == other.__dict__[k].label()
                    assert self.hyperparameters == other.hyperparameters
                elif isinstance(self.__dict__[k], pd.DatetimeIndex):
                    # print(k)
                    assert self.__dict__[k].equals(other.__dict__[k])
                else:
                    # print(k)
                    assert np.array_equal(self.__dict__[k], other.__dict__[k])

            except AssertionError:
                if verbose:
                    print(f"Models do not have  the same value of the attribute '{k}'.")
                return False
        return True

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
    def hyperparameters(self, new_hyperparameters: list) -> None:
        for kernel,new_hyperparameter in zip(self.kernel.iterkernels(),new_hyperparameters):
            kernel.hyperparameters = new_hyperparameter
        
    def deep_periodic_predict(self, x=None, y=None, split=None):
        if split is None:
            split = [.25, .5, .75, 1]
        if x is None:
            x = self._xtrain
        if y is None:
            y = self._ytrain

        n = y.size
        probs = []
        results = []

        for p in split:
            mm = GaussianProcessRegressor(x[-int(n * p):], y[-int(n * p):])
            res, prob = mm.is_periodic()            
            probs.append(prob.tolist())
            results.append(res)
        if IsPeriodic.Periodic.name in results:
            probs_filtered = [
                probs[i] for i in range(
                    len(results)) if results[i] == IsPeriodic.Periodic.name]
            probs_max = probs_filtered[np.argmax(
                [prob[1] for prob in probs_filtered])]
            return IsPeriodic.Periodic.name, probs_max
        else:
            probs_max = probs[np.argmax([prob[0] for prob in probs])]
            return IsPeriodic.NonPeriodic.name, probs_max

      
    def score(self,ytest = None):
        return ModelScore.score(self, ytest)
    

    
    def autofit(
            self,
            max_periodic=None,
            stationary_kernel=None,
            size=None,
            train_size=None,
            rayon=.001,
            granularity=99):
        ''' Automatically fit a model to a time series.'''
        if max_periodic is None:
            max_periodic = 3
        x, y = self._xtrain, self._ytrain


        models = []
        comp = []
        sig_list = []
        decision, Prob = GaussianProcessRegressor(x, y).is_linear()
        # print(decision, Prob)
        # print(IsLinear.Linear.name)
        if decision == IsLinear.Linear.name:
            model = GaussianProcessRegressor(x, y, Linear())
            model.fit()
            copy_model = deepcopy(model)
            models.append(copy_model)
            yf, sig = model.predict()
            comp.append(yf)
            sig_list.append(sig)
            y = y - yf

        periodic_number = 0

        while (periodic_number < max_periodic):

            model = GaussianProcessRegressor(x, y, Periodic())


            dec1, Prob1 = self.deep_periodic_predict(x, y)
            # print(dec1 , Prob1)

            period_0 = 10

            if dec1 == IsPeriodic.Periodic.name and max(Prob1) > .999:
                periodic_number += 1

                model.fit(
                    size=size,
                    train_size=train_size,
                    rayon=rayon,
                    granularity=granularity)
                hyps = model.hyperparameters
                period = hyps["period"]
                if np.abs(period - period_0) < .01:
                    model.fit(
                        size=size,
                        train_size=train_size,
                        rayon=rayon,
                        granularity=granularity)

                copy_model = deepcopy(model)
                models.append(copy_model)
                yf, sig = model.predict()
                comp.append(yf)
                sig_list.append(sig)
                y = y - yf
                period_0 = period
            else:
                break
        if stationary_kernel is None:
            stationary_kernel, stationary_prob = self.which_stationarity()
        if stationary_kernel == IsMaternOrRBF.RBF.name:
            model = GaussianProcessRegressor(x, y, RBF())
        elif stationary_kernel == IsMaternOrRBF.Matern.name:
            model = GaussianProcessRegressor(x, y, Matern(nu=1.5))
        model.fit()
        copy_model = deepcopy(model)
        models.append(copy_model)
        yf, sig = model.predict()
        comp.append(yf)
        sig_list.append(sig)
        self._gprs = models
        self.components = comp
        self._yfit = sum(comp)
        self._std_yfit = sum(sig_list)
        self.kernel = sum(map(lambda w:w.kernel, self._gprs))

        
    def fit(
            self,
            method=None,
            max_periodic=None,
            stationary_kernel=None,
            size=None,
            train_size=None,
            rayon=.001,
            granularity=99):
        '''Fit a model to a time series according to a kernel if it is defined at the constructor level. 
        If not, find automatically the best kernel and fit the model. '''
        
        
        xtrain = self._xtrain
        ytrain = self._ytrain  
        if self.kernel is None:
            return self.autofit(
                max_periodic=max_periodic,
                stationary_kernel=stationary_kernel,
                size=size,
                train_size=train_size,
                rayon=rayon,
                granularity=granularity)
        
        # ziped_kernels = filter(lambda w:not inspect.isabstract(w[1]) if w[0]!='ABC' else False, inspect.getmembers(kernels, inspect.isclass))
        # names,classes = zip(*ziped_kernels)
        if not self.kernel.is_kernel():
            raise NotKernelError("The input kernel is not a valid kernel.")
            
        if isinstance(self.kernel,Sum):           
            self._gprs = []
            comp = []
            sig_list = []
            # for ker in self.kernel.decompose_sorted():
            for ker in self.kernel.decompose():

                model = GaussianProcessRegressor(xtrain, ytrain,kernel = ker)
                model.fit(method = method,
                        size=size,
                        train_size=train_size,
                        rayon=rayon,
                        granularity=granularity)
                self._gprs.append(deepcopy(model))
                yf, sig = model.predict()
                comp.append(yf)
                sig_list.append(sig)
                ytrain = ytrain - yf
                self.components = comp
                self._yfit = sum(comp)
                self._std_yfit = sum(sig_list)

        elif isinstance(self.kernel,Prod):
            raise ValueError("Tcholeskyhe input kernel must be a sum of kernels or a simple kernel.")

        else:
            model = GaussianProcessRegressor(xtrain, ytrain, kernel = self._kernel)
            model.fit(
                method,
                size=size,
                train_size=train_size,
                rayon=rayon,
                granularity=granularity)
            self._yfit, self._std_yfit = model.predict()
            self._kernel = model._kernel
            self.components = self._yfit
            self._gprs = deepcopy(model)

    def _retrain(self, kernel=None):
        '''This function automatically recovers the state of a pre-trained model.
        Its main use is the continuation of the training of a model if it is incomplete or inadequate.
        '''
        original = deepcopy(self)
        new_self = deepcopy(self)
        new_self._ytrain = new_self._ytrain - new_self._yfit
        new_self._kernel = kernel
        new_self.fit()
        return new_self, original

    def refit(self, kernel=None):
        '''This function automatically recovers the state of a pre-trained model.
        Its main use is the continuation of the training of a model if it is incomplete or inadequate.
        '''
        new_gp, original = self._retrain(kernel)
        # print(f'original kernel: {original.kernel}')
        # print(f'residu kernel: {new_gp.kernel}')

        new_self = deepcopy(original)
        if isinstance(new_self._gprs, list):
            if isinstance(new_gp._gprs, list):
                new_self._gprs.extend(new_gp._gprs)
                new_self.components.extend(new_gp.components)
            else:
                new_self._gprs.append(new_gp._gprs)
                new_self.components.append(new_gp.components)
        else:
            new_self._gprs = [new_self._gprs]
            new_self.components = [new_self.components]
            if isinstance(new_gp._gprs, list):
                new_self._gprs.extend(new_gp._gprs)
                new_self.components.extend(new_gp.components)
            else:
                new_self._gprs.append(new_gp._gprs)
                new_self.components.append(new_gp.components)
        new_self._kernel =  new_self._kernel + new_gp._kernel 
        new_self.predict(new_self._xtrain)
        new_self._yfit = new_self._ypred
        new_self._std_yfit = new_self._std_ypred
        new_self._ypred = None
        new_self._xtest = None
        return new_self      
                    
    def predict(
            self,
            xt=None,
            return_value = False,
            components = False
            ):

        if xt is None:
            self._xtest = self._xtrain.copy()
            self._ypred, self._std_ypred = self._yfit.copy(), self._std_yfit.copy()
            cmps = self.components
            return self._ypred, self._std_ypred
        else:
            self._xtest = xt
            if isinstance(self._gprs, GaussianProcessRegressor):
                ypred_, std_ = self._gprs.predict(xt)
                components = False
            elif isinstance(self._gprs, list):
                models = self._gprs
                if models is None:
                    raise NotFittedError("This GaussianProcessesRegressor instance is not fitted yet")
                yt_std_list = []
                yt_pred_list = []
                for mdl in models:
                    yt_pred, yt_std = mdl.predict(xt)
                    yt_pred_list.append(yt_pred)
                    yt_std_list.append(yt_std)
                ypred_ = sum(yt_pred_list)
                std_ = sum(yt_std_list)
                cmps = np.array(yt_pred_list)
        self._ypred = ypred_
        self._std_ypred = std_
        if return_value:
            if components:
                return ypred_, std_, cmps
            else:
                return ypred_, std_

    def update(self, x_update=None, y_update=None, method ='sliding'):
        if not isinstance(self._gprs, list):
            self._gprs.update(x_update, y_update, method = method)
        else:
            y = y_update
            for m in self._gprs:
                m.update(x_update, y, method = method)
                yp, _ = m.predict(x_update)
                y = y - yp
    
    def predict_with_update(self,xt,data: dict,return_value):
        if data is None:
            return self.predict(xt,return_value=return_value)
        else:
            self.update(**data)
            return self.predict(xt)

    def set_prediction(self,x_test, ypred,ypred_std):
        l = (len(x_test), len(ypred), len(ypred_std))
        if len(set(l)) == 1:
            self._xtest = x_test
            self._ypred = ypred
            self._std_ypred = ypred_std
        return 


        
    def evaluate(self,xt,yt, horizon = 1):
        chunks_xt = [xt[h:h + horizon] for h in range(0, len(xt), horizon)]
        chunks_yt = [yt[h:h + horizon] for h in range(0, len(yt), horizon)]
        prediction = []
        std_prediction = []
        for i in ProgressBar(range(len(chunks_xt)),  desc="==> Prediction",ascii=False, ncols=75):
            yp,yp_std = self.predict(chunks_xt[i], return_value = True)
            data = dict(x_update = chunks_xt[i],y_update = chunks_yt[i])
            self.update(**data)
            self.predict()
            prediction.extend(yp)
            std_prediction.extend(yp_std)
        self._xtest = xt
        self._ypred = np.array(prediction)
        self._std_ypred = np.array(std_prediction)
        # return prediction
    
    def which_stationarity(self):
        class_names = [x.name for x in IsMaternOrRBF]
        url = API.url_which_stationarity()
        y = self._ytrain
        y_resampled = resample(y, Size.SMALL_SIZE.value)
        y_sd = (y_resampled - y_resampled.mean()) / y_resampled.std()
        data = {"inputs": y_sd.reshape(1, -1).tolist()}
        response = requests.post(url, data=json.dumps(data))
        values = np.array(response.json()["outputs"][0])
        pred_test = np.argmax(values)
        decision = class_names[pred_test]
        return decision, softmax(values)
    

        

if '__main__'==__name__:
    import datetime
    kernel = White(.01) + RBF(length_scale=.8) 
    size = 1000
    x = np.linspace(0,10,size)
    start = datetime.datetime.now()
    end = start + pd.Timedelta(days=7)
    x = pd.date_range(datetime.datetime.now(), end = end, freq ='10min')
    seed = np.random.seed(314)
    y = kernel.simulate(x, seed = seed)

    trainSize = int(.7*len(x))
    x_train, y_train = x[:trainSize], y[:trainSize]
    x_test, y_test = x[trainSize:], y[trainSize:]

    print(Sum.mro())
    pattern1  = RBF() + RBF()
    pattern2 = sum(2*[RBF()])
    pattern2 == pattern1
    model1 = GaussianProcessTimeSerie(x_train,y_train,kernel = pattern2 )
    model1.fit()
    print(model1.kernel.hyperparameters)

    model1.predict(x_test)
    plt.figure(figsize = (15,8))
    plt.plot(model1._xtrain,model1._ytrain,'k')
    plt.plot(model1._xtrain,model1._yfit,'r')
    plt.plot(model1._xtest,y_test,'g')
    plt.plot(model1._xtest,model1._ypred,'b')
    plt.show()

