import enum
import os
# from .kernels import *
# from .tools import LargeKernelNames, SmallKernelNames
from urllib.parse import urljoin
import re



class Instance(enum.Enum):
    SIMPLE = os.environ.get("BASE_URL", 'https://simple.makeprediction.com/')
    PERIODIC = os.environ.get("BASE_URL", 'https://periodic.makeprediction.com/')

class ContainerName(enum.Enum):
    pass

class PeriodicContainerName(ContainerName):

    Periodic = "periodic_model"
    PeriodicPlusMatern = "tf_PeriodicPlusMatern"
    PeriodicPlusRBF = "tf_PeriodicPlusRBF"
    PeriodicModel = "periodic_model"
    which_stationarity = "stationary_kernel_predict"
    @classmethod
    def attrmatch(cls,value):
        return hasattr(cls,value)
    @classmethod
    def get(cls,value):
        if cls.attrmatch(value):  
            return getattr(cls,value)

    
class PeriodicLengthScaleIID(ContainerName):
    
    LS = "tf_Periodic_ls"
    IID = "iid_periodic_300"
    
class SimpleContainerName(ContainerName):

    model_expression_predict = "model_expression_predict"
    gp_kernel_predict_300 = "gp_kernel_predict_300"
    gp_kernel_predict_simple_300 = "gp_kernel_predict_simple_300"
    is_periodic = "periodic_kernel_predict"
    is_linear = "linear_kernel_predict"
    RBF = "rbf_1d"
    Polynomial = "polynomial_1d"
    Linear = "linear_1d"
    Matern32 = "matern32_1d"
    Matern52 = "matern52_1d"
    Matern12 = "matern12_1d"

    @classmethod
    def attrmatch(cls,value):
        condition = (re.match("^Polynomial\([0-9]+\)$", value) is None),\
        (re.match("^Matern\([0-9]/[0-9]\)$", value) is None)
        if all(condition):
            return hasattr(cls,value)
        return True
    @classmethod
    def get(cls,value):
        if cls.attrmatch(value):
            if 'Polynomial' in value:
                return cls.Polynomial
            elif value == 'Matern(1/2)':
                return cls.Matern12
            elif value == 'Matern(3/2)':
                return cls.Matern32
            elif value == 'Matern(5/2)':
                return cls.Matern52            
            return getattr(cls,value)

class API:
    @classmethod
    def simple_url(cls,name):
        if SimpleContainerName.attrmatch(name):
            value = SimpleContainerName.get(name).value
            path = f"/{value.replace('_1d','')}/v1/models/{value}:predict"
            return urljoin(Instance.SIMPLE.value, path)
        
    @classmethod
    def periodic_url(cls,name):
        if PeriodicContainerName.attrmatch(name):
            value = PeriodicContainerName.get(name).value
            LS = PeriodicLengthScaleIID.LS.value
            IID = PeriodicLengthScaleIID.IID.value
            path1 = f"/{value}/v1/models/{value}:predict"
            path2 = f"/{LS}/v1/models/{LS}:predict"
            path3 = f"/{IID}/v1/models/{IID}:predict"
            urls = [urljoin(Instance.PERIODIC.value, path) for path in (path1,path2,path3)]
            return urls

    @classmethod
    def url_which_stationarity(cls):
        which_stationarity = PeriodicContainerName.which_stationarity.value
        url = f"{which_stationarity}/v1/models/{which_stationarity}:predict"
        return urljoin(Instance.PERIODIC.value, url)

    @classmethod
    def url_is_periodic(cls):
        is_periodic = SimpleContainerName.is_periodic.value
        url = f"/{is_periodic}/v1/models/{is_periodic}:predict"
        return urljoin(Instance.SIMPLE.value, url)

    @classmethod
    def url_is_linear(cls):
        is_linear = SimpleContainerName.is_linear.value
        url = f"/{is_linear}/v1/models/{is_linear}:predict"
        return urljoin(Instance.SIMPLE.value, url)

    
    @classmethod
    def get_url(cls,name):
        if SimpleContainerName.attrmatch(name):
            value = SimpleContainerName.get(name).value
            path = f"/{value.replace('_1d','')}/v1/models/{value}:predict"
            return urljoin(Instance.SIMPLE.value, path)
        elif PeriodicContainerName.attrmatch(name):
            value = PeriodicContainerName.get(name).value
            LS = PeriodicLengthScaleIID.LS.value
            IID = PeriodicLengthScaleIID.IID.value
            path1 = f"/{value}/v1/models/{value}:predict"
            path2 = f"/{LS}/v1/models/{LS}:predict"
            path3 = f"/{IID}/v1/models/{IID}:predict"
            urls = [urljoin(Instance.PERIODIC.value, path) for path in (path1,path2,path3)]
            return tuple(urls)
        return cls.url_which_stationarity()

