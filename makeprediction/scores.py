
from abc import ABC, abstractmethod
import numpy as np
from makeprediction.exceptions import NotFittedError
import pandas as pd 
from makeprediction.metrics import Metrics



class IModelScore(ABC):   
    @abstractmethod
    def score():
        pass

class ModelScore(IModelScore):
    @classmethod
    def score(cls,model, ytest = None):
        """Compute  all  metrics for model evaluation in regression:
        - R Square/Adjusted R Square.
        - Mean Square Error(MSE).
        - Root Mean Square Error(RMSE).
        - Normalized Root Mean Square Error(NRMSE)
        - Mean Absolute Error(MAE).
        """
    
        try:
            mae = Metrics.mae(model._ytrain, model._yfit)
            mse = Metrics.mse(model._ytrain, model._yfit)
            rmse = Metrics.rmse(model._ytrain, model._yfit)
            nrmse = Metrics.nrmse(model._ytrain, model._yfit)
            r2 = Metrics.r2(model._ytrain, model._yfit)
            mbe = Metrics.mbe(model._ytrain, model._yfit)
        except (AttributeError):
            raise NotFittedError("Model is not fitted yet.")

        L_score = [
                mbe,
                mae,
                mse,
                rmse,
                nrmse,
                r2,
                ]
        d1 = dict(zip(["MBE", "MAE", "MSE", "RMSE", "NRMSE", "R2"], L_score))

        if ((model._ypred is None) | (ytest is None)):
            result = {"TrainErrors": d1}
            return pd.DataFrame(result)
        else:
            mae_pred = Metrics.mae(ytest, model._ypred)
            mse_pred = Metrics.mse(ytest, model._ypred)
            rmse_pred = Metrics.rmse(ytest, model._ypred)
            nrmse_pred = Metrics.nrmse(ytest, model._ypred)
            r2_pred = Metrics.r2(ytest, model._ypred)
            mbe_pred = Metrics.mbe(ytest, model._ypred)

            L_score_pred = [mbe_pred, mae_pred, mse_pred, rmse_pred, nrmse_pred, r2_pred]
            d2 = dict(zip(["MBE", "MAE", "MSE", "RMSE", "NRMSE", "R2"], L_score_pred))
            result = {"TrainErrors": d1, "TestErrors": d2}
            return pd.DataFrame(result)
    