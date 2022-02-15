
<!-- ![alt text](assets/logo.png)
 -->
<img src="assets/logo.png" alt="makeprediction logo" width="300px"/>
<!-- <img src="assets/logo_1.png" alt="makeprediction logo" width="300px"/>
 -->


MakePrediction is a package for building an automatic Gaussian process regression (GPR) models for time series prediction in Python. It was originally created by [Hanany Tolba].
 
 * MakePrediction is an open source project. If you have relevant skills and are interested in contributing then please do contact us (hananytolba@yahoo.com).*

Gaussian process regression (GPR):
=====================================
The advantages of this Gaussian processes package:
* Very fast training.
* Very fast prediction.
* The prediction can be interpolated as desired. 
* The training of the model is automatic, no kernel function needs to be specified. An optimal choice of kernel is automatically elaborated.
* Possibility to choose a kernel function manually.
* The prediction is probabilistic (Gaussian) so that confidence intervals can be calculated and used to decide whether to make a strategic decision.  
* The package provides an API for deployment. 

   
## Where do you find time series?
* Energy
* Finance 
* Medical, Biotech, and Healthcare
* IoT Monitoring 
* Supply Chain
* Agriculture
* Retail


## What does makeprediction do?
* Modelling and analysis time series.
* Automatic time-series prediction (forecasting).
* Real-Time time series prediction.
* Deploy on production the fitted (or saved) makeprediction model.

### Applications:
* Energy consumption prediction. 
* Energy demand prediction.
* Stock price prediction.
* Stock market prediction.
* ...
### Latest release from PyPI

* pip install makeprediction

### Latest source from GitHub

*Be aware that the `master` branch may change regularly, and new commits may break your code.*

[MakePrediction GitHub repository](https://github.com/HananyTolba/MakePrediction.git), run:

* pip install .

Example
==========================

Here is a simple example:

```python
import pandas as pd
import numpy as np

from makeprediction.gpts import GaussianProcessTimeSerie
from makeprediction.kernels import RBF, White
from makeprediction.visualization import Visualizer

#generate time series
###############################
  
x = pd.date_range(start = datetime.datetime(2021,1,1), periods=1000, freq = '3s' )
time2num = date2num(x)

# Simulate a data
date = pd.date_range(start = '2022',periods = 1000, freq = '30T')
# As sum of RBF kernel and Gaussian noise kernel
kernel = RBF() + White(variance = .01) 
# add mean and variance 
data = 1000 + 50*kernel.simulate(date, seed = np.random.seed(1234))
# create a dataframe with data
df = pd.DataFrame(data = data, index = date, columns=['value'])
df.head()


# split time serie into train and test
TRAIN_SIZE = int(.8*len(df))
df_train, df_test = df[:TRAIN_SIZE], df[TRAIN_SIZE:]

# Create an instance of the class GaussianProcessTimeSerie with train data:
#########################################
model = GaussianProcessTimeSerie(df_train.index, df_train.value)
# Show train data with test data 
Visualizer.iplot(model, df_test.index, df_test.value)

```
<img src="assets/fig1.svg" alt="makeprediction logo" width="700px"/>

```python
# fit the model
model.fit()
```


```python
#predict with model and plot result
model.predict(df_test.index)
Visualizer.iplot(model, df_test.index, df_test.value)

```
<img src="assets/fig2.svg" alt="makeprediction logo" width="700px"/>


```python
# Show initial score 
model.score(df_test.value)

# Online prediction with updating 
ypred = np.empty(shape = (0,))
ypred_std = np.empty(shape = (0,))

for x,y in df_test.itertuples():
    # predict for x value
    yp,yp_std = model.predict(x,return_value = True)
    ypred = np.append(ypred,yp)
    ypred_std = np.append(ypred_std,yp_std)
    # update the model for (x,y)
    data = {'x_update': x, 'y_update': y}
    model.update(**data)
#Set all new prediction to the model    
model.set_prediction(df_test.index, ypred, ypred_std)

# Show new prediction 

Visualizer.iplot(model, df_test.index, df_test.value)
```
<img src="assets/fig_pred.svg" alt="makeprediction logo" width="700px"/>

The previous prediction with updating, can be obtained simply by the "predict" method as follows:


<img src="assets/fig3.svg" alt="makeprediction logo" width="700px"/>

```python
# Show new score 
model.score(df_test.value)

{'TrainErrors': {'MBE': 1.9652784019209927e-05,
  'MAE': 3.8085808450980743,
  'MSE': 22.963876468774956,
  'RMSE': 4.792063904913514,
  'NRMSE': 0.004642410409421008,
  'R2': 0.9867205336559287},
 'TestErrors': {'MBE': -0.3680353774898197,
  'MAE': 4.749355093190354,
  'MSE': 34.36287534118008,
  'RMSE': 5.8619856142078755,
  'NRMSE': 0.005905883435157759,
  'R2': 0.9879230611904987}}
```