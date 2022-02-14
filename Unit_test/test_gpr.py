#!/usr/bin/env python
# -*- coding: utf-8 -*-

from makeprediction.api import PeriodicContainerName
from makeprediction.kernels import RBF, Matern, Linear, Polynomial, Periodic
from makeprediction import kernels
from makeprediction.kernels import Sum, Prod
from makeprediction.invtools import fast_pd_inverse as pdinv

from makeprediction.gaussianprocess import GaussianProcessRegressor as GPR

import pytest
import numpy as np
from numpy.testing import assert_almost_equal
import inspect

x = np.linspace(-3, 3, 10)


def f(s):
    return np.sin(s)


y = f(x).ravel()

ziped_kernels = filter(lambda w:not inspect.isabstract(w[1]) if w[0]!='ABC' else False, inspect.getmembers(kernels, inspect.isclass))
names,classes = zip(*ziped_kernels)

# kernel = RBF()
kers = [
    "periodic",
    "matern12",
    "linear",
    "matern32",
    "rbf",
    "matern52",
    "polynomial"]
kers = [RBF(), Periodic(), Matern(), Polynomial(), Linear()]


@pytest.mark.parametrize('kernel', kers)
def test_kernel_name(kernel):
    gpr = GPR(x, y, kernel=kernel)
    assert gpr.kernel.label()== kernel.label()

# @pytest.mark.parametrize('kernel', kers)
# def test_tf_model(kernel):
#     gpr = GPR(x,y)
#     gpr.kernel_choice = kernel
#     #print(type(gpr._model))
#     assert isinstance(gpr._model,tensorflow.keras.Model)


@pytest.mark.parametrize('kernel', kers)
def test_get_hyperparameters(kernel):
    gpr = GPR(x, y, kernel = kernel)
    parms = gpr.kernel.hyperparameters
    if isinstance(kernel, Periodic):
        assert parms["length_scale"] == 1
        assert parms["period"] == 1
        assert parms["variance"] == 1
    elif isinstance(kernel, RBF):
        assert parms["length_scale"] == 1
        assert parms["variance"] == 1
    elif isinstance(kernel, Matern):
        assert parms["length_scale"] == 1
        assert parms["variance"] == 1
        assert parms["nu"] == .5
    elif isinstance(kernel, Polynomial):
        assert parms["offset"] == 0
        assert parms["variance"] == 1
        assert parms["degree"] == 2
    elif isinstance(kernel, Linear):
        assert parms["variance"] == 1

@pytest.mark.parametrize('kernel', kers)
def test_set_hyperparameters(kernel):
    gpr = GPR(x, y, kernel = kernel)
    parms_per = {"length_scale": .5, "period": .5, "variance": 2}
    parms = {"length_scale": .5, "variance": 2}

    if isinstance(kernel, Periodic):
        gpr.kernel.hyperparameters = parms_per
        assert gpr.kernel.hyperparameters == parms_per
    if isinstance(kernel, RBF):
        gpr.kernel.hyperparameters = parms
        assert gpr.hyperparameters == parms



kers = [RBF(), Matern()]
@pytest.mark.parametrize('kernel', kers)
def test_prediction(kernel):
    gpr = GPR(x, y,kernel = kernel)
    gpr.fit()
    gpr.std_noise = .0001
    xtrainTransform, a, b = gpr.x_transform()
    K_noise = gpr._kernel.count(
        xtrainTransform,
        xtrainTransform)
    np.fill_diagonal(K_noise, K_noise.diagonal() + gpr._sigma_n**2)

    invK_noise = pdinv(K_noise)

    gpr._invK = invK_noise

    y_pred, y_cov = gpr.predict()
    # print(y_pred,"y : ", y)

    assert_almost_equal(y_pred, y, decimal=5)
    assert_almost_equal(y_cov**2, 0., decimal=5)


# A = np.random.RandomState(314).normal(0, 1, (4,4))
# A = A + A.T
# m = np.random.RandomState(314).normal(0, 1, (4,1))
# r = np.array([1.])

# inv_A = np.linalg.inv(A)

# A_augmented = np.block([[A, m], [m.T, r]])
# gpr = GPR(x,y)
# inv_A_augmented = gpr._inv_add_update(inv_A, m, r)
# def test_invupdate():
#     A_augmented
#     assert_almost_equal(inv_A_augmented,np.linalg.inv(A_augmented))
