#!/usr/bin/env python
# -*- coding: utf-8 -*-

from makeprediction.kernels import RBF, Matern, Cosine
from makeprediction.kernels import Periodic, Polynomial, Linear
from makeprediction.kernels import Sum, Prod

import pytest
import numpy as np


kernels = [RBF(length_scale=1.0),
           # Cosine(length_scale=1.0),
           Matern(length_scale=1.0),
           Matern(length_scale=1.0),
           Matern(length_scale=1.0),
           # Exponential(length_scale=1.0),
           # Linear(0),
           Periodic(length_scale=1.0, period=1.0),

           ]


kernels_labels = [RBF(),
                  Matern(),
                  Matern(),
                  Matern(),
                  Polynomial(),
                  Linear(),
                  Periodic(),

                  ]


# @pytest.mark.parametrize('kernel', kernels_labels)
# def test_getLabel(kernel):
#     assert kernel.__class__.__name__ == kernel.label()


# @pytest.mark.parametrize('kernel', kernels)
# def test_get_hyperparameters(kernel):

#     parms = kernel.get_hyperparameters()

#     if kernel.__class__.__name__ == "Periodic":
#         assert parms["length_scale"] == 1.
#         assert parms["period"] == 1.
#         assert parms["variance"] == 1.

#     else:
#         assert parms["length_scale"] == 1.
#         assert parms["variance"] == 1.


# parms_per = {"length_scale": .5, "period": .3, "variance": 2}
# parms = {"length_scale": .1, "variance": 2}


# @pytest.mark.parametrize('kernel', kernels)
# def test_set_hyperparameters(kernel):

#     if kernel.__class__.__name__ == "Periodic":
#         kernel.set_hyperparameters(parms_per)
#         assert kernel._length_scale == .5
#         assert kernel._period == .3
#         assert kernel._variance == 2

#     else:
#         kernel.set_hyperparameters(parms)
#         assert kernel._length_scale == .1
#         assert kernel._variance == 2


x1 = np.random.RandomState(314).normal(0, 1, (5, 1))
x2 = np.random.RandomState(314).normal(0, 1, (6, 1))

# xx = X
# xy = Y
lin = Linear()


def test_lin_count():

    Kx = lin.count(x1)
    Kxx = lin.count(x1, x1)
    Kx1x2 = lin.count(x1, x2)
    Kx2x1 = lin.count(x2, x1)

    assert np.array_equal(Kx, Kxx)
    assert np.array_equal(Kx, Kx.T)
    assert np.array_equal(Kx1x2, Kx2x1.T)


kernels_ = [RBF(length_scale=1.0),
            Cosine(length_scale=1.0),
            Matern(length_scale=1.0),
            Matern(length_scale=1.0, nu = 1.5),
            Matern(length_scale=1.0, nu = 2.5),
            Periodic(length_scale=1.0, period=1.0),
            ]


@pytest.mark.parametrize('kernel', kernels_)
def test_radial_dist(kernel):
    # if kernel.__class__.__name__ != "Linear":
    Kx = kernel.radial_dist(x1)
    Kxx = kernel.radial_dist(x1, x1)
    Kx1x2 = kernel.radial_dist(x1, x2)
    Kx2x1 = kernel.radial_dist(x2, x1)

    assert np.array_equal(Kx, Kxx)
    assert np.array_equal(Kx, Kx.T)
    assert np.array_equal(Kx1x2, Kx2x1.T)


k1 = RBF(length_scale=1)
k2 = Periodic(length_scale=1, period=.5)

SumKer = Sum(k1, k2)


def test_KernelSum_count():
    K = SumKer.count(x1, x2)
    K1plusK2 = k1.count(x1, x2) + k2.count(x1, x2)
    assert np.array_equal(K, K1plusK2)


ProdKer = Prod(k1, k2)


def test_KernelProd_count():
    K = ProdKer.count(x1, x2)
    K1prodK2 = k1.count(x1, x2) * k2.count(x1, x2)
    assert np.array_equal(K, K1prodK2)


# kernels = [RBF(length_scale=1.0),
#            # Cosine(length_scale=1.0),
#            Matern12(length_scale=1.0),
#            Matern32(length_scale=1.0),
#            Matern52(length_scale=1.0),
#            # Exponential(length_scale=1.0),
#            # Linear(),
#            Periodic(length_scale=1.0, period=1.0),

#            ]


# @pytest.mark.parametrize('kernel', kernels)
# def test_kernel_get_length_scale(kernel):
#     parms = kernel.get_length_scale()
#     assert parms == 1.0


# hyperparms = .3


# @pytest.mark.parametrize('kernel', kernels)
# def test_kernel_set_length_scale(kernel):
#     kernel.set_length_scale(hyperparms)
#     assert kernel.get_length_scale() == .3


# def test_periodic_get_period():
#     period = Periodic(length_scale=1, period=1.3).get_period()
#     assert period == 1.3


# def test_periodic_set_period():
#     per = Periodic(length_scale=1, period=1)
#     per.set_period(.3)
#     assert per.get_period() == .3


# # @pytest.mark.parametrize('kernel', kernels)
# # def test_set_hyperparameters(kernel):

# #     parms_per = {"length_scale":.5,"period":.5}
# #     parms = {"length_scale":.5}

# #     if kernel.__class__.__name__ == "Periodic":
# #         kernel.set_hyperparameters(parms_per)
# #         assert kernel._length_scale == .5
# #         assert kernel._period == .5
# #     else:
# #         assert kernel._length_scale == .5

#     # assert all(v == actual_dict[k] for k,v expected_dict.items()) and
#     # len(expected_dict) == len(actual_dict)

#     # else:
#     #    kernel.set_hyperparameters(parms)
# #     assert  kernel.get_hyperparameters() == parms


# # @pytest.mark.parametrize('kernel', kernels)
# # def test_kernel_get_parms(kernel):
# # 	assert kernel.get_length_scale() == 1.

# # @pytest.mark.parametrize('kernel', kernels)
# # def test_kernel_set_parms(kernel):

# #     hyperparms = np.array([0.5])
# #     kernel.set_length_scale(hyperparms)
# #     assert kernel.get_length_scale() == hyperparms
