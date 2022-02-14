import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from makeprediction.invtools import fast_pd_inverse as pdinv
from makeprediction.invtools import inv_col_add_update, inv_col_pop_update

########################################################
A = np.random.RandomState(314).normal(0, 1, (5, 5))
A = A @ A.T
m = np.random.RandomState(314).normal(0, 1, (5, 1))
r = np.array([1.])

inv_A = np.linalg.inv(A)


def test_pdinv():
    assert_almost_equal(inv_A, pdinv(A))


A_1 = A[1:, 1:]


def test_inv_col_pop_update():
    assert_almost_equal(inv_col_pop_update(inv_A, 0), np.linalg.inv(A_1))


A_augmented = np.block([[A, m], [m.T, r]])
# gpr = GPR(x,y)
inv_A_augmented = inv_col_add_update(inv_A, m, r)


def test_inv_col_add_update():
    A_augmented
    assert_almost_equal(inv_A_augmented, np.linalg.inv(A_augmented))
