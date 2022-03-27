#!/usr/bin/env python
# -*- coding: utf-8 -*-


# __author__ = "Hanany Tolba"
# __copyright__ = "Copyright 2020, Guassian Process by Deep Learning Project"
# __credits__ = ["Hanany Tolba"]
# __license__ = "Apache License 2.0"
# __version__ = "0.0.1"
# __maintainer__ = "Hanany Tolba"
# __email__ = "hanany100@gmail.com"
# __status__ = "Production"
'''
This tools for speed positive symmetric matrix inversion.
'''
# from dateutil.parser import parse
# from dateutil.parser import ParserError

import pandas as pd
import datetime

from scipy.linalg import lapack
import numpy as np
# from numpy.testing import assert_almost_equal
# from matplotlib import dates as dts
# from matplotlib.dates import datestr2num

inds_cache = {}


def uppertriangular_2_symmetric(ut):
    n = ut.shape[0]
    try:
        inds = inds_cache[n]
    except KeyError:
        inds = np.tri(n, k=-1, dtype=np.bool)
        inds_cache[n] = inds
    ut[inds] = ut.T[inds]


def fast_pd_inverse(m: np.ndarray) -> np.ndarray:
    '''
    This method calculates the inverse of a A real symmetric positive definite (n × n)-matrix
    It is much faster than Numpy's "np.linalg.inv" method for example.
    '''
    try:
        cholesky, info = lapack.dpotrf(m)
        if info != 0:
            raise ValueError('dpotrf failed on input {}'.format(m))
            # print("cas 1")
        inv, info = lapack.dpotri(cholesky)
        if info != 0:
            raise ValueError('dpotri failed on input {}'.format(cholesky))
            # print("cas 2")
    except BaseException:
        inv = np.linalg.inv(m)

    uppertriangular_2_symmetric(inv)
    return inv


def inv_col_add_update(
        A: np.ndarray,
        x: np.ndarray,
        r: float) -> np.ndarray:
    '''
    This method update the inverse of a matrix appending one column and one row.
    Assume we have a  kernel matrix (A) and we known its inverse. Now,
    for prediction reason in GPR model, we expand A with one coulmn and one row, A_augmented = [A x;x.T r]
    and wish to know the inverse of A_augmented. This function calculate the inverse of
    A_augmented using block matrix inverse formular,
    hence much faster than direct inverse using for example
    Numpy function "np.linalg.inv(A_augmented)".
    '''
    x = x.reshape(-1, 1)
    # x.T = x.reshape(1, -1)

    (n, m) = A.shape
    if n != m:
        raise('Matrix should be square.')

    # if (A,A.T,decimal=7)
    #     raise('Matrix should be symmetric.')

    Ax = np.dot(A, x)

    q = 1 / (r - np.dot(Ax.T, x))

    M = np.block([[A + np.dot(q * Ax, Ax.T), -q * Ax], [-q * Ax.T, q]])

    return M


def inv_col_pop_update(A: np.ndarray, c: int) -> np.ndarray:
    '''
    This method update the inverse of a matrix  when the i-th row and column are removed.

    '''
    (n, m) = A.shape
    if n != m:
        raise('Matrix should be square.')

    q = A[c, c]
    Ax = np.delete(A, c, axis=0)[:, c]
    Ax = Ax.reshape(-1, 1)
    yA = np.delete(A, c, axis=1)[c, :]
    yA = yA.reshape(1, -1)

    M = np.delete(np.delete(A, c, axis=1), c, axis=0) - (Ax / q) @ yA

    return M


def date2num(x):

    # if isinstance(x,pd.Series):
    #         if x.dtype == np.float64:
    #             return np.array(x)
    #         elif x.dtype == np.int64:
    #             return np.array(x)

    if isinstance(x, list):
        x = np.array(x)
    if isinstance(x, (int, float)):
        x = np.array(x)

    if isinstance(x, (datetime.datetime, pd.DatetimeIndex, pd.Series, str)):
        date = pd.to_datetime(x)
        # print(type(date))
        try:
            dt = date.astype(np.int64) / 10 ** 9
            dt = np.array(dt)
        except BaseException:
            dt = pd.to_datetime(x).timestamp()
            dt = np.array(dt)

    elif isinstance(x, np.ndarray):
        if x.dtype == np.float64:
            dt = np.array(x)
        elif x.dtype == np.int64:
            dt = x.astype(np.float64)
        elif np.issubdtype(x.dtype, np.datetime64):
            dt = x.astype(np.int64) / 10**9
        elif np.issubdtype(x.dtype, np.str_):
            date = pd.to_datetime(x)
            dt = date.astype(np.int64) / 10 ** 9
            dt = np.array(dt)
        elif np.issubdtype(x.dtype, np.object_):
            date = pd.to_datetime(x)
            dt = date.astype(np.int64) / 10 ** 9
            dt = np.array(dt)

        else:
            raise ValueError('error')

    return dt
