import numpy as np
import pdb

def gauss(mu, covar, x):

    n, d = 1, 1 #x.shape #TODO Fix
    j, k = 1, 1 #covar.shape #TODO Fix

    invcov = np.linalg.inv(covar)

    x = x - np.ones(n) * mu
    fact = np.sum(np.matmul(x, invcov) * x)

    y = np.exp(-0.5 * fact)
    y = y / np.sqrt((2*np.pi)**d * np.linalg.det(covar))

    return y
