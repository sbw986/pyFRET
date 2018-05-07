import numpy as np
import pdb

def dist2(x, c):

    if len(x.shape) == 1:
        ndata = len(x)
        dimx = 1
    else:
        ndata, dimx = x.shape()

    if len(c.shape) == 1:
        ncentres = len(c)
        dimc = 1
    else:
        ncentres, dimc = c.shape()

    if dimx != dimc:
        print('error')

    n2 = np.ones([1, ncentres]) * np.transpose([x**2]) + \
         np.ones([ndata,1]) * (c ** 2) - 2 * np.transpose([x]) * c

    #TODO This is going to have problems.  Translation not good.
    if n2.any().any() < 0:
        n2 = [ival if ival >= 0 else 0 for ival in n2]

    return n2
