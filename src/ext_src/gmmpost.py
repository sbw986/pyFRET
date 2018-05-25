import numpy as np
import pdb
from .gmmactiv import gmmactiv

def gmmpost(mix, x):
    ndata = len(x)

    a = gmmactiv(mix, x)

    post = np.ones([ndata,1]) * mix.priors * a
    s = np.sum(post, 1)
    if any(s == 0):
        print('Some zero posterior probabilities')
        zero_rows = np.where(s == 0)
        s = s + (s == 0)
        post[zero_rows, :] = 1/mix.ncentres
    post = post/(np.transpose([s]) * np.ones(mix.ncentres))

    return post, a
