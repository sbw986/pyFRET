import numpy as np
import pdb

def normalise(A, dim = None):

    if dim == None:
        z = np.sum(A)
        s = z + (z == 0)
        M = A / s
    elif dim == 1:
        z = np.sum(A)
        s = z + (z == 0)
        #TODO using repmat, but matlab has REPMATC??
        M = A / np.tile(s, A.shape[0])
    else:
        z = np.sum(A, axis = dim - 1)
        s = z + (z == 0)
        L = A.shape[dim - 1]
        d = len(A.shape)
        v = np.ones(d)
        v[dim - 1] = L
        c = np.tile(np.transpose([s]), v.astype('int'))
        M = A / c
    return M, z
