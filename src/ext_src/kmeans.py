import numpy as np
from .dist2 import dist2
import pdb

def kmeans(centres, data, options):
    if len(data.shape) == 1:
        ndata = len(data)
        data_dim = 1
    else:
        ndata, data_dim = data.shape

    if len(centres.shape) == 1:
        ncentres = len(centres)
        dim = 1
    else:
        ncentres, dim = centres.shape

    if options[13]:
        niters = options[13]
    else:
        niters = 100

    if options[4] == 1:
        perm = np.random.permutation(ndata)
        perm = perm[0:ncentres]

        centres = data[perm]

    id_ = np.eye(ncentres)

    for n in range(0,niters):
        old_centres = centres
        d2 = dist2(data, centres)

        index = np.argmin(d2, axis = 1)
        minvals = [d2[i,k] for i, k in enumerate(index)] #d2[index]
        post = id_[index,:]

        num_points = np.sum(post, axis = 0)
        for j in range(0, ncentres):
            if num_points[j] > 0:
                centres[j] = np.sum(data[np.where(post[:,j] != 0)])/num_points[j]

        e = np.sum(minvals)

        if n > 0:
            if np.max(np.abs(centres - old_centres)) < options[1] and np.abs(old_e - e) < options[2]:
                options[7] = e
                return centres, options, post
        old_e = e

    options[7] = e
    return centres, options, post
