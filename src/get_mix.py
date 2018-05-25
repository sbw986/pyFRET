from sklearn import mixture
import numpy as np
import sys
sys.path.append('../ext_src')
from gmm import gmm
from vbfret_gmminit import gmminit
from gmmem import gmmem
import pdb

def get_gmm_mix(x, start_guess):
    """
    :type x:
    :type start_guess
    :rtype
    """

    if len(start_guess.shape) > 1:
        ncentres, D = start_guess.shape
    else:
        ncentres = start_guess.shape[0]
        D = 1

    options = [0, 1e-4, 1e-4, 1e-6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1e-8, 0.1, 0]

    options[0] = -1
    options[2] = 0.1
    options[13] = 10
    options[4] = 1

    # TODO making a bunch of assumptions here... Removing lots of other functions for now

    if D == 1:
        mix = gmm(D, ncentres, 'spherical')
    else:
        mix = gmm(D, ncentres, 'full')

    #Set mix.centres to starting guess
    mix.centres = start_guess

    #initialize with hard K-means algorithm
    mix = gmminit(mix, x, options)


    #initialize with soft K-means algorithm
    mix, options, errlog = gmmem(mix, x, options, 3)

    if D == 1:
        mix.covars = np.reshape(mix.covars, [1, 1, ncentres])

    return mix
