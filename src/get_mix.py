from sklearn import mixture
import numpy as np
import sys
sys.path.append('/Users/Steven/PycharmProjects/pyFRET/ext_src')
from gmm import gmm
from vbfret_gmminit import gmminit
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

    options[1] = -1
    options[3] = 0.1
    options[14] = 10
    options[5] = 1

    # TODO making a bunch of assumptions here... Removing lots of other functions for now

    if D == 1:
        mix = gmm(D, ncentres, 'spherical')
    else:
        mix = gmm(D, ncentres, 'full')

    #Set mix.centres to starting guess
    mix.centres = start_guess

    #initialize with hard K-means algorithm
    mix = gmminit(mix, x, options)

    return mix
