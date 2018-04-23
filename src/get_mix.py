from sklearn import mixture
import numpy as np

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
        mix = mixture.BayesianGaussianMixture(n_components = ncentres, covariance_type = 'spherical')
    else:
        mix = mixture.BayesianGaussianMixture(n_components = ncentres, covariance_type = 'full')
    mix.fit(np.rot90([x]))

    return mix
