import sklearn

def get_gmm_mix(x, start_guess):
    """
    :type x:
    :type start_guess
    :rtype
    """

    ncentres, D = start_guess.shape

    options = [0, 1e-4, 1e-4, 1e-6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1e-8, 0.1, 0]

    options[1] = -1
    options[3] = 0.1
    options[14] = 10
    options[5] = 1

    if D == 1:
        mix = sklearn.mixture.GMM()
