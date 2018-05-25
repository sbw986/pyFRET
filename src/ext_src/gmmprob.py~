import numpy as np
from gmmactiv import gmmactiv

def gmmprob(mix, x):

    a = gmmactiv(mix, x)

    return a * np.transpose(mix.priors)
