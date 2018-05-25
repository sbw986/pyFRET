import numpy as np
from scipy.special import gammaln
from scipy.special import digamma as psi
import pdb

def kldirichlet(vecP, vecQ):
    alphaP = np.sum(vecP)
    alphaQ = np.sum(vecQ)

    res = gammaln(alphaP) - gammaln(alphaQ) - \
          np.sum(gammaln(vecP) - gammaln(vecQ)) + \
          np.sum((vecP - vecQ) * (psi(vecP) - psi(alphaP)))

    return res
