import numpy as np

def forwbackFRET(A, px_z, pz, data):
    """
    :type A:
    :type px_z:
    :type pz:
    :type data:
    :rtype ...
    """

    K = A.shape[1]
    D, T = data.shape
    Xi = np.zeros([K, K])
    GammaInit = np.zeros([1, K])
    Nk = np.zeros([K, 1])
    xbar = np.zeros([D, K])
    S = np.zeros([D, D] * K)

    Gamma = np.zeros([T, K])
    alpha = np.zeros([T, K])
    beta = np.zeros([T, K])
    scale = np.zeros([1, T])

    #Forward pass (with scaling)
    alpha[1][:] = pz * px_z[1][:]
    scale[1] = np.sum(alpha[1][:])
    alpha[1][:] = alpha[1][:]/scale[1]
    for t in range(2, T + 1):
        alpha[t][:] = (alpha[t-1][:] * A) * px_z[t][:]
        scale[t] = np.sum(alpha[t][:])
        alpha[t][:] = alpha[t][:]/scale[t]

    #Backward pass (with scaling)
    beta[T][:] = np.ones([1,K])/scale[T]
    for t in range(T-1, 0, -1):
        beta[t][:] = (beta[t + 1][:] * px_z[t+1][:]) * np.conj(A) / scale[t]

    #Another pass gives us the joint probabilities
    for t in range(1, T):
        Xi = Xi + A * (np.conj(alpha[t][:]) * \
             beta[t + 1][:] * px_z[t+1][:])

    #Compute Gamma
    Gamma = alpha * beta
    Gamma = Gamma / np.matlib.repmat(np.sum(Gamma,2), 1, K)

    GammaInit = GammaInit + Gamma[1][:]
    lnZv = np.sum(np.log(scale))

    Nk = Nk + np.conj(np.sum(Gamma, 1))
    for k in range(1, K + 1):
        xbar[:][k] = np.sum(np.matlib.repmat(np.conj(Gamma[:][k]), D, 1) \
                      * data, 2)
    lnZ = np.sum(lnZv, 1)
    wa = Xi
    wpi = GammaInit

    Nk = Nk + 1e-10

    xbar = xbar / np.matlib.repmat(np.conj(Nk), D, 1)

    for k in range(1, K + 1):
        diff1 = data - np.matlib.repmat(xbar[:][k], 1, T)
        diff2 = np.matlib.repmat(np.conj(Gamma[:][k]), D, 1) * diff1
        S[:][:][k] = S[:][:][k] + (diff2 * np.conj(diff1))

    Nk3d = np.zeros([D, D] * K)

    for k in range(1, K + 1):
        Nk3d[:][:][k] = Nk[k]

    S = S / Nk3d

    return [wa, wpi, xbar, S, Nk, lnZ]
