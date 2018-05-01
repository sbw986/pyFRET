import numpy as np
import pdb

def forwbackFRET(A, px_z, pz, data):
    """
    :type A:
    :type px_z:
    :type pz:
    :type data:
    :rtype ...
    """

    K = A.shape[1]
    if len(data.shape) == 1:
        T = data.shape[0]
        D = 1
    else:
        D, T = data.shape
    Xi = np.zeros([K, K])
    GammaInit = np.zeros([K])
    Nk = np.zeros([K])
    xbar = np.zeros([D, K])
    S = np.zeros([D, D, K])

    Gamma = np.zeros([T, K])
    alpha = np.zeros([T, K])

    beta = np.zeros([T, K])
    scale = np.zeros([T])

    #Forward pass (with scaling)
    pdb.set_trace()
    alpha[0,:] = np.rot90(pz) * px_z[0,:]
    scale[0] = np.sum(alpha[0,:])
    alpha[0,:] = alpha[0,:]/scale[0]
    for t in range(1, T):
        alpha[t,:] = (np.matmul(alpha[t-1,:], A) * px_z[t,:])
        scale[t] = np.sum(alpha[t,:])
        alpha[t,:] = alpha[t,:]/scale[t]



    #Backward pass (with scaling)
    beta[T-1,:] = np.ones([K])/scale[T-1]
    for t in range(T-2, -1, -1):
        beta[t,:] = np.matmul((beta[t+1,:] * px_z[t+1,:]), np.transpose(A)) / scale[t]

    #Another pass gives us the joint probabilities
    for t in range(T-1):
        Xi = Xi + A * (np.matmul(np.rot90([alpha[t,:]],3), [beta[t+1,:] * px_z[t+1,:]]))

    #Compute Gamma
    Gamma = alpha * beta
    Gamma = Gamma / np.transpose(np.matlib.repmat(np.sum(Gamma,1), K, 1))

    GammaInit = GammaInit + Gamma[0,:]
    lnZv = np.sum(np.log(scale))

    Nk = Nk + np.sum(Gamma, 0)
    for k in range(0, K):
        xbar[:,k] = np.sum(np.matlib.repmat(Gamma[:,k], D, 1) * data, 1)
    lnZ = np.sum(lnZv, 0)
    wa = Xi
    wpi = GammaInit

    Nk = Nk + 1e-10

    xbar = xbar / np.matlib.repmat(np.conj(Nk), D, 1)

    for k in range(0, K):
        diff1 = data - np.matlib.repmat(xbar[:,k], 1, T)
        diff2 = np.matlib.repmat(np.conj(Gamma[:,k]), D, 1) * diff1
        S[:,:,k] = S[:,:,k] + np.matmul(diff2, np.transpose(diff1))

    Nk3d = np.zeros([D, D, K])

    for k in range(0, K):
        Nk3d[:,:,k] = Nk[k]

    S = S / Nk3d

    return [wa, wpi, xbar, S, Nk, lnZ]
