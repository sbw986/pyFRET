import numpy as np
import sys
sys.path.append('/Users/Steven/PycharmProjects/pyFRET/ext_src')
import normalise
import gauss
import pdb

def chmmViterbi(out, x):
    D, K = out.m.shape
    T = len(x)
    omega = np.zeros([T,K])
    bestPriorZ = np.zeros([T,K])
    z_hat = np.zeros(T)

    # Get parameters from out structure
    pZ0, _ = normalise.normalise(out.Wpi)
    A = out.Wa

    #Convert A from matrix of counts to a probability matrix
    A, _ = normalise.normalise(A, 2)
    mus = out.m
    W = out.W
    v = out.v
    covarMtx = np.zeros([D,D,K])
    for k in range(0,K):
        covarMtx[:,:,k] = np.linalg.inv(W[:,:,k])/(v[k] - D - 1)
    #Compute values for timestep 1
    for k in range(0,K):
        omega[0,k] = np.log(pZ0[k]) + np.log(gauss.gauss(mus[:,k], covarMtx[:,:,k], x[0]))

    bestPriorZ[0,:] = 0
    for t in range(1,T):
        for k in range(0,K):
            tmp = np.log(A[:,k]) + omega[t-1,:]
            bestPriorZ[t,k] = np.argmax(tmp)
            omega[t,k] = tmp[int(bestPriorZ[t,k])]
            omega[t,k] = omega[t,k] + np.log(gauss.gauss(mus[:,k], covarMtx[:,:,k], x[t]))
    z_hat[T-1] = np.argmax(omega[T-1,:])
    logLikelihood = omega[T-1,:][int(z_hat[T-1])]
    for t in range(T-2, 0, -1):
        z_hat[t] = bestPriorZ[t+1, int(z_hat[t+1])]
    x_hat = [mus[0][int(zval)] for zval in z_hat] #mus[:, z_hat]

    return x_hat, z_hat
