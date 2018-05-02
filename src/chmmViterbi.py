import numpy as np

def chmmViterbi(out, x):
    D, K = out.m.shape
    T = len(x)
    omega = np.zeros(T,K)
    bestPriorZ = np.zeros(T,K)
    z_hat = np.zeros(T)
    
