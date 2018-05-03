import math
import numpy as np
import scipy
import sys
sys.path.append('/Users/Steven/PycharmProjects/pyFRET/ext_src')
import dirrnd as dirrnd
import forwbackFRET as forwbackFRET
import kldirichlet as kldirichlet
import pdb

class Out:
    def __init__(self, Wa, Wpi, beta, m, W, v, F):
        self.Wa = Wa
        self.Wpi = Wpi
        self.beta = beta
        self.m = m
        self.W = W
        self.v = v
        self.F = F

def pyFRET_VBEM(x, mix, prior_par, options):
    """
    :type x:
    :type mix:
    :type priorPar:
    :type options:
    :rtype out
    """



    # Initialize variables
    if len(x.shape) > 1:
        D, T = x.shape
    else:
        T = x.shape[0]
        D = 1

    K = mix.n_components #mix.ncentres #SBW edit
    Fold = -1 * math.inf
    logLambdaTilde = np.zeros([K])
    trW0invW = np.zeros([K])
    lnZ = np.zeros([options.max_iter])
    Fa = np.zeros([options.max_iter])
    Fpi = np.zeros([options.max_iter])
    Fgw = np.zeros([options.max_iter])
    F = np.zeros([options.max_iter])

    # Hyperparameter priors
    upi = prior_par.upi
    upi_vec = upi * np.ones([K])
    ua = prior_par.ua
    uad = prior_par.uad
    ua_mtx = ua + uad * np.eye(K)
    m0 = prior_par.mu
    beta0 = prior_par.beta
    W0 = prior_par.W
    v0 = prior_par.v
    W0inv = np.linalg.inv(W0)

    # Use 'responsibilities' from initialization to set sufficient statistics
    #Nk = T * np.conj(mix.mean_prior_) #T * np.conj(mix.priors) # SBW Edit
    #TODO Fix means_prior_ in Nk
    Nk = T * np.rot90(mix.means_)[0]
    xbar = np.rot90(mix.means_)[0] #np.conj(mix.centres) #SBW Edit
    S = mix.covariances_ #mix.covars #SBW Edit

    #TODO remove this stuff that was placed for debugging
    Nk = np.array([9, 9])
    xbar = np.array([.6274, 0.3726])
    S = np.array([[[.2338, .2338]]])

    # Use above sufficient statistics for M step update equations


    beta = beta0 + Nk

    v = v0 + Nk
    m = ((beta0 * m0) * np.ones([1,K]) + np.ones([D, 1]) * Nk * xbar) / (np.ones([D,1]) * beta)
    W = np.zeros([D,D,K]) # * K)
    for k in range(0, K):
        mult1 = beta0 * Nk[k] / (beta0 + Nk[k])
        diff3 = xbar[k] - m0
        # TODO S is being indexed differently
        W[:,:,k] = np.linalg.inv(W0inv + Nk[k] * S[:,:,k] + \
                     mult1 * diff3 * np.rot90([diff3]))
    Wpi = Nk/T + upi

    # Initialize transition matrix using random values drawn from Dirichlet
    wa = np.zeros([K,K])

    for k in range(0, K):
        wa[k,:] = dirrnd.dirrnd(ua_mtx[k,:], 1) * (T - 1) / K # TODO write dirrnd function

    #TODO remove wa define for debugging
    wa = np.array([[6.0158, 2.4842], [1.6071, 6.8929]])
    Wa = wa + ua_mtx


    #Pre-calculate constant term used in lower bound estimation
    logB0 = -1 * (v0 / 2) * np.log(np.linalg.det(W0)) - (v0 * D / 2) * np.log(2) - \
            (D * (D - 1) / 4) * np.log(np.pi) - \
            np.sum(scipy.special.gammaln(0.5 * (v0 + 1 - (np.arange(1,D+1)))))

    # Main loop of algorithm
    for iterv in range(0, options.max_iter):
        astar = np.exp(scipy.special.digamma(Wa) - \
                        np.matmul(np.transpose([scipy.special.digamma(np.sum(Wa, axis = 1))]),[np.ones([K])]))
        pistar = np.exp(scipy.special.digamma(Wpi) - \
                        scipy.special.digamma(np.sum(Wpi, axis = 0)))

        for k in range(0, K):
            logLambdaTilde[k] = np.sum(scipy.special.digamma((v[k] * np.ones([1,D]) \
                                       + 1 - np.arange(1, D+1)) / 2 )) + D * np.log(2) \
                                       + np.log(np.linalg.det(W[:,:,k]))

        #Calculate E

        xWx = np.zeros([T,K])
        xWm = np.zeros([T,K])
        mWm = np.zeros(K)
        for d1 in range(0, D):
            if D == 1:
                m1 = m[:]
                x1 = x[:]
            else:
                m1 = m[d1,:]
                x1 = x[d1,:]
            for d2 in range(0, D):
                if D == 1:
                    m2 = m[:]
                    x2 = x[:]
                else:
                    m2 = m[d2,:]
                    x2 = x[d2,:]
                W12 = np.reshape(W[d1,d2,:], (1, K))
                xWx = xWx + np.matmul(np.rot90([x1 * x2],3), W12)
                xWm = xWm + np.rot90([x1],3) * W12 * m2
                mWm = mWm + (m1 * W12 * m2)

        E = np.matmul((xWx - 2 * xWm + np.ones([T,1]) * mWm), np.diag(v)) + \
            np.matmul(np.ones([T,1]), ([D/beta]))

        pXgivenZtilde = ((2 * np.pi) ** (-D/2)) * \
                        np.exp(0.5 * (np.rot90(np.ones([1,T])) * logLambdaTilde - E))

        #Forward-back algorithm
        #TODO Start from here for testunit debugging

        wa, wpi, xbar, S, Nk, lnZ[iterv] = forwbackFRET.forwbackFRET(astar, pXgivenZtilde, pistar, x) # TODO write forwbackFRET fun

        #Compute F
        H = 0

        for k in range(0, K):
            logBk = -(v[k]/2) * np.log(np.linalg.det(W[:,:,k])) - \
                    (v[k] * D / 2) * np.log(2) - (D * (D - 1) / 2) * np.log(np.pi) - \
                    np.sum(scipy.special.gammaln(0.5 * (v[k] + 1 - (range(1, D+1)))))
            H = H - logBk - 0.5 * (v[k] - D - 1) * logLambdaTilde[k] + 0.5 * v[k] * D
            diff = m[:,k] - m0
            mWm[0][k] = np.matmul(np.matmul(np.transpose(diff), W[:,:,k]), diff)
            trW0invW[k] = np.trace(np.matmul(W0inv,W[:,:,k]))



        Lt41 = 0.5 * np.sum(D * np.log(beta0 / (2 * np.pi)) + logLambdaTilde - \
                            D * beta0 / beta - beta0 * v * mWm)
        Lt42 = K * logB0 + 0.5 * (v0 - D - 1) * np.sum(logLambdaTilde) - \
               0.5 * np.sum(v * trW0invW)
        Lt4 = Lt41 + Lt42
        Lt7 = 0.5 * np.sum(logLambdaTilde + D * np.log(beta/(2 * np.pi))) - \
              0.5 * D * K - H
        Fgw[iterv] = Lt4 - Lt7
        for kk in range(0, K):
            uad_vec = np.zeros([K])
            uad_vec[kk] = uad
            Fa[iterv] = Fa[iterv] - kldirichlet.kldirichlet(Wa[kk,:], ua_mtx[kk,:]) # TODO write kldirichlet function

        Fpi[iterv] = - kldirichlet.kldirichlet(Wpi, upi_vec)
        F[iterv] = Fa[iterv] + Fgw[iterv] + Fpi[iterv] + lnZ[iterv]
        if iterv > 2 and (F[iterv] < F[iterv - 1] - .000001):
            print('Warning!!: Lower bound decreased')

        #M Step
        beta = beta0 + Nk
        v = v0 + Nk
        m = ((beta0 * m0) * np.ones(K) + \
             np.ones([D]) * Nk * xbar) / (np.ones([D]) * beta)
        for k in range(0, K):
            mult1 = beta0 * Nk[k] / (beta0 + Nk[k])
            diff3 = xbar[:,k] - m0
            W[:,:,k] = np.linalg.inv(W0inv + Nk[k] * S[:,:,k] + \
                         mult1 * diff3 * np.rot90([diff3]))
        Wa = wa + ua_mtx
        Wpi = wpi + upi

        if iterv > 2:
            if abs((F[iterv] - F[iterv - 1])/F[iterv - 1]) < options.threshold: # TODO isfinite needs to be added
                lnZ = lnZ[0:iterv + 1]
                Fa = Fa[0:iterv + 1]
                Fpi = Fpi[0:iterv + 1]
                Fgw = Fgw[0:iterv + 1]
                F = F[0:iterv + 1]
                break

    out = Out(Wa, Wpi, beta, m, W, v, F)

    return out
