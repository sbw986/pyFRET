import math
import numpy as np
import scipy

def pyFRET_VBEM(x, mix, priorPar, options):
    """
    :type x:
    :type mix:
    :type priorPar:
    :type options:
    :rtype out
    """

    # Initialize variables
    D, T = x.shape
    K = mix.ncentres
    Fold = -1 * math.inf
    logLambdaTilde = np.zeros([1, K])
    trW0invW = np.zeros([1, K])
    lnZ = np.zeros([1, options.maxIter])
    Fa = np.zeros([1, options.maxIter])
    Fpi = np.zeros([1, options.maxIter])
    Fgw = np.zeros([1, options.maxIter])
    F = np.zeros([1, options.maxIter])

    # Hyperparameter priors
    upi = PriorPar.upi
    upi_vec = upi * np.ones([1,K])
    ua = PriorPar.ua
    uad = PriorPar.uad
    ua_mtx = ua + uad * np.eye(K)
    m0 = PriorPar.mu
    beta0 = PriorPar.beta0
    W0 = PriorPar.W
    v0 = PriorPar.v
    W0inv = np.linalg.inv(W0)

    # Use 'responsibilities' from initialization to set sufficient statistics
    Nk = T * np.conj(mix.priors)
    xbar = np.conj(mix.centres)
    S = mix.covars

    # Use above sufficient statistics for M step update equations
    beta = beta0 + Nk
    v = v0 + Nk
    m = ((beta0 * m0) * np.ones([1,K]) + np.ones([D, 1]) * np.conj(Nk) * xbar) / (np.ones([D,1]) * np.conj(beta))
    W = np.zeros([D,D] * K)
    for k in range(1, K + 1):
        mult11 = beta0 * Nk(k) / (beta0 + Nk(k))
        diff3 = xbar[:][k] - m0
        W[:][:][k] = np.linalg.inv(w0inv + Nk(k) * S[:][:][k] + \
                     mult1 * diff3 * np.conj(diff3))
    Wpi = np.conj(Nk)/T + upi

    # Initialize transition matrix using random values drawn from Dirichlet
    wa = np.zeros([1,K])
    for k in range(1, K + 1):
        wa[k][:] = dirrnd(ua_mtrx[k][:], 1) * (T - 1) / K # TODO write dirrnd function
    Wa = wa + ua_mtx

    #Pre-calculate constant term used in lower bound estimation
    logB0 = -1 * (v0 / 2) * np.log(np.linalg.det(W0)) - (v0 * D / 2) * log(2) \
            - (D * (D - 1) / 4) * np.log(pi) - \
            np.sum(scipy.special.gammaln(0.5 * (v0 + 1 - (range(1:D + 1)))))

    # Main loop of algorithm
    for iterv in range(1, options.maxIter + 1):
        astart = np.exp(scipy.special.digamma(Wa) - \
                        scipy.special.digamma(np.sum(Wa, 2)) * np.ones([1,K]))
        pistar = np.exp(scipy.special.digamma(Wpi) - \
                        scipy.special.digamma(np.sum(Wpi, 2)))
        for k in range(1, K + 1):
            logLambdaTilde(k) = np.sum(scipy.special.digamma((v[k] * np.ones([1,D]) \
                                       + 1 - range(1, D + 1)) / 2 )) + D * np.log(2) \
                                       + np.log(np.linalg.det(W[:][:][k]))

        #Calculate E
        xWx = np.zeros([T,K])
        xWm = np.zeros([T,K])
        mWm = np.zeros([1,K])
        for d1 in range(1, D + 1):
            m1 = m[d1][:]
            x1 = x[d1][:]
            for d2 in range(1, D + 1):
                m2 = m[d2][:]
                x2 = x[d2][:]
                W12 = np.reshape(W[d1][d2][:], (1, K))
                xWx = xWx + np.conj((x1 * x2)) * W12
                xWm = xWm + np.conj(x1) * W12 * m2
                mWm = mWm + (m1 * W12 * m2)

        E = (xWx - 2 * xWm + np.ones([T,1]) * mWm) * np.diag(v) + \
            np.ones([T,1]) * np.conj((D/beta))

        pXgivenZtilde = ((2 * np.pi) ** (-D/2)) * \
                        np.exp(0.5 * (logLambdaTilde[np.ones([1,T])][:] - E))

        #Forward-back algorithm
        wa, wpi, xbar, S, Nk, lnZ[iterv] = forwbackFRET(astar, pXgivenZtilde, pistar, x) # TODO write forwbackFRET fun

        #Compute F
        H = 0
        for k in range(1, K + 1):
            logBk = -(v[k]/2) * np.log(np.linalg.det(W[:][:][k])) - \
                    (v[k] * D / 2) * np.log(2) - (D * (D - 1) / 2) * np.log(np.pi) - \
                    np.sum(scipy.special.gammaln(0.5 * (v[k] + 1 - (range(1, D+1)))))
            Hd = H - logBk - 0.5 * (v[k] - D - 1) * logLambdaTilde[k] + 0.5 * v[k] * D
            diff = m[:][k] - m0
            trW0invW[k] = np.trace(W0inv*W[:][:][k])

        Lt41 = 0.5 * np.sum(D * np.log(beta0 / (2 * np.pi)) + np.conj(logLambdaTilde) - \
                            D * beta0 / beta - beta0 * v * np.conj(mWm))
        Lt42 = K * logB0 + 0.5 * (v0 - D - 1) * np.sum(logLambdaTilde) - \
               - 0.5 * np.sum(v * np.conj(trW0invW))
        Lt4 = Lt41 + Lt42
        Lt7 = 0.5 * np.sum(np.conj(logLambdaTilde) + D * np.log(beta/(2 * np.pi))) \
              -0.5 * D * K - H
        Fgw[iterv] = Lt4 - Lt7
        for kk in range(1, K + 1):
            uad_vec = np.zeros([1,K])
            uad_vec[kk] = uad
            Fa[iterv] = Fa[iterv] - kldirichlet(Wa[kk][:], ua_mtx[kk][:]) # TODO write kldirichlet function
        Fpi[iterv] = - kldirichlet(Wpi, upi_vec)
        F[iterv] = Fa[iterv] + Fgw[iterv] + Fpi[iterv] + lnZ[iterv]
        if iterv > 2 and (F[iterv] < F[iterv - 1] - .000001):
            print('Warning!!: Lower bound decreased')

        #M Step
        beta = beta0 + Nk
        v = v0 + Nk
        m = ((beta0 * m0) * np.ones([1,K]) + \
             np.ones([D,1]) * np.conj(Nk) * xbar) / (np.ones([D, 1]) * np.conj(beta))
        for k in range(1, K + 1):
            mult1 = beta0 * Nk[k] / (beta0 + Nk[k])
            diff3 = xbar[:][k] - m0
            W[:][:][k] = np.linalg.inv(W0inv + Nk[k] * S[:][:][k] + \
                         mult1 * diff3 * np.conj(diff3))
        Wa = wa + ua_mtx
        Wpi = wpi + upi

        if iterv > 2:
            if abs((F[iterv] - F[iterv - 1])/F[iterv - 1]) < options.threshold: # TODO isfinite needs to be added
                lnZ[iterv + 1::] = []
                Fa[iterv + 1::] = []
                Fpi[iterv + 1::] = []
                Fgw[iterv + 1::] = []
                F[iterv + 1::] = []
                break

    out.Wa = Wa
    out.Wpi = Wpi
    out.beta = beta
    out.m = m
    out.W = W
    out.v = v
    out.F = F

    return out
