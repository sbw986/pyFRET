import numpy as np
import pdb
from dist2 import dist2

def gmmactiv(mix, x):

    ndata = len(x)

    a = np.zeros([ndata, mix.ncentres])

    if mix.covar_type == 'spherical':
        n2 = dist2(x, mix.centres)

        wi2 = np.ones([ndata,1]) * (2 * mix.covars)
        normal = (np.pi * wi2)**(mix.nin/2)
        a = np.exp(-1 * (n2 / wi2)) / normal

    elif mix.covar_type == 'diag':
        normal = (2 * np.pi)**(mix.nin/2)
        s = np.prod(np.sqrt(mix.covars),1)
        for j in range(0, mix.ncentres):
            diffs = x - np.matmul(np.ones(ndata), mix.centres[j,:])
            a[:,j] = np.exp(-0.5 * np.sum((diffs ** 2)/np.ones(ndata) * mix.covars[j,:], 2)/(normal * s[j]))

    elif mix.covar_type == 'full':
        normal = (2 * np.pi) ** (mix.nin/2)
        for j in range(0,mix.ncentres):
            diffs = x - np.matmul(np.ones(ndata), mix.centres[j,:])
            c = np.linalg.cholesky(mix.covars[:,:,j])
            temp = diffs/c
            a[:,j] = np.exp(-0.5 * np.sum(temp * temp, 1))/ (normal * np.prod(np.diag(c)))

    elif mix.covar_type == 'ppca':
        log_normal = np.matmul(mix.nin, np.log(2*pi))
        d2 = np.zeros(ndata, mix.ncentres)
        logZ = np.zeros(mix.ncentres)
        for i in range(0, ncentres):
            k = 1 - mix.covars[i] / mix.lambda_[i,:]
            logZ[i] = log_normal + mix.nin * np.log(mix.covars[i]) - np.sum(np.log(1-k))
            diffs = np.matmul(np.ones(ndata), mix.centres[i,:])
            proj = np.matmul(diffs, mix.U[:, :, i])
            d2[:, i] = (np.sum(diffs * diffs, 2)) - \
                        np.sum((proj * np.matmul(np.ones(ndata), k)), 2) / mix.covars[i]
        a = np.exp(-0.5 * (d2 + np.matmul(np.ones(ndata), logZ)))
    else:
        print('Unknown covariance type ')

    return a
