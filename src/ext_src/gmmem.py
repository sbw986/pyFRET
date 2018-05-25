import numpy as np
from .gmmprob import gmmprob
import pdb
from .gmmpost import gmmpost
from .dist2 import dist2

def gmmem(mix, x, options, nargout):

    if len(x.shape) == 1:
        ndata = x.shape[0]
        xdim = 1
    else:
        ndata, xdim = x.shape

    if options[13] != 0:
        niters = options[13]
    else:
        niters = 100

    display = options[0]
    store = 0
    if nargout > 2:
        store = 1
        errlog = np.zeros(niters)
    test = 0
    if options[2] > 0.0:
        test = 1

    check_covars = 0
    if options[4] >= 1:
        if display >= 0:
            print('check_covars is on')
        check_covars = 1
        MIN_COVAR = np.finfo(float).eps
        init_covars = mix.covars


    for n in range(0, niters):
        post, act = gmmpost(mix, x)

        if display or store or test:
            prob = np.matmul(act, mix.priors)
            e = -1 * np.sum(np.log(prob))
            if store:
                errlog[n] = e
            if display > 0:
                print('1')
            if test:
                if n > 0 and np.abs(e - eold) < options[2]:
                    options[7] = e
                    return mix, options, errlog
                else:
                    eold = e


        new_pr = np.sum(post, 0)
        new_c = np.matmul(np.transpose(post),x)

        mix.priors = new_pr / ndata
        mix.centres = new_c / (new_pr * np.ones(mix.nin))
        if mix.covar_type == 'spherical':
            n2 = dist2(x, mix.centres)
            v = []
            for j in range(0, mix.ncentres):
                v.append(np.matmul(post[:,j], np.transpose(n2[:,j])))
            mix.covars = (v/new_pr)/mix.nin
            if check_covars:
                for j in range(0, mix.ncentres):
                    if mix.covars[j] < MIN_COVAR:
                        mix.covars[j] = init_covars[j]
        elif mix.covar_type == 'diag':
            for j in range(0, mix.ncentres):
                diffs = x - (np.ones(ndata) * mix.centres[j,:])
                mix.covars[j,:] = np.sum((diffs * diffs) * (post[:,j] * np.ones(mix.nin)),1)/new_pr[j]
            if check_covars:
                for j in range(0, mix.ncentres):
                    if np.min(mix.covars[j]) < MIN_COVAR:
                        mix.covars[j,:] = init_covars[j,:]
        elif mix.covar_type == 'full':
            for j in range(0, mix.ncentres):
                diffs = x - (np.ones(ndata) * mix.centres[j,:])
                diffs = diffs * (np.sqrt(post[:,j]) * np.ones(mix.nin))
                mix.covars[:,:,j] = np.transpose(diffs) * diffs / new_pr
            if check_covars:
                for j in range(0, mix.ncentres):
                    if np.min(np.linalg.svd(mix.covars[:,:,j])) < MIN_COVAR:
                        mix.covars[:,:,j] = init_covars[:,:,j]
        elif mix.covar_type == 'ppca':
            for j in range(0, mix.ncentres):
                diffs = x - (np.ones(ndata) * mix.centres[j,:])
                diffs = diffs * (np.sqrt(post[:, j]) * np.ones(mix.nin))
                #TODO Add PPCA function
                tempcovars, tempU, templambda = ppca(np.transpose(diffs) * diffs / new_pr(j), mix.ppca_dim)
                if len(templambda) != mix.ppca_dim:
                    print('error')
                else:
                    mix.covars[j] = tempcovars
                    mix.U[:,:,j] = tempU
                    mix.lambda_[j,:] = templambda
            if check_covars:
                if mix.covars[j] < MIN_COVAR:
                    mix.covars[j] = init_covars[j]
        else:
            print('Unknown covariance type')

    options[7] = -1 * np.sum(np.log(gmmprob(mix,x)))

    if display >= 0:
        print(maxitmess)
    return mix, options, errlog
