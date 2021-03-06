#from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
from .kmeans import kmeans
import pdb
import numpy as np
from .dist2 import dist2

def gmminit(mix, x, options):
    if len(x.shape) == 1:
        ndata = len(x)
        xdim = 1
    else:
        ndata, xdim = x.shape()

    GMM_WIDTH = 1.0
    mix.centres, options, post = kmeans(mix.centres, x, options)

    cluster_sizes = np.sum(post, 0)
    cluster_sizes[cluster_sizes < 1] = 1 #TODO SBW addition for no prior zero, seems consistent with matlab but need to check
    mix.priors = cluster_sizes / np.sum(cluster_sizes)

    if mix.covar_type == 'spherical':
        if mix.ncentres > 1:
            cdist = dist2(mix.centres, mix.centres)
            cdist = cdist + np.diag(np.ones(mix.ncentres) * np.finfo('d').max)
            mix.covars = np.min(cdist, 0) #TODO Should this be 1?
            mix.covars = mix.covars + GMM_WIDTH * (mix.covars < np.finfo(float).eps)
        else:
            mix.covars = np.mean(np.diag(np.cov(x)))
    #TODO full not tested yet
    elif mix.covar_type == 'full':
        for j in range(0, mix.ncentres):
            c = x[np.argwhere(post[:,j]), :]
            diffs = c - np.matmul((np.ones(c.shape[0])), mix.centres[j,:])
            mix.covars[:, :, j] = np.matmul(np.transpose(diffs), diffs) / c.shape[0]
            if rank[mix.covars[:,:,j]] < mix.nin:
                mix.covars[:,:,j] = mix.covars[:,:,j] + GMM_WIDTH * np.eye(mix.nin)
    # ppca not tested.  May not be used
    """
    elif mix.covar_type == 'ppca':
        for j in range(0, mix.ncentres):
            c = x[np.argwhere(post[:,j]), :]
            diffs = c - np.matmul((np.ones(c.shape[0])), mix.centres[j,:])
            #TODO replace PCA with PPCA
            #tempcovars, tempU, templambda = ppca(np.matmul(np.transpose(diffs), diffs)/c.shape[0], mix.ppca_dim)
            pca_ = PCA()
            pca_.fit(np.matmul(np.transpose(diffs), diffs)/c.shape[0], mix.ppca_dim)
            tempcovars = pca_.get_covariance() # TODO check
            tempU = None # TODO this is incorrect
            templambda = pca_.get_params() #TODO this is likely incorrect too...


            if len(templambda) != mix.ppca_dim:
                print('Unable to extract enough components')
            else:
                mix.covars[j] = tempcovars
                mix.U[:, :, j] = tempU
                mix.lambda_[j, :] = templambda
    """
    return mix
