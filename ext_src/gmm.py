import numpy as np

class Mix:
    def __init__(self, type_, nin, ncentres):
        self.type = type_
        self.nin = nin
        self.ncentres = ncentres
        self.priors = None
        self.ppca_dim = None
        self.centres = None
        self.covar_type = None
        self.nwts = None

def gmm(dim, ncentres, covar_type, ppca_dim = None):
    if ncentres < 1:
        assert('Number of centres must be greater than zero')

    var_types = ('spherical', 'diag', 'full', 'ppca')
    if covar_type not in var_types:
        assert('Undefined covariance type')
    else:
        mix.covar_type = covar_type

    mix = Mix('gmm', dim, ncentres)

    if covar_type == 'ppca':
        if ppca_dim == None:
            ppca_dim = 1
        elif ppca_dim > dim:
            assert('Dimension of PPCA subspace must be less than data.')
        mix.ppca_dim = ppca_dim

    mix.priors = np.ones([1,mix.ncentres]) / mix.ncentres

    if mix.covar_type == 'spherical':
        mix.covars = np.ones([1, mix.ncentres])
        mix.nwts = mix.ncentres + mix.ncentres * mix.nin + mix.ncentres
    elif mix.covar_type == 'diag':
        mix.covars = np.ones([mix.ncentres, mix.nin])
        mix.nwts = mix.ncentres + mix.ncentres * mix.nin + mix.ncentres*mix.nin
    elif mix.covar_type == 'full':
        mix.covars = np.tile(np.eye(mix.nin), [1, 1, mix.ncentres])
        mix.nwts = mix.ncentres + mix.ncentres * mix.nin \
                   + mix.ncentres * mix.nin * mix.nin
    elif case == 'ppca':
        mix.covars = 0.1 * np.ones([1, mix.ncentres])
        init_space = np.eye(mix.nin)
        init_space = init_space[:][0:mix.ppca_dim - 1]
        init_space[mix.ppca_dim:mix.nin-1][:] = \
            np.ones([mix.nin - mix.ppca_dim, mix.ppca_dim])
        mix.U = np.tile(init_space, [1, 1, mix.ncentres])
        mix.lambda_ = np.ones([mix.ncentres, mix.ppca_dim])
        mix.nwts = mix.ncentres + mix.ncentres * mix.nin + mix.ncentres + \
                   mix.ncentres * mix.ppca_dim + mix.ncentres * mix.nin * mix.ppca_dim
    else:
        assert('Unknown covariance type ', mix.covar_type)

    return mix
