def gmmem(mix, x, options):

    if len(x.shape) == 1:
        ndata = x.shape[0]
        xdim = 1
    else:
        ndata, xdim = x.shape

    if options[14] != 0:
        niters = options[14]
    else:
        niters = 100

    # TODO add missing store stuff...

    check_covars = 0
    if options[5] >= 1:
        check_covars = 1
        MIN_COVAR = eps
        init_covars = mix.covars

    for n in range(0,niters):
        post, act = gmmpost(mix, x)
        
