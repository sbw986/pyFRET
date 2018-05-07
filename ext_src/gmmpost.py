import numpy as np

def gmmpost(mix, x):

    ndata = len(x)

    #TODO write gmmactiv
    a = gmmactiv(mix, x)
    post = np.matmul(np.ones(ndata)mmix.priors) * a
    s = np.sum(post, 1)
    if any(s == 0):
        print('Some zero posterior probabilities')
        zero_rows = np.where(s == 0)
        s = s + (s == 0)
        post[zero_rows, :] = 1/mix.ncentres

    post = post/np.matmul(s, np.ones(mix.ncentres))

    return post, a
