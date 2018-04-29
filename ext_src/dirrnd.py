import numpy as np

def dirrnd(pu, n = None):
    s = pu.shape[0]
    #if len(pu.shape) == 2:
#        s = pu.shape[1]
#    else:
#        s = 1

    if n == None:
        n = pu.shape[0]
    else:
        #if pu.shape[1] == 1:
        pu = np.ones((n,1)) * pu
        #else:
            #return

    samples = np.random.gamma(pu)
    #TODO sum is missing dimensionality factor
    samples = samples / (np.sum(samples) * np.ones(s))

    return samples
