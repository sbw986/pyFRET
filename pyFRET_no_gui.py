import numpy as np
import src.get_mix as get_mix
import src.pyFRET_VBEM as pyFRET_VBEM
import src.chmmViterbi as chmmViterbi
import pdb

class Priors:
    def __init__(self, D):
        """
        :type D: Int representing dimension of data set
        """
        self.upi = 1
        self.mu = 0.5 * np.ones([D,1])
        self.beta = 0.25
        self.W = 50 * np.eye(D)
        self.v = 5
        self.ua = 1.0
        self.uad = 0

class RunParams:
    def __init__(self, D, kmin, K, I, data):
        """
        :type D: Int, dimension of data set
        :type kmin: Int, min num of states to try
        :type K: Int, max num of states to try
        :type I: Int, max num of restarts
        :type data: List[List[floats]], raw data
        """
        self.D = D
        self.kmin = kmin
        self.K = K
        self.I = I
        self.N = len(data)

class VBopts:
    def __init__(self, max_iter, threshold, display_fig, \
                 display_nrg, display_iter, display_iters_to_converg):
        self.max_iter = max_iter
        self.threshold = threshold
        self.display_fig = display_fig
        self.display_nrg = display_nrg
        self.display_iter = display_iter
        self.display_iters_to_converg = display_iters_to_converg

# Parameter settings
def main(run_params, prior_params, vb_options, data):
    bestOut = [[[] for _ in range(run_params.K)] for _ in range(run_params.N)]
    outF = -1 * np.inf * np.ones([run_params.N, run_params.K])
    best_idx = np.zeros([run_params.N, run_params.K])

    for n in range(0, run_params.N):
        fret = data[n]
        for k in range(run_params.kmin, run_params.K+1):
            ncentres = k
            init_mu = (np.arange(1, ncentres+1))/(ncentres + 1)
            i = 1
            maxLP = -1 * np.inf
            while i < run_params.I + 1:
                if k == 1 and i > 3:
                    break
                if i > 1:
                    init_mu = np.random.rand(ncentres, 1)
                mix = get_mix.get_gmm_mix(fret, init_mu)
                out = pyFRET_VBEM.pyFRET_VBEM(fret, mix, prior_params, vb_options)

                if out.F[-1] > maxLP:
                    maxLP = out.F[-1]
                    #bestMix[n][k] = mix
                    bestOut[n][k-1] = out
                    outF[n][k-1] = out.F[-1]
                    best_idx[n][k-1] = i

                i += 1
    pdb.set_trace()
    z_hat = [[[] for _ in range(run_params.K)] for _ in range(run_params.N)]
    x_hat = [[[] for _ in range(run_params.K)] for _ in range(run_params.N)]
    for n in range(run_params.N):
        for k in range(run_params.kmin, run_params.K + 1):
            x_hat[n][k-1], z_hat[n][k-1] = chmmViterbi.chmmViterbi(bestOut[n][k-1], data[n])
    pdb.set_trace()

if __name__ == '__main__':

    D = 1
    #x = np.array([[1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0]])
    x = np.array([[1, 0, 0, 1, 0, 1, 2, 2, 1, 1, 0, 2, 1, 1, 0, 1, 1, 0]])

    run_pars = RunParams(D, 3, 3, 10, x)
    prior_pars = Priors(D)
    vb_opts = VBopts(100, 1e-5, False, False, False, False)

    main(run_pars, prior_pars, vb_opts, x)
