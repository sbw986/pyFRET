import numpy as np
from src.get_mix import get_gmm_mix
from src.vbpy_VBEM import vbpy_VBEM
from src.chmmViterbi import chmmViterbi
import pdb

class Priors:
    def __init__(self, D):
        """
        :type D: Int representing dimension of data set
        """
        self.upi = 1
        self.mu = 0.5 * np.ones([D, 1])
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

class vbpy:
    def __init__(self):
        D = 1
        self.run_params = RunParams(D, 2, 2, 10, x)
        self.prior_params = Priors(D)
        self.vb_options = VBopts(100, 1e-5, False, False, False, False)

        self.reset()

    def reset(self):
        self.bestOut = None
        self.outF = None
        self.best_idx = None

        self.x_hat = None
        self.z_hat = None
        self.data = None
        
    def fit(self, data):
        self.reset()
        
        self.data = data
        self.bestOut = [[[] for _ in range(self.run_params.K)] for _ in range(self.run_params.N)]
        self.outF = -1 * np.inf * np.ones([self.run_params.N, self.run_params.K])
        self.best_idx = np.zeros([self.run_params.N, self.run_params.K])

        for n in range(0, self.run_params.N):
            fret = self.data[n]
            for k in range(self.run_params.kmin, self.run_params.K+1):
                ncentres = k
                init_mu = (np.arange(1, ncentres+1))/(ncentres + 1)
                i = 1
                maxLP = -1 * np.inf
                while i < self.run_params.I + 1:
                    if k == 1 and i > 3:
                        break
                    if i > 1:
                        init_mu = np.random.rand(ncentres, 1)
                    mix = get_gmm_mix(fret, init_mu)
                    out = vbpy_VBEM(fret, mix, self.prior_params, self.vb_options)

                    if out.F[-1] > maxLP:
                        maxLP = out.F[-1]
                        #bestMix[n][k] = mix
                        self.bestOut[n][k-1] = out
                        self.outF[n][k-1] = out.F[-1]
                        self.best_idx[n][k-1] = i

                    i += 1

    def predict(self, data=None):
        if data is not None:
            transform_data = data
        else:
            transform_data = self.data
        self.z_hat = [[[] for _ in range(self.run_params.K)] for _ in range(self.run_params.N)]
        self.x_hat = [[[] for _ in range(self.run_params.K)] for _ in range(self.run_params.N)]
        for n in range(self.run_params.N):
            for k in range(self.run_params.kmin, self.run_params.K + 1):
                self.x_hat[n][k-1], self.z_hat[n][k-1] = chmmViterbi(self.bestOut[n][k-1], transform_data[n])

if __name__ == '__main__':

    D = 1
    #x = np.array([[1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0]])
    x = np.array([[1.0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0]])
    for i, d in enumerate(x[0]):
        x[0, i] = d + np.random.normal(0, 0.1)
    pf = vbpy()
    pf.fit(x)
    pf.predict()
    print(pf.z_hat)
    print(pf.x_hat)
