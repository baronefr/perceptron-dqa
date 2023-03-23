# %%

import numpy as np

from lib.tenn import *
from dQA_mps import mydQA


def Hz_mu_singleK_with_ancilla(N, mu, K, f_FT_, patterns, extra_ancillary):
    """ Build factorized Hz^{mu,k} (bond dimension = 1) on N sites"""

    d = 2
    Hz_i = []
    for i in range(1,N+1):
        tens = np.zeros((1,1,d,d), dtype=np.complex128)
        for s_i in range(d):
            tens[0,0,s_i,s_i] = np.power(f_FT_[K]/np.sqrt(N+1), 1/N) * np.exp(1.0j * (np.pi/(N+1)) * K * (1-patterns[mu,i-1]*(-1)**s_i))
        Hz_i.append(tens)

    # add identities
    for i in range(extra_ancillary):
        tens = np.zeros((1,1,d,d), dtype=np.complex128)
        for s_i in range(d):
            tens[0,0,s_i,s_i] = 1
        Hz_i.append(tens)

    return Hz_i



class mydQA_ancilla(mydQA):

    def __init__(self, dataset, P : int, dt : float, max_bond : int = 10, device = None, n_ancilla = 0):
        super().__init__(dataset, P, dt, max_bond, device)
        self.n_ancilla = n_ancilla

    def compute_loss(self, psi):
        N_tens = len(psi) - self.n_ancilla
        eps = 0.0
        for mu in range(self.N_xi):
            for kk in range(self.N+1):
                mpo = Hz_mu_singleK_with_ancilla(self.N, mu, kk, self.fxft, self.dataset, self.n_ancilla)

                psiH = apply_mpsmpo(psi, mpo)

                E = braket(psiH, psi)
                eps += E/N_tens
        return eps[0,0]

    def run():
        raise Exception('do not use this class to run dQA!!')



if __name__== "__main__":
    obj = mydQA_ancilla('data/patterns_17-21.1.npy', 100, 1, max_bond=10, n_ancilla=3)
    obj.init_fourier()
    # then use  obj.compute_loss(psi)  to compute the loss

# %%