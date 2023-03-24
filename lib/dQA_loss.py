

# ====================================================
#  Quantum Information and Computing exam project
#
#   UNIPD Project |  AY 2022/23  |  QIC
#   group : Barone, Coppi, Zinesi
# ----------------------------------------------------
#   > description                                    |
#
#   dQA utilities: MPS evaluation of the loss
# ----------------------------------------------------
#   coder : Barone Francesco, Zinesi Paolo
#         :   github.com/baronefr/perceptron-dqa/
#   dated : 17 March 2023
#     ver : 1.0.0
# ====================================================


# %%

import numpy as np

from lib.tenn import *
from dQA_mps import mydQA


def Hz_mu_singleK_with_ancilla(N, mu, K, f_FT_, patterns, extra_ancillary):
    """Build factorized Hz^{mu,k} (bond dimension = 1) on N sites and add identities on ancilla qubits."""

    d = 2
    Hz_i = []
    for i in range(1,N+1):
        tens = np.zeros((1,1,d,d), dtype=np.complex128)
        for s_i in range(d):
            tens[0,0,s_i,s_i] = np.power(f_FT_[K]/np.sqrt(N+1), 1/N) * np.exp(1.0j * (np.pi/(N+1)) * K * (1-patterns[mu,i-1]*(-1)**s_i))
        Hz_i.append(tens)

    # add identities for ancillary qubits
    for i in range(extra_ancillary):
        tens = np.zeros((1,1,d,d), dtype=np.complex128)
        for s_i in range(d):
            tens[0,0,s_i,s_i] = 1
        Hz_i.append(tens)

    return Hz_i






class mydQA_ancilla(mydQA):

    def __init__(self, dataset, P : int, dt : float, device = None, n_ancilla : int = 0, flip_endian : bool = False):

        assert n_ancilla >= 0, 'number of ancillas must be non negative'

        if flip_endian:
            dataset = dataset[:,::-1]

        # initialize the mother class and call the fourier transform right now
        super().__init__(dataset, P, dt, 10, device)
        super().init_fourier()
        # define the number of ancillas
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

    def run(self) -> None:
        raise Exception('do not use this class to run dQA!')
    
    def single_step(self) -> None:
        raise Exception('do not use this class to run dQA!')



if __name__== "__main__":
    obj = mydQA_ancilla('data/patterns_17-21.1.npy', P = 100, dt = 1, n_ancilla=3)
    # then use  obj.compute_loss(psi)  to compute the loss given the circuit MPS psi

