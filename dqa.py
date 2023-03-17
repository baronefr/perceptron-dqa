#!/usr/bin/python3

# ====================================================
#  Quantum Information and Computing exam project
#
#   UNIPD Project |  AY 2022/23  |  QIC
#   group : Barone, Coppi, Zinesi
# ----------------------------------------------------
#   > description                                    |
#
#   class setup of dQA execution
# ----------------------------------------------------
#   coder : Barone Francesco, Zinesi Paolo
#         :   github.com/baronefr/
#   dated : 17 March 2023
#     ver : 1.0.0
# ====================================================


# %%

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

import jax
from tqdm import tqdm

# custom functions
from tenn import *
from dqa_general import *

# %%


class mydQA():

    def __init__(self, dataset, P : int, dt : float, max_bond : int = 10, device = None):

        if device is None:
            self.device = jax.devices('cpu')[0]
        
        if isinstance(dataset, str):
            self.dataset = np.load(dataset)
        else:
            self.dataset = dataset

        self.N_xi = self.dataset.shape[0]
        self.N = self.dataset.shape[1]
        
        self.P = P
        self.dt = dt
        self.tau = dt * P
        self.max_bond = max_bond

        self.pp = 0 # internal steps counter

    def init_fourier(self):
        self.Uk_FT = np.zeros((self.N+1,self.P), dtype=np.complex128)
        for p in range(0,self.P):
            self.Uk_FT[:,p] = fft.fft( np.exp(-1.0j*((p+1)/self.P)*(self.dt)*PerceptronHamiltonian.f_perceptron(range(self.N+1), self.N)), norm="ortho")
        self.fxft = fft.fft( PerceptronHamiltonian.f_perceptron(range(self.N+1), self.N), norm="ortho" )

    def compute_loss(self, psi):
        N_tens = len(psi)
        eps = 0.0
        for mu in range(self.N_xi):
            for kk in range(self.N+1):
                mpo = PerceptronHamiltonian.Hz_mu_singleK(self.N, mu, kk, self.fxft, self.dataset)
                # TODO: eventually store these matrices, since they are all the same

                psiH = apply_mpsmpo(psi, mpo)

                E = braket(psiH, psi)
                eps += E/N_tens
        return eps[0,0]


    def single_step(self):

        psi = self.psi

        s_p = (self.pp+1)/self.P
        beta_p = (1-s_p)*self.dt

        # loop over patterns
        for mu in range(self.N_xi):

            Uz = PerceptronHamiltonian.make_Uz(self.N, self.Uk_FT[:,self.pp], self.dataset[mu])
            psi = apply_mpsmpo(psi, Uz)

            #preprocess = compress_svd_normalized(psi, max_bd=max_bond)
            #psi = right_canonicalize(preprocess, 1) # makes loss much more stable

            preprocess = right_canonize(psi, 1)     # makes loss much more stable
            psi = compress_svd_normalized(preprocess, max_bd=self.max_bond)

            curr_bdim = psi[int(self.N/2)].shape[0]
            self.bd_monitor.append( curr_bdim )

        Ux = PerceptronHamiltonian.make_Ux(self.N, beta_p = beta_p)
        psi = apply_mpsmpo(psi, Ux)

        # evaluate  <psi | H | psi>
        expv = self.compute_loss(psi)

        # do not return implicitly, but update internal objects
        self.psi = psi
        self.loss.append( (s_p, expv) )
        self.pp += 1

        return expv


    def run(self, skip_jit = None) -> float:

        print('dQA---')
        print(' tau = {}, P = {}, dt = {}'.format(self.tau, self.P, self.dt) )
        print(' dataset :  N = {}, N_xi = {}'.format(self.N, self.N_xi) )
        
        # initialize state and internal counter
        self.psi = [ np.array([[[2**-0.5], [2**-0.5]]], dtype=np.complex128) ] * self.N
        self.pp = 0

        # reset trackers
        self.loss = []
        self.bd_monitor = []
        
        l = self.compute_loss(self.psi)  # evaluate loss for first time
        self.loss.append( (0, l) )

        pbar = tqdm(total=self.P, desc='QAnnealing')

        # exe without jit (useful at the beginning...)
        if skip_jit is not None:
            with jax.disable_jit():
                for _ in range(skip_jit):
                    expv = self.single_step()

                    # etc
                    pbar.update(1)
                    pbar.set_postfix_str("loss = {}, bd = {}".format( np.around(expv, 5), self.bd_monitor[-1] ) )

        else:
            skip_jit = 0

        # EXE
        with jax.default_device(self.device):
                for _ in range(self.P - skip_jit):
                    expv = self.single_step()

                    # etc
                    pbar.update(1)
                    pbar.set_postfix_str("loss = {}, bd = {}".format( np.around(expv, 5), self.bd_monitor[-1] ) )

# %%

#    usage example:
#obj = mydQA('data/patterns_17-21.npy', 100, 1, max_bond=10)
#obj.init_fourier()
#obj.run()
#plot_loss( obj.loss )

# %%
