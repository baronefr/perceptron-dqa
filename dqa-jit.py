#!/usr/bin/python3

# ====================================================
#  Quantum Information and Computing exam project
#
#   UNIPD Project |  AY 2022/23  |  QIC
#   group : Barone, Coppi, Zinesi
# ----------------------------------------------------
#   > description                                    |
#
#   library functions
# ----------------------------------------------------
#   coder : Barone Francesco, Zinesi Paolo
#         :   github.com/baronefr/
#   dated : 17 March 2023
#     ver : 1.0.0
# ====================================================


import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

import jax
from tqdm import tqdm

# custom functions
from tenn import *
from dqa_general import *

backend = 'cpu' # or 'gpu' if available
device = jax.devices(backend)[0]




# %% [markdown]
# #### parameters of the dQA

N = 21     # number of spins/sites/parameters/qubits  (21)
P = 100   # total number of QA steps  // should be 100/1000
dt = 1   # time interval  (1 or 0.1)

# tau (annealing time) will be fixed as P*dt

max_bond = 10  # MPS max bond dimension

N_xi = 17  # dataset size (number of patterns)  17


# load the dataset
xi = np.load('data/patterns_17-21.npy')
assert N_xi == xi.shape[0], 'number of patterns does not match'
assert N == xi.shape[1], 'number of features does not match'


# %% [markdown]
# ## init the dqa algo

# compute the fourier transform of perceptron hamiltonian
Uk_FT = np.zeros((N+1,P), dtype=np.complex128)
for p in range(0,P):
    Uk_FT[:,p] = fft.fft( np.exp(-1.0j*((p+1)/P)*dt*PerceptronHamiltonian.f_perceptron(range(N+1), N)), norm="ortho")
fx_FT = fft.fft(PerceptronHamiltonian.f_perceptron(range(N+1), N), norm="ortho")


# %% utility function for our model

def compute_loss(psi, N, fxft, xi):
    N_tens = len(psi)
    eps = 0.0
    for mu in range(N_xi):
        for kk in range(N+1):
            mpo = PerceptronHamiltonian.Hz_mu_singleK(N, mu, kk, fxft, xi)
            # TODO: eventually store these matrices, since they are all the same

            psiH = apply_mpsmpo(psi, mpo)
            E = braket(psiH, psi)
            eps += E/N_tens
    return eps[0,0]

# %%

psi = [ np.array([[[2**-0.5], [2**-0.5]]], dtype=np.complex128) ] * N
l = compute_loss(psi, N, fx_FT, xi)
print('initial loss value:', l)

loss = []
cc = []




# %% [markdown]
# ## run the algo

crop_p = None

loss.append( (0, l) )
tau = dt * P

print('dQA ---')
print(' tau = {}, P = {}, dt = {}'.format(tau, P, dt) )
print(' device :', backend)

if crop_p is not None:
    print(' [!] simulation will be stopped at iter', crop_p)


#  finally... RUN!
with jax.default_device(device):
    with tqdm(total=P, desc='QAnnealing') as pbar:

        for pp in range(P):

            s_p = (pp+1)/P
            beta_p = (1-s_p)*dt

            # loop over patterns
            for mu in range(N_xi):
                #print('mu =', mu) # for DEBUG

                Uz = PerceptronHamiltonian.make_Uz(N, Uk_FT[:,pp], xi[mu])
                psi = apply_mpsmpo(psi, Uz)

                #step = compress_svd_normalized(psi, max_bd=max_bond)
                #psi = right_canonicalize(step 1) # makes loss much more stable

                step = right_canonize(psi, 1)     # makes loss much more stable
                psi = compress_svd_normalized(step, max_bd=max_bond)

                curr_bdim = psi[int(N/2)].shape[0]
                cc.append( curr_bdim )

            Ux = PerceptronHamiltonian.make_Ux(N, beta_p = beta_p)
            psi = apply_mpsmpo(psi, Ux)

            # evaluate  <psi | H | psi>
            expv = compute_loss(psi, N, fx_FT, xi)
            loss.append( (s_p, expv) )

            # etc
            pbar.update(1)
            pbar.set_postfix_str("loss = {}, bd = {}".format( np.around(expv, 5), curr_bdim ) )
            
            if crop_p is not None:
                if pp == crop_p:   break



# %% [markdown]
# #### plot loss

plt.plot( *zip( *np.real_if_close(loss) ) )
plt.yscale('log')
plt.title('dQA')
plt.show()

# %% [markdown]
# #### plot bond dimension trend

plt.plot(cc)
plt.title('bond dimension monitor')
plt.show()

