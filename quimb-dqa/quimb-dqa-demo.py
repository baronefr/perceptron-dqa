# %%
import quimb.tensor as qtn

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

#from tqdm import trange
import os
from datetime import datetime
import pytz

os.system("rm -f events.log")

# %%
# parameters
tau = 1000
P = 1000
dt = tau/P
max_bond_dim = 10
N = 21           # qubits (features) to simulate
N_csi = 17       # training samples
d = 2

# training data (all labels are set =1 for simplicity)
csi_patterns = np.random.choice([-1,1], size=(N_csi,N))
#csi_patterns

# %% [markdown]
# ## Definitions

# %% [markdown]
# ### Initial state

# %%
def make_psi0(N, state):
    """
    Build factorized psi0 (bond dimension = 1) on N sites starting from the same 'state'
    """

    mps = qtn.MPS_product_state([state for i in range(N)], site_ind_id='s{}')#, site_tag_id='psi0_{}')
    return mps


# %% [markdown]
# ### $\hat{U}_x$ operator

# %%
def Ux_p(N, d, beta_p):
    """ Build factorized Ux(beta_p) (bond dimension = 1) on N sites"""

    Ux_i = np.identity(d)*np.cos(beta_p) + 1.0j*(np.ones(d)-np.identity(d))*np.sin(beta_p) # single site operator
    Ux = qtn.MPO_product_operator([Ux_i]*N, upper_ind_id='u{}', lower_ind_id='s{}')
    return Ux


# %% [markdown]
# ### $\hat{U}_z$ operator

# %%
def h_perceptron(m, N):
    """ Cost function to be minimized in the perceptron model, depending on the overlap m.
        The total H_z Hamiltonian is obtained as a sum of these cost functions evaluated at each pattern csi_mu.

        h(m) = 0 if m>=0 else -m/sqrt(N)
    """
    m = np.array(m)
    return np.where(m>=0, 0, -m/np.sqrt(N)).squeeze()


def f_perceptron(x, N):
    """ Cost function to be minimized in the perceptron model, depending on the Hamming distance x.
        The total H_z Hamiltonian is obtained as a sum of these cost functions evaluated at each pattern csi_mu.

        f(x) = h(N - 2x) = h(m(x)) with m(x) = N - 2x
    """

    m = N - 2*np.asarray(x)
    return h_perceptron(m, N)

# %%
# Fourier transform of U_z of dimension (N+1,P)
Uz_FT = np.zeros((N+1,P), dtype=np.complex128)

for p in range(1,P+1):
    Uz_FT[:,p-1] = fft.fft(np.exp(-1.0j*(p/P)*dt*f_perceptron(range(N+1), N)), norm="ortho")

# %%
def Uz_p_mu(N, d, p, mu, Uz_FT_, patterns):
    """ Build Uz^mu(gamma_p) (bond dimension = N+1) on N sites
        - p in range(1,P+1)
    """

    Uz_i = []

    # leftermost tensor (i = 1)
    i = 1
    tens = np.zeros((N+1,d,d), dtype=np.complex128)
    for s_i in range(d):
        tens[:,s_i,s_i] = np.power(Uz_FT_[:,p-1]/np.sqrt(N+1), 1/N) * np.exp(1.0j * (np.pi/(N+1)) * np.arange(N+1) * (1-patterns[mu,i-1]*(-1)**s_i))
    Uz_i.append(tens.copy())

    # bulk tensors (2 <= i <= N-1)
    for i in range(2,N):
        tens = np.zeros((N+1,N+1,d,d), dtype=np.complex128)
        for s_i in range(d):
            np.fill_diagonal(tens[:,:,s_i,s_i], 
                             np.power(Uz_FT_[:,p-1]/np.sqrt(N+1), 1/N) * np.exp(1.0j * (np.pi/(N+1)) * np.arange(N+1) * (1-patterns[mu,i-1]*(-1)**s_i)))
        Uz_i.append(tens.copy())
    

    # rightermost tensor (i = N)
    i = N
    tens = np.zeros((N+1,d,d), dtype=np.complex128)
    for s_i in range(d):
        tens[:,s_i,s_i] = np.power(Uz_FT_[:,p-1]/np.sqrt(N+1), 1/N) * np.exp(1.0j * (np.pi/(N+1)) * np.arange(N+1) * (1-patterns[mu,i-1]*(-1)**s_i))
    Uz_i.append(tens.copy())


    Uz = qtn.tensor_1d.MatrixProductOperator(Uz_i, upper_ind_id='u{}', lower_ind_id='s{}')
    return Uz


# %% [markdown]
# ### $\hat{U}_z$ operator as a sum of smaller MPOs

# %%
def Uz_p_mu_singleK(N, d, p, mu, K, Uz_FT_, patterns):
    """ Build factorized Uz^{mu,k}(gamma_p) (bond dimension = 1) on N sites"""

    Uz_i = []
    for i in range(1,N+1):
        tens = np.zeros((d,d), dtype=np.complex128)
        for s_i in range(d):
            tens[s_i,s_i] = np.power(Uz_FT_[K,p-1]/np.sqrt(N+1), 1/N) * np.exp(1.0j * (np.pi/(N+1)) * K * (1-patterns[mu,i-1]*(-1)**s_i))
        Uz_i.append(tens.copy())

    Uz = qtn.MPO_product_operator(Uz_i, upper_ind_id='u{}', lower_ind_id='s{}')
    return Uz


# %% [markdown]
# **Eventually try to implement the variational compression algorithm**

# %% [markdown]
# # dQA of Perceptron model

# %%
# Fourier transform of f_perceptron(x) of dimension (N+1,)
fx_FT = fft.fft(f_perceptron(range(N+1), N), norm="ortho")

# %%
def Hz_mu_singleK(N, d, mu, K, f_FT_, patterns):
    """ Build factorized Hz^{mu,k} (bond dimension = 1) on N sites"""

    Hz_i = []
    for i in range(1,N+1):
        tens = np.zeros((d,d), dtype=np.complex128)
        for s_i in range(d):
            tens[s_i,s_i] = np.power(f_FT_[K]/np.sqrt(N+1), 1/N) * np.exp(1.0j * (np.pi/(N+1)) * K * (1-patterns[mu,i-1]*(-1)**s_i))
        Hz_i.append(tens.copy())

    Hz = qtn.MPO_product_operator(Hz_i, upper_ind_id='u{}', lower_ind_id='s{}')
    return Hz

# %%
def compute_energy_density(psi, H_z_mu_K):
    """
    Compute energy density (a figure of merit) of the MPS 'psi' w.r.t. the MPO Hamiltonian H_z.
    Energy density is computed as eps = (1/N) \bra{psi} H_z^{mu,k} \ket{psi}.
    """

    N_tens = psi.num_tensors
    psi_H = psi.reindex({f"s{i}":f"u{i}" for i in range(N_tens)}).H
    tn = (psi_H & H_z_mu_K & psi)
    eps = tn.contract()/N_tens

    return eps

# %%
# start by creating psi0
psi0 = make_psi0(N, state=[d**-0.5]*d)
#psi0.draw()

# %%
# 'training' loop
psi = psi0.copy()
FoM_time = []

# starting FoM
eps = 0.0
for mu in range(N_csi):
    for k in range(N+1):
        eps += compute_energy_density(psi0, Hz_mu_singleK(N, d, mu, k, fx_FT, csi_patterns))
FoM_time.append(eps)

for p in range(1,P+1):
    for mu in range(N_csi):

        # apply Uz to psi and compress it
        Uz_p_mu_ = Uz_p_mu(N, d, p, mu, Uz_FT, patterns=csi_patterns)
        psi = Uz_p_mu_.apply(psi, compress=True, max_bond=max_bond_dim, method="svd").copy()
    
    Ux_p_ = Ux_p(N, d, beta_p=(1-p/P)*dt)
    psi = Ux_p_.apply(psi).copy()

    # evaluate FoM (energy density) of psi
    eps = 0.0
    for mu in range(N_csi):
        for k in range(N+1):
            eps += compute_energy_density(psi, Hz_mu_singleK(N, d, mu, k, fx_FT, csi_patterns))
    FoM_time.append(eps)

    if(p % 25 == 0):
        nowtime = datetime.now(tz=pytz.timezone("Europe/Rome")).strftime("%d/%m/%Y %H:%M:%S")
        os.system(f"echo Finished iteration p={p} at {nowtime} >> events.log")

FoM_time = np.real_if_close(FoM_time)


# save results
np.savetxt(f"eps_s_P{P}.txt", FoM_time)

