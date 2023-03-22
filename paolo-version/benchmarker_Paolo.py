#!/usr/bin/python3

# ====================================================
#  Quantum Information and Computing exam project
#
#   UNIPD Project |  AY 2022/23  |  QIC
#   group : Barone, Coppi, Zinesi
# ----------------------------------------------------
#   > description                                    |
#
#   utility script
# ----------------------------------------------------
#   coder : Barone Francesco
#         :   github.com/baronefr/
#   dated : 17 March 2023
#     ver : 1.0.0
# ====================================================

import os
from datetime import datetime

import numpy as np
import numpy.fft as fft
from scipy.special import binom

from quimb import tensor as qtn



# ----------------------
#   TARGET DEFINITIONS
# ----------------------

# function to run -------------------
def function_to_run(P, dt, max_bond_dim, filepath_patterns, **ignore ):
    
    # definition of parameters
    csi_patterns = np.load(filepath_patterns).astype(int)
    tau = P*dt
    N_csi, N = csi_patterns.shape
    d = 2

    # Fourier transform of U_z of dimension (N+1,P)
    Uz_FT = np.zeros((N+1,P), dtype=np.complex128)
    for p in range(1,P+1):
        Uz_FT[:,p-1] = fft.fft(np.exp(-1.0j*(p/P)*dt*f_perceptron(range(N+1), N)), norm="ortho")

    # Fourier transform of f_perceptron(x) of dimension (N+1,)
    fx_FT = fft.fft(f_perceptron(range(N+1), N), norm="ortho")

    # start by creating psi0
    psi0 = make_psi0(N, state=[d**-0.5]*d)

    # 'training' loop
    psi = psi0.copy()


    for p in range(1,P+1):
        for mu in range(N_csi):

            # apply Uz to psi and compress it
            Uz_p_mu_ = Uz_p_mu(N, d, p, mu, Uz_FT, patterns=csi_patterns)
            psi = Uz_p_mu_.apply(psi, compress=True, max_bond=max_bond_dim, method="svd").copy()
        
        Ux_p_ = Ux_p(N, d, beta_p=(1-p/P)*dt)
        psi = Ux_p_.apply(psi).copy()


    # evaluate final energy density of psi
    eps = 0.0
    for mu in range(N_csi):
        for k in range(N+1):
            eps += compute_energy_density(psi, Hz_mu_singleK(N, d, mu, k, fx_FT, csi_patterns))

    return np.real_if_close(eps)



# ----------------------
#   USEFUL DEFINITIONS
# ----------------------
def make_psi0(N, state):
    """
    Build factorized psi0 (bond dimension = 1) on N sites starting from the same 'state'
    """

    mps = qtn.MPS_product_state([state for i in range(N)], site_ind_id='s{}')
    return mps


def Ux_p(N, d, beta_p):
    """ Build factorized Ux(beta_p) (bond dimension = 1) on N sites"""

    Ux_i = np.identity(d)*np.cos(beta_p) + 1.0j*(np.ones(d)-np.identity(d))*np.sin(beta_p) # single site operator
    Ux = qtn.MPO_product_operator([Ux_i]*N, upper_ind_id='u{}', lower_ind_id='s{}')
    return Ux


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




# parameters to test ----------------
#
#  NOTE: format as list of dictionaries, which will be passed to input function as arguments
#
parameter_combinations = [
    {'P' : 1000,
     'dt' : np.round(dt,3),
     'max_bond_dim' : 10,
     'filepath_patterns' : 'patterns_17-21.npy' }
    
    for dt in np.arange(start = 1.0, stop=2.0, step = 0.1)
]

# target file to log results
benchmark_file = 'test.csv'





# ----------------------
#         SETUP
# ----------------------

# benchmark settings ----------------
MAX_FAILURE_TOLERANCE = 4      # maximum number of failures to tolerate before abort benchmarking

# log settings ----------------------
error_log_file = 'errors.log'  # set to None to disable this feature

# create head of log file, if it does not exist
if not os.path.isfile(benchmark_file):
    print('making header of benchmark file')
    benf = open(benchmark_file, "w")
    benf.write( ','.join( '{}'.format(v) for v in parameter_combinations[0].keys()) )
    benf.write(',output\n')
    benf.close()
else:
    print('benchmark file exists, do not print header')





# ----------------------
#        EXECUTE
# ----------------------
exe_failure_counter = 0
for settings in parameter_combinations:

    try:
        # run the target function, but catch exceptions ...
        value_to_log = function_to_run(**settings)

    except Exception as e:
        value_to_log = None  # None marks a failure
        exe_failure_counter += 1

        # write this error to log file, if argument error_log_file valid
        if error_log_file is not None:
            errf = open(error_log_file, "a")
            errf.write("\n[{}] params {} failed ----------\n".format(
                datetime.now().strftime("%d/%m/%Y %H:%M:%S"), str(settings) )
            )
            errf.write( str(e) )
            errf.close()
    
    # write to file the result of this run
    benf = open(benchmark_file, "a")
    benf.write( ','.join( '{}'.format(v) for v in settings.values()) )
    benf.write(',' + str(value_to_log) + '\n')
    benf.close()

    if exe_failure_counter >= MAX_FAILURE_TOLERANCE:
        raise Exception('multiple failures occurred in benchmarks, aborting...')


print('operations completed')
#os.exit(0)
