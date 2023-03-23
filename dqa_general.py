

# ====================================================
#  Quantum Information and Computing exam project
#
#   UNIPD Project |  AY 2022/23  |  QIC
#   group : Barone, Coppi, Zinesi
# ----------------------------------------------------
#   > description                                    |
#
#   main functions for dQA: hamiltonians, MPOs,
#   dataset, ...
# ----------------------------------------------------
#   coder : Barone Francesco, Zinesi Paolo
#         :   github.com/baronefr/
#   dated : 17 March 2023
#     ver : 1.0.0
# ====================================================

import numpy as np
import matplotlib.pyplot as plt


class PerceptronHamiltonian:

    def make_Ux(N, beta_p, dtype = np.complex128):
        """Return as MPO the U_x evolution operator at time-parameter beta_p."""

        tb = np.array( [[np.cos(beta_p), 1j*np.sin(beta_p)],[1j*np.sin(beta_p), np.cos(beta_p)]], dtype=dtype)
        return [ np.expand_dims(tb, axis=(0,1)) for _ in range(N) ]

    def Wz(N, Uk : np.array, xi : int, marginal = None, dtype = np.complex128):
        """The tensors of Eq. 17 of reference paper."""
        
        bond_dim = len(Uk)

        if marginal == 'l':
            shape = (1,bond_dim,2,2)
        elif marginal == 'r':
            shape = (bond_dim,1,2,2)
        else:
            shape = (bond_dim,bond_dim,2,2)

        tensor = np.zeros( shape, dtype = dtype )

        coeff = np.power( Uk/np.sqrt(N+1), 1/N)
        exx = 1j * np.arange(bond_dim) * np.pi / (N + 1)    # check: N+1

        for kk in range(bond_dim):
            spin_matrix = np.diag(
                [ coeff[kk]*np.exp(exx[kk]*(1-xi)), 
                coeff[kk]*np.exp(exx[kk]*(1+xi)) ] 
            )
            if marginal == 'l':
                tensor[0,kk,:,:] = spin_matrix
            elif marginal == 'r':
                tensor[kk,0,:,:] = spin_matrix
            else:
                tensor[kk,kk,:,:] = spin_matrix

        return tensor

    def make_Uz(N : int, Uk : np.array, xi : np.array, dtype = np.complex128):
        """Return as MPO the U_z evolution operator at time s_p (defined indirectly by Uk)."""

        # Uk must be a vector for all k values, while p is fixed 
        # xi must be a single sample from dataset
        
        assert len(xi) == N, 'not matching dims'

        arrays = [ PerceptronHamiltonian.Wz(N, Uk, xi[0], marginal = 'l', dtype = dtype) ] + \
                [ PerceptronHamiltonian.Wz(N, Uk, xi[i+1], dtype = dtype) for i in range(N-2) ] + \
                [ PerceptronHamiltonian.Wz(N, Uk, xi[N-1], marginal = 'r', dtype = dtype) ]

        return arrays
    

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
        return PerceptronHamiltonian.h_perceptron(m, N)


    def Hz_mu_singleK(N, mu, K, f_FT_, patterns):
        """ Build factorized Hz^{mu,k} (bond dimension = 1) on N sites"""

        d = 2
        Hz_i = []
        for i in range(1,N+1):
            tens = np.zeros((1,1,d,d), dtype=np.complex128)
            for s_i in range(d):
                tens[0,0,s_i,s_i] = np.power(f_FT_[K]/np.sqrt(N+1), 1/N) * np.exp(1.0j * (np.pi/(N+1)) * K * (1-patterns[mu,i-1]*(-1)**s_i))
            Hz_i.append(tens)

        #Hz = qtn.MPO_product_operator(Hz_i)#, upper_ind_id='u{}', lower_ind_id='s{}')
        return Hz_i



def create_dataset(N : int, features : int):
    """Create dataset as described by ref. paper, i.e. random +-1 values."""
    x = np.random.randint(2, size=(N, features))
    x[ x == 0 ] = -1  # data is encoded as +- 1
    return x

def plot_loss(loss):
    plt.plot( *zip( *np.real_if_close(loss) ) )
    plt.yscale('log')
    plt.title('dQA')
    return plt
