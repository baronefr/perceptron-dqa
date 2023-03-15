
# %%

import numpy as np
import quimb as qu
import quimb.tensor as qtn

from tenso import *

# %% create tensors as numpy arrays

def make_Ux(N, beta_p, dtype = np.complex128):
    """Return as MPO the U_x evolution operator at time-parameter beta_p."""

    tb = np.array( [[np.cos(beta_p), 1j*np.sin(beta_p)],[1j*np.sin(beta_p), np.cos(beta_p)]], dtype=dtype)
    
    arrays = [ np.expand_dims(tb, axis=0) ] + \
             [ np.expand_dims(tb, axis=(0,1)) for _ in range(N-2) ] + \
             [ np.expand_dims(tb, axis=0) ]

    return arrays

def Wz(N, Uk : np.array, xi : int, marginal = False, dtype = np.complex128):
    """The tensors of Eq. 17 of reference paper."""
    
    bond_dim = len(Uk)

    shape = (bond_dim,2,2) if marginal else (bond_dim,bond_dim,2,2)
    tensor = np.zeros( shape, dtype = dtype )

    coeff = np.power( Uk/np.sqrt(N+1), 1/N)
    exx = 1j * np.arange(bond_dim) * np.pi / (N + 1)    # check: N+1

    for kk in range(bond_dim):
        spin_matrix = np.diag(
            [ coeff[kk]*np.exp(exx[kk]*(1-xi)), 
              coeff[kk]*np.exp(exx[kk]*(1+xi)) ] 
        )
        if marginal:  tensor[kk,:,:] = spin_matrix
        else:         tensor[kk,kk,:,:] = spin_matrix
    
    return tensor

def make_Uz(N : int, Uk , xi ): #, bond_dim = None, dtype = np.complex128):
    bond_dim = None
    dtype = np.complex128
    """Return as MPO the U_z evolution operator at time s_p (defined indirectly by Uk)."""

    # Uk must be a vector for all k values, while p is fixed 
    # xi must be a single sample from dataset
    
    assert len(xi) == N, 'not matching dims'

    arrays = [ Wz(N, Uk, xi[0], marginal = True, dtype = dtype) ] + \
             [ Wz(N, Uk, xi[i+1], dtype = dtype) for i in range(N-2) ] + \
             [ Wz(N, Uk, xi[N-1], marginal = True, dtype = dtype) ]

    return arrays


# %% create tensors with Quimb routine

def make_Ux_tn(N, beta_p, dtype = np.complex128):
    """Return as MPO the U_x evolution operator at time-parameter beta_p."""

    tb = np.array( [[np.cos(beta_p), 1j*np.sin(beta_p)],[1j*np.sin(beta_p), np.cos(beta_p)]], dtype=dtype)
    
    arrays = [ np.expand_dims(tb, axis=0) ] + \
             [ np.expand_dims(tb, axis=(0,1)) for _ in range(N-2) ] + \
             [ np.expand_dims(tb, axis=0) ]

    return qtn.tensor_1d.MatrixProductOperator( arrays )

def make_Uz_tn(N : int, Uk , xi ): #, bond_dim = None, dtype = np.complex128):
    bond_dim = None
    dtype = np.complex128
    """Return as MPO the U_z evolution operator at time s_p (defined indirectly by Uk)."""

    # Uk must be a vector for all k values, while p is fixed 
    # xi must be a single sample from dataset
    
    assert len(xi) == N, 'not matching dims'

    arrays = [ Wz(N, Uk, xi[0], marginal = True, dtype = dtype) ] + \
             [ Wz(N, Uk, xi[i+1], dtype = dtype) for i in range(N-2) ] + \
             [ Wz(N, Uk, xi[N-1], marginal = True, dtype = dtype) ]

    return qtn.tensor_1d.MatrixProductOperator( arrays )





# %% load test data

N = 4

Uk_FT = np.ones((N+1,10), dtype=np.complex128)
xi = np.loadtxt('dataxi.txt')[:,0:N]






# %% TEST 1: compute values with Quimb

TN_psi = qu.tensor.tensor_builder.MPS_product_state( 
    [ np.array([[2**-0.5, 2**-0.5]], dtype=np.complex128) ] * N, tags=['psi'],
)
TN_uz = make_Uz_tn(N, Uk_FT[:,0], xi[0])

quimb_psi_1 = TN_uz.apply(TN_psi)
quimb_psi_2 = TN_uz.apply(quimb_psi_1)
quimb_psi_3 = TN_uz.apply(quimb_psi_2)

# %% TEST 1: compute values with tenso and check

psi = [ np.array([[2**-0.5, 2**-0.5]], dtype=np.complex128) ] + [ np.array([[[2**-0.5, 2**-0.5]]], dtype=np.complex128) for i in range(N-2) ] + [ np.array([[2**-0.5, 2**-0.5]], dtype=np.complex128) ]
uz = make_Uz(N, Uk_FT[:,0], xi[0])

my_psi_1 = apply_mpsmpo(psi, uz)
my_psi_2 = apply_mpsmpo(my_psi_1, uz)
my_psi_3 = apply_mpsmpo(my_psi_2, uz)



for i in range(N):
    assert np.allclose( my_psi_1[i] , np.array( quimb_psi_1.tensors[i].data ) )

for i in range(N):
    assert np.allclose( my_psi_2[i] , np.array( quimb_psi_2.tensors[i].data ) )

for i in range(N):
    assert np.allclose( my_psi_3[i] , np.array( quimb_psi_3.tensors[i].data ) )


# %% TEST 2: compute values with my lib

dty = np.complex128

psi = [ np.array([[2**-0.5, 2**-0.5]], dtype=dty) ] + [ np.array([[[2**-0.5, 2**-0.5]]], dtype=dty) for i in range(N-2) ] + [ np.array([[2**-0.5, 2**-0.5]], dtype=dty) ]
uz = make_rand_mpo(N, bond_dim=5)

my_psi_1 = apply_mpsmpo(psi, uz)
my_psi_2 = apply_mpsmpo(my_psi_1, uz)
my_psi_3 = apply_mpsmpo(my_psi_2, uz)

# %% TEST 2: compute values with Quimb

TN_psi = qu.tensor.tensor_builder.MPS_product_state( 
    [ np.array([[2**-0.5, 2**-0.5]], dtype=dty) ] * N, tags=['psi'],
)
TN_uz = qtn.tensor_1d.MatrixProductOperator( uz )

quimb_psi_1 = TN_uz.apply(TN_psi)
quimb_psi_2 = TN_uz.apply(quimb_psi_1)
quimb_psi_3 = TN_uz.apply(quimb_psi_2)




for i in range(N):
    assert np.allclose( my_psi_1[i] , np.array( quimb_psi_1.tensors[i].data ) )

for i in range(N):
    assert np.allclose( my_psi_2[i] , np.array( quimb_psi_2.tensors[i].data ) )

for i in range(N):
    assert np.allclose( my_psi_3[i] , np.array( quimb_psi_3.tensors[i].data ) )


# %%

