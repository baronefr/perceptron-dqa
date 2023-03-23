

# ====================================================
#  Quantum Information and Computing exam project
#
#   UNIPD Project |  AY 2022/23  |  QIC
#   group : Barone, Coppi, Zinesi
# ----------------------------------------------------
#   > description                                    |
#
#   testing the custom tenn functions, comparing
#   with Quimb output
# ----------------------------------------------------
#   coder : Barone Francesco
#         :   github.com/baronefr/
#   dated : 17 March 2023
#     ver : 1.0.0
# ====================================================


# %%

import numpy as np
from tenn import *

# reference lib to compare against
import quimb as qu
import quimb.tensor as qtn

from bak.lrp_utility import compress_svd_normalized as csvd_test

# %% parameters of this test

N = 5
mpo_bond_dim = 10
mps_bond_dim = 3


# %%

# create random MPO and MPS
mpo = make_rand_mpo_complex(N, bond_dim = mpo_bond_dim)
mps = make_rand_mps_complex(N, bond_dim = mps_bond_dim)

# take the product
my_res = apply_mpsmpo(mps, mpo)

# %%  compute in quimb

qmpo = qtn.tensor_1d.MatrixProductOperator( mpo )
qmps = qtn.tensor_1d.MatrixProductState( mps , shape='lpr') # force correct reshape lpr -> lrp
quimb_res = qmpo.apply(qmps)

# %% check the result

i = 0
for mm, qq in zip(my_res, quimb_res.tensors):
    # remember to convert my result from (l,p,r) to (l,r,p), which is the format used in quimb

    # check numerical equivalence
    assert np.allclose( np.transpose(mm, (0,2,1)), qq.data), "site {} not matching".format(i)
    i+=1

# %%

qmps.right_canonize(start=N,stop=0)
myrc = right_canonize(mps, 1, True)

# %%

i = 1
for mm, qq in zip(myrc[1:], qmps.tensors[1:]):
    # check numerical equivalence
    assert np.allclose( mm.transpose(0,2,1), qq.data), "site {} not matching".format(i)
    i+=1


# %%

print('test passed')


# %%


mps_tenso = convert_from_quimb(quimb_res)
mps_tenso = convert(mps_tenso)

# %%
tenso_svd = csvd_test(mps_tenso, 4)

# %% 

buuh = compress_svd_normalized(my_res, max_bd = 4)

# %%

i = 0
for mm, qq in zip(buuh, tenso_svd):
    # check numerical equivalence
    assert np.allclose( np.transpose(mm, (0,2,1)), qq.data), "site {} SVD not matching".format(i)
    i+=1

# %%
