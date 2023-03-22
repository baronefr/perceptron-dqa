

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
#   coder : Barone Francesco
#         :   github.com/baronefr/
#   dated : 17 March 2023
#     ver : 1.0.0
# ====================================================


import numpy as np

from typing import List, Union
import jax
import jax.numpy as jnp
from functools import partial
import quimb.tensor as qtn



# -----------------------------------------------
#    general MPS/MPO operations
# -----------------------------------------------

@jax.jit
def convert(mps) -> List[np.ndarray]:
    """reshape MPS from (l,r,phys) to (l,phys,r)"""
    return [ np.transpose(tensor, (0,2,1)) for tensor in mps ]

def convert_from_quimb(mps):
    """reshape MPS from Quimb(l,r,phys) to (l,phys,r)"""
    return [ np.transpose(tensor.data, (0,2,1)) for tensor in mps.tensors ]

def remove_margin_extra_dim_in_lrp(mps) -> None:
    """
    Considering an MPS in format (l,r,phys), removes the sites with 
    bond dimension == 1 on the leftmost and rightmost site.
    """
    mps[0] = mps[0][0]
    mps[-1] = mps[-1][:,0,:]

def add_margin_extra_dim_in_lrp(mps) -> None:
    """
    Considering an MPS generically in format (l,r,phys), with first and last sites 
    having shape (1,phys), add an extra dimension to match the (l,r,phys) format on all sites.
    """
    mps[0] = np.expand_dims(mps[0], 0)
    mps[-1] = np.expand_dims(mps[-1], 1)

def to_dense_quimb(mps) -> None:
    """Use Quimb to retrieve a dense state from a MPS."""
    transposed = [ np.transpose(el, (0,2,1)) for el in mps ]
    return qtn.tensor_1d.MatrixProductState( transposed ).to_dense()

# -----------------------------------------------
#    MPS MPO operations
# -----------------------------------------------

@jax.jit
def contract_mps_mpo_single_site(state_site : List[np.ndarray], operator_site : List[np.ndarray]) -> List[np.ndarray]:
    """Given a single site of MPS and MPO, return the contraction merging the left and right dimensions."""
    #
    #                l
    #                |                MPS ~ (l,phys,r)
    #           j -- O -- k             
    #                | b (m)          MPO ~ (l,r,up,down)
    #           a -- @ -- c
    #
    tensor = jnp.einsum('abc,jklb->ajckl', state_site, operator_site)
    tensor = tensor.reshape( 
        (state_site.shape[0]*operator_site.shape[0], state_site.shape[2]*operator_site.shape[1], operator_site.shape[3])
    )
    return np.transpose(tensor, (0,2,1) ) # using (l,phys,r) ordering

@jax.jit
def apply_mpsmpo(mps : List[np.ndarray], mpo : List[np.ndarray]) -> List[np.ndarray]:
    """
    Apply an MPO to a MPS.
    The operators are provided as tensor lists of equal number of sites.
    """
    assert len(mps) == len(mpo), 'must have same number of sites'
    return [ contract_mps_mpo_single_site(ss, oo) for ss,oo in zip(mps,mpo) ]


@jax.jit
def braket(bra : List[np.ndarray], ket : List[np.ndarray]) -> np.ndarray:
    """Compute <bra|ket> contraction of the given MPS."""
    assert len(bra) == len(ket)

    prev = None

    for bb, kk in zip(bra, ket):

        mm = jnp.einsum('axb,cxd->acbd', bb, jnp.conj(kk))

        if prev is not None:
            prev = jnp.einsum('xy,xyab->ab', prev, mm)
        else:
            prev = mm[0,0,:,:]

    return prev


def make_rand_mpo_complex(N : int, bond_dim = 1):
    """Return a random (complex128) MPO in format (l,r,up,down)."""

    assert N > 1

    shape_left   = (1,bond_dim,2,2)
    shape_right  = (bond_dim,1,2,2)
    shape_middle = (bond_dim,bond_dim,2,2)

    arrays = [ np.random.uniform(low=-1.0, size=shape_left  ) + 1.j * np.random.uniform(low=-1.0, size=shape_left  ) ] + \
             [ np.random.uniform(low=-1.0, size=shape_middle) + 1.j * np.random.uniform(low=-1.0, size=shape_middle) for _ in range(N-2)] + \
             [ np.random.uniform(low=-1.0, size=shape_right ) + 1.j * np.random.uniform(low=-1.0, size=shape_right ) ]

    return arrays


def make_rand_mps_complex(N : int, bond_dim = 1):
    """Return a random (complex128) MPS in format (l,r,phys)."""

    assert N > 1

    shape_margin = (1,2,bond_dim)
    shape_middle = (bond_dim,2,bond_dim)

    arrays = [ np.random.uniform(low=-1.0, size=shape_margin) + 1.j * np.random.uniform(low=-1.0, size=shape_margin) ] + \
             [ np.random.uniform(low=-1.0, size=shape_middle) + 1.j * np.random.uniform(low=-1.0, size=shape_middle) for _ in range(N-2)] + \
             [ np.random.uniform(low=-1.0, size=shape_margin[::-1]) + 1.j * np.random.uniform(low=-1.0, size=shape_margin[::-1]) ]

    return arrays







# -----------------------------------------------
#    CANONIZE
# -----------------------------------------------

@jax.jit
def right_canonize_twosites(
         A : List[np.ndarray], B : List[np.ndarray]
    ) -> Union[List[np.ndarray],List[np.ndarray]]:
    """
    Execute a QR decomposition on a couple of adjacent sites and merge R to left site.
    """

    # input tensor is reshaped (l,phys,r) -> (phys,r,l)
    this = B.transpose( (1,2,0) )
    original_shape = this.shape
    # merge r dimension to first index, returning a matrix of size (phys*r, l)
    this = this.reshape( (this.shape[0]*this.shape[1],this.shape[2]) )

    isometry, rest = qr_stabilized(this) 
    #         ^^ rest has shape  (r*, r)

    # isometry is reshaped to (phys,r,l*), then to (l*,phys,r) via transpose (2,0,1)
    B = jnp.transpose( isometry.reshape(original_shape[0],original_shape[1], -1 ), (2,0,1) )
    A = jnp.einsum('abc,dc->abd', A, rest)
    
    # note: this normalization could be optional, but in order
    #        to jit this function I let it be executed by default
    A /= jnp.linalg.norm(A)
    return A, B

# do not jit this
def right_canonize(mps : List[np.ndarray], site : int) ->  List[np.ndarray]:
    """Compute right canonization of MPS up to given site."""
    assert site < len(mps)
    assert site >= 1

    # create a copy to be modified FIXME
    mps = [ el for el in mps ]

    # loop over each site, from last to second
    for n in reversed(range(site, len(mps))):
        mps[n-1], mps[n] = right_canonize_twosites(mps[n-1], mps[n])
    return mps







# -----------------------------------------------
#     NORMALIZE via SVD
# -----------------------------------------------

#  FIXME
def normalize_onesite(A, A_next):
    A_mat = np.transpose(A, (0, 2, 1)).reshape((-1, A.shape[1]))
    U, S, V = np.linalg.svd(A_mat, full_matrices=False)
    A = np.transpose(U.reshape((A.shape[0], A.shape[2], A.shape[1])), (0, 2, 1))
    A_next = np.tensordot(np.diag(S).dot(V), A_next, axes=(1, 0))
    return A, A_next

#  FIXME
def normalize_mps_via_svd(mps):
    N = len(mps)
    for l in range(N):
        if l == N - 1:
            mps[l] = mps[l] / np.linalg.norm(mps[l].ravel())
        else:
            mps[l], mps[l + 1] = normalize_onesite(mps[l], mps[l + 1])




# -----------------------------------------------
#     SVD COMPRESSION
# -----------------------------------------------

# do not jit this
def compress_normalize_onesite(A : np.ndarray, A_next : np.ndarray, max_bd : int) -> Union[np.ndarray, np.ndarray]:
    #   A = (l,r,phys)   ->   (l,phys,r)  -> (l*phys, r)
    # 
    #A_mat = np.transpose(A, (0, 2, 1)).reshape((-1, A.shape[1])) # merge

    #   A = (l,phys,r)   ->  (l*phys, r)
    this = A.reshape((-1, A.shape[2]))
    #print('i', A.shape, this.shape)

    U, V = svd_truncated(this, cutoff = 1e-10, absorb = 1, max_bond = max_bd, cutoff_mode=4, renorm = 2)
    #print('ii', U.shape, V.shape)

    #  (l,phys,r*) -> (l,r*,phys)
    #A = np.transpose(U.reshape((A.shape[0], A.shape[2], -1)), (0, 2, 1)) # -1 was a A.shape[1]
    #A = np.transpose( U.reshape((A.shape[0], A.shape[1], -1)) , (0, 2, 1))
    A = U.reshape((A.shape[0], A.shape[1], -1))
    #print('iii', A.shape)

    #A_next = np.tensordot(V, A_next, axes=(1, 0))
    A_next = jnp.tensordot(V, A_next, axes=(1,0) )
    #print('iv', A_next.shape)
    #raise Exception('stop')

    return A, A_next

# do not jit this
def compress_svd_normalized(mps : List[np.ndarray], max_bd : int) -> List[np.ndarray]:
    N = len(mps)
    for l in range(N):
        if l == N - 1:
            mps[l] = mps[l] / jnp.linalg.norm(mps[l].ravel())
        else:
            mps[l], mps[l + 1] = compress_normalize_onesite(mps[l], mps[l + 1], max_bd=max_bd)
    return mps










# -----------------------------------------------
#     ETC, for everything I have to reorganize
# -----------------------------------------------

@jax.jit
def rdmul(x : np.ndarray, d : np.ndarray):
    """Right-multiplication a matrix by a vector representing a diagonal."""
    return x * jnp.reshape(d, (1, -1))

@jax.jit
def rddiv(x : np.ndarray, d : np.ndarray):
    """Right-multiplication of a matrix by a vector representing an inverse
    diagonal.
    """
    return x / jnp.reshape(d, (1, -1))

@jax.jit
def ldmul(d : np.ndarray, x : np.ndarray):
    """Left-multiplication a matrix by a vector representing a diagonal."""
    return x * jnp.reshape(d, (-1, 1))

@jax.jit
def qr_stabilized(x : np.ndarray) -> Union[np.ndarray,np.ndarray]:
    """QR-decomposition, with stabilized R factor."""

    Q, R = jnp.linalg.qr(x)
    # stabilize the diagonal of R
    rd = jnp.diag(R)
    s = jnp.sign(rd)
    Q = rdmul(Q, s)
    R = ldmul(s, R)
    return Q, R

# do not jit this
def _trim_and_renorm_svd_result(
    U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm
):
    """Give full SVD decomposion result ``U``, ``s``, ``VH``, optionally trim,
    renormalize, and absorb the singular values. See ``svd_truncated`` for
    details.
    """
    if (cutoff > 0.0) or (renorm > 0):
        if cutoff_mode == 1:  # 'abs'
            n_chi = jnp.count_nonzero( s > cutoff)

        elif cutoff_mode == 2:  # 'rel'
            n_chi = jnp.count_nonzero( s > cutoff * s[0])

        elif cutoff_mode in (3, 4, 5, 6):
            if cutoff_mode in (3, 4):
                pow = 2
            else:
                pow = 1

            sp = s**pow
            csp = jnp.cumsum(sp, 0)
            tot = csp[-1]

            if cutoff_mode in (4, 6):
                n_chi = jnp.count_nonzero(csp < (1 - cutoff) * tot) + 1
            else:
                n_chi = jnp.count_nonzero((tot - csp) > cutoff) + 1

        n_chi = max(n_chi, 1)
        if max_bond > 0:
            n_chi = min(n_chi, max_bond)

    elif max_bond > 0:
        # only maximum bond specified
        n_chi = max_bond
    else:
        # neither maximum bond dimension nor cutoff specified
        n_chi = s.shape[0]

    if n_chi < s.shape[0]:
        s = s[:n_chi]
        U = U[:, :n_chi]
        VH = VH[:n_chi, :]

        if renorm > 0:
                norm = (tot / csp[n_chi - 1]) ** (1 / pow)
                s *= norm

    if absorb is None:
        return U, s, VH
    if absorb == -1:
        U = rdmul(U, s)
    elif absorb == 1:
        VH = ldmul(s, VH)
    else:
        s = jnp.sqrt(s)
        U = rdmul(U, s)
        VH = ldmul(s, VH)

    #return U, None, VH
    return U, VH

# do not jit this 
# TODO check custom function
def svd_truncated( x,
    cutoff=-1.0, cutoff_mode=3,
    max_bond=-1,
    absorb=0, renorm=0,
):
    """Truncated svd or raw array ``x``.

    Parameters
    ----------
    cutoff : float
        Singular value cutoff threshold, if ``cutoff <= 0.0``, then only
        ``max_bond`` is used.
    cutoff_mode : {1, 2, 3, 4, 5, 6}
        How to perform the trim:

            - 1: ['abs'], trim values below ``cutoff``
            - 2: ['rel'], trim values below ``s[0] * cutoff``
            - 3: ['sum2'], trim s.t. ``sum(s_trim**2) < cutoff``.
            - 4: ['rsum2'], trim s.t. ``sum(s_trim**2) < sum(s**2) * cutoff``.
            - 5: ['sum1'], trim s.t. ``sum(s_trim**1) < cutoff``.
            - 6: ['rsum1'], trim s.t. ``sum(s_trim**1) < sum(s**1) * cutoff``.

    max_bond : int
        An explicit maximum bond dimension, use -1 for none.
    absorb : {-1, 0, 1, None}
        How to absorb the singular values. -1: left, 0: both, 1: right and
        None: don't absorb (return).
    renorm : {0, 1}
        Whether to renormalize the singular values (depends on `cutoff_mode`).
    """
    U, s, VH = np.linalg.svd(x,full_matrices=False)
    return _trim_and_renorm_svd_result(
            U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm
    )

