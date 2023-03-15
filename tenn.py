import numpy as np
import opt_einsum as oe
from autoray import do
import tensornetwork as tn



def convert(mps):
    """reshape MPS from (l,r,phys) to (l,phys,r)"""
    return [ np.transpose(tensor, (0,2,1)) for tensor in mps ]

def convert_from_quimb(mps):
    """reshape MPS from Quimb(l,r,phys) to (l,phys,r)"""
    return [ np.transpose(tensor.data, (0,2,1)) for tensor in mps.tensors ]

def remove_margin_extra_dim_in_lrp(mps) -> None:
    mps[0] = mps[0][0]
    mps[-1] = mps[-1][:,0,:]

def add_margin_extra_dim_in_lrp(mps) -> None:
    mps[0] = np.expand_dims(mps[0], 0)
    mps[-1] = np.expand_dims(mps[-1], 1)

# -----------------------------------------------
#    MPS MPO operations
# -----------------------------------------------


def contract_mps_mpo_single_site(state_site, operator_site):
    """Given the sites of MPS and MPO, return the contraction."""
    #
    #                l
    #                |
    #           j -- O -- k
    #                | b (m)
    #           a -- @ -- c
    #
    tensor = oe.contract('abc,jklb->ajckl', state_site, operator_site)
    tensor = tensor.reshape( 
        (state_site.shape[0]*operator_site.shape[0], state_site.shape[2]*operator_site.shape[1], operator_site.shape[3])
    )

    return np.transpose(tensor, (0,2,1) ) # using (l,p,r) ordering

def apply_mpsmpo(mps, mpo) -> list:
    """Apply an MPO to a MPS."""
    assert len(mps) == len(mpo), 'must have same number of sites'
    return [ contract_mps_mpo_single_site(mps[i], mpo[i]) for i in range( len(mps) ) ]


def make_rand_mpo_complex(N : int, bond_dim = 1):
    """Return a random complex MPO."""

    assert N > 1

    shape_left   = (1,bond_dim,2,2)
    shape_right  = (bond_dim,1,2,2)
    shape_middle = (bond_dim,bond_dim,2,2)

    arrays = [ np.random.uniform(low=-1.0, size=shape_left  ) + 1.j * np.random.uniform(low=-1.0, size=shape_left  ) ] + \
             [ np.random.uniform(low=-1.0, size=shape_middle) + 1.j * np.random.uniform(low=-1.0, size=shape_middle) for _ in range(N-2)] + \
             [ np.random.uniform(low=-1.0, size=shape_right ) + 1.j * np.random.uniform(low=-1.0, size=shape_right ) ]

    return arrays


def make_rand_mps_complex(N : int, bond_dim = 1):
    """Return a random complex MPS."""

    assert N > 1

    shape_margin = (1,2,bond_dim)
    shape_middle = (bond_dim,2,bond_dim)

    arrays = [ np.random.uniform(low=-1.0, size=shape_margin) + 1.j * np.random.uniform(low=-1.0, size=shape_margin) ] + \
             [ np.random.uniform(low=-1.0, size=shape_middle) + 1.j * np.random.uniform(low=-1.0, size=shape_middle) for _ in range(N-2)] + \
             [ np.random.uniform(low=-1.0, size=shape_margin[::-1]) + 1.j * np.random.uniform(low=-1.0, size=shape_margin[::-1]) ]

    return arrays






# -----------------------------------------------
#    LIBRARY INTERFACES
# -----------------------------------------------

def convert_mps_from_google(mpsg):
    N = len(mpsg.tensors)

    array = [ mpsg.tensors[0][0,:,:].copy() ] + \
            [ np.swapaxes(mpsg.tensors[i+1], 1,2).copy() for i in range(N-2)] + \
            [ mpsg.tensors[-1][:,:,0].copy() ]
    return array

def google_random_mps(N, bond_dim = 3, canon = None):
    """I use the Google library to make a random mps."""
    phys_dim = 2

    mpstate = tn.FiniteMPS.random(
        d = [phys_dim for _ in range(N)],
        D = [bond_dim for _ in range(N-1)],
        dtype = np.float32,
        canonicalize=True
    )

    if canon is not None:
        mpstate.center_position = canon
        mpstate.canonicalize(normalize = True)
    
    return convert_mps_from_google(mpstate)



# -----------------------------------------------
#    CANONICALIZATION
# -----------------------------------------------

def right_canonicalize(mps, site : int, normalize : bool = True):
    """Compute right canonicalization of MPS up to given site."""
    assert site < len(mps)
    assert site >= 1

    # create a copy to be modified
    mps = [ el.copy() for el in mps ]

    # loop over each site, from last to second
    for n in reversed(range(site, len(mps))):

        # input tensor is reshaped (l,phys,r) -> (phys,r,l)
        this = mps[n].transpose( (1,2,0) )
        original_shape = this.shape
        # merge r dimension to first index, so this is a matrix now of size (phys*r, l)
        this = this.reshape( (this.shape[0]*this.shape[1],this.shape[2]) )

        isometry, rest = qr_stabilized(this) 
        # rest has shape  (r*, r)
        #print(rest.shape)

        # isometry is reshaped to (phys,r,l*), then to (l*,phys,r) via (2,0,1)   # dev: broken (2,1,0)
        mps[n] = np.transpose( isometry.reshape(original_shape[0],original_shape[1], -1 ), (2,0,1) )
        #if n  >= 1: 
        mps[n-1] = oe.contract('abc,dc->abd', mps[n-1], rest)
        
        if normalize:
                Z = np.linalg.norm(mps[n - 1])
                mps[n - 1] /= Z
    
    return mps



# -----------------------------------------------
#     SVD COMPRESSION (first, untested version)
# -----------------------------------------------

def compress_svd_naive(mps, max_bd = 3):

    N = len(mps)
    mps = [ el.copy() for el in mps ]

    first_site = True

    def truncate_svd(site):
        if site.ndim < 3:
            if first_site:
                #print('first!')
                this = np.expand_dims(site, 0) # must be left tensor, bring to format (l,r,phys)
                this = np.swapaxes(this, 1, 2) # -> (l,phys,r)
            else:
                this = np.expand_dims(site, 2)
        else:
            this = np.swapaxes(site, 1, 2) # reshapes tensor (l,r,phys) -> (l,phys,r)

        original_shape = this.shape
        #print('  parsed:', original_shape, '->', (this.shape[0]*this.shape[1], this.shape[2]) )

        # merge left and physical index
        this = this.reshape( (this.shape[0]*this.shape[1], this.shape[2]) )

        # compute SVD and crop to max bond dim
        #u, s, vh = scipy.linalg.svd(this, full_matrices=True)
        #print('SVD shapes:', u.shape, s.shape, vh.shape )
        #u, s = u[:,:max_bd], s[:max_bd]#, vh[:max_bd,:]
        #s = s/np.sum(s)
        #vh = vh[:min(s.size,max_bd),:]
        #print(' SVD*  -> ', u.shape, s.shape, vh.shape )

        u, _, vh = svd_truncated(this, cutoff = 1e-10, absorb = 1, 
                                 max_bond = max_bd, cutoff_mode=4, renorm = 2)
        #print( u.shape, vh.shape )
        # reshape u
        u_shape = (original_shape[0], original_shape[1], u.shape[1])
        #print('new =', u_shape )
        new_u = np.swapaxes( u.reshape( u_shape ), 1, 2)

        return new_u, vh

    compressed_mps = []


    for ii in range(N-1):
        
        Ustar, fwd = truncate_svd(mps[ii])

        if ii < N-2:
            mps[ii+1] = oe.contract('mX,Xrp->mrp', fwd, mps[ii+1])
        else:
            #print('\n\nstep', ii, fwd.shape, mps[ii+1].shape)
            mps[ii+1] = oe.contract('mX,Xp->mp', fwd, mps[ii+1])


        if first_site:
            Ustar = Ustar[0]
            #Ustar = Ustar/np.linalg.norm(Ustar)
            first_site = False

        #Ustar = Ustar/np.linalg.norm(Ustar)
        compressed_mps.append(Ustar)
        
    #Ustar = mps[-1]/np.linalg.norm(mps[-1])

    compressed_mps.append(mps[-1])

    return compressed_mps







# -----------------------------------------------
#     NORMALIZE via SVD
# -----------------------------------------------

def normalize_onesite(A, A_next):
    A_mat = np.transpose(A, (0, 2, 1)).reshape((-1, A.shape[1]))
    U, S, V = np.linalg.svd(A_mat, full_matrices=False)
    A = np.transpose(U.reshape((A.shape[0], A.shape[2], A.shape[1])), (0, 2, 1))
    A_next = np.tensordot(np.diag(S).dot(V), A_next, axes=(1, 0))
    return A, A_next

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

def compress_normalize_onesite(A, A_next, max_bd : int):
    #   A = (l,r,phys)   ->   (l,phys,r)  -> (l*phys, r)
    # 
    #A_mat = np.transpose(A, (0, 2, 1)).reshape((-1, A.shape[1])) # merge

    #   A = (l,phys,r)   ->  (l*phys, r)
    this = A.reshape((-1, A.shape[2]))
    #print('i', A.shape, this.shape)

    U, _, V = svd_truncated(this, cutoff = 1e-10, absorb = 1, 
                                 max_bond = max_bd, cutoff_mode=4, renorm = 2)
    #print('ii', U.shape, V.shape)

    #  (l,phys,r*) -> (l,r*,phys)
    #A = np.transpose(U.reshape((A.shape[0], A.shape[2], -1)), (0, 2, 1)) # -1 was a A.shape[1]
    #A = np.transpose( U.reshape((A.shape[0], A.shape[1], -1)) , (0, 2, 1))

    A = U.reshape((A.shape[0], A.shape[1], -1))
    #print('iii', A.shape)
    #raise Exception()
    #A_next = np.tensordot(V, A_next, axes=(1, 0))
    #print(V.shape, A_next.shape)
    A_next = np.tensordot(V, A_next, axes=(1,0) )
    #print('iv', A_next.shape)
    #raise Exception('stop')

    return A, A_next

def compress_svd_normalized(mps, max_bd : int):
    N = len(mps)
    for l in range(N):
        #print('STEP', l)
        if l == N - 1:
            mps[l] = mps[l] / np.linalg.norm(mps[l].ravel())
        else:
            mps[l], mps[l + 1] = compress_normalize_onesite(mps[l], mps[l + 1], max_bd=max_bd)

    return mps











# -----------------------------------------------
#     ETC, for everything I have to reorganize
# -----------------------------------------------


def rdmul(x, d):
    """Right-multiplication a matrix by a vector representing a diagonal."""
    return x * np.reshape(d, (1, -1))


def rddiv(x, d):
    """Right-multiplication of a matrix by a vector representing an inverse
    diagonal.
    """
    return x / np.reshape(d, (1, -1))


def ldmul(d, x):
    """Left-multiplication a matrix by a vector representing a diagonal."""
    return x * np.reshape(d, (-1, 1))

def _trim_and_renorm_svd_result(
    U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm
):
    """Give full SVD decomposion result ``U``, ``s``, ``VH``, optionally trim,
    renormalize, and absorb the singular values. See ``svd_truncated`` for
    details.
    """
    if (cutoff > 0.0) or (renorm > 0):
        if cutoff_mode == 1:  # 'abs'
            n_chi = do("count_nonzero", s > cutoff)

        elif cutoff_mode == 2:  # 'rel'
            n_chi = do("count_nonzero", s > cutoff * s[0])

        elif cutoff_mode in (3, 4, 5, 6):
            if cutoff_mode in (3, 4):
                pow = 2
            else:
                pow = 1

            sp = s**pow
            csp = do("cumsum", sp, 0)
            tot = csp[-1]

            if cutoff_mode in (4, 6):
                n_chi = do("count_nonzero", csp < (1 - cutoff) * tot) + 1
            else:
                n_chi = do("count_nonzero", (tot - csp) > cutoff) + 1

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

    # XXX: tensorflow can't multiply mixed dtypes
    # omit this

    if absorb is None:
        return U, s, VH
    if absorb == -1:
        U = rdmul(U, s)
    elif absorb == 1:
        VH = ldmul(s, VH)
    else:
        s = do("sqrt", s)
        U = rdmul(U, s)
        VH = ldmul(s, VH)

    return U, None, VH

def svd_truncated(
    x,
    cutoff=-1.0,
    cutoff_mode=3,
    max_bond=-1,
    absorb=0,
    renorm=0,
    backend=None,
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
    U, s, VH = do("linalg.svd", x)
    return _trim_and_renorm_svd_result(
            U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm
    )

def qr_stabilized(x):
    """QR-decomposition, with stabilized R factor."""

    Q, R = do("linalg.qr", x)
    # stabilize the diagonal of R
    rd = do("diag", R)
    s = np.sign(rd)
    Q = rdmul(Q, s)
    R = ldmul(s, R)
    return Q, R
