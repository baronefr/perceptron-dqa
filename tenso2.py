import numpy as np
import opt_einsum as oe
import scipy
from autoray import do
import tensornetwork as tn


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

    for n in reversed(range(site + 1, len(mps))):

        this = mps[n]

        if this.ndim < 3:
            this = np.expand_dims(this, 2) # must be right tensor, bring to format (l,phys,r)
            this = np.swapaxes(this, 0, 2) # swap to format (r,phys,l)
            this = this.reshape( (this.shape[0]*this.shape[1], this.shape[2]) ) 
            mode_swap = False
        else:
            this = np.swapaxes(this, 0, 2) # reshapes tensor (l,r,phys) -> (phys,r,l)
            original_shape = this.shape
            this = this.reshape( (this.shape[0]*this.shape[1],this.shape[2]) )
            mode_swap = True

        isometry, rest = qr_stabilized(this)#np.linalg.qr( this , mode='complete')
        #isometry = isometry[:this.shape[0], :this.shape[1]]
        #rest = rest[:this.shape[1], :]
        #print('info', isometry.shape, rest.shape)

        if mode_swap:
            #print('!!', n, original_shape, isometry.shape )
            els = isometry.shape[0]*isometry.shape[1]
            mps[n] = np.swapaxes( isometry.reshape(original_shape[0],original_shape[1], -1 ), 0, 2)

            #print('prev update', mps[n-1].shape, rest.shape)
            mps[n-1] = oe.contract('abc,db->adc', mps[n-1], rest)
        else:
            mps[n] = isometry
            #print('!!!', mps[n-1].shape, rest.shape )
            mps[n-1] = oe.contract('abc,db->acd', mps[n-1], rest)
        
        
        if normalize:
            Z = np.linalg.norm(mps[n - 1])
            mps[n - 1] /= Z
    
    return mps



# -----------------------------------------------
#     SVD COMPRESSION (first, untested version)
# -----------------------------------------------

def compress_svd_naive(mps, max_bd = 3):

    N = len(mps)
    mps = [ p.copy() for p in mps ]
    #print('sites:', N)
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
#     SVD COMPRESSION (better version)
# -----------------------------------------------

def compress_normalize_onesite(A, A_next, max_bd : int):
    A_mat = np.transpose(A, (0, 2, 1)).reshape((-1, A.shape[1]))
    U, _, V = svd_truncated(A_mat, cutoff = 1e-10, absorb = 1, 
                                 max_bond = max_bd, cutoff_mode=4, renorm = 2)
    
    A = np.transpose(U.reshape((A.shape[0], A.shape[2], -1)), (0, 2, 1)) # -1 was a A.shape[1]
    A_next = np.tensordot(V, A_next, axes=(1, 0))
    return A, A_next

def compress_svd_normalized(mps, max_bd : int):
    N = len(mps)
    for l in range(N):
        if l == N - 1:
            mps[l] = mps[l] / np.linalg.norm(mps[l].ravel())
        else:
            mps[l], mps[l + 1] = compress_normalize_onesite(mps[l], mps[l + 1], max_bd=max_bd)

    return mps









# -----------------------------------------------
#     VARIATIONAL COMPRESSION (slow as hell)
# -----------------------------------------------
def variational_compression_sweep(mps, psi_opti, cache_right = None):
    """A single sweep of the compression algo."""
    
    sweep_normalize_LR = True

    N = len(mps)
    
    # caching first sweep from right
    if cache_right is None:
        R_cache = []
        prev = None
        first_site = True

        for ii in reversed(range(2,N)):
            if first_site:
                #     mps*     (b)   == O
                #                       | (x)
                #     psi      (a)   -- r
                prev = oe.contract('ax,bx->ab', psi_opti[ii], np.conj(mps[ii]) )
                first_site = False
            else:
                #     mps*     (c)   == O ==   (d)           1  == @ ==  3
                #                       | (x)            ->        @
                #     psi      (a)   -- r --   (b)           0  -- @ --  2
                tmp = oe.contract('abx,cdx->acbd', psi_opti[ii], np.conj(mps[ii]) )
                #         c  == @ ==  y           (y)   == O
                #               @         with             | (x)
                #         a  -- @ --  x           (x)   -- r
                prev = oe.contract('xy,acxy->ac', prev, tmp)
            
            #print('caching R site {} of size {}'.format(ii, prev.shape) )
            if sweep_normalize_LR: prev = prev/np.linalg.norm(prev)
            #print('INFO norm', np.linalg.norm(prev) )
            R_cache.append( prev.copy() )

    else:
        R_cache = cache_right

    #print('Rc len', len(R_cache))
    #     mps*        O == (b)
    #            (x)  | 
    #     psi         r -- (a)
    L = oe.contract('ax,bx->ab', psi_opti[0], np.conj(mps[0]) )
    if sweep_normalize_LR: L = L/np.linalg.norm(L)
    #print('init L site of size {}'.format(L.shape) )
    L_cache = [ L ]


    # getting ready for first swipe
    jj = 0
    for ii in range(1,N-1):
        jj += 1
        R = R_cache[-jj]

        #print( np.conj(mps[ii]).shape, R.shape )
        #             (x)
        #   a  == O ==  == @
        #       b |        @
        #             c -- @   
        tmp = oe.contract('axb,cx->abc', np.conj(mps[ii]), R )
        #                             (x)
        #       # == y          y  == O ==  == @
        #       #                   b |        @
        #       # -- x                    c -- @   
        psi_opti[ii] = oe.contract('xy,ybc->xcb', L, tmp )

        #print('written site {} of shape {}'.format(ii,psi_opti[ii].shape))

        # prepare next step -------- 

        #        c == O == d
        #             |  x
        #        a -- @ -- b
        tmp = oe.contract('abx,cdx->acbd', psi_opti[ii], np.conj(mps[ii]) )
        L = oe.contract('xy,xyab->ab', L, tmp)
        if sweep_normalize_LR: L = L/np.linalg.norm(L)
        L_cache.append( L.copy() )


    #print('Lcache size =', len(L_cache) )

    # optimizing the last site ------------
    #       @ == y       y  == O 
    #       @                b |
    #       @ -- x             ?
    #print('MIST', L.shape, np.conj(mps[-1]).shape)
    psi_opti[-1] = oe.contract('xy,yb->xb', L, np.conj(mps[-1]))
    #print('written final site ({}) of shape {}'.format(N-1,psi_opti[N-1].shape))

    # prepare the reverse sweep -----------
    #              b  == O 
    #                  y |
    #              x  -- @
    #print('WARN', psi_opti[-1].shape, np.conj(mps[-1]).shape)
    R = oe.contract('xy,by->xb', psi_opti[-1], np.conj(mps[-1]))
    if sweep_normalize_LR: R = R/np.linalg.norm(R)
    R_cache = [ R ]
    jj = 1

    #return psi_opti, None
    #print('ccc', psi_opti)

    for ii in reversed( range(1,N-1) ):
        jj += 1
        L = L_cache[-jj]

        #             (x)
        #   a  == O ==  == @
        #       b |        @
        #             c -- @
        tmp = oe.contract('axb,cx->abc', np.conj(mps[ii]), R )
        #                             (x)
        #       # == y          y  == O ==  == @
        #       #                   b |        @
        #       # -- x                    c -- @   
        psi_opti[ii] = oe.contract('xy,ybc->xcb', L, tmp )

        #print('written site {} of shape {}'.format(ii,psi_opti[ii].shape))

        # prepare next step -------- 

        #        c == O == d
        #             |  x
        #        a -- @ -- b
        tmp = oe.contract('abx,cdx->acbd', psi_opti[ii], np.conj(mps[ii]) )
        R = oe.contract('xy,acxy->ac', R, tmp)
        if sweep_normalize_LR: R = R/np.linalg.norm(R)
        R_cache.append( R.copy() )

        #return psi_opti, None

    # optimizing the first site ------------
    #       @ == y       y  == O     O ==  y    y == @
    #       @                b |     | b             @
    #       @ -- x             ?     ?          x -- @
    psi_opti[0] = oe.contract('xy,yb->xb', R, np.conj(mps[0]))
    #print('written first site ({}) of shape {}'.format(0,psi_opti[0].shape))

    return psi_opti, R_cache[:-1]


def variational_compression(mps, target_bond_dim, N_sweep = 1, guess = None):
    """Perform N sweeps of variational compression starting from a guessed, lower dimentional, MPS."""

    if guess is None:
        guess = google_random_mps(len(mps), bond_dim=target_bond_dim, canon = 1)

    cache = None

    pss = guess.copy()
    for i in range(N_sweep):
        #print('sweep', i)
        pss, cache = variational_compression_sweep(mps, guess, cache)
        
    return pss







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
