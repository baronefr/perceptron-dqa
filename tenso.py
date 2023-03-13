import numpy as np
import opt_einsum as oe
import tensornetwork as tn





def contract_mps_mpo_margin(state_site, operator_site):
    return oe.contract('ax,byx->aby', state_site, operator_site).reshape( (-1, state_site.shape[1]) )

def contract_mps_mpo(state_site, operator_site):
    return oe.contract('abc,ijlc->aibjl', state_site, operator_site).reshape( 
        (state_site.shape[0]*operator_site.shape[0], state_site.shape[1]*operator_site.shape[1], operator_site.shape[3])
    )

def apply_mpsmpo(mps, mpo):
    assert len(mps) == len(mpo), 'must have same number of sites'
    assert len(mps) >= 2

    N = len(mps)
    out = []

    out.append( contract_mps_mpo_margin(mps[0], mpo[0]) )

    for i in range(N-2):
        out.append(
            contract_mps_mpo(mps[i+1], mpo[i+1])
        )
    
    out.append( contract_mps_mpo_margin(mps[-1], mpo[-1]) )

    return out


def make_rand_mpo(N : int, bond_dim = 1, dtype = np.complex128 ): #, bond_dim = None, dtype = np.complex128):
    """Return a random MPO."""

    shape_margin = (bond_dim,2,2)
    shape_middle = (bond_dim,bond_dim,2,2)

    arrays = [ np.random.uniform(low=-1.0, size=shape_margin) + 1.j * np.random.uniform(low=-1.0, size=shape_margin) ] + \
             [ np.random.uniform(low=-1.0, size=shape_middle) + 1.j * np.random.uniform(low=-1.0, size=shape_middle) for _ in range(N-2)] + \
             [ np.random.uniform(low=-1.0, size=shape_margin) + 1.j * np.random.uniform(low=-1.0, size=shape_margin) ]

    return arrays


# same structure as https://github.com/google/TensorNetwork/blob/e12580f1749493dbe05f474d2fecdec4eaba73c5/tensornetwork/matrixproductstates/base_mps.py#L139

def left_canonicalize(mps, site : int, normalize : bool = True):

    assert site < len(mps)

    for n in range(0, site):

        this = mps[n]

        if this.ndim < 3:
            this = np.expand_dims(this, 0) # must be left tensor, bring to format (l,phys,r)
            mode_swap = False
        else:
            this = np.swapaxes(this, 1, 2) # reshapes tensor (l,r,phys) -> (l,phys,r)
            original_shape = this.shape
            mode_swap = True

        this = this.reshape( (this.shape[0]*this.shape[1], this.shape[2]) ) # merge left and phys dim index

        isometry, rest = np.linalg.qr( this )

        if mode_swap:
            # revert the axis swap
            mps[n] = np.swapaxes( isometry.reshape( original_shape ), 1, 2)
        else:
            mps[n] = isometry
        
        mps[n+1] = oe.contract('ab,bcd->acd', rest, mps[n+1])

        if normalize:
            Z = np.linalg.norm( mps[n + 1] )
            mps[n + 1] /= Z

    return mps



def right_canonicalize(mps, site : int, normalize : bool = True):

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


        isometry, rest = np.linalg.qr( this , mode='complete')
        isometry = isometry[:this.shape[0], :this.shape[1]]
        rest = rest[:this.shape[1], :]

        if mode_swap:
            mps[n] = np.swapaxes( isometry.reshape( original_shape ), 0, 2)
            mps[n-1] = oe.contract('abc,bd->adc', mps[n-1], rest)
        else:
            mps[n] = isometry
            mps[n-1] = oe.contract('abc,cd->abd', mps[n-1], rest)
        
        
        if normalize:
            Z = np.linalg.norm(mps[n - 1])
            mps[n - 1] /= Z
    
    return mps


def canonicalize_around(mps, center : int, normalize : bool = True, copy : bool = True):
    if copy:
        arg = [ ss.copy() for ss in mps ]
        tmp = left_canonicalize(arg,center,normalize)   
    else:
        tmp = left_canonicalize(mps,center,normalize)

    return right_canonicalize(tmp,center,normalize)



def contract_braket_left(bra, ket):
    assert len(bra) == len(ket)

    first_site = True
    mm = None

    for bb, kk in zip(bra, ket):
        if first_site:
            assert bb.ndim == 2

            mm = oe.contract('ax,bx->ab', bb, np.conj(kk))
            first_site = False
            print('ok')
        else:
            tmp = oe.contract('abx,cdx->acbd', bb, np.conj(kk))
            mm = oe.contract('xy,xyab->ab', mm, tmp)
            print('okk')

    return mm


def contract_braket_right(bra, ket):
    assert len(bra) == len(ket)

    first_site = True
    mm = None

    for bb, kk in reversed(zip(bra, ket)):
        if first_site:
            assert bb.ndim == 2

            mm = oe.contract('ax,bx->ab', bb, np.conj(kk))
            first_site = False
        else:
            tmp = oe.contract('abx,cdx->acbd', bb, np.conj(kk))
            mm = oe.contract('xy,abxy->ab', mm, tmp)

    return mm



def contract_mps_braket(bra, ket):
    assert len(bra) == len(ket)

    prev = None

    for bb, kk in zip(bra, ket):

        if bb.ndim < 3:
            mm = oe.contract('ax,bx->ab', bb, np.conj(kk))
            if prev is not None:
                prev = oe.contract('xy,xy->', prev, mm)
            else:
                prev = mm
        else:
            mm = oe.contract('abx,cdx->acbd', bb, np.conj(kk))
            if prev is not None:
                prev = oe.contract('xy,xyab->ab', prev, mm)
            else:
                prev = mm

    return float(prev)



def convert_mps_from_google(mpsg):
    N = len(mpsg.tensors)

    array = [ mpsg.tensors[0][0,:,:].copy() ] + \
            [ np.swapaxes(mpsg.tensors[i+1], 1,2).copy() for i in range(N-2)] + \
            [ mpsg.tensors[-1][:,:,0].copy() ]
    return array

def google_random_mps(N, bond_dim = 3, canon = None):
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






def compression_sweep(mps, psi_opti, cache_right = None):
    
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




def compression_algo(mps, target_bond_dim, N_sweep = 1, guess = None):

    if guess is None:
        guess = google_random_mps(len(mps), bond_dim=target_bond_dim, canon = 1)

    cache = None

    pss = guess.copy()
    for i in range(N_sweep):
        #print('sweep', i)
        pss, cache = compression_sweep(mps, guess, cache)
        
    return pss