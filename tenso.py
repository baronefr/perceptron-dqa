import numpy as np
import opt_einsum as oe
import tensornetwork as tn
import scipy




def contract_mps_mpo_margin(state_site, operator_site):
    #      @ -- a
    #   x  |
    #      O -- b
    #      |                MPO = [(l),r,down,up]
    #      y
    return oe.contract('ax,byx->aby', state_site, operator_site).reshape( (-1, state_site.shape[1]) )

def contract_mps_mpo(state_site, operator_site):
    #   a -- @ -- b
    #        | c
    #   i -- O -- j
    #        |               
    #        l
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
        print( isometry.shape, rest.shape)

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






def mpo_apply_compress_density_matrix(mps, mpo, max_bd):
    """INCOMPLETE! Apply MPO to MPS with density matrix algo,
    as described in https://tensornetwork.org/mps/algorithms/denmat_mpo_mps/
    """

    assert len(mps) == len(mpo)

    def crop_svd(u,s,vh):
        return u[:,:max_bd], s[:max_bd], vh[:max_bd,:], np.sum(s[:max_bd])/np.sum(s)
    
    def crop_svd_4D(u,s,vh):
        return u[..., :max_bd], s[..., None, :], vh # TODO, what about vh?

    first_site = True
    LL = []
    UPPER = []

    for ss, oo in zip( mps[:-1], mpo[:-1] ):
        if first_site:
            #         @ -- a
            #      x  |
            #         O -- b     =  upper
            #         |
            #         y
            upper = oe.contract('ax,byx->aby', ss, oo )
            tmp = oe.contract('aby,cdy->abdc', upper, np.conj(upper) )  # permute cd to have
            first_site = False
            #      @ --  a
            #      @
            #      @ --  b
            #      @
            #      @ --  d    \
            #      @          | ->  permuted cd here!
            #      @ --  c    /
        else:
            #   a -- @ -- b
            #        | c
            #   i -- O -- j
            #        |               
            #        l
            upper = oe.contract('abc,ijlc->aibjl', ss, oo)
            block = oe.contract('abcde,fghie->abgfcdih', upper, np.conj(upper) )
            tmp = oe.contract('abcd,abcdlmno->lmno', tmp, block )

        UPPER.append( upper.copy() )
        LL.append( tmp.copy() )

    # contract with last block
    upper = oe.contract('ax,byx->aby', mps[-1], mpo[-1] )
    rho = oe.contract('abcd,abx,dcy->xy', tmp, upper, np.conj(upper) )

    #UPPER = UPPER.reverse()

    u,s,vh = np.linalg.svd(rho)
    u,s,vh, err = crop_svd(u,s,vh)
    print(rho)

    upper_crop = oe.contract('abx,xc->abc', upper, u )

    #print(UPPER[-1].shape)
    #print(LL[-2].shape)

    rho56 = oe.contract('abcd,abjkE,dcmlF,jkG,mlH->EFGH', 
                LL[-2], UPPER[-1], np.conj(UPPER[-1]), 
                upper_crop, np.conj(upper_crop)
    )
    #         @ -- a
    #         @
    #         @ -- b 
    #         |
    #         x


    # TODO what is an SVD of a 4D tensor??

    return rho56


def left_canonicalize_compress(mps, site : int, max_bd : int, normalize : bool = True):

    assert site < len(mps)

    for n in range(0, site):

        this = mps[n]

        if this.ndim < 3:
            this = np.expand_dims(this, 0) # must be left tensor, bring to format (l,phys,r)
            original_shape = this.shape
            mode_swap = False
        else:
            this = np.swapaxes(this, 1, 2) # reshapes tensor (l,r,phys) -> (l,phys,r)
            original_shape = this.shape
            mode_swap = True

        this = this.reshape( (this.shape[0]*this.shape[1], this.shape[2]) ) # merge left and phys dim index

        isometry, rest = np.linalg.qr( this )

        print('site QR:', n,  isometry.shape, rest.shape)
        print('   to origin ', original_shape)

        if mode_swap:
            # revert the axis swap
            print(' before:', isometry )
            mps[n] = np.swapaxes( isometry.reshape( original_shape ), 1, 2)
            print(' after:', isometry.reshape( original_shape ) )
        else:
            mps[n] = isometry
        
        mps[n+1] = oe.contract('ab,bcd->acd', rest, mps[n+1])

        if normalize:
            Z = np.linalg.norm( mps[n + 1] )
            mps[n + 1] /= Z

    return mps