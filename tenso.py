import numpy as np
import opt_einsum as oe



# adapted from https://github.com/google/TensorNetwork/blob/e12580f1749493dbe05f474d2fecdec4eaba73c5/tensornetwork/matrixproductstates/base_mps.py#L139

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


def canonicalize(mps, site : int, normalize : bool = True, copy : bool = True):
    if copy:
        arg = [ ss.copy() for ss in mps ]
        tmp = left_canonicalize(arg,site,normalize)   
    else:
        tmp = left_canonicalize(mps,site,normalize)

    return right_canonicalize(tmp,site,normalize)

