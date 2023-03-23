from tqdm import tqdm
from importlib.util import find_spec
GPU = True


if GPU:
    if find_spec('cupy') is not None:
        import cupy as ncp
    else:
        print('Selected device is GPU but cupy is not installed, falling back to numpy')
        import numpy as ncp
else:
    import numpy as ncp


def kronecker_prod(operators):

    result = operators[0]
    for op in operators[1:]:

        result = ncp.kron(result, op)

    return result

def ReLU(x):
    return x * (x > 0)

def H_perc_diag(data, labels):
    ## ASSUMING H_PERC IS DIAGONAL, WHICH IS IN THE COMPUTATIONAL BASIS ##
    n_data, n = data.shape
    identity = ncp.ones((2,), 'float64')
    sigma_z  = ncp.array([1., -1.])

    h_perc = ncp.zeros((2**n,), dtype='float64')
    
    for i in tqdm(range(n_data), desc='Constructing H_perc'):

        op = ncp.zeros((2**n,), dtype='float64')
        for j in range(n):

            op += kronecker_prod([identity]*j+[data[i, j] * sigma_z]+[identity]*(n-j-1))

        h_perc += ReLU(-labels[i]*op)
        del op

    return h_perc / ncp.sqrt(n)

def H_x(n):
    identity = ncp.diag([1., 1.])
    sigma_x  = ncp.array([[0., 1.], [1., 0.]])

    op = ncp.zeros((2**n, 2**n), dtype='float64')
    for j in range(n):
        op += kronecker_prod([identity]*j+[sigma_x]+[identity]*(n-j-1))

    return -op

def H_QA(p, P, Hz, Hx):
    frac = p/P
    return (frac*Hz + (1-frac)*Hx).astype('complex128')


def init_state(n):
    return (ncp.ones((2**n,))/ncp.sqrt(2**n)).astype('complex128')


def ed_qa_step(state, Ht, dt):

    eigvals, eigvecs = ncp.linalg.eigh(Ht)
    # define time evolution operator
    U = ncp.exp(-1.j*eigvals*dt)

    # rewrite state in eigenvector basis, apply time evolution operator, project back to computational basis
    evolved = ncp.dot(eigvecs, U * ncp.dot(eigvecs.transpose().conjugate(), state))
    return evolved