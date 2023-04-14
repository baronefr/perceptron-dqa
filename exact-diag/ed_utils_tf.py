import tensorflow as tf
import tensorflow.linalg as tfl

from tqdm import tqdm



# set GPUs
physical_devices = tf.config.list_physical_devices('GPU')
print('GPU devices:', physical_devices)

if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

        
sigma_x = tfl.LinearOperatorFullMatrix([[0., 1.], [1., 0.]])
sigma_z = tfl.LinearOperatorFullMatrix([[1., 0.], [0., -1.]])
id      = tfl.LinearOperatorIdentity(2)

def H_perc_nobatch(data, labels):
    n_data, n = data.shape

    h_perc = tf.zeros((2**n, 2**n), dtype='float32')
    
    for i in tqdm(range(n_data), desc='Constructing H_perc'):

        op = tf.zeros((2**n, 2**n), dtype='float32')
        for j in range(n):

            data_op = tfl.LinearOperatorDiag(tf.repeat(data[i, :], [2], axis=0)[2*j:2*(j+1)])
            op     += tfl.LinearOperatorKronecker([id]*j+[tfl.LinearOperatorComposition([data_op, sigma_z])]+[id]*(n-j-1)).to_dense()
            del data_op

        h_perc += tf.nn.relu(-labels[i]*op)
        del op

    return h_perc / tf.sqrt(n+0.)

def H_x(n):
    sigma_xs = [tfl.LinearOperatorKronecker([id]*i+[sigma_x]+[id]*(n-i-1)).to_dense() for i in range(n)]
    return -tf.reduce_sum(tf.stack(sigma_xs), axis=0)

def H_z(n):
    sigma_zs = [tfl.LinearOperatorKronecker([id]*i+[sigma_z]+[id]*(n-i-1)).to_dense() for i in range(n)]
    return tf.reduce_sum(tf.stack(sigma_zs), axis=0)

def H_QA(p, P, Hz, Hx):
    frac = p/P
    return tf.cast(frac*Hz + (1-frac)*Hx, dtype='complex128')


def init_state(n):
    return tf.ones((2**n,), dtype='complex128')/tf.sqrt((2.**n+0.j))


def ed_qa_step(state, Ht, dt):

    eigvals, eigvecs = tfl.eigh(Ht)
    # define time evolution operator
    U = tf.exp(-1.j*eigvals*dt)

    # rewrite state in eigenvector basis, apply time evolution operator, project back to computational basis
    evolved = tfl.matvec(eigvecs, U * tfl.matvec(eigvecs, state, adjoint_a=True))
    return evolved