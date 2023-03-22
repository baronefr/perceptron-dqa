import numpy as np
from tqdm import tqdm

import qiskit.quantum_info as qi
from qiskit.extensions.simulator import snapshot
from qiskit import Aer, transpile, QuantumCircuit


####################################
#### GET PERCEPTRON HAMILTONIAN ####
####################################
sigma_z  = np.diag([1., -1.])
identity = np.diag([1., 1.])

def kronecker_prod(operators):

    result = operators[0]
    for op in operators[1:]:

        result = np.kron(result, op)

    return result

def ReLU(x):
    return x * (x > 0)

def H_perc_nobatch(data, labels):
    n_data, n = data.shape

    h_perc = np.zeros((2**n, 2**n), dtype='float32')
    
    for i in tqdm(range(n_data), desc='Constructing H_perc'):

        op = np.zeros((2**n, 2**n), dtype='float32')
        for j in range(n):

            op += kronecker_prod([identity]*j+[data[i, j] * sigma_z]+[identity]*(n-j-1))

        h_perc += ReLU(-labels[i]*op)
        del op

    return (h_perc / np.sqrt(n)).astype('complex')


####################################
######### LOSS FUNCTIONS ###########
####################################
def get_statevec(qc, num_qubits):

    qc.snapshot("weight_statevector", qubits=range(num_qubits))

def loss(statevec, h_perc):

    return np.vdot(statevec, np.dot(h_perc, statevec))

def get_losses_from_results(results, data, labels, num_qubits, representation='same'):

    statevectors = np.stack(results.data()['snapshots']['statevector']['weight_statevector'])[:,:2**num_qubits]

    if representation != 'same':
        # invert data components if the circuit was constructed in big endian mode
        # NOT SURE IF THIS WORKS
        h_perc = H_perc_nobatch(data[:,::-1], labels)
    else:
        h_perc = H_perc_nobatch(data, labels)

    return np.apply_along_axis(loss, axis=1, arr=statevectors, h_perc=h_perc)



if __name__ == '__main__':

    # example code

    def your_evolution(qc, p, num_qubits, num_ancillae):
    
        # implement yout evolution here
        if p == 1:
            qc.h(range(num_qubits))
            return
        if p == 2:
            qc.h(range(num_qubits))
            qc.x(range(num_qubits, num_qubits+num_ancillae))
            return
        if p == 3:
            qc.x(3)
            return
        if p == 4:
            qc.x(2)

    N_data = 3
    N_feat = 4

    data = np.array([[1., 1., 1., 1.], 
                 [1., -1., 1., -1.], 
                 [1., -1., 1., 1.]])

    labels = np.ones((N_data, ))

    P = 4

    num_qubits   = 4
    num_ancillae = 2

    qc = QuantumCircuit(num_qubits + num_ancillae)

    # apply evoltion 
    for p in range(P):

        your_evolution(qc, p+1, num_qubits, num_ancillae)
        get_statevec(qc, num_qubits)


    qc.draw()

    simulator = Aer.get_backend('aer_simulator_statevector')
    circ = transpile(qc, simulator)

    # Run and get counts
    result = simulator.run(circ).result()

    print(get_losses_from_results(result, data, labels, num_qubits, representation='same'))