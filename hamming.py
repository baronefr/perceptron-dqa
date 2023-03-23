import math
import numpy as np
from qiskit import QuantumCircuit

# useful links:
#  https://qiskit.org/documentation/tutorials/circuits/3_summary_of_quantum_operations.html


def qft_dagger(qc, n, shift = 0):
    """n-qubit QFTdagger on n qubits in circ, shifting by arg value"""
    for qubit in range(n//2):
        qc.swap(shift+qubit, shift+n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-math.pi/float(2**(j-m)), shift+m, shift+j)
        qc.h(shift+j)




def circuit_heaviside(data_qubits : int, # how many quibits are used for data encoding
                      data : np.array,   # the data array (of values +-1)
                      init_flips = None, # if qc is None, initialize the circuit with this flips
                      ancillary_qubits = None, # how many ancillaries to use, if None it is automatically computed
                      theta = None, # rotation to use in qubit (automatic is better)
                      qc = None):   # if None, init a new circuit
    
    assert len(data) == data_qubits

    # compute the number of ancillaries and optimal rotation value
    if ancillary_qubits is None:
        ancillary_qubits = int( np.ceil( np.log2(data_qubits) ) + 1 )

    if theta is None:
        theta = math.pi/(2**(ancillary_qubits-1))
    
    # create a new circuit if not prompted as arg
    if qc is None:
        qc = QuantumCircuit(data_qubits + ancillary_qubits)

        # init the qubits
        if init_flips is not None:
            assert len(init_flips) == data_qubits

            for ii, qb in enumerate(init_flips):
                if qb == +1:  qc.x(ii)
        
            #qc.barrier()

    # rotate data qubits if the data feature is one (will be reverted later)
    for ii, qb in enumerate(data):
        if qb == +1:  qc.x(ii)

    # prepare ancillary qubits with Hadamard gate
    for ii in range(ancillary_qubits):
        qc.initialize([1, 0], data_qubits + ii)
        qc.h(data_qubits + ii)

    # linking the phase shifts in Fourier basis
    for target_qubit in range(data_qubits):  #ancillary_qubits
        repetitions = 1
        for counting_qubit in range(ancillary_qubits):
    #        #for _ in range(repetitions):
    #        #    qc.cp(theta, data_qubits + counting_qubit, target_qubit)
            qc.cp(theta*repetitions, data_qubits + counting_qubit, target_qubit)
            repetitions *= 2

    # apply QFT inverse
    #qc.barrier()
    qft_dagger(qc, ancillary_qubits, shift = data_qubits)
    #qc.barrier()

    # revert rotation of data qubits
    for ii, qb in enumerate(data):
        if qb == +1:  qc.x(ii)

    # trick: cnot the last ancillary with previous one to evaluate if > N/2
    qc.cx(data_qubits+ancillary_qubits-2, data_qubits+ancillary_qubits-1)
    
    return qc







class Hamming:

    def __init__(self, n_qubit, n_ancillary, dataset):
        self.n_qubit = n_qubit
        self.n_ancillary = n_ancillary
        self.dataset = dataset
        assert dataset.shape[1] == n_qubit, 'dataset size does not match number of qubits in the circuit'

        self.data_size = dataset.shape[0] # size of dataset
        self.control_q = n_qubit + n_ancillary - 1

    def hamming_circuit_for_sample(self, mu : int, qc = None):
        return circuit_heaviside(data_qubits = self.n_qubit,
                      data = self.dataset[mu,:],
                      init_flips = None, # do not init the circuit spins
                      ancillary_qubits = self.n_ancillary,
                      qc = qc) # override if needed
    
    def control_unitaries(self, mu : int, gamma, qc):
        #for ii in range( self.n_qubit ):
        #    # syntax:  lambda, control_qubit, target_qubit
        #    qc.cp( -2 * gamma * self.dataset[mu, ii], self.control_q, ii)

        # da Paolo (adapted) ----------------------
        for jj in range( self.n_qubit ):
            qc.crz(-2*gamma*self.dataset[mu, jj]/np.sqrt(self.n_qubit),  self.control_q, jj)

        # da Paolo (original) ---------------------
        #for j in range(data_qubits):
        #    qc.crz(-2*gamma_t*((-1)**(data[j]+1))/np.sqrt(data_qubits), qc.num_qubits-1, j)

        return qc