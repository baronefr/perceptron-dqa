

# ====================================================
#  Quantum Information and Computing exam project
#
#   UNIPD Project |  AY 2022/23  |  QIC
#   group : Barone, Coppi, Zinesi
# ----------------------------------------------------
#   > description                                    |
#
#   class setup of dQA execution
# ----------------------------------------------------
#   coder : Zinesi Paolo
#   dated : 27 March 2023
#     ver : 1.0.0
# ====================================================


from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import QFT, IntegerComparator
import numpy as np


class HammingEvolution:
    """
    Class to generate all the modules of Heaviside evolution circuit consistently.
    """
    def __init__(self, num_data_qubits : int) -> None:

        # infer number ancillas used to count, number of ancillas used to compare Hamming distance
        self._num_data_qubits = num_data_qubits
        self._num_count_ancillas = int(np.ceil(np.log2(self._num_data_qubits+1)))

        # in this situation the comparison is really simple
        self._simple_compare = (self._num_data_qubits + 1 == 2**self._num_count_ancillas)

        # circuit initializer
        self._data_qubits = QuantumRegister(self._num_data_qubits)
        self._count_ancillas = AncillaRegister(self._num_count_ancillas)
        self._qc = QuantumCircuit(self._data_qubits, self._count_ancillas)

        # intialize comparison ancillas if necessary
        if not self._simple_compare:
            self._num_comp_ancillas = self._num_count_ancillas
            self._comp_ancillas = AncillaRegister(self._num_count_ancillas)
            self._qc.add_register(self._comp_ancillas)

        # ancilla in which the Heaviside control will be stored
        if self._simple_compare:
            self._control_ancilla = self._count_ancillas[-1]
        else:
            self._control_ancilla = self._comp_ancillas[0]


    @property
    def num_data_qubits(self):
        return self._num_data_qubits
    
    @property
    def num_count_ancillas(self):
        return self._num_count_ancillas
    
    @property
    def simple_compare(self):
        return self._simple_compare
    
    @property
    def data_qubits(self):
        return self._data_qubits
    
    @property
    def count_ancillas(self):
        return self._count_ancillas
    
    @property
    def qc(self):
        return self._qc.copy()
    
    @property
    def num_comp_ancillas(self):
        if self._simple_compare:
            return 0
        else:
            return self._num_comp_ancillas

    @property
    def comp_ancillas(self):
        if self._simple_compare:
            return []
        else:
            return self._comp_ancillas
    
    @property
    def num_ancillas(self):
        return self.num_count_ancillas + self.num_comp_ancillas
    
    @property
    def ancillas(self):
        return list(self.count_ancillas) + list(self.comp_ancillas)
    
    @property
    def qubits(self):
        return list(self.data_qubits) + list(self.count_ancillas) + list(self.comp_ancillas)
    
    @property
    def control_ancilla(self):
        return self._control_ancilla
    

    

    def init_state_plus(self):
        """
        Generate a circuit where all the qubits are initialized at |+> = H|0> intead of simply |0>.
        """

        # return a new copy of the circuit, but with the same number of qubits for consistency
        circ = self.qc.copy()

        for iq in range(self.num_data_qubits):
            circ.h(self.data_qubits[iq])

        return circ


    def Hamming_count(self, train_data):
        """ 
        Generate circuit of `self.num_data_qubits` qubits that counts the Hamming distance from the training data.
        The count is stored in the `self.count_ancillas` qubits. 
            - train_data: vector of training data.

            Conventions:
            - (1,-1) <--> (|0>,|1>)
            - little endians: least significant bit is the last one of the string
        """

        assert len(train_data) == self.num_data_qubits, "Wrong dimension of training data"

        # return a new copy of the circuit, but with the same number of qubits for consistency
        circ = self.qc.copy()


        # flip only when the training data is -1: in this way the circuit can simply count the number 
        # of states that are |1>
        # little endians convention is applied !!! train_data[::-1] !!!
        for iq, train_data_i in enumerate(train_data[::-1]):
            if train_data_i == -1:
                circ.x(self.data_qubits[iq])

        # initial Hadamards to create superposition in the counter register
        for ia in range(self.num_count_ancillas):
            circ.h(self.count_ancillas[ia])
        
        
        # Phase estimation
        for ia in range(self.num_count_ancillas):
            # the order is from the lowest index of the ancilla to the highest
            n_reps = 2**ia

            # repeat n_reps times the application of the unitary gate controlled on the ancillary qubit
            for rep_idx in range(n_reps):
                for iq in range(self.num_data_qubits):
                    circ.cp(2*np.pi/2**self.num_count_ancillas, self.count_ancillas[ia], self.data_qubits[iq])


        # invert flip applied previously to count the number of |1>
        # little endians convention is applied !!! train_data[::-1] !!!
        for iq, train_data_i in enumerate(train_data[::-1]):
            if train_data_i == -1:
                circ.x(self.data_qubits[iq])

        circ.barrier()
        qft_circ = QFT(self.num_count_ancillas, inverse=True).decompose(reps=1)

        circ = circ.compose(qft_circ, self.count_ancillas)

        # add an additional comparison circuit if needed
        if not self.simple_compare:
            circ = circ.compose(IntegerComparator(self.num_count_ancillas, int(np.ceil(self.num_data_qubits/2.0)), geq=True).decompose(reps=1),
                                qubits=self.ancillas)

        return circ


    def U_z(self, train_data, gamma):
        """
        Generate circuit for Uz evolution according to the training data and the value of gamma.
            - train_data: vector of training data.
            - gamma: multiplicative float in the time evolution definition.

            Conventions:
            - (1,-1) <--> (|0>,|1>)
            - little endians: least significant bit is the last one of the string
        """
        assert len(train_data) == self.num_data_qubits, "Wrong dimension of training data"

        # return a new copy of the circuit, but with the same number of qubits for consistency
        circ = self.qc.copy()
        circ.barrier()

        # define controlled operation on the 'ancilla_index'
        # little endians convention is applied !!! iq and idata goes on opposite directions !!!
        for iq, idata in zip(range(self.num_data_qubits),range(len(train_data)-1,-1,-1)):
            circ.crz(-2*gamma*train_data[idata]/np.sqrt(self.num_data_qubits), self.control_ancilla, self.data_qubits[iq])

        circ.barrier()
        return circ


    def U_x(self, beta):
        """
        Generate circuit for Ux evolution according to the value of beta.
            - beta: multiplicative float in the time evolution definition.

        """

        # return a new copy of the circuit, but with the same number of qubits for consistency
        circ = self.qc.copy()
        circ.barrier()

        for iq in range(self.num_data_qubits):
            circ.rx(-2*beta, self.data_qubits[iq])

        return circ


    def single_step_composer(self, qc, dataset, beta_p : float, gamma_p : float, tracking_function = None):
        """Define how a circuit is composed for each step in dQA."""

        for mu in range( dataset.shape[0] ):
        
            # create Hamming error counter circuit based on the given pattern
            qc_counter = self.Hamming_count(train_data = dataset[mu,:])
            qc_counter_inverse = qc_counter.inverse()
        
            # create Uz evolution circuit
            qc_Uz = self.U_z(train_data = dataset[mu,:], gamma=gamma_p)
        
            # compose all circuits to evolve according to Uz
            qc.compose(qc_counter, inplace=True)
            qc.compose(qc_Uz, inplace=True)
            qc.compose(qc_counter_inverse, inplace=True)

        # create and apply Ux evolution circuit
        qc_Ux = self.U_x(beta_p)
        qc.compose(qc_Ux, inplace=True)

        if tracking_function is not None:
            tracking_function( [qc_counter, qc_Uz, qc_counter_inverse, qc_Ux], compose=True)

        return qc