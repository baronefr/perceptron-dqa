import numpy as np
from tqdm import tqdm

import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister

GPU = True
from importlib.util import find_spec

if GPU:
    if find_spec('cupy') is not None:
        import cupy as ncp
    else:
        print('Selected device is GPU but cupy is not installed, falling back to numpy')
        import numpy as ncp
else:
    import numpy as ncp


####################################
#### GET PERCEPTRON HAMILTONIAN ####
####################################

def kronecker_prod(operators):

    result = operators[0]
    for op in operators[1:]:

        result = np.kron(result, op)

    return result

def ReLU(x):
    return x * (x > 0)

def H_perc_nobatch(data, labels):
    n_data, n = data.shape
    sigma_z  = np.diag([1., -1.])
    identity = np.diag([1., 1.])

    h_perc = np.zeros((2**n, 2**n), dtype='float32')
    
    for i in tqdm(range(n_data), desc='Constructing H_perc'):

        op = np.zeros((2**n, 2**n), dtype='float32')
        for j in range(n):

            op += kronecker_prod([identity]*j+[data[i, j] * sigma_z]+[identity]*(n-j-1))

        h_perc += ReLU(-labels[i]*op)
        del op

    return (h_perc / np.sqrt(n)).astype('complex')

def H_perc_diag(data, labels):
    ## ASSUMING H_PERC IS DIAGONAL, WHICH IS IN THE COMPUTATIONAL BASIS ##
    n_data, n = data.shape
    identity = np.ones((2,), 'float32')
    sigma_z  = np.array([1., -1.])

    h_perc = np.zeros((2**n,), dtype='float32')
    
    for i in tqdm(range(n_data), desc='Constructing H_perc'):

        op = np.zeros((2**n,), dtype='float32')
        for j in range(n):

            op += kronecker_prod([identity]*j+[data[i, j] * sigma_z]+[identity]*(n-j-1))

        h_perc += ReLU(-labels[i]*op)
        del op

    return ncp.array((h_perc / np.sqrt(n)).astype('complex128'))



########################
##### LOSS TRACKER #####
########################

class LossTracker:
    def __init__(self, num_qubits, num_ancillae, init_state):

        self.n_qubits   = num_qubits
        self.n_ancillae = num_ancillae
        if type(init_state) is qi.Statevector:
            self._statevecs = [init_state]
        elif type(init_state) is QuantumCircuit:
            self._statevecs = [qi.Statevector.from_instruction(init_state)]
        else:
            print('type', type(init_state), 'of init_state is not valid')
        self._h_perc    = None
        self._little_endian = True
        self._statevecs_arr = None


    def track(self, qc, compose=True):

        if compose:
            # create a copy of the current circuit internally
            
            self.current_qc = QuantumCircuit(QuantumRegister(self.n_qubits), AncillaRegister(self.n_ancillae))
            
            # compose the circuit
            if type(qc) is list:
                
                for circuit in qc:
                    self.current_qc.compose(circuit, inplace=True)
                
            elif type(qc) is QuantumCircuit:
                self.current_qc.compose(qc, inplace=True)
            else:
                print('Error: type of qc is', type(qc))
                return
        else:
            self.current_qc = qc

        # track the state        
        self._statevecs.append(self._statevecs[-1].evolve(self.current_qc))
       
        del self.current_qc

    
    @property
    def statevecs(self):
        return self._statevecs
    
    @statevecs.setter
    def set_statevecs(self, value):
        self._statevecs = [value]
    
    @property
    def h_perc(self):
        return self._h_perc

    def reset(self, num_qubits=None, num_ancillae=None):
        self._statevecs.clear()
        self._statevecs_arr = None

        if num_qubits:
            self.n_qubits = num_qubits
        if num_ancillae:
            self.n_ancillae = num_ancillae

    def finalize(self):

        ## convert statevectors to arrays, keep only qubits of interest
        arr_list = []
        
        for state in tqdm(self._statevecs, desc='finalizing LossTracker'):
            out_red = qi.partial_trace(state, range(self.n_qubits, self.n_qubits + self.n_ancillae))
            prob, st_all = ncp.linalg.eigh(ncp.array(out_red.data))
            idx    = ncp.argmax(prob) 
            arr_list.append(st_all[:, idx].copy())
            del out_red, prob, st_all, idx
        
        self._statevecs_arr = ncp.stack(arr_list)
        del arr_list

    def finalize_opt(self):

        ## instead of tracing outs qubit we can simply modify the hamiltonian by putting
        ## identities on the ancillary qubits, remembering that qiskit uses little endians.
        self._statevecs_arr = ncp.array(np.stack([state.data for state in self._statevecs]))

    def __loss(self, statevec, h_perc):

        return ncp.vdot(statevec, h_perc * statevec)
    
    def get_losses(self, data, little_endian=True, labels=None):

        if len(self._statevecs) == 0:
            print('Error: no statevectors has been tracked down, please call track() before')
            return
        
        if labels is None:
            labels = np.ones((data.shape[0],))
        
        if self._statevecs_arr is None:
            print('LossTracker was not finalized, finalizing...')
            self.finalize()
            print('Done!')
            
        if self._h_perc is None or self._little_endian != little_endian:

            if little_endian:
                self._h_perc = H_perc_diag(data, labels)
            else:
                # invert data components if the circuit was constructed in big endian mode
                # NOT SURE IF THIS WORKS
                self._h_perc = H_perc_diag(data[:,::-1], labels)
            self._little_endian = little_endian
           
        result = ncp.real_if_close(ncp.apply_along_axis(self.__loss, axis=1, arr=self._statevecs_arr, h_perc=self._h_perc))
        
        if type(result) is np.ndarray:  return result
        else:                           return result.get()

    def get_losses_opt(self, data, little_endian=True, labels=None):

        if len(self._statevecs) == 0:
            print('Error: no statevectors has been tracked down, please call track() before')
            return
        
        if labels is None:
            labels = np.ones((data.shape[0],))
        
        if self._statevecs_arr is None:
            print('LossTracker was not finalized, finalizing...')
            self.finalize_opt()
            print('Done!')
            
        if self._h_perc is None or self._little_endian != little_endian:

            if little_endian:
                self._h_perc = kronecker_prod([ncp.ones((2,))]*self.n_ancillae + [H_perc_diag(data, labels)])
            else:
                # invert data components if the circuit was constructed in big endian mode
                # NOT SURE IF THIS WORKS
                self._h_perc = kronecker_prod([ncp.ones((2,))]*self.n_ancillae + [H_perc_diag(data[:,::-1], labels)])
            self._little_endian = little_endian
           
        result = ncp.real_if_close(ncp.apply_along_axis(self.__loss, axis=1, arr=self._statevecs_arr, h_perc=self._h_perc))
        
        if type(result) is np.ndarray:  return result
        else:                           return result.get()
    
    def get_edensity(self, data, little_endian=True, opt=True, labels=None):

        if opt:
            losses = self.get_losses_opt(data, little_endian=little_endian, labels=labels)
        else:
            losses = self.get_losses(data, little_endian=little_endian, labels=labels)
        e0 = np.real_if_close(np.sort(self._h_perc if type(self._h_perc) is np.ndarray else self._h_perc.get())[0])
        self.e0 = e0
        return (losses-e0) / data.shape[1]
    


if __name__ == '__main__':

    # example code

    def your_evolution(p, num_qubits, num_ancillae):
    
        # implement yout evolution here
        qc = QuantumCircuit(num_qubits+num_ancillae)
        if p == 1:
            qc.h(range(num_qubits))
        if p == 2:
            qc.h(range(num_qubits))
            qc.x(range(num_qubits, num_qubits+num_ancillae))
        if p == 3:
            qc.x( 3)
        if p == 4:
            qc.x(2)

        return qc
    
    N_data = 3
    N_feat = 4

    data = np.array([[1., 1., 1., 1.], 
                 [1., -1., 1., -1.], 
                 [1., -1., 1., 1.]])

    labels = np.ones((N_data, ))

    P = 4

    num_qubits   = 4
    num_ancillae = 2

    qc_tot = QuantumCircuit(num_qubits + num_ancillae)
    loss_tracker = LossTracker(num_qubits, num_ancillae, init_state=qc_tot)

    # apply evoltion 
    for p in range(P):

        qc = your_evolution(p+1, num_qubits, num_ancillae)
        loss_tracker.track(qc, compose=True)
        qc_tot = qc_tot.compose(qc)


    qc_tot.draw()


    print(loss_tracker.get_losses(data, little_endian=True))