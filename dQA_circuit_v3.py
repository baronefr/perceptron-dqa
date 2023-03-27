#!/usr/bin/python3

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
#   coder : Barone Francesco, Coppi Alberto, Zinesi Paolo
#         :   github.com/baronefr/
#   dated : 17 March 2023
#     ver : 1.0.0
# ====================================================

# %%
import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange

from lib.HammingEvolution import *
from lib.loss_tracker import *
from lib.dQA_loss import *

import qtealeaves.observables as obs
from qmatchatea import QCConvergenceParameters, QCOperators, QCBackend
from qmatchatea.py_emulator import run_py_simulation




# %%


class mydQA_circuit:

    def __init__(self, dataset : np.ndarray | str, 
                 P : int, dt : float, 
                 backend : str = 'qiskit') -> None:
        
        assert backend in ['qiskit', 'matcha', 'matcha_single_step'], 'not valid backend identifier'

        # retrieve dataset dimension and infer number of qubits
        if isinstance(dataset, str):
            self.dataset = np.load(dataset)
        else:
            self.dataset = dataset
        self.num_csi, self.num_data_qubits = dataset.shape

        self.annealing_param = {'P' : P, 'dt' : dt}
        self.backend = backend
        self.qc_generator = HammingEvolution(num_data_qubits = self.num_data_qubits)

        # link function to call for simulation execution
        if backend == 'qiskit':
            self.run = self._run_qiskit 
        elif backend == 'matcha':
            self.run = self._run_matcha
        else:
            self.run = self._run_matcha_stepbystep




    def _run_qiskit(self):
        assert self.backend == 'qiskit', 'not valid backend! please use {}'.format(self.backend)

        # retrieve simulation parameters
        P, dt  = self.annealing_param['P'], self.annealing_param['dt']

        # initialize the circuit
        qc = self.qc_generator.init_state_plus()

        # setting the loss tracker
        self.loss_tracker = LossTracker( 
            self.qc_generator.num_data_qubits, 
            self.qc_generator.num_ancillas,
            init_state=qc
        )

        for pp in trange(P):
            s_p = (pp+1)/P
            gamma_p = s_p*dt
            beta_p = (1-s_p)*dt

            qc = self.qc_generator.single_step_composer(qc, self.dataset, beta_p , gamma_p, tracking_function=self.loss_tracker.track)

        self.loss = self.loss_tracker.get_edensity(self.dataset, little_endian=True)
        return self.loss




    def _run_matcha(self, max_bond = 10):
        assert self.backend == 'matcha', 'not valid backend! please use {}'.format(self.backend)

        # retrieve simulation parameters
        P, dt  = self.annealing_param['P'], self.annealing_param['dt']

        # initialize the circuit
        qc = self.qc_generator.init_state_plus()
        
        self.loss = []

        # build the full circuit
        for pp in trange(P, desc = 'composing the circuit '):
            s_p = (pp+1)/P
            gamma_p = s_p*dt
            beta_p = (1-s_p)*dt

            qc = self.qc_generator.single_step_composer(qc, self.dataset, beta_p, gamma_p)
        
        # define matcha simulation observables and parameters
        operators = QCOperators()
        sigma_z = np.array([[1, 0], [0, -1]])
        operators.ops["sz"] = sigma_z
        observables = obs.TNObservables()
        observables += obs.TNObsBondEntropy()
        observables += obs.TNState2File("state.txt", "F")
        observables += obs.TNObsLocal('label', 'sz')
        conv_params = QCConvergenceParameters(max_bond_dimension=max_bond, singval_mode="C")
        backend = QCBackend(backend="PY")

        print('running matcha simulation')
        res = run_py_simulation( qc,
            convergence_parameters=conv_params,
            operators=operators,
            observables=observables,
            backend=backend
        )

        print('computing loss')
        loss_obj = mydQA_ancilla(self.dataset, P, dt, 
            n_ancilla=self.qc_generator.num_count_ancillas,
            flip_endian=True
        )
        self.loss.append( loss_obj.compute_loss( res.observables["mps_state"] ) )

        return self.loss






    def _run_matcha_stepbystep(self, max_bond = 10):
        assert self.backend == 'matcha_single_step', 'not valid backend! please use {}'.format(self.backend)
        print('WARNING: run experimental matcha step by step')  # FIXME


        # retrieve simulation parameters
        P, dt  = self.annealing_param['P'], self.annealing_param['dt']

        # initialize the circuit
        qc = self.qc_generator.init_state_plus()

        # define matcha simulation observables and parameters
        operators = QCOperators()
        sigma_z = np.array([[1, 0], [0, -1]])
        operators.ops["sz"] = sigma_z

        # setting up loss tracker
        loss_obj = mydQA_ancilla(self.dataset, P, dt, 
            n_ancilla=self.qc_generator.num_count_ancillas,
            flip_endian=True
        )

        self.loss = []
        initial_state = None # save here the state (MPS) at each step


        # build the full circuit
        for pp in trange(P, desc='execute dQA steps'):
            s_p = (pp+1)/P
            gamma_p = s_p*dt
            beta_p = (1-s_p)*dt

            qc = self.qc_generator.single_step_composer(qc, self.dataset, beta_p, gamma_p)

            observables = obs.TNObservables()
            observables += obs.TNObsBondEntropy()
            observables += obs.TNState2File("state.txt", "F")
            observables += obs.TNObsLocal('label', 'sz')
            conv_params = QCConvergenceParameters(max_bond_dimension=max_bond, singval_mode="C")
            backend = QCBackend(backend="PY")

            # ref : https://quantum_matcha_tea.baltig-pages.infn.it/py_api_quantum_matcha_tea/chapters/py_emulator.html#qmatchatea.py_emulator.run_py_simulation
            res = run_py_simulation( qc,
                convergence_parameters=conv_params,
                operators=operators,
                observables=observables,
                initial_state = initial_state,
                backend=backend
            )

            initial_state = res.observables["mps_state"]
            #initial_state = [ np.array(el) for el in initial_state ]
            self.loss.append( loss_obj.compute_loss( initial_state ) )

        return self.loss





# %%

if __name__== "__main__":

    # create a dataset
    N_xi, N_features = 5, 7
    csi_patterns = np.random.choice([1,-1], size=(N_xi, N_features))

    # initialize the dQA wrapper
    dd = mydQA_circuit(csi_patterns, P = 100, dt = 1, backend = 'matcha')

    # run the simulation
    loss = dd.run()
    print('final loss value: {}'.format(loss[-1]) )

    # TODO check loss len and format before plotting
    plt.plot(loss)
    plt.yscale('log')
    plt.show()



# %%

