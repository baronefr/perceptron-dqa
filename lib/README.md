# lib

```markdown
├─ dQA_loss.py           loss management for MPS simulation
├─ dQA_utils.py          core of MPS dQA for perceptron model
├─ HammingEvolution.py   core of circuit formulation of dQA
├─ loss_tracker.py       loss management for Qiskit simulation
└─ tenn.py               Tensor Network operations
```



## Quantum Tea

Quantum TEA is required in order to run the MPS simulation of the circuit dynamics. Please take a look at https://baltig.infn.it/quantum_tea/quantum_tea .

### setup hints

```bash
git clone https://baltig.infn.it/quantum_tea_leaves/py_api_quantum_tea_leaves.git leaves/
git clone https://baltig.infn.it/quantum_matcha_tea/py_api_quantum_matcha_tea.git matcha/
pip3 install -e leaves/
pip3 install -e matcha/
```

### documentation

**Tea Leaves**: https://quantum_tea_leaves.baltig-pages.infn.it/py_api_quantum_tea_leaves/index.html
* repo: https://baltig.infn.it/quantum_tea_leaves/py_api_quantum_tea_leaves/-/tree/master/

**Matcha**: https://quantum_matcha_tea.baltig-pages.infn.it/py_api_quantum_matcha_tea/
* repo: https://baltig.infn.it/quantum_matcha_tea/py_api_quantum_matcha_tea/-/tree/master
