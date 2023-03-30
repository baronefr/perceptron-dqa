<h1 align="center"> <b>digitized Quantum Annealing via TN simulations</b> </h1>

<p align="center"><b>Project</b> // Barone Francesco, Coppi Alberto, Zinesi Paolo<br></p>


### In this repo

```markdown
├── data/            dataset and benchmarks
├── img/             plots
├── quimb-dqa/       first implementation of MPS dQA (with Quimb)
│
├─ dQA_mps.py        implementation of dQA with MPS (jax)
└─ dQA_circuit.py    implementation of dQA with Matcha/Qiskit
```


### Setup

Python requirements are listed in `requirements.txt`.

**[Quantum Tea Leaves](https://baltig.infn.it/quantum_tea/quantum_tea)** and **Quantum Matcha Tea** are required to run the `dQA_circuit` implementation. To setup a custom environment, we suggest to take a look at the script `setup-env.sh`, as it should work in most cases.

Our software has been tested with the following software version:
```
qiskit==0.38.0
qmatchatea==0.4.7
qtealeaves==0.4.15
```

Furthermore, in order to use the dQA circuit simulation via Matcha in step-by-step mode, it is necessary to hot-fix the Matcha library itself.
```python
#   fix line 840 of file  qtealeaves.emulator.mps_simulator.py  as
obj = cls(len(tensor_list), 0, conv_params, local_dim)
```

<br><br>

## Bibliography

* G. Lami et al. **"Quantum Annealing for Neural Network optimization problems: a new approach via Tensor Network simulations"**, in SciPost Physics, 2022 ([submitted](https://arxiv.org/abs/2208.14468))

---

<h5 align="center">Quantum Information and Computing<br>AY 2022/2023 - University of Padua</h5>

<p align="center">
  <img src="https://raw.githubusercontent.com/baronefr/baronefr/main/shared/2022_unipd.png" alt="" height="70"/>
  &emsp;
  <img src="https://raw.githubusercontent.com/baronefr/baronefr/main/shared/2022_pod.png" alt="" height="70"/>
</p>
