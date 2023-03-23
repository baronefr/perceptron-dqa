#!/usr/bin/bash -e

# create a pip environment for dqa and activate it
python3 -m venv mydqa
source mydqa/bin/activate

# install requirements
pip3 install -r requirements.txt

# download the Quantum Matcha library
git clone https://baltig.infn.it/quantum_tea_leaves/py_api_quantum_tea_leaves/-/tree/master/ lib/
git clone https://baltig.infn.it/quantum_matcha_tea/py_api_quantum_matcha_tea/-/tree/master  lib/

# install the Q libraries
pip3 install -e lib/py_api_quantum_tea_leaves/
pip3 install -e lib/py_api_quantum_matcha_tea/