#!/usr/bin/bash -e

# create a pip environment for dqa and activate it
python3 -m venv mydqa
source mydqa/bin/activate

# install requirements
pip3 install -r requirements.txt

# download the Quantum Matcha library (qmatchatea and qtealeaves)
git clone https://baltig.infn.it/quantum_tea_leaves/py_api_quantum_tea_leaves.git lib/leaves/
git clone https://baltig.infn.it/quantum_matcha_tea/py_api_quantum_matcha_tea.git lib/matcha/

# install the q* libraries
pip3 install -e lib/leaves/
pip3 install -e lib/matcha/