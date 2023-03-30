#!/usr/bin/python3

# ====================================================
#  Quantum Information and Computing exam project
#
#   UNIPD Project |  AY 2022/23  |  QIC
#   group : Barone, Coppi, Zinesi
# ----------------------------------------------------
#   > description                                    |
#
#   utility script
# ----------------------------------------------------
#   coder : Barone Francesco
#         :   github.com/baronefr/
#   dated : 17 March 2023
#     ver : 1.0.0
# ====================================================

import numpy as np
import os
from datetime import datetime

from dQA_circuit_v3 import *

# ----------------------
#   TARGET DEFINITIONS
# ----------------------

save_curves = True

# function to run -------------------
def function_to_run(P, dt, max_bond_dim, datafile, **ignore ):

    csi_patterns = np.load(datafile)
    #csi_patterns = csi_patterns[0:5,:]  # crop dataset (optional)

    # initialize the dQA wrapper
    dd = mydQA_circuit(csi_patterns, P = P, dt = dt, backend = 'matcha_single_step')

    # run the simulation
    loss = dd.run(max_bond=max_bond_dim)
    print('final loss value: {}'.format(loss[-1]) )

    if save_curves:
        loss = np.array(loss)
        np.save('data/benchmark/circuit_12-15_loss/dt{}_bd{}.npy'.format(dt, max_bond_dim), loss)

    return np.real( loss[-1] )


# parameters to test ----------------
#
#  NOTE: format as list of dictionaries, which will be passed to input function as arguments
#
parameter_combinations = [
    {'P' : 100, 'dt' : 1.2, 'max_bond_dim' : 10, 'datafile' : 'data/patterns_12-15.npy'}
    # default syntax: P,dt,max_bond_dim,datafile,output
] + [
    {'P' : 100, 'dt' : 1.2, 'max_bond_dim' : int(bd), 'datafile' : 'data/patterns_12-15.npy'} 
    for bd in np.arange(start = 40, stop=61, step = 20)
    # default syntax: P,dt,max_bond_dim,datafile,output
] 

# target file to log results
benchmark_file = 'test-cbd.csv'





# ----------------------
#         SETUP
# ----------------------

# benchmark settings ----------------
MAX_FAILURE_TOLERANCE = 4      # maximum number of failures to tolerate before abort benchmarking

# log settings ----------------------
error_log_file = 'errors.log'  # set to None to disable this feature

# create head of log file, if it does not exist
if not os.path.isfile(benchmark_file):
    print('making header of benchmark file')
    benf = open(benchmark_file, "w")
    benf.write( ','.join( '{}'.format(v) for v in parameter_combinations[0].keys()) )
    benf.write(',output\n')
    benf.close()
else:
    print('benchmark file exists, do not print header')





# ----------------------
#        EXECUTE
# ----------------------
exe_failure_counter = 0
for settings in parameter_combinations:

    try:
        # run the target function, but catch exceptions ...
        value_to_log = function_to_run(**settings)

    except Exception as e:
        value_to_log = None  # None marks a failure
        exe_failure_counter += 1

        # write this error to log file, if argument error_log_file valid
        if error_log_file is not None:
            errf = open(error_log_file, "a")
            errf.write("\n[{}] params {} failed ----------\n".format(
                datetime.now().strftime("%d/%m/%Y %H:%M:%S"), str(settings) )
            )
            errf.write( str(e) )
            errf.close()
    
    # write to file the result of this run
    benf = open(benchmark_file, "a")
    benf.write( ','.join( '{}'.format(v) for v in settings.values()) )
    benf.write(',' + str(value_to_log) + '\n')
    benf.close()

    if exe_failure_counter >= MAX_FAILURE_TOLERANCE:
        raise Exception('multiple failures occurred in benchmarks, aborting...')


print('\noperations completed')
exit(0)
