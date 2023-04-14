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
import tensorflow as tf
import tensorflow.linalg as tfl
import os
from datetime import datetime
from ed_utils_tf import *

from tqdm import tqdm



# ----------------------
#   TARGET DEFINITIONS
# ----------------------

# function to run -------------------
def function_to_run(h_perc, h_x, E0, N_feat, P, dt):
    
    state = init_state(N_feat)
    loss  = []

    pbar = tqdm(range(P), desc='ED QA')
    for i in pbar:

        h_t   = H_QA(i+1, P, h_perc, h_x)
        state = ed_qa_step(state, h_t, dt)

        loss.append(tf.cast((tf.tensordot(tf.math.conj(state), tfl.matvec(tf.cast(h_perc, 'complex128'), state), axes=1)-E0)/N_feat, 'float32').numpy())

        pbar.set_postfix({'loss':loss[-1], 'norm':tf.cast(tf.norm(state), 'float32').numpy()})

    np.save(f'losses/loss_history_P{P}_dt{dt:.1f}_E{E0:.1f}.npy', np.stack(loss))

    return loss[-1]


# parameters to test ----------------
#
#  NOTE: format as list of dictionaries, which will be passed to input function as arguments
#
parameter_combinations = [
    {'P' : 1000, 'dt' : np.round(dt,3)} for dt in np.arange(start = 0.1, stop=2.0, step = 0.1)
]

# target file to log results
benchmark_file = 'test.csv'





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


#read data and prepare hamiltonians

N_data = 8
N_feat = 10

data   = np.load('../data/patterns8-10.npy')
labels = tf.ones((N_data), 'float32')

h_perc = tf.cast(H_perc_nobatch(data, labels), 'complex128')
E0     = tfl.eigh(h_perc)[0][0]
h_x    = tf.cast(H_x(N_feat), 'complex128')




# ----------------------
#        EXECUTE
# ----------------------
exe_failure_counter = 0
for settings in tqdm(parameter_combinations, desc='Grid search'):

    print('Running with: ', settings)
    try:
        # run the target function, but catch exceptions ...
        value_to_log = function_to_run(h_perc, h_x, E0, N_feat, **settings)

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


print('operations completed')
os.exit(0)