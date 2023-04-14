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

import os
import numpy as np
from datetime import datetime
from ed_utils_ncp import *
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

        loss.append(ncp.real_if_close((ncp.tensordot(state.conjugate(), ncp.dot(h_perc, state), axes=1)-E0)/N_feat))

        pbar.set_postfix({'loss':loss[-1].astype('float32'), 'norm':ncp.linalg.norm(state).astype('float32')})

    
    return loss[-1], np.stack(loss)


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


# read data and prepare hamiltonians
# put out of grid search as it is needed once per datafile
data_fname = '../data/patterns_8-10.npy'

data   = np.load(data_fname)
N_data, N_feat = data.shape
labels = np.ones((N_data), 'float32')

h_perc_diag = H_perc_diag(data, labels)
E0     = np.sort(h_perc_diag)[0]
h_x    = H_x(data.shape[1])
h_perc = np.diag(h_perc_diag)




# ----------------------
#        EXECUTE
# ----------------------
exe_failure_counter = 0
date_time = datetime.now().strftime('%Y%m%d%H%M%S')
for settings in tqdm(parameter_combinations, desc='Grid search'):

    print('Running with: ', settings)
    try:
        # run the target function, but catch exceptions ...
        value_to_log, loss_history = function_to_run(h_perc, h_x, E0, N_feat, **settings)

    except Exception as e:
        value_to_log = None  # None marks a failure
        loss_history = np.zeros((1,))
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

    # save loss history
    np.save('losses/loss_history_{}_P{}_dt{:.1f}_E{:.1f}.npy'.format(datetime.now().strftime('%Y%m%d%H%M%S'), settings['P'], settings['dt'], E0), loss_history)


    if exe_failure_counter >= MAX_FAILURE_TOLERANCE:
        raise Exception('multiple failures occurred in benchmarks, aborting...')


print('operations completed')
os.exit(0)