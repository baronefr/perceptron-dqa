#!/usr/bin/python3

# just generate data


# %%

import numpy as np

N_xi, N_features = 8, 21 
#N_xi, N_features =  8, 10

x = np.random.randint(2, size=(N_xi, N_features))
x[ x == 0 ] = -1  # data is encoded as +- 1

np.save('data_{}-{}.npy'.format(N_xi, N_features), x)

# %%