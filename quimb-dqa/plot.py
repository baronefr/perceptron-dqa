# %%

import matplotlib.pyplot as plt
import numpy as np

file = 'eps_s_P100.txt'

data = np.loadtxt(file)

plt.plot(data)
plt.yscale('log')
# %%
