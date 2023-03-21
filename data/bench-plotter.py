# %%

import pandas as pd
import matplotlib.pyplot as plt

useLaTeX = True  # if True, use LaTeX backend
if useLaTeX:
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rc('text', usetex=True)

plt.rcParams['font.size'] = 22
plt.rcParams['figure.figsize'] = (9,6)



# %%

b_numpy_17 = pd.read_csv('benchmark/8-10_numpy.csv')
b_numpy_17 = b_numpy_17[  b_numpy_17['P'] == 1000 ]



# %%

color_palette = ['#006ec2', 'orange']

#plt.figure(figsize=(6, 4), dpi=80)
plt.plot(b_numpy_17['dt'], b_numpy_17['output'], linewidth=0.4, c=color_palette[0])
plt.scatter(b_numpy_17['dt'], b_numpy_17['output'], marker="*", s=120, c=color_palette[0], label='Numpy')
plt.yscale('log')
plt.legend()
plt.ylabel(r"$\varepsilon(1)$")
plt.xlabel(r"$\delta t$")
plt.title('N = 21')



# %%
