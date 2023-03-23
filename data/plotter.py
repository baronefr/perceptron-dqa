

# ====================================================
#  Quantum Information and Computing exam project
#
#   UNIPD Project |  AY 2022/23  |  QIC
#   group : Barone, Coppi, Zinesi
# ----------------------------------------------------
#   > description                                    |
#
#   data plotting
# ----------------------------------------------------
#   coder : Barone Francesco
#         :   github.com/baronefr/
#   dated : 23 March 2023
#     ver : 1.0.0
# ====================================================

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

useLaTeX = True  # if True, use LaTeX backend
if useLaTeX:
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rc('text', usetex=True)

plt.rcParams['font.size'] = 22
plt.rcParams['figure.figsize'] = (9,6)

color_palette = ['#006ec2', '#ff7b00']


# %% utilities

def symbols_line(pplot, x, y, color, marker, msize, label) -> None:
    """Plot a dashed line with symbols."""

    pplot.scatter(x, y, linewidth=3, marker=marker, s=msize,
            c=color, label=label)
    pplot.plot(x, y, linewidth=2, linestyle='--', alpha=0.7, c=color)




# %% [markdown]
# # large dataset (N=21)
#  note: ED data has been taken from reference paper



# %% quimb backend

mps_qtn = pd.read_csv('benchmark/quimb_17-21.csv')
mps_qtn_100 = mps_qtn[  mps_qtn['P'] == 100 ]
mps_qtn_1000 = mps_qtn[  mps_qtn['P'] == 1000 ]

paper_ed_100 = pd.read_csv('paper_7a_ed100.csv')
paper_ed_1000 = pd.read_csv('paper_7a_ed1000.csv')

# paper ED
plt.plot(paper_ed_100['dt'], paper_ed_100['output'], linewidth=4, c=color_palette[0], label='ED (paper) $P=100$')
plt.plot(paper_ed_1000['dt'], paper_ed_1000['output'], linewidth=4, c=color_palette[1], label='ED (paper) $P=1000$')

# our data 
symbols_line(plt, mps_qtn_100['dt'], mps_qtn_100['output'], color = color_palette[0], 
             marker = '*', msize = 120, label = 'MPS $P=100$')


plt.scatter(mps_qtn_1000['dt'], mps_qtn_1000['output'], linewidth=3, marker='s', s=100,
            c=color_palette[1], label='MPS $P=1000$')
plt.plot(mps_qtn_1000['dt'], mps_qtn_1000['output'], linewidth=2, linestyle='--', alpha=0.7, c=color_palette[1])

# plot setup
plt.yscale('log')
plt.tight_layout()
plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.15), fancybox=True, ncol=2)
plt.ylabel(r"$\varepsilon(1)$")
plt.xlabel(r"$\delta t$")
plt.title('quimb MPS ($N = 21$)')


# %% numpy backend

mps_npy = pd.read_csv('benchmark/numpy_17-21.csv')
mps_npy_100 = mps_npy[  mps_npy['P'] == 100 ]
mps_npy_1000 = mps_npy[  mps_npy['P'] == 1000 ]

#paper_mps_100 = pd.read_csv('paper_7a_mps100.csv')    # basically the same!
#paper_mps_1000 = pd.read_csv('paper_7a_mps1000.csv')
paper_ed_100 = pd.read_csv('paper_7a_ed100.csv')
paper_ed_1000 = pd.read_csv('paper_7a_ed1000.csv')

# paper ED
plt.plot(paper_ed_100['dt'], paper_ed_100['output'], linewidth=4, c=color_palette[0], label='ED (paper) $P=100$')
plt.plot(paper_ed_1000['dt'], paper_ed_1000['output'], linewidth=4, c=color_palette[1], label='ED (paper) $P=1000$')

# our data 
symbols_line(plt, mps_npy_100['dt'], mps_npy_100['output'], color = color_palette[0], 
             marker = '*', msize = 120, label = 'MPS $P=100$' )
symbols_line(plt,mps_npy_1000['dt'], mps_npy_1000['output'], color = color_palette[1], 
             marker = 's', msize = 100, label = 'MPS $P=1000$')


# plot setup
plt.yscale('log')
plt.tight_layout()
plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.15), fancybox=True, ncol=2)
plt.ylabel(r"$\varepsilon(1)$")
plt.xlabel(r"$\delta t$")
plt.title('numpy MPS ($N = 21$)')









# %% [markdown]
# # smaller dataset (N=10)
#  this case allows to compare with ED on our modest PCs

exact_diag = pd.read_csv('benchmark/ED_8-10.csv')
exact_diag_100 = exact_diag[ exact_diag['P'] == 100 ]
exact_diag_1000 = exact_diag[ exact_diag['P'] == 1000 ]

plt.plot(exact_diag_100['dt'], exact_diag_100['output'], linewidth=3, c=color_palette[0], label='$P=100$')
plt.plot(exact_diag_1000['dt'], exact_diag_1000['output'], linewidth=3, c=color_palette[1], label='$P=1000$')
plt.yscale('log')
plt.legend()
plt.ylabel(r"$\varepsilon(1)$")
plt.xlabel(r"$\delta t$")
plt.title('exact diagonalization ($N = 10$)')





# %% quimb backend

mps_npy = pd.read_csv('benchmark/quimb_8-10.csv')
mps_npy_100 = mps_npy[  mps_npy['P'] == 100 ]
mps_npy_1000 = mps_npy[  mps_npy['P'] == 1000 ]

# MPS plots
plt.scatter(mps_npy_100['dt'], mps_npy_100['output'], linewidth=3, marker='*', s=120,
            c=color_palette[0], label='MPS $P=100$')
plt.plot(mps_npy_100['dt'], mps_npy_100['output'], linewidth=1, c=color_palette[0])

plt.scatter(mps_npy_1000['dt'], mps_npy_1000['output'], linewidth=3, marker='s', s=100,
            c=color_palette[1], label='MPS $P=1000$')
plt.plot(mps_npy_1000['dt'], mps_npy_1000['output'], linewidth=1, c=color_palette[1])

# compare with ED
plt.plot(exact_diag_100['dt'], exact_diag_100['output'], linewidth=3, c=color_palette[0], label='ED $P=100$')
plt.plot(exact_diag_1000['dt'], exact_diag_1000['output'], linewidth=3, c=color_palette[1], label='ED $P=1000$')

plt.yscale('log')
plt.tight_layout()
plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.15), fancybox=True, ncol=2)
plt.ylabel(r"$\varepsilon(1)$")
plt.xlabel(r"$\delta t$")
plt.title('quimb MPS ($N = 10$)')




# %% numpy backend

mps_npy = pd.read_csv('benchmark/numpy_8-10_2.csv') # use this dataset for comparison with single shot
mps_npy_100 = mps_npy[  mps_npy['P'] == 100 ]
mps_npy_1000 = mps_npy[  mps_npy['P'] == 1000 ]


plt.scatter(mps_npy_100['dt'], mps_npy_100['output'], linewidth=3, marker='*', s=120,
            c=color_palette[0], label='MPS $P=100$')
plt.plot(mps_npy_100['dt'], mps_npy_100['output'], linewidth=1, c=color_palette[0])

plt.scatter(mps_npy_1000['dt'], mps_npy_1000['output'], linewidth=3, marker='s', s=100,
            c=color_palette[1], label='MPS $P=1000$')
plt.plot(mps_npy_1000['dt'], mps_npy_1000['output'], linewidth=1, c=color_palette[1])

plt.plot(exact_diag_100['dt'], exact_diag_100['output'], linewidth=3, c=color_palette[0], label='ED $P=100$')
plt.plot(exact_diag_1000['dt'], exact_diag_1000['output'], linewidth=3, c=color_palette[1], label='ED $P=1000$')

plt.yscale('log')
plt.tight_layout()
plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.15), fancybox=True, ncol=2)
plt.ylabel(r"$\varepsilon(1)$")
plt.xlabel(r"$\delta t$")
plt.title('numpy MPS ($N = 10$)')








# %% numpy backend, but averaged

files = [
    'benchmark/numpy_8-10_1.csv',
    'benchmark/numpy_8-10_2.csv',
    'benchmark/numpy_8-10_3.csv'
]

tmp_100 = []
tmp_1000 = []

for f in files:
    mps_npy = pd.read_csv( f )
    mps_npy_100 = mps_npy[  mps_npy['P'] == 100 ]['output'].to_numpy()
    mps_npy_1000 = mps_npy[  mps_npy['P'] == 1000 ]['output'].to_numpy()

    bak_dt = mps_npy[  mps_npy['P'] == 100 ]['dt'].to_numpy()

    tmp_100.append(mps_npy_100)
    tmp_1000.append(mps_npy_1000)

data_100 = np.average( np.matrix(tmp_100), axis=0).A1 
data_1000 = np.average(np.matrix(tmp_1000), axis=0).A1 


# plot data
plt.scatter(bak_dt, data_100, linewidth=4, c=color_palette[0], label='MPS $P=100$')
plt.plot(bak_dt, data_100, linewidth=0.6, c=color_palette[0])

plt.scatter(bak_dt, data_1000, linewidth=4, c=color_palette[1], label='MPS $P=1000$')
plt.plot(bak_dt, data_1000, linewidth=0.6, c=color_palette[1])

# plot exact diag results
plt.plot(exact_diag_100['dt'], exact_diag_100['output'], linewidth=3, c=color_palette[0], label='ED $P=100$')
plt.plot(exact_diag_1000['dt'], exact_diag_1000['output'], linewidth=3, c=color_palette[1], label='ED $P=1000$')

#plt.plot(data_1000, linewidth=.8, c=color_palette[1])
plt.yscale('log')
plt.legend()
plt.title('numpy averaged comparison ($N=10$)')





# %% test with 9-12

mps_npy = pd.read_csv('benchmark/numpy_9-12.csv') # use this dataset for comparison with single shot
mps_npy_100 = mps_npy[  mps_npy['P'] == 100 ]
mps_npy_1000 = mps_npy[  mps_npy['P'] == 1000 ]


plt.scatter(mps_npy_100['dt'], mps_npy_100['output'], linewidth=3, marker='*', s=120,
            c=color_palette[0], label='MPS $P=100$')
plt.plot(mps_npy_100['dt'], mps_npy_100['output'], linewidth=1, c=color_palette[0])

plt.scatter(mps_npy_1000['dt'], mps_npy_1000['output'], linewidth=3, marker='s', s=100,
            c=color_palette[1], label='MPS $P=1000$')
plt.plot(mps_npy_1000['dt'], mps_npy_1000['output'], linewidth=1, c=color_palette[1])

plt.plot(exact_diag_100['dt'], exact_diag_100['output'], linewidth=3, c=color_palette[0], label='ED $P=100$')
plt.plot(exact_diag_1000['dt'], exact_diag_1000['output'], linewidth=3, c=color_palette[1], label='ED $P=1000$')

plt.yscale('log')
plt.tight_layout()
plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.15), fancybox=True, ncol=2)
plt.ylabel(r"$\varepsilon(1)$")
plt.xlabel(r"$\delta t$")
plt.title('numpy MPS ($N = 12$)')




# %% [markdown]
# # bond dimension scaling

# %% loading data

files = [
    'benchmark/numpy_17-21_loss/17-21_full-loss_bd2.npy',
    'benchmark/numpy_17-21_loss/17-21_full-loss_bd4.npy',
    'benchmark/numpy_17-21_loss/17-21_full-loss_bd8.npy',
    'benchmark/numpy_17-21_loss/17-21_full-loss_bd10.npy',
    'benchmark/numpy_17-21_loss/17-21_full-loss_bd15.npy',
    'benchmark/numpy_17-21_loss/17-21_full-loss_bd20.npy',
    'benchmark/numpy_17-21_loss/17-21_full-loss_bd40.npy',
    'benchmark/numpy_17-21_loss/17-21_full-loss_bd60.npy',
    'benchmark/numpy_17-21_loss/17-21_full-loss_bd80.npy',
]

labels = [ '2', '4', '8', '10', '15', '20', '40', '60', '80']

bd_data = []
for f in files:
    bd_data.append( np.load(f) )



# %% plotting curves (function of s)

cmap = plt.get_cmap('plasma')

for ii, curve in enumerate(bd_data):
    plt.plot( np.linspace(0,1,len(curve)), curve, label="${}$".format(labels[ii]),
              color = cmap((ii)/len(bd_data)), linewidth=2,
    )

plt.ylabel(r"$\varepsilon(s)$")
plt.xlabel(r"$s$")
plt.yscale('log')
plt.legend()
plt.title('bond dimension check ($N = 21$, $P = 100$)')


# %% plotting deviation curves

data_matrix = np.matrix(bd_data)
average = np.average(data_matrix, axis=0).A1

for ii, curve in enumerate(bd_data):
    plt.plot( np.linspace(0,1,len(curve)), curve - average, label="${}$".format(labels[ii]),
              color = cmap((ii)/len(bd_data)), linewidth=2,
    )

plt.ylabel(r"$\Delta\varepsilon(s)$")
plt.xlabel(r"$s$")
plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.15), fancybox=True, ncol=5)
plt.title('deviation from mean ($N = 21$, $P = 100$)')







# %%


