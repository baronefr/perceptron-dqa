

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

FILE_PREFIX = '../img/'
DO_SAVEFIG = False

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
lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.15), fancybox=True, ncol=2)
plt.ylabel(r"$\varepsilon(1)$")
plt.xlabel(r"$\delta t$")
plt.title('quimb MPS ($N = 21$)')
if DO_SAVEFIG: plt.savefig(FILE_PREFIX + 'quimb-21.svg', transparent=True, bbox_extra_artists=(lgd,), bbox_inches='tight')


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
lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.15), fancybox=True, ncol=2)
plt.ylabel(r"$\varepsilon(1)$")
plt.xlabel(r"$\delta t$")
plt.title('numpy MPS ($N = 21$)')
if DO_SAVEFIG: plt.savefig(FILE_PREFIX + 'numpy-21.svg', transparent=True, bbox_extra_artists=(lgd,), bbox_inches='tight')








# %% [markdown]
# # smaller dataset (N=12)
#  this case allows to compare with ED on our modest PCs

exact_diag = pd.read_csv('benchmark/ED_9-12.csv')
exact_diag_100 = exact_diag[ exact_diag['P'] == 100 ]
exact_diag_1000 = exact_diag[ exact_diag['P'] == 1000 ]

plt.plot(exact_diag_100['dt'], exact_diag_100['output'], linewidth=3, c=color_palette[0], label='$P=100$')
plt.plot(exact_diag_1000['dt'], exact_diag_1000['output'], linewidth=3, c=color_palette[1], label='$P=1000$')
plt.yscale('log')
plt.legend()
plt.ylabel(r"$\varepsilon(1)$")
plt.xlabel(r"$\delta t$")
plt.title('exact diagonalization ($N = 12$)')
if DO_SAVEFIG: plt.savefig(FILE_PREFIX + 'ed-12.svg', transparent=True)



# %% numpy backend

mps_npy = pd.read_csv('benchmark/numpy_9-12.1.csv') # use this dataset for comparison with single shot
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
lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.15), fancybox=True, ncol=2)
plt.ylabel(r"$\varepsilon(1)$")
plt.xlabel(r"$\delta t$")
plt.title('numpy MPS ($N = 12$)')
if DO_SAVEFIG: plt.savefig(FILE_PREFIX + 'numpy-12.svg', transparent=True, bbox_extra_artists=(lgd,), bbox_inches='tight')







# %% numpy backend, but averaged

files = [
    'benchmark/numpy_9-12.1.csv',
    'benchmark/numpy_9-12.2.csv',
    'benchmark/numpy_9-12.3.csv'
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

plt.yscale('log')
plt.legend()
plt.title('numpy averaged comparison ($N=12$)')
if DO_SAVEFIG: plt.savefig(FILE_PREFIX + 'numpy-12_averaged.svg', transparent=True)





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
if DO_SAVEFIG: plt.savefig(FILE_PREFIX + 'bond_dim-21.svg', transparent=True)



# %% plotting deviation curves

data_matrix = np.matrix(bd_data)
average = np.average(data_matrix, axis=0).A1

for ii, curve in enumerate(bd_data):
    plt.plot( np.linspace(0,1,len(curve)), curve - average, label="${}$".format(labels[ii]),
              color = cmap((ii)/len(bd_data)), linewidth=2,
    )

plt.ylabel(r"$\Delta\varepsilon(s)$")
plt.xlabel(r"$s$")
lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.15), fancybox=True, ncol=5)
plt.title('deviation from mean ($N = 21$, $P = 100$)')
if DO_SAVEFIG: plt.savefig(FILE_PREFIX + 'bond_dim-21_delta.svg', transparent=True, bbox_extra_artists=(lgd,), bbox_inches='tight')






# %%

ed = np.load('benchmark/ed_9-12_loss/loss_history_20230324093415_P100_dt0.1_E0.0.npy')
ed = np.concatenate( (np.array([ed[0]]), ed) ) # fix missing value of ED benchmark for s=0

files = [
    'benchmark/numpy_9-12_loss/dt0.1_bd10.npy',
    'benchmark/numpy_9-12_loss/dt0.1_bd20.npy',
    'benchmark/numpy_9-12_loss/dt0.1_bd40.npy',
    'benchmark/numpy_9-12_loss/dt0.1_bd60.npy',
    'benchmark/numpy_9-12_loss/dt0.1_bd80.npy',
]
labels = [ '10', '20', '40', '60', '80']

bd_data = []
for f in files:
    bd_data.append( np.load(f) )

for ii, curve in enumerate(bd_data):
    plt.plot( np.linspace(0,1,len(curve)), curve, label="${}$".format(labels[ii]),
              color = cmap((ii)/len(bd_data)), linewidth=2,
    )

plt.plot(np.linspace(0,1,len(ed)), ed, label='ED', c='k')
plt.yscale('log')
plt.ylabel(r"$\varepsilon(s)$")
plt.xlabel(r"$s$")
plt.legend()
plt.title("$N = 12$")
if DO_SAVEFIG: plt.savefig(FILE_PREFIX + 'bond_dim-12.svg', transparent=True)


# %% [markdown]
# ## single dQA plots

ed = np.load('benchmark/ed_9-12_loss/loss_history_20230324100157_P100_dt0.5_E0.0.npy')

plt.plot( np.linspace(0,1,len(ed)), ed)
plt.yscale('log')
plt.ylabel(r"$\varepsilon(s)$")
plt.xlabel(r"$s$")
plt.title('Exact Diagonalization')


# %%

mps = np.load('benchmark/numpy_9-12_loss/dt0.5_bd10.npy')

plt.plot( np.linspace(0,1,len(mps)), mps)
plt.yscale('log')
plt.ylabel(r"$\varepsilon(s)$")
plt.xlabel(r"$s$")
plt.title('MPS')

# %%

mps_quimb = np.load('benchmark/quimb_9-12_loss_dt0.5_bd10.npy')

plt.plot( np.linspace(0,1,len(mps_quimb)), mps_quimb, c='#fda85a', linewidth=3.4)
plt.yscale('log')
plt.ylabel(r"$\varepsilon(s)$")
plt.xlabel(r"$s$")
plt.title('MPS (Quimb)')
if DO_SAVEFIG: plt.savefig(FILE_PREFIX + 'dqa_quimb_only.svg', transparent=True)



# %% comparison dQA plots

fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2, sharey=ax1)

# plot dQA for both implementations
l1 = ax1.plot( np.linspace(0,1,len(mps_quimb)), mps_quimb, label = 'MPS (quimb)', linewidth=3, c='#fda85a')
r1 = ax2.plot( np.linspace(0,1,len(mps)), mps, label = 'MPS (jax)', linewidth=3, c='purple')

# plot ED in both sides
l0 = ax1.plot( np.linspace(0,1,len(ed)), ed, label = 'ED', linewidth=2, c = '#00a645', linestyle='--')
ax2.plot( np.linspace(0,1,len(ed)), ed, linewidth=2, c = '#00a645', linestyle='--')

plt.yscale('log')
ax1.set_ylabel(r"$\varepsilon(s)$")
ax1.set_xlabel(r"$s$")
ax2.set_xlabel(r"$s$")
fig.suptitle('dQA')

lgd = fig.legend( loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=False, shadow=False, ncol=3)

if DO_SAVEFIG: plt.savefig(FILE_PREFIX + 'dqa_comparison_12_05_100.svg',
                           transparent=True, bbox_extra_artists=(lgd,), bbox_inches='tight')



# %% [markdown]
# ## execution time benchmarks

bdata = pd.read_csv('benchmark/bench.csv')
bdata = bdata[ bdata['bd'] != 80 ] # remove 80 from dataset
bdf_bybackend = bdata.pivot_table(index = ['backend'], 
                      columns = ['bd'], values = 'time')
bdf_bybond = bdata.pivot_table(index = ['bd'], columns = ['backend'], values = 'time')


# %%

cmap = plt.get_cmap('viridis')

ax = bdf_bybackend.iloc[ reversed([3,0,2,4,1]) ].plot.barh(figsize=(9,7), 
        color = [cmap(ii/3) for ii in range(4) ]
)
ax.legend(loc='lower right', title=r"$\chi$", ncol=2)
ax.set_xlabel(r"time per iteration [$s/it$]")
plt.tight_layout()
if DO_SAVEFIG: plt.savefig(FILE_PREFIX + 'bench-by-backend.svg', transparent=True)

# %%

ax = bdf_bybond.plot.barh(figsize=(9,7), rot=0, 
                         color=['red', 'darkorange', 'gold', 'dodgerblue', 'deeppink'],
                        edgecolor='white', linewidth=1, width=0.70
)
ax.set_ylabel(r"$\chi$")
ax.set_xlabel(r"time per iteration [$s/it$]")
#ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.24), fancybox=True, 
#        ncol=2, prop={'size': 18} )
ax.legend(loc='lower right', prop={'size': 22})
plt.tight_layout()
if DO_SAVEFIG: plt.savefig(FILE_PREFIX + 'bench-by-bond.svg', transparent=True)



# %% [markdown]
# ## circuit bd test

# %%
files = [
    'benchmark/circuit_12-15_loss/dt1.2_bd10.npy',
    'benchmark/circuit_12-15_loss/dt1.2_bd20.npy',
    'benchmark/circuit_12-15_loss/dt1.2_bd40.npy',
    'benchmark/circuit_12-15_loss/dt1.2_bd60.npy'
]
labels = ['10', '20', '40', '60']

bd_data = []
for f in files:
    bd_data.append( np.load(f) )

for ii, curve in enumerate(bd_data):
    plt.plot( np.linspace(0,1,len(curve)), curve, label="${}$".format(labels[ii]),
              color = cmap((ii)/len(bd_data)), linewidth=3,
    )

plt.yscale('log')
plt.legend()
plt.ylabel(r"$\varepsilon(s)$")
plt.xlabel(r"$s$")
#plt.title(r"circuit Matcha")
plt.tight_layout()
if DO_SAVEFIG: plt.savefig(FILE_PREFIX + 'circuit-bond.svg', transparent=True)



# %% comparison with JAX

matcha = np.load('benchmark/circuit_12-15_loss/dt1.2_bd60.npy')
jaxloss = np.load('benchmark/numpy_12-15_loss_dt1.2_bd10.npy')

plt.plot( np.linspace(0,1,len(jaxloss)), jaxloss, label="JAX MPS",
              color = 'purple', linewidth=3,
    )
plt.plot( np.linspace(0,1,len(matcha)), matcha, label="Matcha",
              color = '#00a645', linewidth=3,
    )

plt.yscale('log')
plt.legend()
plt.ylabel(r"$\varepsilon(s)$")
plt.xlabel(r"$s$")
#plt.title(r"circuit with Matcha")
plt.tight_layout()
if DO_SAVEFIG: plt.savefig(FILE_PREFIX + 'circuit-matcha-jax.svg', transparent=True)


# %%
