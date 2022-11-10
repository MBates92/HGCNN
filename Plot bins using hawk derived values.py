import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

font = {'size' : 5}

matplotlib.rc('font', **font)

PATH = 'D:/Workshop/PhD/Data/ViaLactea/Temperature Analysis/Whole Plane Analysis'

N_mins = [3,5,7,10,14,20,28]
T_crits = [12.5,14.4,16.7,19.3,22.4,25.9]
N_mins.reverse()

f, axes = plt.subplots(nrows=len(N_mins), ncols=len(T_crits), figsize=(15,15),sharex=True, sharey=True)

for i,N_min in enumerate(N_mins):
    bins = pd.read_csv(f'{PATH}/{N_min:02}/bins.csv', header = None).values
    bins = [bin[0] for bin in bins]
    for j,T_crit in enumerate(T_crits):
        try:
            if j == 0:
                axes[i,j].set_ylabel(fr'$\mathrm{{N_{{min}}}} = {N_min}$')
            if N_min == N_mins[-1]:
                axes[i,j].set_xlabel(fr'$\mathrm{{T_{{crit}}}} = {T_crit}$')
            df = pd.read_csv(f'{PATH}/{N_min:02}/{int(np.round(T_crit))}_hist.csv')
            axes[i,j].hist(bins[:-1], bins, weights = df['n_all'],
                log=True,
                histtype = u'step',
                color='k',
                label= 'ALL T')
            axes[i,j].hist(bins[:-1], bins, weights = df['n_hot'],
                log=True,
                histtype = u'step',
                color='b',
                label= fr'$T \geq {T_crit}$')
            axes[i,j].hist(bins[:-1], bins, weights = df['n_cold'],
                log=True,
                histtype = u'step',
                color='r',
                label= fr'$ T < {T_crit}$')
            axes[i,j].set_xscale('log')
            axes[i,j].legend()
            axes[i,j].set_ylim(1e1, 1e8)
            axes[i,j].set_xlim(3e20)
        except:
            axes[i,j].hist(bins[:-1], bins, weights = df['n_all'],
                log=True,
                histtype = u'step',
                color='k',
                label= 'ALL T')
f.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', 
                which='both', 
                top=False, 
                bottom=False, 
                left=False, 
                right=False, 
                pad=20)
plt.xlabel(r'$N(H_2) [cm^{-2}]$')
plt.ylabel(r'$dN/d\logN(H_2)[\mathrm{per bin}]$')
plt.savefig(f'{PATH}/Montage.svg', format='svg', dpi=1200)