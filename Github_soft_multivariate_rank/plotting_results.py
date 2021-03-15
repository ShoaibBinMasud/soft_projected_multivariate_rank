import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import matplotlib as mpl
import seaborn as sns
import re
import os
#mpl.rcParams['figure.dpi'] = 400
pd.set_option('display.float_format', lambda x: '%.6f' % x)
pd.set_option('display.float_format', lambda x: '%.6f' % x)

with open('0_500_prof_dim_stats.pkl', 'rb') as f:
    rank_eng_dim = pickle.load(f)
with open('prof_0_500_all_dists_sRE.pkl', 'rb') as f:
    rank_energy = pickle.load(f)
    
    
with open('0_1000_prof_null_stats.pkl', 'rb') as f:
    null_dist = pickle.load(f)  

reg_RE = ['0sRE', '0.001sRE', '0.01sRE', '0.1sRE', '1sRE']
nrow = 1
ncol = 5
fig, axs = plt.subplots(nrow, ncol, figsize = (24,5.2))
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=.4, wspace=.3)
for j in range(nrow):
        for i in range(len(reg_RE)):
            reg = reg_RE[i]
            x = re.findall('[0.0-9]+', reg)[0]
            sns.kdeplot(data=null_dist['v1'].astype('float64'), x = reg, color= 'b',linestyle ='-', linewidth = 3, label ='v1',ax = axs[i])
            sns.kdeplot(data=null_dist['v2'].astype('float64'), x = reg, color= 'g',linestyle =':', linewidth = 3, label ='v2',ax = axs[i])
            sns.kdeplot(data=null_dist['v3'].astype('float64'), x = reg, color= 'r',linestyle ='-', linewidth = 3, label ='v3',ax = axs[i])
            sns.kdeplot(data=null_dist['v4'].astype('float64'), x = reg, color= 'c',linestyle =':', linewidth = 3, label ='v4',ax = axs[i])
            sns.kdeplot(data=null_dist['v5'].astype('float64'), x = reg, color= 'm',linestyle ='-', linewidth = 3, label ='v5',ax = axs[i])
            sns.kdeplot(data=null_dist['v6'].astype('float64'), x = reg, color= 'y',linestyle ='-', linewidth = 3, label ='v6',ax = axs[i])
            sns.kdeplot(data=null_dist['v7'].astype('float64'), x = reg, color= 'k',linestyle =':', linewidth = 3, label ='v7',ax = axs[i])
            sns.kdeplot(data=null_dist['v11'].astype('float64'),x = reg, color= 'b',linestyle ='-', linewidth = 3, label ='v11',ax = axs[i])
            axs[i].set_title("Non Normlized KDE values $\lambda= $"+ x , fontsize = 8)
            axs[i].set_ylabel('Distribution',  fontsize=10, fontweight = 'bold')
            axs[i].set_xlabel('Rank Energy Statistics', fontsize=10, fontweight = 'bold')
            axs[i].legend()
#plt.savefig('prof_null_dist')
           
'''
# w.r.t. dimension
mean_RE_dim = pd.DataFrame()
std_RE_dim = pd.DataFrame()
for key in rank_eng_dim.keys():
    mean_RE_dim[key] = rank_eng_dim[key].filter(regex = 'sRE').mean()
    std_RE_dim[key] = rank_eng_dim[key].filter(regex = 'sRE').std()
      
mean_RE = pd.DataFrame()
std_RE = pd.DataFrame()

for key in rank_energy.keys():
    mean_RE[key] = rank_energy[key].filter(regex = 'sRE').mean()
    std_RE[key] = rank_energy[key].filter(regex = 'sRE').std()
    
sample_type = ['v1', 'v2', 'v3', 'v5', 'v6', 'v9', 'v11', 'v12']
idx = 0
for sample in sample_type:
    mean_RE_dim.insert(idx, column = (sample, 3), value = mean_RE[sample])
    idx +=6

ncol = 4
nrow = 2
fig, axs = plt.subplots(nrow, ncol, figsize = (24,10))
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=.3, wspace=.3)
dim_list = [3, 8, 20, 50, 100, 200]
m =  0
for i in range(nrow):
    for j in range(ncol):
        v = sample_type[i  * ncol + j]
        axs[i, j].plot(np.arange(6), mean_RE_dim.iloc[0, m * 6: m *6 + 6 ], 'go', ls='-', linewidth = 3, ms=8, color = 'blue',
                       label = 'RE: reg = ' + str(0.01))
        axs[i, j].legend(loc = 1)
        axs[i, j].set_title(v, fontsize=20)
        axs[i, j].set_ylabel('Rank Energy Statistics',  fontsize=10, fontweight='bold')
        axs[i, j].set_xlabel('Dimension $n$', fontsize=10, fontweight='bold')
        axs[i, j].set_title(v,  fontsize=20)
        plt.sca(axs[i, j])
        plt.xticks([0, 1, 2, 3, 4, 5], dim_list) 
        m+=1
plt.savefig('prof_0_500_dims_sRE')


# all distribution  
mean_RE = pd.DataFrame()
std_RE = pd.DataFrame()
mean_Rmmd = pd.DataFrame()
std_Rmmd = pd.DataFrame()
for key in rank_energy.keys():
    mean_RE[key] = rank_energy[key].astype('float64').filter(regex = 'sRE').mean()
    std_RE[key] = rank_energy[key].astype('float64').filter(regex = 'sRE').std()
    mean_Rmmd[key] = rank_energy[key].astype('float64').filter(regex = 'sRE').mean()
    std_Rmmd[key] = rank_energy[key].astype('float64').filter(regex = 'sRE').std()
    
ncol = 4
nrow = 3
reg_list = [0.0, 1e-4, 0.001, 0.01, 0.1,1, 5, 10]
fig, axs = plt.subplots(nrow, ncol, figsize = (16,12))
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=.25, wspace=.25)

for i in range(nrow):
    for j in range(ncol):
        col = mean_RE.columns[i  * ncol + j]
        axs[i, j].plot(np.arange(8), mean_RE[col], 'o', ls='-', linewidth = 5, ms=8, color = 'royalblue', label = "Rank Energy")
        axs[i, j].set_ylabel('Rank Energy Statistics',  fontsize=8, color = 'black', fontweight='bold')
        axs[i, j].set_title(col,  fontsize=12, fontweight='bold')
        plt.sca(axs[i, j])
        plt.xticks( [0, 1, 2, 3, 4, 5, 6, 7], reg_list)
        axs[i, j].set_xlabel('reg parameter $\lambda$', fontsize=10,  fontweight='bold')
        axs[i, j].legend(loc = 1)
plt.savefig('prof_0_500_all_dists_sRE')
'''