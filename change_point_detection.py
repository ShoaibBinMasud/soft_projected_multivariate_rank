import numpy as np
import ot
import ghalton
from scipy.io import loadmat
from sRmmd import energy_test_statistics
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import matplotlib.pyplot as plt

def Compute2SampleSoftRankEnergy(dat, halton_points, window = 250, stride = 1, lambd = 1e-3):
    lenDat = len(dat)
    out = np.zeros((int(np.floor(lenDat/stride)), 1))
    count = 0
    for i in range(0,lenDat-stride,stride):
        if (i<window or i >= lenDat-window):
            out[count, ]=0
        else:        
            win1 = dat[i- window:i, :]
            win2 = dat[i: i+window, :]
            concat_data = np.concatenate((win1, win2), axis = 0)
            out[count, ] = soft_rank_energy(concat_data, halton_points)
        count = count+1
    return out

def soft_rank_energy(data, halton_points, window = 250, lambd = 1e-3):
    a, b = ot.unif(2 * window), ot.unif(2 * window)
    M = ot.dist(data, halton_points, metric ='sqeuclidean')
    M /= M.max()       
    G = ot.bregman.sinkhorn_stabilized(a, b, M, lambd)
    row_sum = G.sum(axis = 1)
    scaled_G = G / row_sum[:, np.newaxis]
    r = np.matmul(scaled_G, halton_points)
    rx = r[0 : window]
    ry = r[window :]
    out = energy_test_statistics(rx, ry)
    return out

lambd = 0.001
dataset = loadmat('HASCdataset/person020051.mat')
xdata = dataset['Y']
label = dataset['Lc']


window = 250
dim = xdata.shape[1]
sequencer = ghalton.Halton(dim)
halton_points = np.array(sequencer.get(2 * window))
sRE = Compute2SampleSoftRankEnergy(xdata, halton_points)
convFilter=loadmat('HASCdataset/TwoSampConvFilter')

sRE_filter = convFilter['filter'].flatten()
sRE_filter =sRE_filter[0::int(np.ceil(len(sRE)/(2*window)))] -0.166
sRE_filter = sRE_filter /np.sum(sRE_filter)
sRE_filter = np.convolve(sRE_filter.flatten(), sRE_filter, mode = 'same')


fig, ax1 = plt.subplots( figsize = (16,6))
t = np.arange(xdata[:, 0].shape[0])
ax2 = ax1.twinx()
ax1.plot(t, xdata[:, 0], color = 'darkgrey', label = 'Raw data')
ax2.plot(t, sRE, 'r--',label = 'sRmmd', linewidth = 2)
ax2.plot(t, sRE_filter, 'g-', label = 'sRmmd_Filtered', linewidth = 3)
ax2.set_ylabel('Change Point Statistic')
ax1.set_xlabel('Samples(100Hz)')
ax1.set_title('HASC-PAC2016')
plt.legend(loc='upper right')
plt.xlim(0, 28000)
plt.savefig('change_point')