import numpy as np
import ghalton
import ot
import pandas as pd
import pickle
from scipy.stats import multivariate_normal as mvn
from scipy.stats import bernoulli as ber
from scipy.stats import cauchy
from energy_stat import energy_test_statistics
from sinkhorn_balanced_Aeron import plan
import torch

def rank_generator(G, X, h, thr):
    thr =  1e-8
    maxx = G.max()
    ind = list()
    for i in range(X.shape[0]):
        for j in range(h.shape[0]):
            if G[i, j] / maxx > thr:
                ind.append(j)
    return h[ind]

def ot_rank(data, halton_points, a, b, n):
    M = ot.dist(data, halton_points, metric ='sqeuclidean')
    M /= M.max()       
    n = M.shape[0] // 2
    G_ot = ot.emd(a, b, M)  
    r = rank_generator(G_ot, data, halton_points, thr = 1e-8)
    r_x = r[0:n]
    r_y = r[n:]
    return r_x, r_y

def sinkhorn_rank(data, halton_points, a, b, n,lambd = 1e-4):
    M = ot.dist(data, halton_points, metric ='sqeuclidean')
    M /= M.max()       
    G = ot.bregman.sinkhorn_stabilized(a, b, M, lambd)
    row_sum = G.sum(axis = 1)
    scaled_G = G / row_sum[:, np.newaxis]
    r = np.matmul(scaled_G, halton_points)
    r_x = r[0:n]
    r_y = r[n:]
    return r_x, r_y

def sample_generator(n, d, v):
    
    if v =='v1':
        X, Y = np.zeros((n, d)), np.zeros((n, d))
        for i in range(n):
            for j in range(d):
                X[i, j] = cauchy.rvs(0, 1, 1)
                Y[i, j] = cauchy.rvs(0.2, 1, 1)
            
    elif v =='v2':
        X, Y = np.zeros((n, d)), np.zeros((n, d))
        for i in range(n):
            X[i, 1] = np.random.uniform(0, 1, 1)
            Y[i, 1] = np.random.uniform(0, 1, 1)
            for j in range(2, d):
                X[i, j] = 0.25 + 0.35 * X[i, j-1] + np.random.uniform(0, 1, 1)
                Y[i, j] = 0.25 + 0.5 *  Y[i, j-1] + np.random.uniform(0, 1, 1)
            
    elif v == 'v3':
        mu_v, mu_w = np.zeros(d), np.zeros(d)
        cov_v, cov_w = np.zeros((d, d)), np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                cov_v[i, j] = 0.35 ** np.abs(i - j)
                cov_w[i, j] = 0.65 ** np.abs(i - j)
        X, Y = mvn.rvs(mu_v, cov_v, n), mvn.rvs(mu_w, cov_w, n)
        
    elif v== 'v4':
        mu_v, mu_w = np.zeros(d), np.zeros(d)
        cov_v, cov_w = np.zeros((d, d)), np.zeros((d, d))
        
        for i in range(d):
            for j in range(d):
                if i != j : cov_v[i, j], cov_w[i, j] = 0.2, 0.5
                else: cov_v[i, j], cov_w[i, j] = 1, 1
        X, Y = mvn.rvs(mu_v, cov_v, n), mvn.rvs(mu_w, cov_w, n)
        
    elif v == 'v5':
        mu_v, mu_w = np.zeros(d), np.zeros(d)
        cov_v, cov_w = np.zeros((d, d)), np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                cov_v[i, j] = 0.35 ** np.abs(i - j)
                cov_w[i, j] = 0.75 ** np.abs(i - j)
        V, W = mvn.rvs(mu_v, cov_v, n), mvn.rvs(mu_w, cov_w, n)
        X, Y = np.exp(V), np.exp(W)
        
    elif v == 'v6':
        mu_v, mu_w = np.zeros(d), np.zeros(d)
        cov_v, cov_w = np.zeros((d, d)), np.zeros((d, d))
        
        for i in range(d):
            for j in range(d):
                if i != j : cov_v[i, j], cov_w[i, j] = 0.25, 0.75
                else: cov_v[i, j], cov_w[i, j] = 1, 1
        V, W = mvn.rvs(mu_v, cov_v, n), mvn.rvs(mu_w, cov_w, n)
        X, Y = np.exp(V), np.exp(W)  
   
    elif v == 'v7':
        mu_x, mu_y = 0.25 * np.ones(d), np.zeros(d)
        cov_x, cov_y = 3 * np.eye(d), 3 * np.eye(d)
        X, Y = mvn.rvs(mu_x, cov_x, n), mvn.rvs(mu_y, cov_y, n)
        
    elif v == 'v8':
        X, Y = np.zeros((n, d)), np.zeros((n, d))
        for i in range(n):
            for j in range(d):
                X[i, j] = np.random.gamma(2, 0.1, 1)
                v = np.random.gamma(2, 0.1, 1)
                w =np.exp(np.random.normal(0,1))
                Y[i, j] = v* w
            
  
    elif v == 'v9':
        mu_z1, mu_z2 = np.ones(d), np.ones(d)
        cov_z1, cov_z2 = np.eye(d), np.eye(d)
        z1, z2 = mvn.rvs(mu_z1, cov_z1, n), mvn.rvs(mu_z2, cov_z2, n)
        A = ber.rvs(0.2, loc=0, size = n, random_state = None)
        A= np.broadcast_to(A, (d, n)).T
        W = np.zeros((n, d))
        for i in range(n):
            for j in range(d):
                W[i, j] = np.random.uniform(10, 11, 1) 
        X = z1
        Y = A * z2 + (1 - A) * W
        
    elif v == 'v10':
        mu_z1, mu_z2 = np.ones(d), np.ones(d)
        cov_z1, cov_z2 = np.eye(d), np.eye(d)
        z1, z2 = mvn.rvs(mu_z1, cov_z1, n), mvn.rvs(mu_z2, cov_z2, n)
        A = ber.rvs(0.2, loc=0, size = n, random_state = None)
        A= np.broadcast_to(A, (d, n)).T
        W = np.zeros((n, d))
        for i in range(n):
            for j in range(d):
                W[i, j] = np.random.normal(10, 0.1, 1)
        X = z1
        Y = A*z2 + (1 - A) * W
        
    elif v == 'v11': 
        X = np.array([np.random.laplace(0, 1, n)  for i in range(d)]).T
        y1 = np.random.laplace(1, 1, n)
        Y = np.array([y1] + [np.random.laplace(0, 1, n)  for i in range(d-1)]).T
        
    elif v == 'v12': # multivariate gaussian separated by greater mean
       X, Y = np.zeros((n, d)), np.zeros((n, d))
       for i in range(d):
           X[:, i] = np.random.normal(0, 1, n)
           Y[:, i] = np.random.normal(0, 1, n)
          
       X[:, i] = np.random.normal(0, 4, n)        
    return X, Y

n = 200 
dim = [3, 8, 20, 50, 100, 200] 


noSimulation = 1000
sampleDict = {}
stat_dict = {}
lambd_list = [0.01]
sampleType = ['v1', 'v2', 'v3', 'v5', 'v6', 'v9', 'v11', 'v12']
for key in sampleType:
    for d in dim:
        sampleDict[key, d] = [sample_generator(n, d, v = key) for i in range(noSimulation)]
        

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  

for key in sampleDict.keys():
    print(key)
    gen_samples = sampleDict[key]
    n, d = gen_samples[0][0].shape
    a, b = ot.unif(2 * n), ot.unif(2 *n) 
    a_cuda = torch.from_numpy(a).to(device)
    b_cuda = torch.from_numpy(b).to(device)
    sequencer = ghalton.Halton(d)        
    halton_points = np.array(sequencer.get(2 * n))
    halton_points_cuda = torch.from_numpy(halton_points).to(device)
    results = pd.DataFrame()
    for j in lambd_list:
        print(j)
        sRE = list()
        for i in range(len(gen_samples)):
            X, Y = gen_samples[i][0], gen_samples[i][1]
            data = np.concatenate((X, Y), axis = 0)
            data_cuda = torch.from_numpy(data).to(device)
            if j == 0:
                rx, ry = ot_rank(data, halton_points, a, b, n)
                rx_cuda = torch.from_numpy(rx).to(device)
                ry_cuda = torch.from_numpy(ry).to(device)
            else:
                G = plan(a_cuda, data_cuda, b_cuda, halton_points_cuda, p=2, eps = j)
                row_sum = G.sum(axis = 1)
                scaled_G = G / row_sum[:, np.newaxis]
                r = torch.mm(scaled_G, halton_points_cuda)
                rx_cuda = r[0: n]
                ry_cuda = r[n: ]
                
            sRE.append(energy_test_statistics(rx_cuda, ry_cuda))
            if i % 1 ==0: print('sRE=  %0.6f' %(sRE[i]))
        results[str(j) +'sRE'] = sRE
    stat_dict[key] = results

with open('0_500_prof_dim_stats.pkl','wb') as f:
    pickle.dump(stat_dict, f)
