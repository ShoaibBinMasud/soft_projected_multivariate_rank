import numpy as np
import ghalton
import ot
import pandas as pd
import torch
import pickle
from scipy.stats import multivariate_normal as mvn
from scipy.stats import bernoulli as ber
from scipy.stats import cauchy
from energy_stat import energy_test_statistics
from sinkhorn_balanced_Aeron import plan


def pairwise_distance(X, Y):
    x_col = np.expand_dims(X, axis = 1)
    
    y_lin = np.expand_dims(Y, axis = 0)
    
    M = np.sqrt(np.sum((x_col - y_lin)**2 , 2))
    return M
 
def energy_test_statistic(X, Y):
    n = X.shape[0]
    m = Y.shape[0]
    coefficient = n * m / (n + m)
    xx = pairwise_distance(X + 1e-16, X) # to avoid 'divide by zero error'
    yy = pairwise_distance(Y + 1e-16 , Y)
    xy = pairwise_distance(X, Y)
    return coefficient * ( 2 * np.mean(xy) - np.mean(xx) - np.mean(yy))

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
'''
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
'''
def sample_generator(n, d, v):
    
    if v =='v1':
        X, Y = np.zeros((n, d)), np.zeros((n, d))
        for i in range(n):
            X[i, :] = [cauchy.rvs(0, 1, 1),cauchy.rvs(0, 1, 1), cauchy.rvs(0, 1, 1)]
            Y[i, :] = [cauchy.rvs(0, 1, 1),cauchy.rvs(0.2, 1, 1), cauchy.rvs(0.2, 1, 1)]
            
    elif v =='v2':
        X, Y = np.zeros((n, d)), np.zeros((n, d))
        for i in range(n):
            x1 = np.random.uniform(0, 1, 1)
            x2 = 0.25 + 0.35 * x1 + np.random.uniform(0, 1, 1)
            x3 = 0.25 + 0.35 * x2 + np.random.uniform(0, 1, 1)
            y1 = np.random.uniform(0, 1, 1)
            y2 = 0.25 + 0.5 * y1 + np.random.uniform(0, 1, 1)
            y3 = 0.25 + 0.5 * y2 + np.random.uniform(0, 1, 1)
            X[i, :] = [x1, x2, x3]
            Y[i, :] = [y1, y2, y3]
            
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
        cov_x, cov_y = 3 * np.eye(3), 3 * np.eye(3)
        X, Y = mvn.rvs(mu_x, cov_x, n), mvn.rvs(mu_y, cov_y, n)
        
    elif v == 'v9':
        X, Y = np.zeros((n, d)), np.zeros((n, d))
        for i in range(n):
            X[i, :] =[np.random.gamma(2, 0.1, 1), np.random.gamma(2, 0.1, 1), np.random.gamma(2, 0.1, 1)]
            v1, v2, v3 = np.random.gamma(2, 0.1, 1), np.random.gamma(2, 0.1, 1), np.random.gamma(2, 0.1, 1)
            w1,w2,w3 =np.exp(np.random.normal(0,1)),np.exp(np.random.normal(0,1)),np.exp(np.random.normal(0,1))
            Y[i, :] = [v1* w1, v2*w2 , v3* w3]
            
  
    elif v == 'v10':
        mu_z1, mu_z2 = np.ones(d), np.ones(d)
        cov_z1, cov_z2 = np.eye(d), np.eye(d)
        z1, z2 = mvn.rvs(mu_z1, cov_z1, n), mvn.rvs(mu_z2, cov_z2, n)
        A = ber.rvs(0.2, loc=0, size = n, random_state = None)
        A= np.broadcast_to(A, (d, n)).T
        w1, w2, w3 = np.random.uniform(10, 11, n), np.random.uniform(10, 11, n), np.random.uniform(10, 11, n)
        w11, w22, w33 = np.expand_dims(w1, axis=1), np.expand_dims(w2, axis=1), np.expand_dims(w3, axis=1)
        W = np.concatenate([w11, w22, w33], axis =1)
        X = z1
        Y = A * z2 + (1 - A) * W
        
    elif v == 'v11':
        mu_z1, mu_z2 = np.ones(d), np.ones(d)
        cov_z1, cov_z2 = np.eye(d), np.eye(d)
        z1, z2 = mvn.rvs(mu_z1, cov_z1, n), mvn.rvs(mu_z2, cov_z2, n)
        A = ber.rvs(0.2, loc=0, size = n, random_state = None)
        A= np.broadcast_to(A, (d, n)).T
        W = np.zeros((200, 3))
        for i in range(200):
            W[i, :] = [np.random.normal(10, 0.1, 1), np.random.normal(10, 0.1, 1), np.random.normal(10, 0.1, 1)]
        X = z1
        Y = A*z2 + (1 - A) * W
        
    elif v == 'v12': 
        X = np.array([np.random.laplace(0, 1, n)  for i in range(d)]).T
        y1 = np.random.laplace(1, 1, n)
        Y = np.array([y1] + [np.random.laplace(0, 1, n)  for i in range(d-1)]).T
        
    elif v == 'v13': # multivariate gaussian separated by greater mean
       X, Y = np.zeros((n, d)), np.zeros((n, d))
       for i in range(d):
           X[:, i] = np.random.normal(0, 1, n)
           Y[:, i] = np.random.normal(0, 1, n)
          
       X[:, i] = np.random.normal(0, 4, n)        
    return X, Y

n = 200
dim = 3
noSimulation = 1000
sampleDict = {}
stat_dict = {}
#lambd_list = [0, 0.0001, 1, 5,10]
lambd_list = [0, 0.0001, 0.1, 1, 5, 10]
#sigma_list = [1, 2, 4, 8, 16, 32, 64]
sampleType = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v9', 'v10', 'v11', 'v12', 'v13']
for key in sampleType:
    sampleDict[key] = [sample_generator(n, dim, v = key) for i in range(noSimulation)]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
for key in sampleDict.keys():
    print(key)
    gen_samples = sampleDict[key]
    a, b = ot.unif(2 * n), ot.unif(2 *n)
    a_cuda = torch.from_numpy(a).to(device)
    b_cuda = torch.from_numpy(b).to(device)
    sequencer = ghalton.Halton(dim)
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
                G = plan(a_cuda, data_cuda, b_cuda, halton_points_cuda, p = 2, eps = j)
                row_sum = G.sum(axis = 1)
                scaled_G = G / row_sum[:, np.newaxis]
                r = torch.mm(scaled_G, halton_points_cuda)
                rx_cuda = r[0:n]
                ry_cuda = r[n:]
            #sRE.append(dcor.homogeneity.energy_test(rx, ry)[1])
            sRE.append(energy_test_statistics(rx_cuda, ry_cuda))
            #sRmmd.append( mix_rbf_mmd2(rx, ry, sigma_list, biased = True))
            print('sRE=  %0.6f' %(sRE[i]))
        results[str(j) +'sRE'] = sRE
       # results[str(j) + 'sRmmd'] = sRmmd    
    stat_dict[key] = results
with open('prof_1000_all_dists_sRE.pkl','wb') as f:
    pickle.dump(stat_dict, f)