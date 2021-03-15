import autograd.numpy as np
import ghalton
import pandas as pd
import pickle
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent
from sRmmd import mix_rbf_mmd2, sinkhorn_stabilized, energy_test_statistics
from scipy.stats import cauchy


def sample_generator(n, d, v):
    # check cauchy
    if v =='cauchy':
        X, Y = np.zeros((n, d)), np.zeros((n, d))
        for i in range(n):
            for j in range(d):
                X[i, j] = cauchy.rvs(0, 1, 1)
                Y[i, j] = cauchy.rvs(0, 1, 1)
                
    elif v =='cauchy_diff':
        X, Y = np.zeros((n, d)), np.zeros((n, d))
        for i in range(n):
            for j in range(d):
                X[i, j] = cauchy.rvs(0, 1, 1)
                Y[i, j] = cauchy.rvs(0.2, 1, 1)
                  
    elif v =='v1':
        X = np.array([np.random.laplace(0, 1, n)  for i in range(d)]).T
        y1 = np.random.laplace(0, 1, n)
        Y = np.array([y1] + [np.random.laplace(0, 1, n)  for i in range(d-1)]).T
        
    elif v == 'v2':
        X = np.array([np.random.laplace(0, 1, n)  for i in range(d)]).T
        y1 = np.random.laplace(1, 1, n)
        Y = np.array([y1] + [np.random.laplace(0, 1, n)  for i in range(d-1)]).T
        
    elif v =='v3':
        X, Y = np.zeros((n, d)), np.zeros((n, d))
        for i in range(d):
            X[:, i] = np.random.normal(0, 1, n)
            Y[:, i] = np.random.normal(0, 1, n)
            
        X[:, i] = np.random.normal(0, 4, n)  
        
    elif v=='v4':
        X, Y = np.zeros((n, d)), np.zeros((n, d))
        for i in range(d):
            X[:, i] = np.random.normal(0, 1, n)
            Y[:, i] = np.random.normal(0, 1, n)   
            
    return X, Y
    
n = 200
dim = [100]
noSimulation = 10
sampleDict = {}
stat_dict = {}
lambd = 0.0001
sigma_list = [1, 2, 4, 8, 16, 32]
k = 8
#sample_type = ['cauchy','cauchy_diff' ,'v1', 'v2', 'v3', 'v4']
sample_type = ['v1', 'v3', 'v4']
for v in sample_type:
    for d in dim:
        gen_samples = [sample_generator(n, d, v) for i in range(noSimulation)]
        sampleDict[(v , d)] = gen_samples

egrad = ehess = None
psRE_dict ={}
for key in sampleDict.keys():
    print(key)
    gen_samples = sampleDict[key]
    n, d = gen_samples[0][0].shape
    manifold = Stiefel(d, k)
    a, b = np.ones(2 * n, ) / (2 * n), np.ones(2 * n, ) / (2 * n)
    sequencer = ghalton.Halton(d)
    halton_points = np.array(sequencer.get(2 * n))
    results = pd.DataFrame()
    
    #sRmmd, psRmmd = list(), list()
    sRE, psRE = list(), list()
    for i in range(len(gen_samples)):
        X, Y = gen_samples[i][0], gen_samples[i][1]
        data = np.concatenate((X, Y), axis = 0)
        
        G = sinkhorn_stabilized(data, halton_points, reg = lambd)
        row_sum = G.sum(axis = 1)
        scaled_G = G / row_sum[:, np.newaxis]
        r = np.matmul(scaled_G, halton_points)
        r_x = r[0:n]
        r_y = r[n:]  
        #sRmmd.append(mix_rbf_mmd2(r_x, r_y, sigma_list))
        sRE.append(energy_test_statistics(r_x, r_y, ))
        
        def cost(U):
            projector = np.matmul(U, U.T)
            proj_X, proj_Y = np.matmul(projector, X.T),   np.matmul(projector, Y.T)
            proj_data = np.concatenate((proj_X.T, proj_Y.T), axis = 0)
            #G = sinkhorn_stabilized(np.matmul(projector, data.T), halton_points, reg = lambd)
            G = sinkhorn_stabilized(proj_data, halton_points, reg = lambd)
            row_sum = G.sum(axis = 1)
            scaled_G = G / row_sum[:, np.newaxis]
            r = np.matmul(scaled_G, halton_points)
            r_x = r[0:n]
            r_y = r[n:]
            #cost = - (mix_rbf_mmd2(r_x, r_y, sigma_list, biased = True))
            cost = - (energy_test_statistics(r_x, r_y))

            return cost
        
        problem = Problem(manifold=manifold, cost=cost, egrad = egrad, ehess = ehess, verbosity = 0)
        solver = SteepestDescent(logverbosity=2,  maxtime= 6000)
        #xoptimal, optlog = solver.solve(problem) 
        xoptimal, optlog = solver.solve(problem) 
        psRE.append(-optlog['final_values']['f(x)'])
        
        #if (i + 1)%10 ==0:print('simulation no: %i' %(i),  'sRE= %0.6f' %(sRE[i]), 'psRE = %0.6f' %(psRE[i]))
        if (i + 1)%1 ==0:print('simulation no: %i' %(i),  'sRE= %0.6f' %(sRE[i]), 'psRE = %0.6f' %(psRE[i]))
    results['sRE'] = sRE
    results['psRE'] = psRE
    
    psRE_dict[key] = results
            
#with open('15_30_psRE_dim_50_8.pkl','wb') as f:
#    pickle.dump(psRE_dict, f)            
          
        
        
    