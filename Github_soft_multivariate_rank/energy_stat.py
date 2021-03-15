import autograd.numpy as np
def pairwise_distance(X, Y):
    x_col = np.expand_dims(X, axis = 1)
    
    y_lin = np.expand_dims(Y, axis = 0)
    
    M = np.sqrt(np.sum((x_col - y_lin)**2 , 2))
    return M
 
def energy_test_statistics(X, Y):
    n = X.shape[0]
    m = Y.shape[0]
    coefficient = n * m / (n + m)
    xx = pairwise_distance(X + 1e-16, X) # to avoid 'divide by zero error'
    yy = pairwise_distance(Y + 1e-16 , Y)
    xy = pairwise_distance(X, Y)
    return coefficient * ( 2 * np.mean(xy) - np.mean(xx) - np.mean(yy))