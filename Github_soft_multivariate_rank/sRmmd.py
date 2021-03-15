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

def _mix_rbf_kernel(X, Y, sigma_list):
    m = X.shape[0]
    Z = np.concatenate((X, Y), 0)
    ZZT = np.matmul(Z, Z.T)
    diag_ZZT = np.expand_dims(np.diag(ZZT), 1)
    Z_norm_sqr = np.tile(diag_ZZT, ZZT.shape[0])
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.T

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += np.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)

def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.shape[0]
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = np.diag(K_XX)                       # (m,)
        diag_Y = np.diag(K_YY)                       # (m,)
        sum_diag_X = np.sum(diag_X)
        sum_diag_Y = np.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(axis =1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(axis =1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(axis =0)                     # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    return mmd2

def sinkhorn_stabilized(X, Y, reg, numItermax=5000, tau=1e3, stopThr=1e-9, warmstart=None):
    
    if X.shape[0] <= X.shape[1]:
        X = X.T
    
    x_col = np.expand_dims(X, axis = 1)
    y_lin = np.expand_dims(Y, axis = 0)
    M = np.sum((x_col - y_lin)**2 , 2)
    M =  M / M.max()
    
    a = np.ones(M.shape[0], ) / M.shape[0]
    b = np.ones(M.shape[0], ) / M.shape[0]
    dim_a = len(a)
    dim_b = len(b)
    u, v = np.ones(dim_a) / dim_a, np.ones(dim_b) / dim_b
    warmstart = None
    if warmstart is None:
        alpha, beta = np.zeros(dim_a), np.zeros(dim_b)
    else : 
        alpha, beta = warmstart

    def get_K(alpha, beta):
        return np.exp(-(M - alpha.reshape((dim_a, 1)) - beta.reshape((1, dim_b))) / reg)
    def get_Gamma(alpha, beta, u, v):
        return np.exp(-(M - alpha.reshape((dim_a, 1)) - beta.reshape((1, dim_b)))
                      / reg + np.log(u.reshape((dim_a, 1))) + np.log(v.reshape((1, dim_b))))
    K = get_K(alpha, beta)
    transp = K
    loop = 1
    cpt = 0
    err = 1  
    while loop:
        uprev = u
        vprev = v
        v = b / (np.dot(K.T, u) + 1e-16)
        u = a / (np.dot(K, v) + 1e-16)
        if np.abs(u).max() > tau or np.abs(v).max() > tau:
            alpha, beta = alpha + reg * np.log(u), beta + reg * np.log(v)
            u, v = np.ones(dim_a) / dim_a, np.ones(dim_b) / dim_b
            K = get_K(alpha, beta)
        transp = get_Gamma(alpha, beta, u, v)
        err = np.linalg.norm((np.sum(transp, axis=0) - b))
        if err <= stopThr:
                loop = False

        if cpt >= numItermax:
                loop = False

        if np.any(np.isnan(u)) or np.any(np.isnan(v)):
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break
        cpt = cpt + 1
    return get_Gamma(alpha, beta, u, v)

