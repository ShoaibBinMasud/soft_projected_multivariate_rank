import autograd.numpy as np
  
def lse_np(v_ij_np):
    V_i_np = np.max(v_ij_np, 1).reshape(-1, 1)
    c_np = V_i_np + np.log(np.sum(np.exp((v_ij_np - V_i_np)), 1)).reshape(-1, 1)
    return c_np

def Sinkhorn_ops_np(p, eps, x_i_np, y_j_np):
    x_y_np = np.expand_dims(x_i_np, axis = 1) - np.expand_dims(y_j_np, axis = 0)
    C_e_np = np.sum(x_y_np**2 , 2)/eps
    CT_e_np = C_e_np.T
    
    def S_x_np(f_i): return  -lse_np(f_i.reshape(1, -1) - CT_e_np) 

    def S_y_np(f_j): return   -lse_np(f_j.reshape(1, -1) - C_e_np)  
    return S_x_np, S_y_np

def sink_np(a_i_np, x_i_np, b_j_np, y_j_np, p=1, eps=.1, nits = 5000, tol=1e-9, assume_convergence=True):
    a_i_log_np, b_j_log_np = np.log(a_i_np), np.log(b_j_np) 
    B_i_np, A_j_np = np.zeros_like(a_i_np), np.zeros_like(b_j_np)

    S_x_np, S_y_np = Sinkhorn_ops_np(p, eps, x_i_np, y_j_np)
    for i in range(nits-1):
        B_i_prev_np = B_i_np
        A_j_np = S_x_np(B_i_np.reshape(1, -1) + a_i_log_np.reshape(1, -1))   # a(y)/ε = Smin_ε,x~α [ C(x,y) - b(x) ]  / ε
        B_i_np = S_y_np(A_j_np.reshape(1, -1) + b_j_log_np.reshape(1, -1))   # b(x)/ε = Smin_ε,y~β [ C(x,y) - a(y) ]  / ε
        err = eps * np.abs((B_i_np - B_i_prev_np)).mean()  # Stopping criterion: L1 norm of the updates
        if err < tol:
            break
    a_y_np, b_x_np = eps*A_j_np.reshape(-1), eps*B_i_np.reshape(-1)
    return a_y_np, b_x_np

def plan_np(a_i_np, x_i_np, b_j_np, y_j_np, p=2, eps=1e-2):
    x_y_np = np.expand_dims(x_i_np, axis = 1) - np.expand_dims(y_j_np, axis = 0)
    C_e_np = np.sum(x_y_np**2 , 2)/eps
    a_y_np, b_x_np  = sink_np(a_i_np, x_i_np, b_j_np, y_j_np, p, eps)
    a_ib_j_np = np.matmul(np.expand_dims(a_i_np, axis = 1) , np.expand_dims(b_j_np, axis = 0))
    plan_np = np.exp((np.expand_dims(b_x_np, axis = 1) + np.expand_dims(a_y_np, axis = 0) - C_e_np *eps)/eps) * a_ib_j_np
    return plan_np