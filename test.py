import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as linalg
from scipy.sparse import eye
import time

from scipy.sparse.linalg import spsolve

from nlp_problems import InvertedPendulum

num_iters=0

def optimize(problem, x0, eps=1e-8, max_iters=1000, callback=None, verbose=False, solver=linalg.spsolve, Mfunc=None):
    """
    Example function to optimize nonlinear programming problem
    """
    
    n, m = problem.nvars, problem.nconstraints

    alpha=0.2
    x = x0
    lam = np.random.randn(m)
    z=np.zeros(n+m)
    for it in range(max_iters):
        
        temp_t1 = time.time()
        if verbose:
            print(f"--iter: {it+1}")

        gx, cx = problem.g(x), problem.c(x)
        
        temp_t2= time.time()
        K = problem.KKT(x, lam)
        time_assembly= time.time()-temp_t2
        
        b = np.block([gx, cx])
        
        if Mfunc:
            M = Mfunc(problem, x, lam)
        else:
            M = None
            
            
        # sK_ilu=linalg.spilu(K); 
        # M = linalg.LinearOperator(K.shape, sK_ilu.solve)  
        global num_iters
        num_iters=0
        temp_t3= time.time()
        z = solver(K, b,z, M=M)
        time_solve= time.time()-temp_t3
        print(num_iters)
        p = -z[:n]
        lam = z[n:]

        x += alpha*p

        if callback:
            callback(x)

        if np.linalg.norm(p) < eps and np.linalg.norm(cx) < eps:
            break
        time_iteration= time.time()-temp_t1
        print("Time per iteration %.2f \t Time Assembly %.2f \t Time solve %.2f"\
        %(time_iteration,time_assembly,time_solve));
    if verbose:
        print(f"Optimized in {it-1} iterations")

    return x


    
    
def P1(problem, x, *args):
    """
    M = inv([[I B^T]
             [B  0 ]])
    """

    n, m = problem.nvars, problem.nconstraints
    shape = (m+n,m+n)

    G = problem.G(x)

    def matvec1(v):
        v1, v2 = v[:n], v[n:]
        res = np.zeros(m+n, dtype=v.dtype)
        res[:n] = v1 - G.T @ v2
        res[n:] = v2
        return res
    
    def matvec2(v):
        v1, v2 = v[:n], v[n:]
        res = np.zeros(m+n, dtype=v.dtype)
        res[:n] = v1
        res[n:] = linalg.cg(G @ G.T, -v2)[0]
        return res

    def matvec3(v):
        v1, v2 = v[:n], v[n:]
        res = np.zeros(m+n, dtype=v.dtype)
        res[:n] = v1
        res[n:] = -G @ v1 + v2
        return res
    
    L = linalg.LinearOperator(shape, matvec=matvec1)
    U = linalg.LinearOperator(shape, matvec=matvec2)
    R = linalg.LinearOperator(shape, matvec=matvec3)

    def matvec(v):
        return L @ (U @ (R @ v))

    LO = -linalg.LinearOperator(shape, matvec=matvec)
    #print(linalg.eigs(LO)[0])
    return LO

def P2(problem, x, lam):
    """
    M = inv([[I B^T]
             [B  0 ]])
    """

    n, m = problem.nvars, problem.nconstraints
    shape = (m+n,m+n)

    
    G = problem.G(x)
    A_s=G[:,:-problem.N]
    A_d=G[:,-problem.N:]
    
    H = problem.H(x, lam);
    Wss=H[:-problem.N,:-problem.N]
    Wsd=H[:-problem.N,-problem.N:]
    Wds=H[-problem.N:,:-problem.N]
    Wdd=H[-problem.N:,-problem.N:]
    def matvec_1_A_firstrow(v):
        v1, v2 = v[:n], v[n:]
        
        xs = v1[:-problem.N]
        us = v1[-problem.N:]
        
        #res = np.zeros(m+n, dtype=v.dtype)
        res = G @v1 
        vs = res[:-problem.N]
        vd = res[-problem.N:]
        return vs+vd
        
    def matvec_1_A_lastrow(v):
        v1, v2 = v[:n], v[n:]
        
        xs = v1[:-problem.N]
        us = v1[-problem.N:]
        
        #res = np.zeros(m+n, dtype=v.dtype)
        res = A_s.T @v2 
        return res
        
        
    def matvec2(v):
        v1, v2 = v[:n], v[n:]
        res = np.zeros(m+n, dtype=v.dtype)
        res[:n] = v1
        res[n:] = linalg.cg(G @ G.T, -v2)[0]
        return res

    def matvec3(v):
        v1, v2 = v[:n], v[n:]
        res = np.zeros(m+n, dtype=v.dtype)
        res[:n] = v1
        res[n:] = -G @ v1 + v2
        return res
    
    L = linalg.LinearOperator(shape, matvec=matvec1)
    U = linalg.LinearOperator(shape, matvec=matvec2)
    R = linalg.LinearOperator(shape, matvec=matvec3)

    def matvec(v):
        return L @ (U @ (R @ v))

    LO = -linalg.LinearOperator(shape, matvec=matvec)
    #print(linalg.eigs(LO)[0])
    return LO
    
def P_simple(problem, x, lam):
    """
    M = inv([[I B^T]
             [B  0 ]])
    """
    K = problem.KKT(x, lam)
    n, m = problem.nvars, problem.nconstraints
    shape = (m+n,m+n)

    
    G = problem.G(x)

    
    H = problem.H(x, lam);
    #sK_ilu=linalg.spilu(H); 
    
    def matvec(v):
        v1, v2 = v[:n], v[n:]
        
        res = np.zeros(m+n, dtype=v.dtype)
        res[:n] = v1
        res[n:] = v2
        res=spsolve(K,v)
        return res
        
    

    LO = linalg.LinearOperator(shape, matvec=matvec)
    #print(linalg.eigs(LO)[0])
    return LO 
    



if __name__ == '__main__':

    N = 20
    h = 1e-5
    problem = InvertedPendulum(N=N, h=h)

    n = problem.nvars
    x0 = np.random.randn(n)
    
    
    
    def callback(x):
        print(f"x[0] = {x[:2]}, x[end] = {x[-N-2:-N]}")
        print(f"||c(x)|| = {np.linalg.norm(problem.c(x))}")
        print(f"F(x) = {problem.F(x)}")
        
    def solver_n_iter_callback(xk):
        global num_iters
        num_iters += 1
    # Modify this to change the solver. Maybe some globalization strategies can be used.
    def solver(A, b,x0, M=None):
        return linalg.minres(A, b,x0, tol=1e-6, M=M, callback=solver_n_iter_callback)[0]
        # return spsolve(A,b)

    Mfunc = P1
    Mfunc = None

    t = time.time()
    
    zstar = optimize(problem, x0, verbose=True, max_iters=100, solver=solver, callback=callback,
                     Mfunc=Mfunc)

    print(f"Total time: {time.time() - t}")
    
    problem.plot(zstar)

    ######################################
    # Some analysis of matrix properties #
    ######################################

    def e_i(n, i):
        e = np.zeros(n)
        e[i] = 1
        return e
    
    # Calculate jacobian of constraints
    G_ = problem.G(x0)
    G = np.zeros(G_.shape)

    for i in range(problem.nconstraints):
        G[:,i] = G_ @ e_i(problem.nvars, i)
        
    print(f"shape(G) = {G.shape}")
    print(f"rank(G) = {np.linalg.matrix_rank(G)}")

    # Calculate Hessian of lagrangian
    lam = np.random.randn(len(G))
    H_ = problem.H(x0, lam)
    H = np.zeros(H_.shape)

    for i in range(len(H)):
        H[:,i] = H_ @ e_i(len(H), i)

    eigvals_H = np.real(np.linalg.eigvals(H))
    abseigvals_H = np.abs(eigvals_H)
    
    print(f"shape(H) = {np.shape(H)}")
    print(f"rank(H) = {np.linalg.matrix_rank(H)}")
    print(f"num zero eigs: {sum(abseigvals_H < 1e-9)}")
    print(f"num neg eigs = {sum(eigvals_H < 0)}")
    print(f"num pos eigs = {sum(eigvals_H > 0)}")

    # Calculate KKT matrix
    K = problem.KKT(x0, lam)
    if Mfunc:
        K = Mfunc(problem, x0, lam) @ K
        
    eigvals = np.real(linalg.eigs(K, k = problem.nvars + problem.nconstraints - 2)[0])
    abseigvals = np.abs(eigvals)
    
    print(f"shape(K) = {K.shape}")
    print(f"cond(K) = {np.max(abseigvals)/np.min(abseigvals)}")
    print(f"num zero eigs = {sum(abseigvals < 1e-9)}")
    print(f"num neg eigs = {sum(eigvals < 0)}")
    print(f"num pos eigs = {sum(eigvals > 0)}")
    
