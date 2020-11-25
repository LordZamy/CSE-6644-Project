import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as linalg
from scipy.sparse import eye
import time

from scipy.sparse.linalg import spsolve

from nlp_problems import InvertedPendulum


def optimize(problem, x0, eps=1e-8, max_iters=1000, callback=None, verbose=False, solver=linalg.spsolve, Mfunc=None):
    """
    Example function to optimize nonlinear programming problem
    """
    
    n, m = problem.nvars, problem.nconstraints

    alpha=0.7
    x = x0
    lam = np.random.randn(m)

    for it in range(max_iters):

        if verbose:
            print(f"--iter: {it+1}")

        gx, cx = problem.g(x), problem.c(x)
        K = problem.KKT(x, lam)
        
        b = np.block([gx, cx])

        if Mfunc:
            M = Mfunc(problem, x, lam)
        else:
            M = None
            
        z = solver(K, b, M=M)

        p = -z[:n]
        lam = z[n:]

        x += alpha*p

        if callback:
            callback(x)

        if np.linalg.norm(p) < eps and np.linalg.norm(cx) < eps:
            break

    if verbose:
        print(f"Optimized in {it-1} iterations")

    return x

def optimize_cg(problem, x0, eps=1e-8, max_iters=1000, callback=None, verbose=False, solver=linalg.spsolve, Mfunc=None):
    """
    Example function to optimize nonlinear programming problem
    """
    
    n, m = problem.nvars, problem.nconstraints

    x = x0
    lam = np.random.randn(m)

    for it in range(max_iters):

        if verbose:
            print(f"--iter: {it+1}")

        gx, cx = problem.g(x), problem.c(x)
        
        
        F = np.block([gx, cx])

        G = problem.G(x)
        H = problem.H(x, lam)+eye(n)*10**-6
        
        
        # print(n)
        # print(H.toarray().shape)
        H0=H;    
        z = cg_positive(problem,H,G,H0,F)

        p = -z[:n]
        lam = z[n:]

        x += p

        if callback:
            callback(x)

        if np.linalg.norm(p) < eps and np.linalg.norm(cx) < eps:
            break

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
 
def cg_positive(problem,H,G,H0,F):
    n, m = problem.nvars, problem.nconstraints
    F1, F2= F[:n], F[n:]
    print("antes")
    F_1=spsolve(H0,F1)
    print("depois")
    F_2=G@F_1-F2
    F_=np.concatenate([F_1,F_2],axis=0);
    
    Z=np.zeros_like(F);
    P=R=F_-M_product_positive(problem, H, G,H0,Z)
    
    rsold=(inner_product_positive_F(problem, H, G,H0,F,F_,P)-inner_product_positive_M(problem, H, G,H0,Z,P))
    for i in range(10000):
        print(i)
        P_P=inner_product_positive_M(problem, H, G,H0,P,P)
        alpha=rsold/P_P
        Z=Z+alpha*P
        R=F_-M_product_positive(problem, H, G,H0,Z)
        
        rsnew=(inner_product_positive_F(problem, H, G,H0,F,F_,P)-inner_product_positive_M(problem, H, G,H0,Z,P))
        if(rsnew<10**-6):
            break
        beta=rsnew/rsold
        P=R+beta*P
        rsold=np.copy(rsnew)
        print(rsnew)
        
    return Z
 
def M_product_positive(problem, H, G,H0,v):
    #https://www.ams.org/journals/mcom/1988-50-181/S0025-5718-1988-0917816-8/S0025-5718-1988-0917816-8.pdf
    """
    """
    n, m = problem.nvars, problem.nconstraints
    X, Y = v[:n], v[n:]
    
    A_U=H@X;
    A0_inv_A_U= spsolve(H0,A_U)
    
    B_T_V=G.T@Y 
    A0_inv_B_T_V=spsolve(H0,B_T_V)
    
    R1=A0_inv_A_U+A0_inv_B_T_V
    
    R2_1=G@spsolve(H0,(H@X-H0@X))
    R2_2=G@spsolve(H0,G.T@Y)
    
    R2=R2_1+R2_2
    
    R=np.concatenate([R1,R2],axis=0);
    return R;
 
def inner_product_positive_F(problem, H, G,H0,F,F_,v2):
    #https://www.ams.org/journals/mcom/1988-50-181/S0025-5718-1988-0917816-8/S0025-5718-1988-0917816-8.pdf
    """

    """
    n, m = problem.nvars, problem.nconstraints
    U = F[:n]
    U_, V_ = F_[:n], F_[n:]
        
    W, X = v2[:n], v2[n:]
    
    shape = (m+n,m+n)


    ################
    A_U=H@U_;
    
    first=(A_U).dot(W)-U.dot(W)
    
    ###################    
    
    second=(V_).dot(X)

    
    return first+second
 
def inner_product_positive_M(problem, H, G,H0,v1,v2):
    #https://www.ams.org/journals/mcom/1988-50-181/S0025-5718-1988-0917816-8/S0025-5718-1988-0917816-8.pdf
    """

    """
    n, m = problem.nvars, problem.nconstraints
    U, V = v1[:n], v1[n:]
    
    W, X = v2[:n], v2[n:]
    
    shape = (m+n,m+n)


    ################
    A_U=H@U;
    
    A0_inv_A_U=spsolve(H0,A_U)
    
    A_A0_inv_A_U=H@A0_inv_A_U
    
    first=(A_A0_inv_A_U-A_U).dot(W)
    
    ###################    
    B_T_V=G.T@V
    
    A0_inv_B_T_V=spsolve(H0,B_T_V)
    
    second=(A0_inv_B_T_V-B_T_V).dot(W)
    ####################

    B_U=G@U
    
    B_A0_inv_A_U=G@spsolve(H0,H@U)
    
    third=(B_A0_inv_A_U-B_U).dot(X)
    
    
    #############################
    
    B_A0_inv_B_T=G@spsolve(H0,G.T@V)
    
    fourth=B_A0_inv_B_T.dot(X)

    
    return first+second+third+fourth


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

    # Modify this to change the solver. Maybe some globalization strategies can be used.
    def solver(A, b, M=None):
        return linalg.minres(A, b, tol=1e-6, M=M)[0]


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
    
