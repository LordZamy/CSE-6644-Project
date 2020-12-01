import numpy as np
import pyamg
import scipy.sparse.linalg as linalg
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix, eye

def is_pos_semidef(x):
    print(np.linalg.eigvals(x))
    return np.all(np.linalg.eigvals(x) >= 0)

def P1(problem, x, *args):
    """
    M = inv([[I B^T]
             [B  0 ]])

    See: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.140.1151&rep=rep1&type=pdf
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
        res[n:] = linalg.spsolve(G @ G.T, -v2)
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

    LO = linalg.LinearOperator(shape, matvec=matvec)
    #print(linalg.eigs(LO)[0])
    return LO


def P2(problem, x, lam, *args):
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


def P_simple(problem, x, lam, *args):
    """
    M = inv(K)
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


def bramble_precond(problem, x, lam, *args):
    """
    Should result in a S.P.D. KKT matrix

    See: https://www.ams.org/journals/mcom/1988-50-181/S0025-5718-1988-0917816-8/S0025-5718-1988-0917816-8.pdf
    """
    H = problem.H(x, lam)
    n = H.shape[0]
    # print(H)
    # smallest_value = min(H.data)
    # if smallest_value <= 0:
    #     H = H + (-smallest_value + 1) * eye(n)
    smallest_value = min(H.diagonal().data)
    if smallest_value <= 0:
        H = H + (-smallest_value + 1.0) * eye(n)
    
    # print(is_pos_semidef(H.toarray()))

    # Hprecond = pyamg.smoothed_aggregation_solver(H).aspreconditioner()
    Hprecond = linalg.inv(H)
    #Hprecond = linalg.spsolve(H, eye(len(x)))
    G = problem.G(x)
    
    n, m = len(x), len(lam)
    
    def matvec(v):
        v1 = v[:n]
        v2 = v[n:]

        res = np.zeros((m+n))

        #res[:n] = v1
        #res[n:] = G @ (Hprecond @ (G.T @ v2))
        res[:n] = Hprecond @ v1
        res[n:] = G @ res[:n] - v2

        return res

    shape = (m+n, m+n)
    return linalg.LinearOperator(shape, matvec=matvec)


def schoberl_precond(problem, x, lam):
    """
    Not complete, not sure what this does exactly

    See: https://www.asc.tuwien.ac.at/~schoeberl/wiki/publications/zulehner_2final.pdf
    """

    H = problem.H(x, lam)

    Hprecond = pyamg.smoothed_aggregation_solver(H).aspreconditioner()
    G = problem.G(x)
    
    n = len(x)
    m = len(lam)
    
    def matvec(v):
        v1 = v[:n]
        v2 = v[n:]

        res = np.zeros(n+m)
        res[:n] = Hprecond @ v1 + G.T @ v2
        res[n:] = G @ v1 + G @ (Hprecond @ (G.T @ v2))

        return res

    shape = (m+n,m+n)
    return linalg.LinearOperator(shape, matvec=matvec)


def block_diag(problem, x, lam):
    """
    For use with MINRES, since it should maintain symmetry pretty well

    See; https://epubs.siam.org/doi/pdf/10.1137/100798491?download=true
    """
    H = problem.H(x, lam)
    n = len(x)
    smallest_value = min(H.diagonal().data)
    if smallest_value <= 0:
        H = H + (-smallest_value + 1.0) * eye(n)

    Hinv = linalg.spsolve(H, eye(len(x), format='csc'))
    G = problem.G(x)

    n = len(x)
    m = len(lam)
    
    def matvec(v):
        v1 = v[:n]
        v2 = v[n:]

        res = np.zeros(n+m)

        res[:n] = Hinv @ v1
        res[n:] = spsolve(G @ Hinv @ G.T, v2)

        return res

    shape = (m+n,m+n)
    return linalg.LinearOperator(shape, matvec=matvec)
    
    
def block_diag_minres(problem, x, lam):
    """
    For use with MINRES, since it should maintain symmetry pretty well

    See; https://epubs.siam.org/doi/pdf/10.1137/100798491?download=true
    """
    H = problem.H(x, lam)
    n = len(x)
    # print(is_pos_semidef(H.toarray())) 
    smallest_value = min(H.diagonal().data)
    if smallest_value <= 0:
        H = H + (-smallest_value + 1.0) * eye(n)

    Hinv = linalg.spsolve(H, eye(len(x), format='csc'))
    G = problem.G(x)

    n = len(x)
    m = len(lam)
    
def composed(problem, x, lam):
    H = problem.H(x, lam)
    Hinv = linalg.spsolve(H, eye(len(x), format='csc'))
    G = problem.G(x)

    n = len(x)
    m = len(lam)
    shape = (m+n, m+n)
    
    def matvec_block(v):
        v1 = v[:n]
        v2 = v[n:]

        res = np.zeros(n+m)

        res[:n] = Hinv @ v1
        res[n:] = spsolve(G @ Hinv @ G.T, v2)

        return res

    P_block=linalg.LinearOperator(shape, matvec=matvec_block)
    
    
    Hprecond = pyamg.smoothed_aggregation_solver(H).aspreconditioner()
    #Hprecond = linalg.spsolve(H, eye(len(x)))
    
    def matvec_bramble(v):
        v1 = v[:n]
        v2 = v[n:]

        res = np.zeros((m+n))

        #res[:n] = v1
        #res[n:] = G @ (Hprecond @ (G.T @ v2))
        res[:n] = Hprecond @ v1
        res[n:] = G @ res[:n] - v2

        return res
        
    P_bramble=linalg.LinearOperator(shape, matvec=matvec_bramble)
    
    def matvec(v):
        return  P_block @ (P_bramble @ v)

    
    
    return linalg.LinearOperator(shape, matvec=matvec)