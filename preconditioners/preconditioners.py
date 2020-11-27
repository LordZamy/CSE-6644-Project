import numpy as np
import scipy.sparse.linalg as linalg
from scipy.sparse.linalg import spsolve


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
