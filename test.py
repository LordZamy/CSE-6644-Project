import numpy as np
import scipy.sparse.linalg as linalg

from nlp_problems import InvertedPendulum


def optimize(problem, x0, eps=1e-8, max_iters=1000, callback=None, verbose=False, solver=linalg.spsolve):
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
        K = problem.KKT(x, lam)
        b = np.block([gx, cx])

        #print(f"b shape: {b.shape}")
        #print(f"K shape: {K.shape}")

        z = solver(K, b)

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


if __name__ == '__main__':

    N = 10
    problem = InvertedPendulum(N=N)

    n = problem.nvars
    x0 = np.random.randn(n)
    

    def callback(x):
        print(f"x[0] = {x[:2]}, x[end] = {x[-N-2:-N]}")
        print(f"||c(x)|| = {np.linalg.norm(problem.c(x))}")
        print(f"F(x) = {problem.F(x)}")
        
        
    def solver(A, b):
        return linalg.minres(A, b, tol=1e-4)[0]
        

    zstar = optimize(problem, x0, verbose=True, max_iters=20, solver=solver, callback=callback)
    pendulum.plot(zstar)
