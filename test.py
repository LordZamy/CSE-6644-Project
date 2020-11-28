import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as linalg
from scipy.sparse.linalg import spsolve
from scipy.sparse import eye
import time

from nlp_problems import InvertedPendulum

from nlp_problems import InvertedPendulum_cart
import preconditioners


def optimize(problem, x0, eps=1e-4, max_iters=1000, callback=None, verbose=False, solver=linalg.spsolve, Mfunc=None):
    """
    Example function to optimize nonlinear programming problem
    """
    
    n, m = problem.nvars, problem.nconstraints

    alpha=0.1
    x = x0
    lam = np.random.randn(m)
    z = np.zeros(n+m)

    # For storing residuals in krylov callback
    norm_info = dict(time=[], c=[], Lx=[], clin=[], Lxlin=[],Res=[],num_Iter=[])
    
    
    #Finite Difference check of gradient
    # gx = problem.g(x)
    # F0=problem.F(x)
    # dx=0.001
    # for i in range(len(x)):
        # x[i]+=dx
        # F=problem.F(x)
        # dF=(F-F0)/dx
        # Error=(gx[i]-dF)/dF
        # x[i]-=dx
        # print("gx %.3e  dF %.3e  Error %.3e"%(gx[i],dF,Error))
    
    for it in range(max_iters):
        
        temp_t1 = time.time()
        
        if verbose:
            print(f"\n--iter: {it+1}")

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
        
        num_iters = [0]
        
        temp_t3= time.time()

        # TODO - determine if we're getting the proper residuals
        cur_x = [x]
        def solver_callback(xk):
            xn = cur_x[0] - alpha*xk[:n]
            #cur_x[0] = xn
            
            lam = xk[n:]
            
            num_iters[0] += 1
            norm_info['time'].append(time.time())
            norm_info['c'].append(np.linalg.norm(problem.c(xn)))
            norm_info['Lx'].append(np.linalg.norm(problem.g(xn) - problem.G(xn).T @ lam))

            res = (K @ xk - b)/np.linalg.norm(b)
            
            
            norm_info['Lxlin'].append(np.linalg.norm(res[:n]))
            norm_info['clin'].append(np.linalg.norm(res[n:]))
            norm_info['Res'].append(np.linalg.norm(res))
        
        z = solver(K, b, z, M=M, callback=solver_callback)
        
        norm_info['num_Iter'].append(num_iters)
        
        print(f"---num solver iters: {num_iters[0]}")
        
        time_solve= time.time()-temp_t3
        
        p = -z[:n]
        lam = z[n:]


        L0 = np.linalg.norm(problem.c(x))
        F0 = problem.F(x)
        
        L=L0+1
        l_M=1.;
        l_m=0.0001
        alpha=l_M
        while ((l_M - l_m) / 2 > 1.e-4):
            alpha=(l_M+l_m)/2
            x_ = x + alpha*p
            L = np.linalg.norm(problem.c(x_))
            F = problem.F(x_)

            if L0 < 1.e-4:
                if L > L0 and F > F0:
                    l_M = alpha
                else:
                    l_m = alpha
            else:
                if L > L0:
                    l_M = alpha
                else:
                    L0 = L
                    l_m = alpha


        print(alpha)    
        x=np.copy(x_)
        
        
        
        if callback:
            callback(x, lam)

        if np.linalg.norm(p) < eps and np.linalg.norm(cx) < eps:
            break
        
        time_iteration = time.time()-temp_t1
        
        print(f"Time per iteration: {time_iteration:.2f} \t Time Assembly: {time_assembly:.2f} \t Time solve: {time_solve:.2f}")
        
    if verbose:
        print(f"Optimized in {it-1} iterations")

    return x, norm_info


    
if __name__ == '__main__':

    PLOT = True
    
    N = 10
    h = 1e-5
    problem = InvertedPendulum(N=N, h=h)

    n = problem.nvars
    x0 = np.random.randn(n)

    t0 = time.time()
    
    def outer_callback(x, lam):
        with np.printoptions(precision=3):
            print(f"x[0] = {x[:2]}, x[end] = {x[-N-2:-N]}")
            print(f"||c(x)|| = {np.linalg.norm(problem.c(x)):.5f}")
            print(f"F(x) = {problem.F(x):.2f}")
            print(f"time = {time.time() - t0:.5f}")
            print(problem.c(x)[-10:])
            print(x[-10-N:-N])
    
    # Modify this to change the solver. Maybe some globalization strategies can be used.
    def solver(A, b,x0, M=None, callback=None):
        #return linalg.gmres(A, b, tol=1e-3, M=M, callback=callback, callback_type='x')[0]
        return linalg.cg(A, b, tol=1e-3, M=M, callback=callback)[0]
        # return linalg.minres(A, b, tol=1e-4, M=M, callback=callback)[0]
    
    # Mfunc = preconditioners.bramble_precond
    Mfunc = preconditioners.P1
    # Mfunc = preconditioners.block_diag
    # Mfunc = preconditioners.schoberl_precond
    # Mfunc = preconditioners.diag_hessian
    # Mfunc = None
    #Mfunc = lambda problem, x, lam: eye(problem.nvars + problem.nconstraints)

    zstar, norm_info = optimize(problem, x0, verbose=True, max_iters=200, solver=solver,
                                callback=outer_callback, Mfunc=Mfunc)

    print(f"Total time: {time.time() - t0}")

    
    if PLOT:
        problem.plot(zstar)

    times = np.array(norm_info['time']) - t0
    
    
    
    
    
    
    plt.semilogy(times, norm_info['c'], label='c(x)')
    plt.semilogy(times, norm_info['clin'], 'x--', label='c+Gp')
    plt.legend()
    plt.xlabel("time (s)")
    plt.show()
    
    index_iter=(np.cumsum(norm_info['num_Iter'])-1).astype(int)
    plt.semilogy(np.array(norm_info['c'])[index_iter], label='c(x) iter')
    plt.semilogy( np.array(norm_info['clin'])[index_iter], 'x--', label='c+Gp iter')
    
    plt.legend()
    plt.xlabel("Optimization Iteration")
    plt.show()
    plt.semilogy(times, norm_info['Lx'], label=r'$L_x$')
    plt.semilogy(times, norm_info['Lxlin'], 'x--', label=r'$g-G^T\lambda + Hp$')
    plt.semilogy(times, norm_info['Res'], 'x--', label=r'Total Residue')
    plt.xlabel("time (s)")
    plt.legend()
    plt.show()
    
    

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
    H_ = problem.H(zstar, lam)
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
        M_ = Mfunc(problem, x0, lam)
        M = np.zeros(M_.shape)
        for i in range(M_.shape[1]):
            M[:,i] = M_ @ e_i(M_.shape[0], i)
        K = M @ K
        #K = Mfunc(problem, x0, lam) @ K
        
    eigvals = np.real(linalg.eigs(K, k = problem.nvars + problem.nconstraints - 2)[0])
    abseigvals = np.abs(eigvals)
    
    print(f"shape(K) = {K.shape}")
    print(f"cond(K) = {np.max(abseigvals)/np.min(abseigvals)}")
    print(f"num zero eigs = {sum(abseigvals < 1e-9)}")
    print(f"num neg eigs = {sum(eigvals < 0)}")
    print(f"num pos eigs = {sum(eigvals > 0)}")
    
