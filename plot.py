import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as linalg
import time
from collections import defaultdict

from test import optimize
from nlp_problems import InvertedPendulum
import preconditioners


N = 20
h = 1e-5
max_iters = 10
problem = InvertedPendulum(N=N, h=h)

n = problem.nvars
x0 = np.random.randn(n)

def get_solver_fn(solver_name, tol=1e-3):
    def solver(A, b, x0, M=None, callback=None):
        if solver_name == 'cg':
            return linalg.cg(A, b, tol=tol, M=M, callback=callback)[0]
        elif solver_name == 'gmres':
            return linalg.gmres(A, b, tol=tol, M=M, callback=callback, callback_type='x')[0]
        elif solver_name == 'minres':
            return linalg.minres(A, b, tol=tol, M=M, callback=callback)[0]
    
    return solver

def get_preconditioner(name):
    if name == 'P1':
        return preconditioners.P1
    elif name == 'P2':
        return preconditioners.P2
    elif name == 'schoberl':
        return preconditioners.schoberl_precond
    
    return None

def incompatible(solver_name, precond_name):
    """
    Returns true if solver and preconditioner are not compatible.
    """
    if solver_name == 'minres':
        if precond_name in ['P1']:
            return True
    
    return False

solvers = ['cg', 'gmres', 'minres']
preconds = [None, 'P1']

outer_norm_info_for_solver = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for solver_name in solvers:
    for precond_name in preconds:
        if incompatible(solver_name, precond_name):
            continue

        solver = get_solver_fn(solver_name)
        Mfunc = get_preconditioner(precond_name)

        curr_iteration = [0]
        def outer_callback(x, lam):
            outer_norm_info_for_solver[solver_name][precond_name]['iteration'].append(curr_iteration[0])
            outer_norm_info_for_solver[solver_name][precond_name]['Lx'].append(np.linalg.norm(problem.g(x) - problem.G(x).T @ lam))
            outer_norm_info_for_solver[solver_name][precond_name]['c'].append(np.linalg.norm(problem.c(x)))
            curr_iteration[0] += 1

        t0 = time.time()
        zstar, norm_info = optimize(problem, x0, verbose=True, max_iters=max_iters, solver=solver,
                                    callback=outer_callback, Mfunc=Mfunc)

        # times = np.array(norm_info['time']) - t0

def plot_lag_grad_z():
    for solver_name in solvers:
        for precond_name in preconds:
            if incompatible(solver_name, precond_name):
                continue

            if precond_name == None:
                label = solver_name + ' (No precond.)'
            else:
                label = solver_name + ' ({} precond.)'.format(precond_name)

            norm_info = outer_norm_info_for_solver[solver_name][precond_name]
            iterations = norm_info['iteration']
            Lx = norm_info['Lx']
            plt.semilogy(iterations, Lx, label=label)

    plt.legend()
    plt.xlabel('Outer Iterations')
    plt.ylabel(r'$\|\nabla L_z(z, \lambda)\|$')
    plt.title('Gradient of Lagrangian w.r.t z' + ' (N = {})'.format(N))
    plt.show()

def plot_lag_grad_lam():
    for solver_name in solvers:
        for precond_name in preconds:
            if incompatible(solver_name, precond_name):
                continue

            if precond_name == None:
                label = solver_name + ' (No precond.)'
            else:
                label = solver_name + ' ({} precond.)'.format(precond_name)

            norm_info = outer_norm_info_for_solver[solver_name][precond_name]
            iterations = norm_info['iteration']
            c = norm_info['c']
            plt.semilogy(iterations, c, label=label)

    plt.legend()
    plt.xlabel('Outer Iterations')
    plt.ylabel(r'$\|\nabla L_\lambda(z, \lambda)\|$')
    plt.title('Gradient of Lagrangian w.r.t ' + r'$\lambda$' + ' (N = {})'.format(N))
    plt.show()

plot_lag_grad_z()
plot_lag_grad_lam()



