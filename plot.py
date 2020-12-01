import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as linalg
import time
from collections import defaultdict

from test import optimize
from nlp_problems import InvertedPendulum, InvertedPendulum_cart
import preconditioners


N = 40
h = 1e-5
max_iters = 20
problem = InvertedPendulum_cart(N=N, h=h)

n = problem.nvars
x0 = np.random.randn(n)

def get_solver_fn(solver_name, tol=1e-4):
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
    elif name == 'bramble':
        return preconditioners.bramble_precond
    elif name == 'block_diag':
        return preconditioners.block_diag
    elif name == 'block_diag_minres':
        return preconditioners.block_diag_minres
    
    return None

def incompatible(solver_name, precond_name):
    """
    Returns true if solver and preconditioner are not compatible.
    """
    if solver_name == 'minres':
        if precond_name in ['P1', 'bramble', 'block_diag']:
            return True
    if solver_name == 'cg':
        if precond_name in ['block_diag_minres']:
            return True
    if solver_name == 'gmres':
        if precond_name in ['block_diag_minres']:
            return True
    
    return False

solvers = [ 'minres', 'cg', 'gmres']
# solvers = ['gmres']
# preconds = [None, 'P1', 'block_diag']
preconds = ['block_diag_minres', 'P1', 'block_diag', None, 'bramble']
# preconds = ['bramble']

# max_iters_dict = {'cg': {'bramble': 2, None: 5}, 'gmres': {None: 1, 'bramble': 1}, 'minres': {None: 15}}
max_iters_dict = {}

outer_norm_info_for_solver = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
inner_norm_info_for_solver = defaultdict(lambda: defaultdict(dict))
for solver_name in solvers:
    for precond_name in preconds:
        if incompatible(solver_name, precond_name):
            continue

        print(solver_name, precond_name)

        solver = get_solver_fn(solver_name)
        Mfunc = get_preconditioner(precond_name)

        curr_iteration = [0]
        def outer_callback(x, lam):
            outer_norm_info_for_solver[solver_name][precond_name]['iteration'].append(curr_iteration[0])
            outer_norm_info_for_solver[solver_name][precond_name]['Lx'].append(np.linalg.norm(problem.g(x) - problem.G(x).T @ lam))
            outer_norm_info_for_solver[solver_name][precond_name]['c'].append(np.linalg.norm(problem.c(x)))
            curr_iteration[0] += 1
        
        if solver_name in max_iters_dict and precond_name in max_iters_dict[solver_name]:
            max_it = max_iters_dict[solver_name][precond_name]
        else:
            max_it = max_iters

        t0 = time.time()
        zstar, norm_info = optimize(problem, x0, verbose=True, max_iters=max_it, solver=solver,
                                    callback=outer_callback, Mfunc=Mfunc)

        inner_norm_info_for_solver[solver_name][precond_name] = norm_info
        times = np.array(norm_info['time']) - t0
        inner_norm_info_for_solver[solver_name][precond_name]['time'] = times
        # print(inner_norm_info_for_solver[solver][precond_name])

# print(inner_norm_info_for_solver)

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

    plt.legend(loc='upper right')
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

    plt.legend(loc='upper right')
    plt.xlabel('Outer Iterations')
    plt.ylabel(r'$\|\nabla L_\lambda(z, \lambda)\|$')
    plt.title('Gradient of Lagrangian w.r.t ' + r'$\lambda$' + ' (N = {})'.format(N))
    plt.show()

def plot_lag_grad_vs_time():
    for solver_name in solvers:
        for precond_name in preconds:
            if incompatible(solver_name, precond_name):
                continue

            if precond_name == None:
                label = solver_name + ' (No precond.)'
            else:
                label = solver_name + ' ({} precond.)'.format(precond_name)

            norm_info = inner_norm_info_for_solver[solver_name][precond_name]
            times = norm_info['time']
            # plt.semilogy(times, norm_info['c'])
            plt.semilogy(times, norm_info['clin'], 'x--', label=label)

    plt.legend(loc='upper right')
    plt.xlabel("time (s)")
    plt.ylabel(r'$\|c(z)\|$')
    plt.title('Norm of constraints in inner loops' + ' (N = {})'.format(N))
    plt.show()

def plot_residual_vs_time():
    for solver_name in solvers:
        for precond_name in preconds:
            if incompatible(solver_name, precond_name):
                continue

            if precond_name == None:
                label = solver_name + ' (No precond.)'
            else:
                label = solver_name + ' ({} precond.)'.format(precond_name)

            norm_info = inner_norm_info_for_solver[solver_name][precond_name]
            times = norm_info['time']
            plt.semilogy(times, norm_info['Res'], 'x--', label=label)

    plt.legend(loc='upper right')
    plt.xlabel("time (s)")
    plt.ylabel(r'$\|Res\|$')
    plt.title('Norm of Residual in inner loops' + ' (N = {})'.format(N))
    plt.show()


plot_lag_grad_z()
plot_lag_grad_lam()
plot_lag_grad_vs_time()
plot_residual_vs_time()



