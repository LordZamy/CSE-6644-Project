import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator

from scipy.sparse import spdiags
from nlp_problems import BaseProblem
import matplotlib.animation as animation

from scipy.sparse import eye

from scipy import sparse

class InvertedPendulum(BaseProblem):

    def __init__(self, N=10, h=1e-5):

        self.N = N  # n steps
        self.dt = 1./N  # time step

        self.h = h  # finite-difference step
        
        self.grav = 9.8
        self.m = self.l = 1  # mass and length of pole
        self.mu = 0.01  # friction constant

        self.state_dim = 2
        self.input_dim = 1

        # Number of variables
        self.n = (self.N+1)*self.state_dim + self.N*self.input_dim

        # Number of constraints
        self.m = (self.N+2)*self.state_dim

        
    @property
    def nvars(self):
        return self.n

    @property
    def nconstraints(self):
        return self.m

    
    def f(self, x, u):
        """
        x = [theta, theta_dot] (State variables?)
        u = torque (Decision variables?)
        """
        x1, x2 = x
        theta_dot = x2
        theta_ddot = self.grav / self.l * np.sin(x1) - self.mu/(self.m * self.l**2) * x2 + 1./(self.m * self.l**2)*u
        return np.array([theta_dot, theta_ddot])

    
    def F(self, z):
        r = 1e-5
        #r = 1
        #Q = np.array([[1, 0],[0,1]])

        xs = z[:-self.N]
        us = z[-self.N:]

        J = 0
        for i in range(self.N):
            #x = xs[i*self.state_dim:(i+1)*self.state_dim]
            J += r * us[i]**2 #+ x.T @ (Q @ x)
        J *= 0.5

        x1_final, x2_final = xs[-2], xs[-1]
        J += 0.5 * (10 * x1_final ** 2 + x2_final**2)

        return J

    
    def g(self, z):

        n = len(z)

        g_ = np.zeros_like(z)
        # dz = np.zeros_like(z)
        
        # for i in range(n):
        #     if i > 0:
        #         dz[i-1] = 0
        #     dz[i] = self.h
        #     g_[i] = 1./(2*self.h) * (self.F(z+dz) - self.F(z-dz))

        for i in range(1, self.N+1):
             g_[-i] = 1e-5 * z[-i]

        g_[-2 - self.N] = 10*z[-2 - self.N]
        g_[-1 - self.N] = z[-1 - self.N]

        return g_

    
    def c(self, z):

        c_ = np.zeros(self.m)

        xs = z[:-self.N]
        us = z[-self.N:]

        dt = 1. / self.N

        # Set dynamic constraints
        k = 0
        for i in range(self.N):
            xnext = xs[k+self.state_dim:k+2*self.state_dim]
            x = xs[k:k+self.state_dim]

            u = us[i]

            c_[k:k+self.state_dim] = xnext - x - dt*self.f(x,u)

            k += self.state_dim

        # ending theta & thetadot should be zero
        c_[-2*self.state_dim:-self.state_dim] = xs[-self.state_dim:]  

        # Starting theta at pi radians
        c_[-2] = xs[0] - np.pi
        
        # Starting thetadot at 0 rad/s
        c_[-1] = xs[1]

        return c_

    
    def G(self, z):

        def matvec(v):
            dv = self.h*v
            return 1./(2*self.h) * (self.c(z + dv) - self.c(z - dv))

        def rmatvec(v):

            res = np.zeros_like(z)
            e = np.zeros_like(z)
            for i in range(len(res)):
                if i > 0:
                    e[i-1] = 0
                e[i] = 1
                res[i] = np.dot(matvec(e), v)
                
            return res

        shape = (self.m, self.n)
        
        
        if(True):
            GG = np.zeros(shape)
            e = np.zeros_like(z)
            for i in range(len(e)):
                if i > 0:
                    e[i-1] = 0
                e[i] = 1
                GG[:,i] = matvec(e)
            
            #print(HH) 
            sG = sparse.csr_matrix(GG) 
            return sG

        
        
        return LinearOperator(shape, matvec=matvec, rmatvec=rmatvec)

        # n = len(z)
        # G_ = np.zeros((self.m, self.n), dtype=z.dtype)

        # dz = np.zeros(n)

        # for i in range(n):
        #     if i > 0:
        #         dz[i-1] = 0
        #     dz[i] = self.h

        #     G_[:,i] = 1./(2*self.h) * (self.c(z + dz) - self.c(z - dz))

        # return G_
        

    def H(self, z, lam):

        def G_L(z):
            return self.g(z) - self.G(z).T @ lam

        def matvec(v):
            dv = self.h*v
            return 1./(2*self.h) * (G_L(z+dv) - G_L(z-dv))
            #return 1./self.h * (G_L(z+dv) - G_L(z))

        n = len(z)
        
        
        if(True):
            HH = np.zeros((n,n))
            e = np.zeros(n)
            for i in range(len(e)):
                if i > 0:
                    e[i-1] = 0
                e[i] = 1
                HH[:,i] = matvec(e)
            
            #print(HH)  
            sH = sparse.csr_matrix(HH) 
            # vals, vecs = sparse.linalg.eigs(sH,k=6)
            # print(vals)
            return sH+0*eye(n)*10**-16
            
        return LinearOperator((n, n), matvec=matvec)

    
    def KKT(self, z, lam):
        
        n, m = len(z), len(lam)

        G = self.G(z)
        H = self.H(z, lam)
        
        def matvec(v):

            res = np.zeros(n+m, dtype=v.dtype)

            v1 = v[:n]
            v2 = v[n:]

            res[:n] = (H @ v1) + (G.T @ v2)
            res[n:] = G @ v1

            return res
            
        def rmatvec(v):

            res = np.zeros(n+m, dtype=v.dtype)

            v1 = v[:n]
            v2 = v[n:]

            res[:n] = (H.T @ v1) + (G.T @ v2)
            res[n:] = G @ v1
                
            return res
        shape = (n+m, n+m)
        return LinearOperator(shape, matvec=matvec,rmatvec=rmatvec)

    
    def plot(self, z):

        us = z[-self.N:]

        xs = np.zeros((self.N+1, self.state_dim), dtype=z.dtype)
        xs[0] = [np.pi, 0]

        for i in range(self.N):
            xs[i+1] = xs[i] + self.dt * self.f(xs[i], us[i])

        thetas = -xs[:,0] + np.pi/2

        # 

        fig = plt.figure()
        ax = plt.axes(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
        line, = ax.plot([], [], lw=3)
        
        N=self.N+1;
        def init():
            line.set_data([], [])
            return line,
        def animate(i):
            x = np.array([0,np.cos(thetas[i%N])])
            y = np.array([0,np.sin(thetas[i%N])])
            line.set_data(x, y)
            return line,

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=200, interval=100, blit=True)

        anim.save('sine_wave.gif', writer='imagemagick')
        
        
        plt.scatter(np.cos(thetas), np.sin(thetas))
        plt.scatter(np.cos(thetas[0]), np.sin(thetas[0]), label='start')
        plt.scatter(np.cos(thetas[-1]), np.sin(thetas[-1]), label='end')
        plt.axis('square')
        plt.ylim([-1.1, 1.1])
        plt.xlim([-1.1, 1.1])
        plt.legend()
        plt.show()
            
        
