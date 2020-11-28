import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator

from scipy.sparse import spdiags
from nlp_problems import BaseProblem
import matplotlib.animation as animation

from scipy.sparse import eye

from scipy import sparse

class InvertedPendulum_cart(BaseProblem):

    def __init__(self, N=10, h=1e-5):

        self.N = N  # n steps
        max_time=1.
        self.dt = max_time/N  # time step
        
        self.time = np.linspace(0,1.2*max_time,np.int(1.2*N))
        
        self.h = h  # finite-difference step
        
        self.grav = 0
        self.m_p = 1  # mass  
        self.l = 2 #length of pole
        self.mu = 0.01  # friction constant

        self.m_cart=10
        self.e=self.m_p/(self.m_cart+self.m_p)
        self.state_dim = 4
        self.input_dim = 1

        # Number of variables
        self.n = (self.N+1)*self.state_dim + self.N*self.input_dim

        # Number of constraints
        self.m = (self.N+2)*self.state_dim

        self.r = 1e-2
        self.q = 100

        
    @property
    def nvars(self):
        return self.n

    @property
    def nconstraints(self):
        return self.m

    
    def f(self, x, u):
        """
        x = [theta, theta_dot] (State variables)
        u = torque (Decision variables)
        """
        x1, x2,x3,x4 = x
        y_dot=x2
        v_dot=-1*np.sin(x3)*self.e+u
        theta_dot = x4
        theta_ddot = 1*np.sin(x3)-u  +   self.grav / self.l * np.cos(x3)*self.e - self.mu/( self.l**2)*self.e * x4
        return np.array([y_dot,v_dot,theta_dot, theta_ddot])

    
    def df(self, x, u):
        """
        x = [theta, theta_dot] (State variables)
        u = torque (Decision variables)
        """
        x1, x2,x3,x4 = x
        
        y_dot_dx=np.array([0,1,0,0]);
        v_dot_dx=np.array([0,0,-self.e*(np.sin(x3)*0+1*np.cos(x3)),0]);
        theta_dot_dx=np.array([0,0,0,1]);
        theta_ddot_dx =np.array([0,0,(np.sin(x3)*0+1*np.cos(x3))-self.grav / self.l * np.sin(x3)*self.e,- self.mu/( self.l**2)*self.e]);
        
        y_dot_du=np.array([0]);
        v_dot_du=np.array([1]);
        theta_dot_du=np.array([0]);
        theta_ddot_du =np.array([-1]);
        
        #
        dot_dx=np.vstack((y_dot_dx,v_dot_dx,theta_dot_dx,theta_ddot_dx))
        dot_du=np.concatenate((y_dot_du,v_dot_du,theta_dot_du,theta_ddot_du))
        return dot_dx,dot_du
    
    
    def F(self, z):
        r = self.r
        Q = self.q * np.eye(self.state_dim)

        xs = z[:-self.N]
        us = z[-self.N:]

        J = 0
        for i in range(self.N):
            x = xs[i*self.state_dim:(i+1)*self.state_dim]
            J += r * us[i]**2 + x.T @ (Q @ x)
        J *= 0.5

        y_final, v_final,theta_final, q_final =xs[-4], xs[-3], xs[-2], xs[-1]
        J += 0.5 * (10 * theta_final ** 2 + q_final**2)

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

        # for i in range(1, self.N+1):
        #     g_[-i] = 1e-5 * z[-i]
        for i in range(n):
            if i < (self.N + 1)*self.state_dim:
                g_[i] = self.q * z[i]
            else:
                g_[i] = self.r * z[i]

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
        c_[-2*self.state_dim:-self.state_dim] = (xs[-self.state_dim:] )

        
        c_[-4] = xs[3]
        c_[-3] = xs[2]
        # Starting theta at pi radians
        c_[-2] = xs[0]+1
        
        # Starting thetadot at 0 rad/s
        c_[-1] = xs[1]

        return c_
        
    def dc(self, z):

        dc_ = np.zeros((self.m,self.n))

        xs = z[:-self.N]
        us = z[-self.N:]

        dt = 1. / self.N

        # Set dynamic constraints
        k = 0
        for i in range(self.N):
            xnext = xs[k+self.state_dim:k+2*self.state_dim]
            x = xs[k:k+self.state_dim]

            u = us[i]

            dot_dx,dot_du=self.df(x,u)
            
            for j in range(self.state_dim):
                dc_[k+j,k+self.state_dim+j] += 1
                
            for j in range(self.state_dim):
                dc_[k+j,k+j] += -1    
                # for jj in range(self.state_dim):
                    # dc_[k+j,k+j+jj] +=-dt*dot_dx[j,jj]
                    
            dc_[k:k+self.state_dim,k:k+self.state_dim]+=-dt*dot_dx#-1+np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])#np.concatenate([d_theta_dot_dx,d_theta_ddot_dx],axis=1) )
            dc_[k:k+self.state_dim,i-self.N]=-dt*dot_du#np.concatenate([d_theta_dot_du,d_theta_ddot_du],axis=0)


            k += self.state_dim

        # ending theta & thetadot should be zero
        for j in range(self.state_dim):
            dc_[-2*self.state_dim+j,-self.state_dim-self.N+j] +=1
        #dc_[-2*self.state_dim:-self.state_dim,-self.state_dim:] = 1

        # Starting theta at pi radians
        dc_[-2,0] += 1
        
        # Starting thetadot at 0 rad/s
        dc_[-1,1] += 1
        
        dc_[-3,2] += 1
        
        dc_[-4,3] += 1
        
        #print(dc_)
        # print(self.N)
        # print(self.m)
        # print("_______________")
        sG = sparse.csr_matrix(dc_) 
        return sG

    
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
            # GG = np.zeros(shape)
            # e = np.zeros_like(z)
            # for i in range(len(e)):
                # if i > 0:
                    # e[i-1] = 0
                # e[i] = 1
                # GG[:,i] = matvec(e)
            
            #print(GG)
            #print(HH) 
            # sG = sparse.csr_matrix(GG)
            sG=self.dc(z) 

            # print(np.amax(np.abs(sG.toarray()-GG)))
            # print(sG.toarray()-GG)
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
            
            
        if (True):
            sK=sparse.vstack([sparse.hstack([H,G.T]),\
            sparse.hstack([G,sparse.csr_matrix((G.shape[0],G.shape[0]), dtype=G.dtype) ])])
            return sK
            
        shape = (n+m, n+m)
        return LinearOperator(shape, matvec=matvec,rmatvec=rmatvec)

    
    def plot(self, z):

        us = z[-self.N:]
        xs_z = z[:-self.N]
        
        xs = np.zeros((len(self.time), self.state_dim), dtype=z.dtype)
        xs[0] = [-1,0,0, 0]

        
        for i in range(len(self.time)-1):
            if(i<self.N):
                xs[i+1] = xs[i] + self.dt * self.f(xs[i], us[i])
                
            else:
                xs[i+1] = xs[i] + self.dt * self.f(xs[i], 0)
            # print(xs[i+0])

        ya = xs[:,0] 
        va = xs[:,1] 
        theta_a = xs[:,2]
        qa = xs[:,3]
        # 

        plt.figure(figsize=(12,10))

        plt.subplot(221)
        plt.plot(self.time[:len(us)],us,'m',lw=2)
        plt.legend([r'$u$'],loc=1)
        plt.ylabel('Force')
        plt.xlabel('Time')
        plt.xlim(self.time[0],self.time[-1])

        plt.subplot(222)
        plt.plot(self.time,va,'g',lw=2)
        plt.legend([r'$v$'],loc=1)
        plt.ylabel('Velocity')
        plt.xlabel('Time')
        plt.xlim(self.time[0],self.time[-1])

        plt.subplot(223)
        plt.plot(self.time,ya,'r',lw=2)
        plt.legend([r'$y$'],loc=1)
        plt.ylabel('Position')
        plt.xlabel('Time')
        plt.xlim(self.time[0],self.time[-1])

        plt.subplot(224)
        plt.plot(self.time,theta_a,'y',lw=2)
        plt.plot(self.time,qa,'c',lw=2)
        plt.legend([r'$\theta$',r'$q$'],loc=1)
        plt.ylabel('Angle')
        plt.xlabel('Time')
        plt.xlim(self.time[0],self.time[-1])

        plt.rcParams['animation.html'] = 'html5'

        x1 = ya
        y1 = np.zeros(len(self.time))

        #suppose that l = 1
        x2 = self.l*np.sin(theta_a)+x1
        x2b = self.l*np.sin(theta_a)+x1
        y2 = self.l*np.cos(theta_a)-y1
        y2b = self.l*np.cos(theta_a)-y1

        fig = plt.figure(figsize=(8,6.4))
        ax = fig.add_subplot(111,autoscale_on=False,\
                             xlim=(-8.5,8.5),ylim=(-1.2*self.l,1.2*self.l))
        ax.set_xlabel('position')
        ax.get_yaxis().set_visible(False)

        crane_rail, = ax.plot([-8.5,8.5],[-0.2,-0.2],'k-',lw=4)
        start, = ax.plot([-1,-1],[-1.5,1.5],'k:',lw=2)
        objective, = ax.plot([0,0],[-0.5,1.5],'k:',lw=2)
        mass1, = ax.plot([],[],linestyle='None',marker='s',\
                         markersize=20,markeredgecolor='k',\
                         color='orange',markeredgewidth=2)
        mass2, = ax.plot([],[],linestyle='None',marker='o',\
                         markersize=20,markeredgecolor='k',\
                         color='orange',markeredgewidth=2)
        line, = ax.plot([],[],'o-',color='orange',lw=4,\
                        markersize=6,markeredgecolor='k',\
                        markerfacecolor='k')
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05,0.9,'',transform=ax.transAxes)
        start_text = ax.text(-1.06,-0.3,'start',ha='right')
        end_text = ax.text(0.06,-0.3,'objective',ha='left')
        ax.set_aspect('equal', adjustable='box')    
        
        def init():
            mass1.set_data([],[])
            mass2.set_data([],[])
            line.set_data([],[])
            time_text.set_text('')
            return line, mass1, mass2, time_text

        def animate(i):
            mass1.set_data([x1[i]],[y1[i]-0.1])
            mass2.set_data([x2b[i]],[y2b[i]])
            line.set_data([x1[i],x2[i]],[y1[i],y2[i]])
            time_text.set_text(time_template % self.time[i])
            return line, mass1, mass2, time_text

        ani_a = animation.FuncAnimation(fig, animate, \
                 np.arange(1,len(self.time)), \
                 interval=100+500/self.N,blit=False,init_func=init)
        plt.show()