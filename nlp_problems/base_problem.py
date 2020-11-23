from abc import ABC, abstractmethod


class BaseProblem(ABC):
    """
    Base nonlinear programming problem
    """

    @abstractmethod
    def f(self, *args):
        """
        Dynamics
        """
        pass
    
    @abstractmethod
    def F(self, x):
        """
        Objective function
        """
        pass

    @abstractmethod
    def c(self, x):
        """
        Constraints
        """
        pass

    @abstractmethod
    def g(self, x):
        """
        Gradient of objective function
        """
        pass

    @abstractmethod
    def G(self, x):
        """
        Jacobian of constraints
        """
        pass

    @abstractmethod
    def H(self, x, lam):
        """
        Hessian of Lagrangrian
        """
        pass

    @abstractmethod
    def KKT(self, x, lam):
        """
        KKT matrix
        """
        pass

    @property
    @abstractmethod
    def nvars(self):
        """
        Length of 'x' in above methods
        """
        pass

    @property
    @abstractmethod
    def nconstraints(self):
        """
        Number of constraints
        """
        pass
    
