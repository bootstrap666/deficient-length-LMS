# -*- coding: utf-8 -*-
"""Class providing statistical model and monte carlo simulation for the deficient length LMS."""
from datetime import datetime
import numpy as np
import scipy.linalg as linalg
from numpy import random as rnd
from numba import njit, prange
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import chi2
from lmsclass import LMS
#import jax.numpy as jnp

class LMS_def_len(LMS):
    """Class providing statistical model and monte carlo simulation for the deficient length LMS."""
    def __init__(self, realizations:int, iterations:int, sigma_x:float, alpha: float,
                 sigma_r:float, step_size:float, system_response:np.ndarray, 
                 initial_solution:np.ndarray, decimation_factor=100, 
                 noise_type='gaussian'):
        """Class constructor"""
        self._realizations = realizations
        self._iterations = iterations
        self._sigma_x = sigma_x
        self._alpha = alpha
        self._sigma_r = sigma_r
        self._step_size = step_size
        self._system_response = system_response
        self._initial_solution = initial_solution
        self._decimation_factor = decimation_factor
        self._noise_type = noise_type
        self._adaptive_filter_length = len(initial_solution)

    def _createRxhxh(self)->np.ndarray:
        """Returns the autocorrelation matrix of an AR1 process"""
        return linalg.toeplitz(self._sigma_x*np.power(self._alpha,
                                                     np.arange(len(self._system_response))
                                                     )
                                )

    def _createC(self)->np.ndarray:
        """Returns the constraint matrix for the deficient length LMS"""
        return np.block([[np.zeros([self._adaptive_filter_length,
                                    len(self._system_response)-self._adaptive_filter_length])],
                        [np.eye(len(self._system_response)-self._adaptive_filter_length)]])

    def _createP(self)->np.ndarray:
        """Returns the orthogonal Projection matrix for the deficient length LMS"""
        C = self._createC()
        return np.eye(len(self._system_response)) - C @ np.linalg.solve(C.T @ C, C.T)

    def _findwopt(self)->np.ndarray:
        """Returns the deficient length LMS optimal solution (internal use)"""
        C = self._createC()
        P = self._createP()
        Rxhxh = self._createRxhxh()
        return P@(self._system_response
                  -np.linalg.solve(
                                   Rxhxh,
                                   C@np.linalg.solve(
                                                     C.T @ np.linalg.solve(
                                                                           Rxhxh,
                                                                           C
                                                                          ),
                                                     C.T@self._system_response
                                                    )
                                  )
                  )

    # def _findwoptnumerically(self)->np.ndarray:
    #     """Returns the deficient length LMS optimal solution (internal use)"""
    #     Nh = len(self._system_response)
    #     Rxhxh = self.__createRxhxh()
    #     C = self.__createC()
    #     h = self._system_response

    #     def objective_fun(w,Rxhxh,h):
    #         return (w-h).T @ Rxhxh @ (w-h)
    #     res = minimize(objective_fun, np.zeros(Nh), (Rxhxh,h),
    #                 constraints=LinearConstraint(C.T,np.zeros(C.shape[1]),
    #                                              np.zeros(C.shape[1])))
    #     return res.x

    # def _findJminnumerically(self)->float:
    #     """Returns the deficient length LMS minimal MSE (internal use)"""
    #     Nh = len(self._system_response)
    #     Rxhxh = self.__createRxhxh()
    #     C = self.__createC()
    #     h = self._system_response

    #     def objective_fun(w,Rxhxh,h):
    #         return (w-h).T @ Rxhxh @ (w-h) + self._sigma_r
    #     res = minimize(objective_fun, np.zeros(Nh), (Rxhxh,h),
    #                 constraints=LinearConstraint(C.T,np.zeros(C.shape[1]),
    #                                              np.zeros(C.shape[1])))
    #     return res.fun

    def findJmin(self)->float:
        """Apagar"""
        return self._findJmin()
    
    def _findJmin(self)->float:
        """Returns the deficient length LMS minimal MSE (internal use)"""
        wopt = self._findwopt()
        Rxhxh = self._createRxhxh()
        return self._sigma_r+(self._system_response-wopt).T@Rxhxh@(self._system_response-wopt)

    def steadystateMSE(self)->float:
        """Returns the estimated steady-state MSE """
        Rxhxh = self._createRxhxh()
        P = self._createP()
        tr = np.trace(P@Rxhxh@P.T)
        Jmin = self._findJmin()

        return Jmin*(1.0+0.5*self._step_size*tr/(1 - 0.5*self._step_size*tr))

    def _get_kurtosis(self)->float:
        if self._noise_type == 'ewedian':
            return 733
        elif self._noise_type == 'laplacian':
            return 6.0
        elif self._noise_type == 'uniform':
            return 9.0/5.0
        else:
            return 3.0

    def predict(self):
        """Predicts the filter behavior from its statistical model"""
        Nh = len(self._system_response)
        N = self._adaptive_filter_length
        Jhist = np.zeros([self._iterations//self._decimation_factor])
        msdhist = np.zeros([self._iterations//self._decimation_factor])
        Vhist = np.zeros([Nh,self._iterations//self._decimation_factor])

        v = np.zeros([Nh,1])
        Rxhxh = self._createRxhxh()
        wopt = self._findwopt()
        Jmin = self._findJmin()

        v[0:N,0] = self._initial_solution - wopt[0:N]

        P = self._createP()

        lamb = np.zeros([Nh,1])
        lam, Q = np.linalg.eig(P@Rxhxh@P.T)
        lamb[:,0] = lam[:]

        k = np.zeros([Nh,1])
        karray = np.diag(np.outer(Q.T@v,Q.T@v))
        k[:,0] = karray[:]

        phiv = np.power(1.0-self._step_size*lam,2) + np.power(self._step_size*lam,2)

        Phi = np.diag(phiv) + (self._step_size**2)*(lamb@lamb.T)

        Psi = np.eye(Nh) - self._step_size* P@Rxhxh@P.T

        for i in range(self._iterations):
            if not i % self._decimation_factor:
                Jhist[i//self._decimation_factor] = Jmin + k.T@lamb
                msdhist[i//self._decimation_factor] = np.sum(k)
                Vhist[:,i//self._decimation_factor] = v[:,0]
            k = Phi @ k + (self._step_size**2)*Jmin*lamb
            v = Psi @ v

        Jinf = self.steadystateMSE()

        return Jinf, Vhist, Jhist, msdhist
    
    def predict2(self):
        """Predicts the filter second order statistics from the statistical model"""
        start_time = datetime.now()
        Nh = len(self._system_response)
        N = self._adaptive_filter_length
#        Jhist = jnp.zeros([self._iterations//self._decimation_factor])
        Jhist = np.zeros([self._iterations//self._decimation_factor])
 #       msdhist = jnp.zeros([self._iterations//self._decimation_factor])

#        v = jnp.zeros([Nh,1])
        v = np.zeros([Nh,1])
        Rxhxh = self._createRxhxh()
        wopt = self._findwopt()
        Jmin = self._findJmin()

        v[0:N,0] = self._initial_solution - wopt[0:N]
#        v = v.at[0:N,0].set(self._initial_solution - wopt[0:N])

        P = self._createP()

#        lamb = jnp.zeros([Nh,1])
        lamb = np.zeros([Nh,1])
#        lam, Q = jnp.linalg.eigh(P@Rxhxh@P.T)
        lam, Q = np.linalg.eigh(P@Rxhxh@P.T)
        lamb[:,0] = lam[:]
#        lamb = lamb.at[:,0].set(lam)

#        k = jnp.zeros([Nh,1])
        k = np.zeros([Nh,1])
#        karray = jnp.diag(np.outer(Q.T@v,Q.T@v))
        karray = np.diag(np.outer(Q.T@v,Q.T@v))
        k[:,0] = karray[:]
#        k = k.at[:,0].set(karray)


#        phiv = jnp.power(1.0-self._step_size*lam,2) + jnp.power(self._step_size*lam,2)
        phiv = np.power(1.0-self._step_size*lam,2) + np.power(self._step_size*lam,2)

#        Phi = jnp.diag(phiv) + (self._step_size**2)*(lamb@lamb.T)
        Phi = np.diag(phiv) + (self._step_size**2)*(lamb@lamb.T)

        for i in range(self._iterations):
            if not i % self._decimation_factor:
                Jhist[i//self._decimation_factor] = Jmin + k.T@lamb
                #msdhist[i//self._decimation_factor] = np.sum(k)
#                Jhist = Jhist.at[i//self._decimation_factor].set((k.T@lamb).item())
#                msdhist = msdhist.at[i//self._decimation_factor].set(np.sum(k))
            k = Phi @ k + (self._step_size**2)*Jmin*lamb

        Jinf = self.steadystateMSE()
        end_time = datetime.now()
        print(f"Duration: {format(end_time - start_time)}")

 #       return Jinf, Jhist, msdhist
        return Jinf, Jhist
    
    
    def run(self)->np.ndarray:
        ''' The monte-carlo simulation that works (wrapper)'''
        start_time = datetime.now()
        Vhist, e2hist, msdhist= self.__do_monte_carlo(self._realizations,
                                                      self._iterations,
                                                      self._sigma_x,
                                                      self._alpha,
                                                      self._adaptive_filter_length,
                                                      self._sigma_r,
                                                      self._step_size,
                                                      self._system_response,
                                                      self._initial_solution,
                                                      self._findwopt(),
                                                      self._decimation_factor,
                                                      self._noise_type)
        end_time = datetime.now()
        print(f"Duration: {format(end_time - start_time)}")
        return Vhist, e2hist, msdhist

    def run2(self)->np.ndarray:
        ''' The monte-carlo simulation that works (wrapper)'''
        start_time = datetime.now()
        e2hist, msdhist= self.__do_monte_carlo2(self._realizations,
                                                self._iterations,
                                                self._sigma_x,
                                                self._alpha,
                                                self._adaptive_filter_length,
                                                self._sigma_r,
                                                self._step_size,
                                                self._system_response,
                                                self._initial_solution,
                                                self._findwopt(),
                                                self._decimation_factor,
                                                self._noise_type)
        end_time = datetime.now()
        print(f"Duration: {format(end_time - start_time)}")
        return e2hist, msdhist

    
    @staticmethod
    @njit(parallel=True)
    def __do_monte_carlo(Nrealizations:int, Niterations:int, sigmax:float, rho: float, N:int,
                       sigmar2:float, mu:float, h:np.ndarray, w0:np.ndarray,
                       wopt:np.ndarray, decimationfactor=100, noise_type='gaussian')->np.ndarray:
        ''' The monte-carlo simulation that works (static method with numba) '''
        # Vector initializations
        Nh = len(h)               # Determine length of response

        Ni = Niterations//decimationfactor

        Vhist = np.zeros((N,Ni), dtype=np.double)
        e2hist = np.zeros(Ni)
        msdhist = np.zeros(Ni)

        # Loop for Monte Carlo realizations
        for _ in prange(Nrealizations):                     # loop for realizations
            w = w0                           # initializes W(0)

            # Input driving noise generation
            sigmau = sigmax*(1.0-rho**2)
            u = np.sqrt(sigmau)*rnd.normal(0,1,size=Niterations+Nh)

            x = np.zeros(Niterations+Nh)
            x[0] = np.sqrt(sigmax)*rnd.randn()
            for k in range(1,Niterations+Nh):    # AR(1) model
                x[k] = rho*x[k-1] + u[k]

            # Generation of the measurement noise sequence
            if noise_type == 'ewedian':
                noise = np.power(np.random.randn(Niterations),5)*np.sqrt(sigmar2/945)
            elif noise_type == 'laplacian':
                noise = np.random.laplace(loc=0.0,scale=np.sqrt(sigmar2/2.0),size=Niterations)
            elif noise_type == 'uniform':
                noise = np.random.uniform(low=-np.sqrt(3.0*sigmar2), high=np.sqrt(3.0*sigmar2), 
                                          size=Niterations)
            else:# assume gaussian
                noise = np.sqrt(sigmar2)*np.random.randn(Niterations)

            # Adaptive algorithm
            for n in range(Niterations):
                xd = np.ascontiguousarray(x[n+Nh:n:-1])
                d = xd @ h + noise[n]
                xy = np.ascontiguousarray(x[n+Nh:n+Nh-N:-1])
                y = xy @ w                          # evaluates adaptive filter output
                e = d - y                           # evaluates error signal
                v = w - wopt[0:N]
                if not n % decimationfactor:
                    Vhist[:,n//decimationfactor] += v
                    e2hist[n//decimationfactor] += e**2
                    msdhist[n//decimationfactor] += v@v

                # Updating the adaptive weights
                w = w + mu * e * xy

                # Update of unknown system response
    #            SystemResponsen = SystemResponsen + np.sqrt(varq) * np.random.randn(Nh)
        Vhist = Vhist/Nrealizations
        e2hist = e2hist/Nrealizations
        msdhist = msdhist/Nrealizations

        return Vhist, e2hist, msdhist

    @staticmethod
    @njit(parallel=True)
    def __do_monte_carlo2(Nrealizations:int, Niterations:int, sigmax:float, rho: float, N:int,
                       sigmar2:float, mu:float, h:np.ndarray, w0:np.ndarray,
                       wopt:np.ndarray, decimationfactor=100, noise_type='gaussian')->np.ndarray:
        ''' The monte-carlo simulation that works (static method with numba) '''
        # Vector initializations
        Nh = len(h)               # Determine length of response

        Ni = Niterations//decimationfactor

        e2hist = np.zeros(Ni)
        msdhist = np.zeros(Ni)

        # Loop for Monte Carlo realizations
        for _ in prange(Nrealizations):                     # loop for realizations
            w = w0                           # initializes W(0)

            # Input driving noise generation
            sigmau = sigmax*(1.0-rho**2)
            u = np.sqrt(sigmau)*rnd.normal(0,1,size=Niterations+Nh)

            x = np.zeros(Niterations+Nh)
            x[0] = np.sqrt(sigmax)*rnd.randn()
            for k in range(1,Niterations+Nh):    # AR(1) model
                x[k] = rho*x[k-1] + u[k]

            # Generation of the measurement noise sequence
            if noise_type == 'ewedian':
                noise = np.power(np.random.randn(Niterations),5)*np.sqrt(sigmar2/945)
            elif noise_type == 'laplacian':
                noise = np.random.laplace(loc=0.0,scale=np.sqrt(sigmar2/2.0),size=Niterations)
            elif noise_type == 'uniform':
                noise = np.random.uniform(a=-np.sqrt(3.0*sigmar2), b=np.sqrt(3.0*sigmar2), 
                                          size=Niterations)
            else:# assume gaussian
                noise = np.sqrt(sigmar2)*np.random.randn(Niterations)

            # Adaptive algorithm
            for n in range(Niterations):
                xd = np.ascontiguousarray(x[n+Nh:n:-1])
                d = xd @ h + noise[n]
                xy = np.ascontiguousarray(x[n+Nh:n+Nh-N:-1])
                y = xy @ w                          # evaluates adaptive filter output
                e = d - y                           # evaluates error signal
                v = w - wopt[0:N]
                if not n % decimationfactor:
                    #e2hist[n//decimationfactor] += (e-noise[n])**2
                    e2hist[n//decimationfactor] += e**2
                    msdhist[n//decimationfactor] += v@v

                # Updating the adaptive weights
                w = w + mu * e * xy

                # Update of unknown system response
    #            SystemResponsen = SystemResponsen + np.sqrt(varq) * np.random.randn(Nh)
        e2hist = e2hist/Nrealizations
        msdhist = msdhist/Nrealizations

        return e2hist, msdhist
    
    def plot_mse(self, mse_simulation:np.ndarray, mse_theoretical:np.ndarray,
                 mse_steady_state:float,mse_variance_theo:float):
        ''' Draw the MSE '''
        fig = plt.figure()
        fig.set_size_inches(8, 6)
        matplotlib.rcParams.update({'font.size': 18})
        t = np.arange(0,self._iterations,self._decimation_factor)
        if not mse_simulation is None:
            _ = plt.plot(t,10*np.log10(mse_simulation),label='simulations')
        if not mse_theoretical is None:
            _ = plt.plot(t,10*np.log10(mse_theoretical),'k',label='theory')
        if not mse_steady_state is None:
            _ = plt.plot([0, self._iterations],[10*np.log10(mse_steady_state),
                                               10*np.log10(mse_steady_state)],
                         'r:',label='steady-state',)
        
        
#        ic = 0.997
#        limits = chi2.interval(ic, self._realizations)
        
        
#        _ = plt.plot([0, self._iterations],[10*np.log10(limits[1]*mse_steady_state/self._realizations),
#                                            10*np.log10(limits[1]*mse_steady_state/self._realizations)],
#                     'r--',label=f'confidence interval({100*ic}%)')
        # if limits[0] > 0:
        #     _ = plt.plot([0, self._iterations],[10*np.log10(limits[0]*mse_steady_state/self._realizations),
        #                                         10*np.log10(limits[0]*mse_steady_state/self._realizations)],
        #                  'r--',label=f'confidence interval({100*ic}%)')
        plt.grid()
        
        _ = plt.plot([0, self._iterations],[10*np.log10(mse_steady_state
                                                        +3*np.sqrt(mse_variance_theo/
                                                                   self._realizations)),
                                            10*np.log10(mse_steady_state
                                                        +3*np.sqrt(mse_variance_theo/
                                                                   self._realizations))],
                        'r--',label='steady-state+3std dev')
        if mse_steady_state > 3*np.sqrt(mse_variance_theo/self._realizations):
            _ = plt.plot([0, self._iterations],[10*np.log10(mse_steady_state
                                                            -3*np.sqrt(mse_variance_theo/
                                                                       self._realizations)),
                                               10*np.log10(mse_steady_state
                                                           -3*np.sqrt(mse_variance_theo/
                                                                      self._realizations))],
                        'r--',label='steady-state-3std dev')

        plt.legend()                           # adds legend with labels defined above
        plt.title(f'Mean Square Error (N={self._adaptive_filter_length})')
        plt.xlabel('iterations')
        plt.ylabel('MSE (dB)')
        ax = plt.gca()                    # grabs current axes and names them ax
        # Setting axes properties
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) # scientific number style
        plt.savefig('MSE.pdf',bbox_inches='tight')
        plt.show()

    # def plot_msd(self, msd_simulation:np.ndarray, msd_theoretical:np.ndarray):
    #     ''' Draw the MSD '''
    #     fig = plt.figure()
    #     fig.set_size_inches(8, 6)
    #     t = np.arange(0,self._iterations,self._decimation_factor)
    #     matplotlib.rcParams.update({'font.size': 18})
    #     _ = plt.plot(t,10*np.log10(msd_simulation), label='MC simulation')
    #     _ = plt.plot(t,10*np.log10(msd_theoretical), label='model')
    #     plt.grid('minor')
    #     plt.legend()
    #     ax = plt.gca()                    # grabs current axes and names them ax
    #     ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) # scientific number style
    #     _=plt.title('MSD (MC simulation)')
    #     _=plt.xlabel('iterations')
    #     _=plt.ylabel('MSD (dB)')
    #     plt.savefig('MSD.pdf',bbox_inches='tight')
    #     plt.show()

    def steadystateerrorvariance(self)->float:
        """Returns the estimated steady-state squared error variance """
        Rxhxh = self._createRxhxh()
        P = self._createP()
        tr = np.trace(P@Rxhxh@P.T)
        tr2 = np.trace(P@Rxhxh@P.T@P@Rxhxh@P.T)
        Jmin = self._findJmin()

        return (self._get_kurtosis() -3.0)*self._sigma_r**2 + (Jmin**2)*(2 +2*self._step_size*tr + ((self._step_size**2)/2) * ((tr**2) + 3*tr2) )
    
    def initialize_at_optimum(self):
        self._initial_solution = self._findwopt()[0:self._adaptive_filter_length]
