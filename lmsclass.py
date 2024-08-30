# -*- coding: utf-8 -*-
"""Class providing statistical model and monte carlo simulation for the LMS algorithm."""
from datetime import datetime
import numpy as np
from scipy import linalg
from numpy import random as rnd
from numba import njit, prange
import matplotlib.pyplot as plt
import matplotlib

class LMS:
    """Class providing statistical model and monte carlo simulation for the LMS."""
    def __init__(self, realizations:int, iterations:int, sigma_x:float, alpha: float,
                 sigma_r:float, step_size:float,system_response:np.ndarray,
                 initial_solution:np.ndarray, decimation_factor=100, noise_type='gaussian'):
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

    def _createRxhxh(self)->np.ndarray:
        """Returns the autocorrelation matrix of an AR1 process"""
        return linalg.toeplitz(self._sigma_x*np.power(self._alpha,
                                                     np.arange(len(self._system_response))
                                                     )
                                )

    def _findwopt(self)->np.ndarray:
        """Returns the deficient length LMS optimal solution (internal use)"""
        return self._system_response

    def _findJmin(self)->float:
        """Returns the deficient length LMS minimal MSE (internal use)"""
        wopt = self._findwopt()
        Rxhxh = self._createRxhxh()
        return self._sigma_r+(self._system_response-wopt).T@Rxhxh@(self._system_response-wopt)

    def steadystateMSE(self)->float:
        """Returns the estimated steady-state MSE """
        Rxhxh = self._createRxhxh()
        tr = np.trace(Rxhxh)
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
        Jhist = np.zeros([self._iterations//self._decimation_factor])
        msdhist = np.zeros([self._iterations//self._decimation_factor])
        Vhist = np.zeros([Nh,self._iterations//self._decimation_factor])

        v = np.zeros([Nh,1])
        Rxhxh = self._createRxhxh()
        wopt = self._findwopt()
        Jmin = self._findJmin()

        v[:,0] = self._initial_solution - wopt

        lamb = np.zeros([Nh,1])
        lam, Q = np.linalg.eig(Rxhxh)
        lamb[:,0] = lam[:]

        k = np.zeros([Nh,1])
        karray = np.diag(np.outer(Q.T@v,Q.T@v))
        k[:,0] = karray[:]

        phiv = np.power(1.0-self._step_size*lam,2) + np.power(self._step_size*lam,2)

        Phi = np.diag(phiv) + (self._step_size**2)*(lamb@lamb.T)

        Psi = np.eye(Nh) - self._step_size* Rxhxh

        for i in range(self._iterations):
            if not i % self._decimation_factor:
                Jhist[i//self._decimation_factor] = Jmin + k.T@lamb
                msdhist[i//self._decimation_factor] = np.sum(k)
                Vhist[:,i//self._decimation_factor] = v[:,0]
            k = Phi @ k + (self._step_size**2)*Jmin*lamb
            v = Psi @ v

        Jinf = self.steadystateMSE()

        return Jinf, Vhist, Jhist, msdhist

    def run(self)->np.ndarray:
        ''' The monte-carlo simulation that works (wrapper)'''
        start_time = datetime.now()
        Vhist, e2hist, msdhist= self.__do_monte_carlo(self._realizations,
                                                      self._iterations,
                                                      self._sigma_x,
                                                      self._alpha,
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

    @staticmethod
    @njit(parallel=True)
    def __do_monte_carlo(Nrealizations:int, Niterations:int, sigmax:float, rho: float,
                       sigmar2:float, mu:float, h:np.ndarray, w0:np.ndarray,
                       wopt:np.ndarray, decimationfactor=100, noise_type='gaussian')->np.ndarray:
        ''' The monte-carlo simulation that works (static method with numba) '''
        # Vector initializations
        Nh = len(h)               # Determine length of response

        Ni = Niterations//decimationfactor

        Vhist = np.zeros((Nh,Ni), dtype=np.double)
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
                xy = np.ascontiguousarray(x[n+Nh:n:-1])
                y = xy @ w                          # evaluates adaptive filter output
                e = d - y                           # evaluates error signal
                v = w - wopt
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

    def plot_msd(self, msd_simulation:np.ndarray, msd_theoretical:np.ndarray):
        ''' Draw the MSD '''
        fig = plt.figure()
        fig.set_size_inches(8, 6)
        t = np.arange(0,self._iterations,self._decimation_factor)
        matplotlib.rcParams.update({'font.size': 18})
        _ = plt.plot(t,10*np.log10(msd_simulation), label='MC simulation')
        _ = plt.plot(t,10*np.log10(msd_theoretical), label='model')
        plt.grid('minor')
        plt.legend()
        ax = plt.gca()                    # grabs current axes and names them ax
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) # scientific number style
        _=plt.title('SD (MC simulation)')
        _=plt.xlabel('iterations')
        _=plt.ylabel('SD (dB)')
        plt.savefig('MSD.pdf',bbox_inches='tight')
        plt.show()
