import numpy as np
import deepxde as dde
import scipy

from . import baseclass


class Convection1D(baseclass.BaseTimePDE):

    def __init__(self, geom=[0, 2*np.pi], time=[0, 1], beta=30.0, nu=0.0, rho=0.0):
        super().__init__()
        # output dim
        self.output_dim = 1
        # domain
        self.geom = dde.geometry.Interval(*geom)
        timedomain = dde.geometry.TimeDomain(*time)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)
        self.bbox = geom + time

        self.beta = beta

        # PDE
        def convection_pde(x, u):
            u_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_t = dde.grad.jacobian(u, x, i=0, j=1)
            return u_t + beta * u_x
            
        def reaction_pde(x, u):
            u_t = dde.grad.jacobian(u, x, i=0, j=1)
            return u_t - rho * u + rho * u ** 2
                       
        def diffusion_pde(x, u):
            u_t = dde.grad.jacobian(u, x, i=0, j=1)
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            return u_t - nu * u_xx
             
        def diffusion_reaction_pde(x, u):
            u_t = dde.grad.jacobian(u, x, i=0, j=1)
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            return u_t - nu * u_xx - rho * u + rho * u ** 2

        self.pde = convection_pde
        self.set_pdeloss()

        # refdata
        self.create_ref_data(nx=126, nt=20) # similar to callback creation of ref data when solution is available (2500 points total)

        # BCs
        def ic_func(x):
            return np.sin(x[:,0:1])

        self.add_bcs([{
            'component': 0,
            'function': ic_func,
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic'
        }, {
            'component': 0,
            'function': (lambda _: -1), # as in Wu et al 2023
            'bc': (lambda _, on_boundary: on_boundary),
            'type': 'dirichlet'
        }])

        # train settings
        self.training_points()  # default

    def create_ref_data(self, nx=256, nt=100): # 256 and 100 as in R3 code
        """
        xgrid: number of X points (for some reason should be even number..)
        nt: number of time steps
        """
        h = 2 * np.pi / nx
        x = np.arange(0, 2 * np.pi, h) # not inclusive of the last point
        t = np.linspace(0, 1, nt).reshape(-1, 1)
        X, T = np.meshgrid(x, t)

        u0 = np.sin(x)
        G = np.zeros_like(u0)

        IKX_pos = 1j * np.arange(0, nx / 2 + 1, 1)
        IKX_neg = 1j * np.arange(- nx / 2 + 1, 0, 1)
        IKX = np.concatenate((IKX_pos, IKX_neg))
        IKX2 = IKX * IKX

        uhat0 = np.fft.fft(u0)
        nu_factor = np.exp(- self.beta * IKX * T)
        A = uhat0 - np.fft.fft(G) * 0 # at t=0, second term goes away
        uhat = A * nu_factor + np.fft.fft(G) * T # for constant, fft(p) dt = fft(p)*T
        u = np.real(np.fft.ifft(uhat))

        u_vals = u.flatten() # why??
        u_vals = u_vals.reshape(nt, nx).T # why?????

        # adapt to how it is in baseclass
        self.ref_data = np.hstack([x.reshape(-1, 1), u_vals])
