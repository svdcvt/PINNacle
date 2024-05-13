import numpy as np
import deepxde as dde
import scipy.io as scio

from . import baseclass


class AllenCahn1D(baseclass.BaseTimePDE):

    def __init__(self, datapath="ref/AC.mat", geom=[-1, 1], time=[0, 1], nu=0.0001):
        super().__init__()
        # output dim
        self.output_dim = 1
        # domain
        self.geom = dde.geometry.Interval(*geom)
        timedomain = dde.geometry.TimeDomain(*time)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)
        self.bbox = geom + time

        # PDE
        def ac_pde(x, u):
            u_t = dde.grad.jacobian(u, x, i=0, j=1)
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            return u_t + 5 * u ** 3 - 5 * u - nu * u_xx

        self.pde = ac_pde
        self.set_pdeloss()

        # refdata
        self.load_ref_data(datapath)

        # BCs
        def ic_func(x):
            xx = x[:, 0:1] # x component
            return xx ** 2 * np.cos(np.pi * xx)

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

    def load_ref_data(self, datapath):
        data = scio.loadmat(datapath)
        XX, TT = np.meshgrid(data['x'], data['tt'])
        ref_inp = np.stack([XX[:,:,None], TT[:,:,None]], -1).reshape(-1, 2)
        ref_out = data['uu'].T.reshape(-1, 1)
        self.ref_data = np.hstack([ref_inp, ref_out]).astype(np.float32)
