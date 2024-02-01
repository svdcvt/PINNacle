import os

os.environ["DDEBACKEND"] = "pytorch"

import deepxde as dde
import numpy as np

from src.pde.burgers import Burgers1D, Burgers2D
from src.utils.callbacks import LossCallback, TesterCallback, PlotCallback, TestR3
from torch.optim import Adam
from deepxde.callbacks import Callback

class TestR3(Callback):
    def __init__(self, method='R3', interval=1, verbose=True):
        super().__init__()
        self.method = method
        self.interval = interval
        self.valid_epoch = 0
        self.log_every = 0
        self.num_bcs_initial = None
        self.num_domain_initial = None
        self.verbose = verbose

    def check(self):
        # a = self.model.data.train_x_bc.shape[0] == sum(self.model.data.num_bcs)
        b = (self.model.data.train_x_all.shape[0], self.model.data.num_domain, sum(self.model.data.num_bcs))
        c = np.all(self.model.data.train_x == np.vstack([self.model.data.train_x_bc, self.model.data.train_x_all]))
        #d = np.all(model.data.train_x_all[:self.num_boundary_initial] == model.data.train_x_bc) 
        if self.verbose:
            print('shapes are right:', b)
            print('train_x = [x_bc, x_all]:', c)
        
        #assert b, f'broken shapes, {} and {}'
        #print('train_x_all = [x_bc, x_pde]', d)

    def resample(self):
        X_res = self.model.data.train_x_all
        pred_per_pde = self.model.predict(X_res, operator=self.model.pde.pde)
        if isinstance(pred_per_pde, list):
            pred_per_pde = np.hstack(pred_per_pde)
        # TODO hotfix mean(-1) issue with several pdes (Burger2D)
        residual_error = np.abs(pred_per_pde).mean(-1) 
        mean_res_err = np.mean(residual_error)

        # we will not touch train_x_bc, as it is in R3 paper. But we will uniformly sample points of geometry (considering all points equal)
        retained = X_res[residual_error > mean_res_err]
        resample_count = X_res.shape[0] - retained.shape[0]
        resampled = self.model.data.geom.random_points(resample_count, 'pseudo')
        self.model.data.train_x_all = np.vstack([resampled, retained])
        # this will update train_x and train_y, but no touching train_x_bc
        self.model.data.resample_train_points(False, False)
        if self.verbose:
            print(f"Resample is done: {resample_count} points with res_loss <= {mean_res_err}")

    def on_train_begin(self):
        self.log_every = self.model.display_every
        self.num_bcs_initial = self.model.data.num_bcs
        self.num_domain_initial = self.model.data.num_domain
        self.num_boundary_initial = sum(self.num_bcs_initial)
        self.bcs_starts = [0] + np.cumsum(self.num_bcs_initial)
        self.bcs_s_e = zip(self.bcs_starts[:-1], self.bcs_starts[1:])

    def on_epoch_end(self):
        self.valid_epoch += 1
        if self.valid_epoch != 0 and self.valid_epoch % self.interval != 0:
            return
        # self.check()
        self.resample()
        # self.check()

    def _resample(self):
        X_all = self.model.data.train_x_all # we take only X_bc and X_dom (X_all), to not count bc twice
        residual_error = np.abs(self.model.predict(X_all, operator=self.model.pde)).squeeze()
        mean_res_err = np.mean(residual_error)
        mean_res_err_bcs = [np.mean(residual_error[slice(*se)]) for se in self.bcs_s_e]
        mean_res_err_dom = np.mean(residual_error[self.num_boundary_initial:])

        # we have three options

        # 1) save the initial numbers of points, each geometry element is independent
        # calculating mean for each element and applying R3 independently

        # 2) save the initial numbers of points, geometry elements are analysed together
        # calculating mean over the geometry, applying R3 to the geometry
        # and resampling uniformly as many points as there left to have initial numbers

        # 3) numbers of points will change, geometry elements are analysed together
        # calculating mean over the geometry, applying R3 to the geometry
        # and resampling uniformly with proportions given initially (e.g., Burg1D: 1:1:4)
        # or depending on ratio of points with (error > average)
        
        # I implement version 1
        def ind_c_gavg(array, avg, start):
            mask = array > avg
            inds = np.where(mask) + start
            count = sum(~mask)
            return inds, count

        # if self.model.


        new_X_bc = np.empty((0, X_all.ndim))

        for i in range(len(self.num_bcs)):
            retain_ind, resample_count = ind_c_gavg(X_all[slice(*self.bcs_s_e[i])], mean_res_err_bcs[i], self.bcs_starts[i])
            new_points = self.data.geom.random_boundary_points(resample_count, 'pseudo')
            new_points = np.vstack([new_points, X_all[retain_ind]])
            new_X_bc = np.vstack([new_X_bc, new_points])
        
        retain_ind, resample_count = ind_c_gavg(X_all[self.num_boundary_initial], mean_res_err_dom, self.num_boundary_initial)
        new_points = self.data.geom.random_points(resample_count, 'pseudo') #boundary false?
        new_X_domain = np.vstack([new_points, X_all[retain_ind]])
        
        self.model.data.train_x_all = np.vstack([new_X_bc, new_X_domain]) 
        self.model.data.train_x_bc = None
        self.model.data.bc_points()
        self.model.data.resample_train_points(False, False)



pde = Burgers2D()
net = dde.nn.FNN([pde.input_dim] + [100 for _ in range(5)]  + [pde.output_dim], "tanh", "Glorot normal")
net = net.float()
opt = Adam(net.parameters(), 1e-3)
model = pde.create_model(net)
model.compile(opt, loss_weights=np.ones(pde.num_loss))
model.train(iterations=10, display_every=1,
        callbacks=[
            TestR3(interval=1, verbose=True),
            LossCallback(verbose=True),
            PlotCallback(log_every=50, fast=True),
            TesterCallback(log_every=10),
            ], model_save_path='runs/test2d')
