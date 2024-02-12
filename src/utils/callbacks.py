import logging
import os
import torch
import numpy as np
import scipy
import itertools

from deepxde.geometry import Hypercube, Interval
from deepxde.callbacks import Callback, PDEPointResampler
from deepxde.utils.internal import list_to_str
from src.utils import plot

logger = logging.getLogger(__name__)


class PDEPointResampler(Callback):
    """Resample the training points for PDE and/or BC losses every given period.

    Args:
        period: How often to resample the training points (default is 100 iterations).
        pde_points: If True, resample the training points for PDE losses (default is
            True).
        bc_points: If True, resample the training points for BC losses (default is
            False; only supported by pytorch backend currently).
    """

    def __init__(self, period=100, pde_points=True, bc_points=False):
        super().__init__()
        self.period = period
        self.pde_points = pde_points
        self.bc_points = bc_points

        self.num_bcs_initial = None
        self.epochs_since_last_resample = 0

    def on_train_begin(self):
        self.num_bcs_initial = self.model.data.num_bcs

    def on_epoch_end(self):
        self.epochs_since_last_resample += 1
        if self.epochs_since_last_resample < self.period:
            return
        self.epochs_since_last_resample = 0
        self.resample()

    def resample(self):
        self.model.data.resample_train_points(self.pde_points, self.bc_points)


class PDEPointAdaptiveResampler(PDEPointResampler):
    def __init__(self, method, period, pde_points=True, bc_points=False, verbose=True, **method_kwargs):
        '''
        method: str, choices=['full' 'rarg' 'rard' 'rad' 'r3']
            Defines which method to use to resample points.
            To resample all points uniformly (Random-R in [1]), choose 'full' (calls deepXde callback PDEPointResampler).
            The methods given in [1] are 'rarg', 'rard', and 'rad'. The method proposed in [2] is 'r3'.

        period: int,
            Frequency to resample = each (period) iterations.
        pde_points, bc_points: bool (default True, False)
            Not implemented, but whether to resample points associated with PDE loss and/or ICBC loss.
        method_kwargs: dict
            'density_mul' : how much more dense the dense collocation set should be compared to already set by model for train set
            'm' (default=1) : number of points to add to initial set for RARG/RARD (RAD defines it from model)
            'k' : exponent of residual error in pdf-equation for RARD/RAD
            'c' : addend of pdf-equation for RARD/RAD

        [1] Wu, et al., 2023
        A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks
        http://dx.doi.org/10.1016/j.cma.2022.115671
        [2] Daw, et al., 2023
        Mitigating Propagation Failures in Physics-informed Neural Networks using Retain-Resample-Release (R3) Sampling
        https://doi.org/10.48550/arXiv.2207.02338

        '''
        super().__init__(period=period, pde_points=pde_points, bc_points=bc_points)
        if method != 'full' and (not pde_points or bc_points):
            raise NotImplementedError("Currently adaptive samplers support update of only points associated with PDE loss")
        self.method = method

        # parsing methods parameters # TODO put default parameters somewhere else?
        if method.startswith("ra"): # rar-g or rar-d or rad
            self.density_mul = method_kwargs['density_mul']
            if method.startswith('rar'): # rar-g or rar-d
                self.m = method_kwargs.get('m', 1)
            if method.endswith('d'): # rar-d(k=2,c=0) or rad(k=1,c=1)
                self.k = method_kwargs.get('k', len(method) // 2) # i am so sorry! but isn't it perfect?
                self.c = method_kwargs.get('c', len(method) % 2)
        # just for debugging
        self.verbose = verbose

    def init(self):
        if self.method in ['rarg', 'rard', 'rad']:
            self.dense_num_domain = self.model.data.train_x_all.shape[0] * self.density_mul
            if self.model.data.train_distribution == "uniform":
                self.dense_set = self.model.data.geom.uniform_points(self.dense_num_domain, boundary=True)
            else:
                self.dense_set = self.model.data.geom.random_points(self.dense_num_domain, random=self.model.data.train_distribution)
            if self.method == 'rad':
                self.m = self.model.data.train_x_all.shape[0]

    def check(self):
        # just debugging
        b = (self.model.data.train_x_all.shape[0], self.model.data.num_domain, sum(self.model.data.num_bcs))
        c = np.all(self.model.data.train_x == np.vstack([self.model.data.train_x_bc, self.model.data.train_x_all]))
        if self.verbose:
            print('shapes are right:', b)
            print('train_x = [x_bc, x_all]:', c)

    def resample(self):
        # for now everything is in here but TODO separate methods
        if self.method == 'full':
            return super().resample()
        
        # else, working with pde residuals
        def compute_residuals(data):
            pred_per_pde = self.model.predict(data, operator=self.model.pde.pde)
            if isinstance(pred_per_pde, list):
                pred_per_pde = np.hstack(pred_per_pde)
            # TODO hotfix mean(-1) issue with several pdes (Burger2D)
            return np.abs(pred_per_pde).mean(-1) # (N, num_pde) -> (N, )

        def to_pdf(res):
            res **= self.k
            res /= res.mean()
            res += self.c
            res -= res.min() # safe division ?
            res /= res.sum()
            return res

        if self.method.startswith('ra'):
            # computeresiduals on a dense set for RAR-G/RAR-D/RAD
            residual_error = compute_residuals(self.dense_set)
            if self.method == 'rarg':
                # top-m added
                idx = np.argsort(residual_error)[-self.m:]
                selected = self.dense_set[idx]
                self.model.data.add_anchors(selected)
            elif self.method == 'rard':
                # m sampled with repetition and added
                idx = np.random.choice(self.dense_num_domain, self.m, p=to_pdf(residual_error))
                selected = self.dense_set[idx]
                self.model.data.add_anchors(selected)
            elif self.method == 'rad':
                # m sampled with repetition and replaced
                idx = np.random.choice(self.dense_num_domain, self.m, p=to_pdf(residual_error))
                selected = self.dense_set[idx]
                self.model.data.replace_with_anchors(selected)
        elif self.method == 'r3':
            X_res = self.model.data.train_x_all
            residual_error = compute_residuals(X_res)
            mean_res_err = np.mean(residual_error)

            # TODO can we use this instead of recomputing?
            #pred_per_pde = self.model.data.last_f_computed
            #pred_per_pde = pred_per_pde[:self.model.pde.num_pde]
            #residual_error_saved = np.abs(np.hstack(pred_per_pde)).mean(-1)
            #mean_res_err_saved = np.mean(residual_error_saved)
            #mask_from_saved = residual_error_saved > mean_res_err_saved

            # we will not touch train_x_bc, as it is in R3 paper. 
            # But we will uniformly sample points in closed geometry (considering all points equal)
            mask = residual_error > mean_res_err
            retained = X_res[mask]
            resample_count = X_res.shape[0] - retained.shape[0]
            resampled = self.model.data.geom.random_points(resample_count, 'pseudo')
            new_train_x_all = np.vstack([resampled, retained])
            self.model.data.train_x_all = new_train_x_all
            # this will update train_x and train_y, but no touching train_x_bc
            self.model.data.resample_train_points(False, False)
            if self.verbose:
                print(f"Resample is done: {resample_count} points with res_loss <= {mean_res_err}")
        elif method == 'breed':
            pass

    def on_epoch_end(self):
        self.epochs_since_last_resample += 1
        if self.epochs_since_last_resample < self.period:
            return
        self.epochs_since_last_resample = 0
        # self.check()
        self.resample()
        # self.check()


class PlotCallback(Callback):

    def __init__(self, log_every=None, verbose=False, fast=False):
        super(PlotCallback, self).__init__()

        self.log_every = log_every
        self.verbose = verbose
        self.fast = fast
        self.valid_epoch = 0

    def plot(self, save_path):
        train_state = self.model.train_state
        plot.plot_state(self.model.pde, train_state, save_path, is_best=False, fast=self.fast)

    def on_train_begin(self):
        self.base_save_path = self.model.model_save_path + "/"
        if not os.path.exists(self.base_save_path):
            os.mkdir(self.base_save_path)

    def on_epoch_end(self):
        self.valid_epoch += 1
        if self.log_every is None or self.valid_epoch % self.log_every != 0:
            return
        if self.verbose:
            print("Plotting at epoch {} ...".format(self.valid_epoch))

        save_path = self.base_save_path + str(self.valid_epoch) + '/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.plot(save_path)

    def on_train_end(self):
        if self.verbose:
            print("Plotting at train end ...")
        self.plot(self.base_save_path)


class LossCallback(Callback):

    def __init__(self, loss_plot=True, verbose=False):
        super(LossCallback, self).__init__()
        self.log_every = None
        self.loss_plot = loss_plot
        self.verbose = verbose
        self.valid_epoch = 0
        self.loss_weights = []

    def on_train_begin(self):
        self.log_every = self.model.display_every
        if self.model.losshistory.loss_weights is not None:
            self.loss_weights.append(self.model.losshistory.loss_weights)
        else:
            self.loss_weights.append(np.ones(self.model.pde.num_loss))
            
    def on_epoch_end(self):
        self.valid_epoch += 1
        if self.valid_epoch % self.log_every != 0:
            return

        if self.model.losshistory.loss_weights is not None:
            self.loss_weights.append(self.model.losshistory.loss_weights.copy())
        else:
            self.loss_weights.append(np.ones(self.model.pde.num_loss))

        if self.verbose:
            loss_weight = self.loss_weights[-1]
            loss_train = self.model.losshistory.loss_train[-1] / loss_weight
            loss_test = self.model.losshistory.loss_test[-1] / loss_weight
            print('Unweighted Loss: {}  {} Weights: {}'.format(
                list_to_str(loss_train),
                list_to_str(loss_test),
                list_to_str(loss_weight),
            ))
        if self.loss_plot:
            self.plot()

    def on_train_end(self):
        self.plot()

    def plot(self):
        save_path = self.model.model_save_path + "/"
        loss_history = self.model.losshistory
        loss_weights = np.array(self.loss_weights)
        loss = np.hstack((
            np.array(loss_history.steps)[:, None],
            np.array(loss_history.loss_train) / loss_weights,
            np.array(loss_history.loss_test) / loss_weights,
            loss_weights,
        ))
        np.savetxt(save_path + "loss.txt", loss, header="step, loss_train, loss_test, loss_weight")
        plot.plot_loss_history(self.model.pde, loss_history, save_path)
        plot.plot_loss_history(self.model.pde, loss_history, save_path, loss_weights=loss_weights)


class TesterCallback(Callback):

    def __init__(self, log_every=100, verbose=True, fRMSE_param={'enable':True, 'iLow':5, 'iHigh':13, 'calc_every':2000}):
        super(TesterCallback, self).__init__()

        self.log_every = log_every
        self.verbose = verbose
        self.fRMSE = fRMSE_param.get('enable', True)
        if self.fRMSE:
            self.fRMSE_l = fRMSE_param.get('iLow', 5)
            self.fRMSE_h = fRMSE_param.get('iHigh', 13)
            self.fRMSE_every = fRMSE_param.get('calc_every', 2000)

        self.indexes = []
        self.maes = []    # Mean Average Error
        self.mses = []    # Mean Square Error
        self.mxes = []    # Maximum Error
        self.l1res = []   # L1 Relative Error
        self.l2res = []   # L2 Relative Error
        self.crmses = []  # CSV_Loss
        self.frmses = []  # Mean Square Error in Fourier Space

        self.epochs_since_last_resample = 0
        self.valid_epoch = 0
        self.disable = False

    def on_train_begin(self):
        self.save_path = self.model.model_save_path + "/"
        pde = self.model.pde

        # Load / Generate Test Data
        if pde.ref_sol is not None: # sample points from geometry
            sample_points = 2500 if pde.input_dim == 2 else 20000
            if getattr(self.model.data.geom, "uniform_points", None) is None:
                logger.warning(f"Method \'Uniform Points\' not found for class {type(self.model.data.geom)}, \
                                 Use random points for testing ...")
                sample_func = self.model.data.geom.random_points
            else:
                sample_func = self.model.data.geom.uniform_points
            
            self.test_x = sample_func(sample_points, boundary=False)
            self.test_y = pde.ref_sol(self.test_x)
        elif pde.ref_data is not None:
            nan_mask = np.isnan(pde.ref_data).any(axis=1)
            self.test_x = pde.ref_data[~nan_mask, :pde.input_dim]
            self.test_y = pde.ref_data[~nan_mask, pde.input_dim:]
        else:
            self.disable = True
            logger.info("No reference solution or data provided, skipping TesterCallback")
            return

        self.solution_l1 = np.abs(self.test_y).mean()
        self.solution_l2 = np.sqrt((self.test_y**2).mean())

        if self.fRMSE:
            self.frmse_init()

    def on_epoch_end(self):
        self.epochs_since_last_resample += 1
        self.valid_epoch += 1
        if self.disable or self.log_every is None or self.epochs_since_last_resample < self.log_every:
            return
        self.epochs_since_last_resample = 0

        with torch.no_grad():
            y = self.model.predict(self.test_x)

        mse = ((y - self.test_y)**2).mean()
        mae = np.abs(y - self.test_y).mean()
        mxe = np.max(np.abs(y - self.test_y))
        l1re = mae / self.solution_l1
        l2re = np.sqrt(mse) / self.solution_l2
        crmse = np.abs((y - self.test_y).mean())
        if self.fRMSE and self.valid_epoch % self.fRMSE_every == 0:
            frmse = self.frmse_calc(y)
        else:
            frmse = (np.nan, np.nan, np.nan)

        self.indexes.append(self.valid_epoch)
        self.mses.append(mse)
        self.maes.append(mae)
        self.mxes.append(mxe)
        self.l1res.append(l1re)
        self.l2res.append(l2re)
        self.crmses.append(crmse)
        self.frmses.append(frmse)

        if self.verbose:
            if np.isnan(frmse[0]):
                print('Validation: epoch {} MSE {:.5f} MAE {:.5f} MXE {:.5f} L1RE {:.5f} L2RE {:.5f} CRMSE {:.5f}'.\
                       format(self.valid_epoch, mse, mae, mxe, l1re, l2re, crmse))
            else:
                print('Validation: epoch {} MSE {:.5f} MAE {:.5f} MXE {:.5f} L1RE {:.5f} L2RE {:.5f} CRMSE {:.5f} FRMSE ({:.5f}, {:.5f}, {:.5f})'.\
                       format(self.valid_epoch, mse, mae, mxe, l1re, l2re, crmse, frmse[0], frmse[1], frmse[2]))

    def on_train_end(self):
        if self.disable:
            return

        self.indexes = np.array(self.indexes)
        self.frmses = np.array(self.frmses)
        np.savetxt(
            self.save_path + 'errors.txt',
            np.array([self.indexes, self.maes, self.mses, self.mxes, self.l1res, self.l2res, self.crmses,\
                      self.frmses[:, 0], self.frmses[:, 1], self.frmses[:, 2]]).T,
            header="epochs, maes, mses, mxes, l1res, l2res, crmses, frmses(low, mid, high)"
        )

        plot.plot_lines([self.indexes, self.maes], xlabel="epochs", labels=['maes'], path=self.save_path + "maes.png", title="mean average error")
        plot.plot_lines([self.indexes, self.mses], xlabel="epochs", labels=['mses'], path=self.save_path + "mses.png", title="mean square error")
        plot.plot_lines([self.indexes, self.mxes], xlabel="epochs", labels=['mxes'], path=self.save_path + "mxes.png", title="maximum error")
        plot.plot_lines([self.indexes, self.l1res, self.l2res],
                        xlabel="epochs",
                        labels=['l1re', 'l2re'],
                        path=self.save_path + "relerr.png",
                        title="relative error")
        X = ~np.isnan(self.frmses).any(axis=1)
        plot.plot_lines([self.indexes[X], self.frmses[X, 0], self.frmses[X, 1], self.frmses[X, 2]], 
                        xlabel="epochs", 
                        labels=['low freq', 'mid freq', 'high freq'], 
                        path=self.save_path + "frmses.png", 
                        title="mean square error in fourier space")

        self.indexes = []
        self.maes = []   
        self.mses = []   
        self.mxes = []   
        self.l1res = []  
        self.l2res = []  
        self.crmses = [] 
        self.frmses = [] 

        self.epochs_since_last_resample = 0
        self.valid_epoch = 0
    
    def frmse_init(self):
        pde = self.model.pde
        if not isinstance(pde.geom, Hypercube) and not isinstance(pde.geom, Interval):
            logger.warning(f"Fourier transform errors are enabled only in Interval / Hypercube and their combination with Time domains. \
                           Type {type(pde.geom).__name__} is not a valid geometry and fRMSE has been disabled")
            self.fRMSE=False
            return
        if pde.input_dim > 3:
            logger.warning(f"For high dimensional PDEs like {type(pde).__name__} with dim {pde.input_dim} is slow to calculate fRMSE. \
                           fRMSE has been disabled")
            self.fRMSE=False
            return 

        # prepare calculation
        self.test_x_delaunay = scipy.spatial.Delaunay(self.test_x)
        ptn = 3e4 # generate about 3e4 uniform sampling points in the domain
        for i in range(pde.input_dim):
            ptn /= pde.bbox[i * 2 + 1] - pde.bbox[i * 2]
        ptn = ptn ** (1 / pde.input_dim)
        xlist = [np.linspace(pde.bbox[i * 2], pde.bbox[i * 2 + 1], int(np.ceil((pde.bbox[i*2+1] - pde.bbox[i*2]) * ptn)) + 1, endpoint=False)[1:] \
                 for i in range(pde.input_dim)]
        self.sample_x = np.stack(np.meshgrid(*xlist), axis=-1)
    
    def frmse_calc(self, y):
        pde = self.model.pde
        res = scipy.interpolate.LinearNDInterpolator(self.test_x_delaunay, y - self.test_y)(self.sample_x.reshape((-1, pde.input_dim)))
        resn = scipy.interpolate.NearestNDInterpolator(self.test_x, y - self.test_y)(self.sample_x.reshape((-1, pde.input_dim)))
        res[np.isnan(res)] = resn[np.isnan(res)]
        err = np.fft.rfftn(res, axes=tuple(range(res.ndim-1))) # transform except the last dim (pde.output_dim)
        err = np.mean(np.abs(err) ** 2 / res.size, axis=-1) # take average through the last dim

        if pde.input_dim == 1:
            err_low = err[:self.fRMSE_l].mean()
            err_mid = err[self.fRMSE_l:self.fRMSE_h].mean()
            err_high = err[self.fRMSE_h:].mean()
        else:
            err_low, err_mid, err_high = 0.0, 0.0, 0.0
            err_low_cnt, err_mid_cnt, err_high_cnt = 0, 0, 0
            for ids in itertools.product(*[range((k+1)//2) for k in err.shape[:-1]]):
                freq2 = sum(i ** 2 for i in ids)
                ilow = min(int(np.sqrt(max(0, self.fRMSE_l**2 - freq2))), err.shape[-1])
                ihigh = min(int(np.sqrt(max(0, self.fRMSE_h**2 - freq2))), err.shape[-1])

                err_low += err[(*ids, slice(None, ilow, None))].sum()
                err_mid += err[(*ids, slice(ilow, ihigh, None))].sum()
                err_high += err[(*ids, slice(ihigh, None, None))].sum()

                err_low_cnt += ilow 
                err_mid_cnt += ihigh - ilow
                err_high_cnt += err.shape[-1] - ihigh
            
            err_low /= err_low_cnt # calculate mean square error
            err_mid /= err_mid_cnt
            err_high /= err_high_cnt

        return err_low, err_mid, err_high
