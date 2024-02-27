import torch
import torch.nn as nn
import numpy as np
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt

class BaseSampler:
    def __init__(self, pde, init_points):
        '''
        pde: `src.pde.BasePDE` object # TODO generic for deepxde
            The object that defines PDE and domain.
        init_points: np.ndarray
            The initial uniform points given by `deepxde.model.Model.data.train_x_all`
        '''
        self._name = pde.__class__.__name__
        self.domain = pde.geomtime if hasattr(pde, 'geomtime') else pde.geom # TimePDE or PDE
        self.domainbbox = np.array(pde.bbox).reshape(-1, 2).T # [x_1_min, x_1_max, x_2_min, x_2_max, ...] -> [[x_1_min, ...],[x_1_max, ...]]
        self.points = init_points # todo check num points is right, it is numpy, etc
        self.size = len(self.points)
        self.dim = self.domain.dim

    def get_points(self):
        '''Method to update training set of PDE object'''
        pass


class Breed(BaseSampler):
    """Class for a sampler proposed by authors. Specially written for DeepXde library.
    Standalone version for Pytorch can be found here: https://gitlab.inria.fr/breed/breed

    Attributes:
     
    sigma: np.array[floats]: (D x D)
        The parameter responsible for neighbourhood widths
    Rs: np.array[floats]
        The array of R-values according to initialization parameters
    distribution: np.array
        The last loss-based distribution
    oob_count: list[int]
        Number of points which were sampled out of domain and needed to be resampled.
        Can help to change sigma value.
   
    Methods:
    
    """
    def __init__(self, pde, init_points, sigma=0.0, start=0.15, end=0.75, breakpoint=10, oob_factor=1.0):
        """
        Args:
        
        pde: `src.pde.BasePDE` object # TODO generic for deepxde
            The object that defines PDE and domain.
        init_points: np.ndarray
            The initial uniform points given by `deepxde.model.Model.data.train_x_all`
        sigma: float, array_like[floats]
            The covariance value for Gaussian neighbourhoods.
            If contains any non-positive values, will be updated to "optimal" in the domain value (in dev).
            If array is given, the values are the main diagonal of cov. matrix. The size must be equal to
            number of dimensions of PDE domain. If float is given, all values of the main diagonal are equal.
        start, end: float in [0,1]
            The first and last R value to create the R values series
        breakpoint: int >= 2
            The value specifies R value scenario, how many values are 
            linearly increasing from start to end, after which R values are constant (=end),
            If value is smaller than 2, it will be changed to 2 (for `start` and `end`)
        oob_factor: float in [0,1]
            The value specifies covariance decrease when a point sampled is out-of-bounds (oob),
            in order to sample new point in narrower Gaussian: sigma' = sigma * oob_factor.
            The decrease will apply until point is inside of boundary, and the new covariance value 
            will be saved for new point (in case of sampling in it's neighbourhood).
        """
        super().__init__(pde, init_points)
        self.sigma_opt_factor = 0.0005 # magic number for sigma being 0.001 for domain length 2 as in Allen Cahn
        self._sigma_init(sigma)
        self.covs = np.full((self.size, self.dim), self.sigma) # covdiag per point!
        self.cov_oob_factor = oob_factor # decrease covariance to sample inside domain if oob happened
        self.Rs = np.linspace(start, end, max(breakpoint, 2), endpoint=True)
        print("Rs: ", self.Rs)
        self.R_i = 0
        self.R = start
        self.oob_count = []
        self.isuniform = np.full((self.size,), True)
        self.indexes = np.arange(self.size)

    def inside(self, x):
        if hasattr(self.domain, 'inside'): # time independant
            return self.domain.inside(x)
        else:
            return np.logical_and(self.domain.geometry.inside(x[:,:-1]), self.domain.timedomain.inside(x[:,-1]))

    def _sigma_init(self, sigma):
        if isinstance(sigma, float):
            sigma = np.full((self.dim, ), sigma)
        elif isinstance(sigma, (np.ndarray, list, tuple)): # array_like but not string
            sigma = np.array(sigma).ravel()
            assert sigma.shape[0] == self.dim, f"The size of array-like `sigma` must be equal to number "\
                                               f"of dimensions of PDE domain: {sigma.size} != {self.dim}"
        else:
            raise RuntimeError(f"Argument `sigma` for `samplers.Breed` is wrong type: {type(sigma)}")

        dim_scales = np.diff(self.domainbbox, axis=0).ravel()
        max_sigma = dim_scales / 8 # six sigma rule to minimize outsiders
        if not np.logical_and(max_sigma >= sigma, sigma > 0).all():
            self.sigma = np.where(sigma > max_sigma, max_sigma, sigma)
            self.sigma = np.where(sigma <= 0, dim_scales * self.sigma_opt_factor, sigma)
            print(f"WARNING (samplers.Breed): Given value of sigma ({sigma}) is updated to {self.sigma}") 
        else:
            self.sigma = sigma

    def _R_step(self):
        self.R_i += 1
        if self.R_i < len(self.Rs):
            self.R = self.Rs[self.R_i]

    def _get_points_boundary(self, loss):
        raise NotImplementedError()

    def get_points(self, residuals, boundary_only=False, plot=True, stats=True):
        if boundary_only:
            return self.get_points_boundry(residuals)
        # else into the whole domain

        residuals -= residuals.min()
        self.distribution = residuals / residuals.sum()
        parental_idx = np.random.choice(self.indexes, size=self.size, p=self.distribution.ravel())

        new_sample_batch = np.empty_like(self.points)
        new_covariances = np.full_like(self.covs, self.sigma)
        new_isuniform = np.full(self.size, True)
        self.oob_count.append([])

        if plot:
            u, i, c = np.unique(parental_idx, return_inverse=True, return_counts=True)
            childcount = c[i]
            spec = []
            plt.figure(figsize=(8,8))
            plt.xlim(*(self.domainbbox[:,0]+[-0.1, 0.1]))
            plt.ylim(*(self.domainbbox[:,1]+[-0.1, 0.1]))
            oob = []
        for i, idx in enumerate(parental_idx):
            # TODO fix uniform to not be sampled by chance
            # because can make uniform an import parent
            if np.random.uniform(0, 1) < self.R:
                if plot:
                    spec.append(childcount[i])
                parent = self.points[idx]
                # preserve parental covariance
                new_covariances[i] = self.covs[idx]
                child = multivariate_normal.rvs(mean=parent, cov=np.diag(new_covariances[i]), size=1)[None] # shape (D, ) -> (1, D)
                # self.inside wants (N, D) array
                parent_oob = 0
                while not self.inside(child)[0]: # TODO decrease sigma only in one dimension!
                    if plot:
                        oob.append(child)
                    parent_oob += 1
                    # decrease covariance to not sample OOB, now for this child and for this child as parent in future
                    new_covariances[i] *= self.cov_oob_factor
                    child = multivariate_normal.rvs(mean=parent, cov=np.diag(new_covariances[i]), size=1)[None]
                if parent_oob != 0:
                    self.oob_count[-1].append(parent_oob)
                new_isuniform[i] = False
            else:
                child = self.domain.random_points(1)
                if plot:
                    spec.append(0)
            new_sample_batch[i] = child
        if plot:
            spec = np.array(spec)
            color = np.where(new_isuniform, 'g', 'b') 
            size = np.where(new_isuniform, 3, spec ** 2.5 * 3)
            plt.scatter(new_sample_batch[:,0], new_sample_batch[:,1], s=size, alpha=0.1, c=color)
            if len(oob):
                oob = np.vstack(oob)
                plt.scatter(oob[:,0], oob[:,1], s=3, alpha=0.3, c='r')
            plt.savefig(f'../tmp/{self._name}_{self.R_i}.png')
            plt.close()

        if stats:
            print('=== Breed: OOB stats ===',
                 f'oob_count = {sum(self.oob_count[-1])}', 
                 f'cov unique, count:',
                 *np.unique(new_covariances, return_counts=True, axis=0),
                 sep='\n')
            
            print('=== Breed: parent/children stats ===',
                  'Num_children distribution:',
                  *np.unique(np.unique(parental_idx, return_counts=True)[1], return_counts=True),
                  sep='\n')

            become_parent = np.in1d(self.indexes, parental_idx)
            become_uniform = ~become_parent
            was_parent = ~self.isuniform
            was_uniform = self.isuniform

            print(f'Parent points  = {become_parent.sum()}',
                 f'Non-parent pts = {become_uniform.sum()}',
                 f'uniform -> uniform   : {np.logical_and(was_uniform, become_uniform).sum()}',
                 f'uniform -> parent    : {np.logical_and(was_uniform, become_parent).sum()}',
                 f'parent  -> parent    : {np.logical_and(was_parent, become_parent).sum()}',
                 f'parent  -> non-parent: {np.logical_and(was_parent, become_uniform).sum()}',
                 sep='\n')

        self.points = new_sample_batch
        self.covs = new_covariances
        self.isuniform = new_isuniform
        self._R_step()
        return self.points

