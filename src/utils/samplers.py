import torch
import torch.nn as nn
import numpy as np
from scipy.stats import multivariate_normal


class BaseSampler:
    def __init__(self, pde, init_points):
        '''
        pde: `src.pde.BasePDE` object # TODO generic for deepxde
            The object that defines PDE and domain.
        init_points: np.ndarray
            The initial uniform points given by `deepxde.model.Model.data.train_x_all`
        '''
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
    def __init__(self, pde, init_points, sigma=0.0, start=0.15, end=0.75, breakpoint=10):
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
        start, end: float
            The first and last R value to create the R values series
        breakpoint: int
            The value specifies R value scenario, how many values are 
            linearly increasing from start to end, after which R values are constant (=end)
        """
        super().__init__(pde, init_points)
        self.sigma_opt_factor = 0.0005 # magic number for sigma being 0.001 for domain length 2 as in Allen Cahn
        self._sigma_init(sigma)
        self.covs = np.full((self.size, self.dim), self.sigma) # covdiag per point!
        self.cov_oob_factor = 0.1 # decrease covariance to sample inside domain if oob happened
        self.Rs = np.linspace(start, end, breakpoint, endpoint=True)
        print("Rs: ", self.Rs)
        self.R_i = 0
        self.R = start
        self.oob_count = []

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

    def get_points(self, residuals, boundary_only=False):
        if boundary_only:
            return self.get_points_boundry(residuals)
        # else into the whole domain

        residuals -= residuals.min()
        self.distribution = residuals / residuals.sum()
        parental_idx = np.random.choice(np.arange(self.size), size=self.size, p=self.distribution.ravel())

        new_sample_batch = np.empty_like(self.points)
        new_covariances = np.full_like(self.covs, self.sigma)
        self.oob_count.append([])
        for i, idx in enumerate(parental_idx):
            if np.random.uniform(0, 1) < self.R:
                parent = self.points[idx]
                # preserve parental covariance
                new_covariances[i] = self.covs[idx]
                child = multivariate_normal.rvs(mean=parent, cov=np.diag(new_covariances[i]), size=1)[None] # shape (D, ) -> (1, D)
                # self.inside wants (N, D) array
                parent_oob = 0
                while not self.inside(child)[0]: # TODO decrease sigma only in one dimension!
                    parent_oob += 1
                    # decrease covariance to not sample OOB, now for this child and for this child as parent in future
                    new_covariances[i] *= self.cov_oob_factor
                    child = multivariate_normal.rvs(mean=parent, cov=np.diag(new_covariances[i]), size=1)[None]
                if parent_oob != 0:
                    self.oob_count[-1].append(parent_oob)
            else:
                child = self.domain.random_points(1)
            new_sample_batch[i] = child
        self.points = new_sample_batch # shuffled naturally?
        self.covs = new_covariances
        
        #print('=== Breed: OOB stats ===',
        #     f'oob_count = {sum(self.oob_count[-1])}', 
        #     f'cov unique = {np.unique(new_covariances, return_counts=True, axis=0)[1]}',
        #     sep='\n')
        self._R_step()
        return self.points


    # in development
    def get_points_new_version(self, residuals, boundary_only=False):
        if boundary_only:
            return self.get_points_boundry(residuals)
        # else into the whole domain

        residuals -= residuals.min()
        self.distribution = residuals / residuals.sum()
        N_g = int(np.ceil(self.size * self.R))
        N_u = self.size - N_g

        uniform_points = self.domain.random_points(N_u)
        if N_u == self.size:
            self.points = uniform_points
            return self.points
        # else we need to sample loss concentrated points
        parental_idx = np.random.choice(np.arange(self.size), size=N_g, p=self.distribution.ravel())
        parents = torch.from_numpy(self.points[parental_idx])
        children = torch.normal(mean=parents, std=self.sigma)    
        outside = ~self.inside(children)
        oob_count = sum(outside) # out of bounds, has to be resampled
        self.oob_count_global = oob_count
        while oob_count != 0:
            print(oob_count)
            children[outside] = torch.normal(mean=parents[outside], std=self.sigma)
            #this is long af
            outside = ~self.inside(children)
            oob_count = sum(outside)
            self.oob_count_global += oob_count
        children = children.numpy()
        self.points = np.vstack([children, uniform_points])
        # shuffle?
        self._R_step()
        return self.points
