import torch
import torch.nn as nn
import numpy as np
from scipy.stats import multivariate_normal


class BaseSampler:
    def __init__(self, pde):
        '''
        pde: `deepxde.data.PDE` object (or child TimePDE)
            The object that defines PDE, its training points and domain.
        '''
        self.domain = pde.geom
        self.points = pde.train_x_all
        self.size = len(self.points)
        self.dim = self.domain.dim

    def get_points(self):
        '''Method to update training set of PDE object'''
        pass


class Breed(BaseSampler):
    """Class for a sampler proposed by authors. Specially written for DeepXde library.
    Standalone version for Pytorch can be found here: https://gitlab.inria.fr/breed/breed

    Attributes:
     
    sigma: float
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
    def __init__(self, pde, sigma=None, start=0.15, end=0.75, breakpoint=10):
        """
        Args:
        
        pde: deepxde.data.PDE class object (or child TimePDE)
            The object that defines PDE, its training points and domain.
        sigma: float or None
            The covariance value for Gaussian neighbourhoods.
            Default value `None` calculates maximum possible in the domain value.
        start, end: float
            The first and last R value to create the R values series
        breakpoint: int
            The value specifies R value scenario, how many values are 
            linearly increasing from start to end, after which R values are constant (=end)
        """
        super().__init__(pde)

        self._sigma_init(sigma)
        self.Rs = np.linspace(start, end, breakpoint, endpoint=True)
        self.R_i = -1
        self.R = start
        self.oob_count = []

    def inside(self, x):
        if hasattr(self.domain, 'inside'):
            return self.domain.inside(x)
        else:
            return np.logical_and(self.domain.geometry.inside(x[:,:-1]), self.domain.timedomain.inside(x[:,-1]))

    def _sigma_init(self, sigma):
        max_sigma = np.diff(self.domain.bbox, axis=0).min() / 8 # six sigma rule to minimize outsiders
        if sigma > max_sigma or sigma is None:
            self.sigma = max_sigma
            print(f"WARNING: Given value of sigma is {'too big' if sigma is not None else 'not given'} ({sigma}), it is updated to {max_sigma}")
        else:
            self.sigma = sigma

    def _R_step(self):
        self.R_i += 1
        if self.R_i < (len(self.Rs) - 1):
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
        self.oob_count.append(0)
        for i, idx in enumerate(parental_idx):
            if np.random.uniform(0, 1) < self.R:
                parent = self.points[idx]
                child = multivariate_normal.rvs(mean=parent, cov=self.sigma, size=1)[None] # shape (1, D), TODO sigma is a vector
                while not self.inside(child)[0]:
                    self.oob_count[-1] += 1
                    child = multivariate_normal.rvs(mean=parent, cov=self.sigma, size=1)[None]
            else:
                child = self.domain.random_points(1)
            new_sample_batch[i] = child
        self.points = new_sample_batch # shuffled naturally?
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
