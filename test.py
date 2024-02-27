import os

os.environ["DDEBACKEND"] = "pytorch"

import deepxde as dde
import numpy as np

from src.pde.burgers import Burgers1D, Burgers2D
from src.utils.callbacks import LossCallback, TesterCallback, PlotCallback, PDEPointAdaptiveResampler
from torch.optim import Adam


pde = Burgers1D()
net = dde.nn.FNN([pde.input_dim] + [100 for _ in range(5)]  + [pde.output_dim], "tanh", "Glorot normal")
net = net.float()
opt = Adam(net.parameters(), 1e-3)
model = pde.create_model(net)
model.compile(opt, loss_weights=np.ones(pde.num_loss))
model.train(
        iterations=10,
        display_every=1,
        callbacks=[
            LossCallback(verbose=True, loss_plot=True),
            TesterCallback(log_every=1),
            ] + [
                PDEPointAdaptiveResampler(
                    verbose=True,
                    **dict(method='breed', period=2, 
                        breed=dict(sigma=0.0, start=0.0, end=1.0, breakpoint=0.5, oob_factor=1.0))
                    )
            ],
        model_save_path='runs/test2d')
