import argparse
import time
import os
import sys
from trainer import Trainer

os.environ["DDEBACKEND"] = "pytorch"

import numpy as np
import torch
import deepxde as dde
from src.model.laaf import DNN_GAAF, DNN_LAAF
from src.optimizer import MultiAdam, LR_Adaptor, LR_Adaptor_NTK, Adam_LBFGS
from src.pde.ac import AllenCahn1D
from src.utils.args import parse_hidden_layers, parse_loss_weight
from src.utils.callbacks import TesterCallback, PlotCallback, LossCallback, PDEPointAdaptiveResampler, wrap_ReduceLROnPlateau
from src.utils.rar import rar_wrapper

pde_list = [AllenCahn1D]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PINNBench trainer')
    parser.add_argument('--name', type=str, default="benchmark")
    parser.add_argument('--device', type=str, default="0")  # set to "cpu" enables cpu training 
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--hidden-layers', type=str, default="100*5")
    parser.add_argument('--loss-weight', type=str, default="")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-decay', action="store_true")
    parser.add_argument('--iter', type=int, default=10000)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--plot-every', type=int, default=2000)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--fourier', action="store_true")
    parser.add_argument('--method', choices=['multiadam', 'lra', 'ntk']) # default is None: FNN net and Adam opt
    parser.add_argument('--resample-method', choices=['breed', 'rarg', 'rard', 'rad', 'r3', 'full']) # default is None: no resampling
    parser.add_argument('--resample-period', type=int, default=1000)
    parser.add_argument('--breed-args', nargs=5, type=float, default=[0.0, 0.0, 1.0, 1.0, 1.0], metavar=('sigma', 'start', 'end', 'breakpoint', 'oob_factor'),
                                        help='''Arguments for resampler method `breed`. All values must be between 0 an 1 (incl.).
                                        `sigma` - neighbourhood width value for sampling, for value <= 0 it is calculated automatically
                                        `start`, `end`, `breakpoint` - R-value scenario parameters. `breakpoint` is a percentage of resample
                                                                       iterations during which R-value linearly grows from `start` to `end`
                                        `oob_factor` - sigma will be updated as sigma*oob_factor for a point that sampled an out-of-bounds point''')
    command_args = parser.parse_args()

    command_args.breed_args[3] = int((command_args.iter // command_args.resample_period) * command_args.breed_args[3])
    seed = command_args.seed
    if seed is not None:
        dde.config.set_random_seed(seed)
    date_str = time.strftime('%m.%d-%H.%M.%S', time.localtime())
    trainer = Trainer(f"{date_str}-{command_args.name}", command_args.device)

    for pde_config in pde_list:

        def get_model_dde():
            if isinstance(pde_config, tuple):
                pde = pde_config[0](**pde_config[1])
            else:
                pde = pde_config()

            def input_transform(data):
                if command_args.fourier:
                    kw = torch.arange(1, 11) * torch.pi
                    x, t = data[:,:1], data[:,1:]
                    out = torch.cat([t, torch.ones_like(t), torch.cos(kw * x), torch.sin(kw * x)], dim=-1)
                    return out
                else:
                    return data

            pde_input_encoding_dim = 22 if command_args.fourier else 2
            net = dde.nn.FNN([pde_input_encoding_dim] + parse_hidden_layers(command_args) + [pde.output_dim], "tanh", "Glorot normal")
            net.apply_feature_transform(input_transform)
            net = net.float()

            loss_weights = parse_loss_weight(command_args)
            if loss_weights is None:
                loss_weights = np.ones(pde.num_loss)
            else:
                loss_weights = np.array(loss_weights)

            opt = torch.optim.Adam(net.parameters(), command_args.lr)
            if command_args.method == "multiadam":
                opt = MultiAdam(net.parameters(), lr=1e-3, betas=(0.99, 0.99), loss_group_idx=[pde.num_pde])
            elif command_args.method == "lra":
                opt = LR_Adaptor(opt, loss_weights, pde.num_pde)
            elif command_args.method == "ntk":
                opt = LR_Adaptor_NTK(opt, loss_weights, pde)

            model = pde.create_model(net)
            model.compile(opt, loss_weights=loss_weights)
            return model

        def get_model_others():
            model = None
            # create a model object which support .train() method, and param @model_save_path is required
            # create the object based on command_args and return it to be trained
            # schedule the task using trainer.add_task(get_model_other, {training args})
            return model

        callbacks = [
                TesterCallback(log_every=command_args.log_every),
                PlotCallback(log_every=command_args.plot_every, fast=True),
                LossCallback(loss_plot=True, verbose=True)
                ]

        if command_args.lr_decay:
            scheduler = lambda opt: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=100, threshold=0.0001, threshold_mode='rel', min_lr=1e-5)
            callbacks.append(wrap_ReduceLROnPlateau(scheduler))

        if command_args.resample_method is not None:
            # TODO hotfix argparse/configparse...
            resampler_params = {
                    "method" : command_args.resample_method,
                    "period" : command_args.resample_period,
                    "density_mul" : 2,
                    "m" : 1, # rarg/rard how many points to add
                    "breed" : dict(zip(["sigma", "start", "end", "breakpoint", "oob_factor"],
                                       command_args.breed_args)) 
                    }
            callbacks.append(
                    PDEPointAdaptiveResampler(
                        verbose=False,
                        plot_verbose=False,
                        **resampler_params)
                    )

        trainer.add_task(
            get_model_dde, {
                "iterations": command_args.iter,
                "display_every": command_args.log_every,
                "callbacks": callbacks 
            }
        )

    commandline = " ".join(f"'{i}'" if " " in i else i for i in sys.argv)
    trainer.setup(__file__, seed, commandline)
    trainer.set_repeat(command_args.repeat)
    trainer.train_all()
    trainer.summary()
