# PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs

This repository is our codebase for [PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs](https://arxiv.org/abs/2306.08827). Our paper is currently under review. We will provide more detailed guide soon.

<p align="center">
  <img width="80%" src="https://raw.githubusercontent.com/i207M/PINNacle/master/resources/pinnacle.png"/>
</p>

### Implemented Methods

This benchmark paper implements the following variants and create a new challenging dataset to compare them,

| Method                                                       | Type                                         |
| ------------------------------------------------------------ | -------------------------------------------- |
| [PINNs](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125) | Vanilla PINNs                                |
| PINNs(Adam+L-BFGS)                                           | Vanilla PINNs                                |
| [PINN-LRA](https://arxiv.org/abs/2001.04536)                 | Loss reweighting                             |
| [PINN-NTK](https://arxiv.org/abs/2007.14527)                 | Loss reweighting                             |
| [RAR](https://arxiv.org/abs/2207.10289)                      | Collocation points resampling                |
| [MultiAdam](https://arxiv.org/abs/2306.02816)                | New optimizer                                |
| [gPINNs](https://arxiv.org/abs/2111.02801)                   | New loss functions (regularization terms)    |
| [hp-VPINNs](https://arxiv.org/abs/2003.05385)                | New loss functions (variational formulation) |
| [LAAF](https://royalsocietypublishing.org/doi/10.1098/rspa.2020.0334) | New architecture (activation)                |
| [GAAF](https://arxiv.org/abs/1906.01170)                     | New architecture (activation)                |
| [FBPINN](https://arxiv.org/abs/2107.07871)                   | New architecture (domain decomposition)      |

See these references for more details,

- [Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125)
- [Understanding and mitigating gradient pathologies in physics-informed neural networks](Understanding and mitigating gradient pathologies in physics-informed neural networks)
- [When and why PINNs fail to train: A neural tangent kernel perspective](https://arxiv.org/abs/2007.14527)
- [A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks](https://arxiv.org/abs/2207.10289)
- [MultiAdam: Parameter-wise Scale-invariant Optimizer for Multiscale Training of Physics-informed Neural Networks](https://arxiv.org/abs/2306.02816)
- [Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems](https://arxiv.org/abs/2111.02801)
- [Sobolev Training for Physics Informed Neural Networks](https://arxiv.org/abs/2101.08932)
- [Variational Physics-Informed Neural Networks For Solving Partial Differential Equations](https://arxiv.org/abs/1912.00873)
- [hp-VPINNs: Variational Physics-Informed Neural Networks With Domain Decomposition](https://arxiv.org/abs/2003.05385)
- [Locally adaptive activation functions with slope recovery for deep and physics-informed neural networks](https://royalsocietypublishing.org/doi/10.1098/rspa.2020.0334)
- [Adaptive activation functions accelerate convergence in deep and physics-informed neural networks](https://arxiv.org/abs/1906.01170)
- [Finite Basis Physics-Informed Neural Networks (FBPINNs): a scalable domain decomposition approach for solving differential equations](https://arxiv.org/abs/2107.07871)



## Installation

```shell
pip install -r requirements.txt
```

## Usage

[📄 Full Documention](https://pinnacle-docs.vercel.app/)

Run all 22 cases with default settings:

```shell
python benchmark.py [--name EXP_NAME] [--seed SEED] [--device DEVICE]
```

<!-- Please read the comments in `benchmark.py` to modify the experiment configuration and apply different methods.

We are actively iterating the CLI design to make it easy to use.

### Multi-GPU Training

One can use `--device` to specify the GPU devices to use. If more than one GPU is specified, the program will run multiple experiments in parallel. -->



## Citation

If you find out work useful, please cite our paper at:

```
@article{hao2023pinnacle,
  title={PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs},
  author={Hao, Zhongkai and Yao, Jiachen and Su, Chang and Su, Hang and Wang, Ziao and Lu, Fanzhi and Xia, Zeyu and Zhang, Yichi and Liu, Songming and Lu, Lu and others},
  journal={arXiv preprint arXiv:2306.08827},
  year={2023}
}

@article{hao2022physics,
  title={Physics-informed machine learning: A survey on problems, methods and applications},
  author={Hao, Zhongkai and Liu, Songming and Zhang, Yichi and Ying, Chengyang and Feng, Yao and Su, Hang and Zhu, Jun},
  journal={arXiv preprint arXiv:2211.08064},
  year={2022}
}
```
