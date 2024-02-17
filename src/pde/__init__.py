from src.pde.burgers import Burgers1D, Burgers2D
from src.pde.chaotic import GrayScottEquation, KuramotoSivashinskyEquation
from src.pde.heat import Heat2D_VaryingCoef, Heat2D_Multiscale, Heat2D_ComplexGeometry, Heat2D_LongTime, HeatND
from src.pde.ns import NS2D_LidDriven, NS2D_BackStep, NS2D_LongTime
from src.pde.poisson import Poisson2D_Classic, PoissonBoltzmann2D, Poisson3D_ComplexGeometry, Poisson2D_ManyArea, PoissonND
from src.pde.wave import Wave1D, Wave2D_Heterogeneous, Wave2D_LongTime

pde_list = \
    [Burgers1D, Burgers2D] + \
    [Poisson2D_Classic, PoissonBoltzmann2D, Poisson3D_ComplexGeometry, Poisson2D_ManyArea] + \
    [Heat2D_VaryingCoef, Heat2D_Multiscale, Heat2D_ComplexGeometry, Heat2D_LongTime] + \
    [NS2D_LidDriven, NS2D_BackStep, NS2D_LongTime] + \
    [Wave1D, Wave2D_Heterogeneous, Wave2D_LongTime] + \
    [KuramotoSivashinskyEquation, GrayScottEquation] + \
    [PoissonND, HeatND]

fast_pde_list = [Burgers1D, Poisson2D_Classic, Wave1D, PoissonBoltzmann2D] # these are the fastest pde to process
