import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/src')
sys.path.append(os.getcwd() + '/externals/Pontryagin-Differentiable-Programming')
sys.path.append(os.getcwd() + '/externals/Pontryagin-Differentiable-Programming/ControlTool')
import OCIL
from JinEnv import JinEnv
from ControlTool import ControlTools


if __name__ == '__main__':
# ------------------------------ Set up dynamic system ------------------------------
    project = "Rocket"
    saveFlag = False
    dynsys = JinEnv.Rocket()
    dynsys.initDyn(Jx=0.5, Jy=1, Jz=1, mass=1, l=1)
    dynsys.initCost(wr=1, wv=1, wtilt=50, ww=1, wsidethrust=1, wthrust=0.1)

    dir = 'examples/PolicyTuning/rocket/data/'
    demoFile = 'rocket_iLQR.mat'

# ------------------------------  ------------------------------
    # initial iLQR solver
    MyiLQR = ControlTools.iLQR(project_name=project)
    