import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/src')
sys.path.append(os.getcwd() + '/externals/Pontryagin-Differentiable-Programming')
import OCIL
from JinEnv import JinEnv


# ------------------------------ Set up dynamic system ------------------------------
project = "CartPole"
mode = "Policy Tuning"
case = "traj"
saveFlag = False
dynsys = JinEnv.CartPole()
dynsys.initDyn(mc=0.5, mp=0.5, l=1)
dynsys.initCost(wx=1, wq=6, wdx=1, wdq=1, wu=0.1)

dir = 'examples/PolicyTuning/cartpole/data/'
demoFile = 'cartpole_demos.mat'

# --------------------------- initilize ----------------------------------------
P = np.eye(369) * 0.000000001
Q = np.eye(369) * 0.
R = np.eye(5) * 0.0000000001

nnFactor = 4

system = OCIL.PolicyTuning(project, mode, case, dynsys, nnFactor, dir, demoFile, saveFlag)
system.initialize_EKF(P, Q, R)

# system.solve()
system.solveAllLoss()
