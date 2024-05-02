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
mode = "SysID"
saveFlag = False
dynsys = JinEnv.CartPole()
dynsys.initDyn()
dynsys.initCost(wx=1,wq=6,wdx=1,wdq=1,wu = 0.1)
dt = 0.05

dir = 'examples/SysID/cartpole/data/'
demoFile = 'cartpole_iodata.mat'

system = OCIL.SysID(project, mode, dynsys, dt, dir, demoFile, saveFlag)

# --------------------------- initilize EKF ----------------------------------------
P = np.eye(3) * 0.0000001
Q = np.eye(3) * 0.
R = np.eye(4) * 0.0000000001

system.initialize_EKF(P, Q, R)

# system.solve()
system.solveAllLoss()

