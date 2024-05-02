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
project = "Rocket"
mode = "SysID"
saveFlag = False
dynsys = JinEnv.Rocket()
dynsys.initDyn()
dynsys.initCost(wr=1, wv=1, wtilt=50, ww=1, wsidethrust=1, wthrust = 0.1)
dt = 0.05

dir = 'examples/SysID/rocket/data/'
demoFile = 'rocket_iodata.mat'

system = OCIL.SysID(project, mode, dynsys, dt, dir, demoFile, saveFlag)

# --------------------------- initilize EKF ----------------------------------------
P = np.eye(5) * 0.0000001
Q = np.eye(5) * 0.
R = np.eye(13) * 0.0000000001

system.initialize_EKF(P, Q, R)

# system.solve()
system.solveAllLoss()

