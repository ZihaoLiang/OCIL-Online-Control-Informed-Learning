import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/src')
sys.path.append(os.getcwd() + '/externals/Pontryagin-Differentiable-Programming')
from OCIL import OCIL
from JinEnv import JinEnv



# ------------------------------ Set up dynamic system ------------------------------
project = "Quadrotor"
mode = "SysID"
saveFlag = False
dynsys = JinEnv.Quadrotor()
dynsys.initDyn(c=0.01)
dynsys.initCost(wr=1, wv=1, wq=5, ww=1, wthrust=0.1)

dir = 'examples/SysID/uav/data/'
demoFile = 'uav_demos.mat'

system = OCIL(project, mode, dynsys, dir, demoFile, saveFlag)

# --------------------------- initilize EKF ----------------------------------------
P = np.eye(5) * 0.0000001
Q = np.eye(5) * 0.
R = np.eye(17) * 0.0000000001

system.initialize_EKF(P, Q, R)

# system.solve()
system.solveAllLoss()

