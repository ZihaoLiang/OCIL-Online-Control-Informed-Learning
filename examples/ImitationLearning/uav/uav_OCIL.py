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
mode = "Imitation Learning"
saveFlag = True
dynsys = JinEnv.Quadrotor()
dynsys.initDyn(Jx=1, Jy=1, Jz=1, mass=1, l=0.4, c=0.01)
dynsys.initCost(wthrust=0.1)

dir = 'examples/ImitationLearning/uav/data/'
demoFile = 'uav_demos.mat'

system = OCIL(project, mode, dynsys, dir, demoFile, saveFlag)

# --------------------------- initilize EKF ----------------------------------------
P = np.eye(4) * 0.0000001
Q = np.eye(4) * 0.
R = np.eye(17) * 0.0000000001

system.initialize_EKF(P, Q, R)

# system.solve()
system.solveAllLoss()

