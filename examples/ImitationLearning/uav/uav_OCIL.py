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
project = "Quadrotor"
mode = "All"
saveFlag = False
dynsys = JinEnv.Quadrotor()
dynsys.initDyn(c=0.01)
dynsys.initCost(wthrust=0.1)

dir = 'examples/ImitationLearning/uav/data/'
demoFile = 'uav_demos.mat'

system = OCIL.ImitationLearning(project, mode, dynsys, dir, demoFile, saveFlag)
system.set_sigma(0.8)

# --------------------------- initilize EKF ----------------------------------------
P = np.eye(9) * 0.0000001
Q = np.eye(9) * 0.
R = np.eye(17) * 0.0000000001

system.initialize_EKF(P, Q, R)

# system.solve()
system.solveAllLoss()

