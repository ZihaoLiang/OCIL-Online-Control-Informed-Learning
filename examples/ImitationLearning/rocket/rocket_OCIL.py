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
mode = "Objective"
saveFlag = False
dynsys = JinEnv.Rocket()
dynsys.initDyn(Jx=0.5, Jy=1, Jz=1, mass=1, l=1)
dynsys.initCost(wthrust = 0.1)

dir = 'examples/ImitationLearning/rocket/data/'
demoFile = 'rocket_demos.mat'

system = OCIL.ImitationLearning(project, mode, dynsys, dir, demoFile, saveFlag)

# --------------------------- initilize EKF ----------------------------------------
P = np.eye(5) * 0.0000001
Q = np.eye(5) * 0.
R = np.eye(16) * 0.0000000001

system.initialize_EKF(P, Q, R)

# system.solve()
system.solveAllLoss()

