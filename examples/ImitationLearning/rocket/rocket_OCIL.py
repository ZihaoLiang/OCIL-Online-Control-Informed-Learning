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
mode = "All"
saveFlag = False
dynsys = JinEnv.Rocket()
dynsys.initDyn()
dynsys.initCost(wthrust=0.1)

dir = 'examples/ImitationLearning/rocket/data/'
demoFile = 'rocket_demos.mat'

system = OCIL.ImitationLearning(project, mode, dynsys, dir, demoFile, saveFlag)
system.set_iteration(333)

# --------------------------- initilize EKF ----------------------------------------
P = np.eye(10) * 0.0000001
Q = np.eye(10) * 0.
R = np.eye(16) * 0.0000000001

system.initialize_EKF(P, Q, R)

system.solve()

