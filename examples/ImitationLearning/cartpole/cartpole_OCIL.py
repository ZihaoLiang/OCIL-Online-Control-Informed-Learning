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
mode = "All"
saveFlag = False
dynsys = JinEnv.CartPole()
# dynsys.initDyn(mc=0.5, mp=0.5, l=1)
dynsys.initDyn()
dynsys.initCost(wu = 0.1)

dir = 'examples/ImitationLearning/cartpole/data/'
demoFile = 'cartpole_demos.mat'

system = OCIL.ImitationLearning(project, mode, dynsys, dir, demoFile, saveFlag)
# system.set_sigma(0.7)
system.set_iteration(333)

# --------------------------- initilize EKF ----------------------------------------
P = np.eye(7) * 0.0000001
Q = np.eye(7) * 0.
R = np.eye(5) * 0.0000000001

system.initialize_EKF(P, Q, R)

system.solve()

