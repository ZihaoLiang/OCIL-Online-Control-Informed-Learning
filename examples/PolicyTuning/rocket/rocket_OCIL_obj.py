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
mode = "Policy Tuning"
case = "Objective"
saveFlag = True
dynsys = JinEnv.Rocket()
dynsys.initDyn(Jx=0.5, Jy=1, Jz=1, mass=1, l=1)
dynsys.initCost(wr=1, wv=1, wtilt=50, ww=1, wsidethrust=1, wthrust=0.4)

dir = 'examples/PolicyTuning/rocket/data/'
demoFile = 'rocket_demos.mat'

# --------------------------- initilize ----------------------------------------
P = np.eye(1147) * 0.000000001
Q = np.eye(1147) * 0.
R = np.eye(16) * 0.0000000001

nnFactor = 2

system = OCIL.PolicyTuning(project, mode, case, dynsys, nnFactor, dir, demoFile, saveFlag)
system.set_iteration(100)
system.initialize_EKF(P, Q, R)
dt = 0.1
horizon = 50
ini_r_I = [10, -8, 5.]
ini_v_I = [-.1, 0.0, -0.0]
ini_q = JinEnv.toQuaternion(1.5, [0, 0, 1])
ini_w = [0, -0.0, 0.0]
ini_state = ini_r_I + ini_v_I + ini_q + ini_w
system.generate_traj(dt, horizon, ini_state)

system.solve()
