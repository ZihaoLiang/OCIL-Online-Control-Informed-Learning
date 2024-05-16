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
mode = "Policy Tuning"
case = "Objective"
saveFlag = False
dynsys = JinEnv.Quadrotor()
dynsys.initDyn(Jx=1, Jy=1, Jz=1, mass=1, l=0.4, c=0.01)
dynsys.initCost(wr=1, wv=1, wq=5, ww=1)

dir = 'examples/PolicyTuning/uav/data/'
demoFile = 'uav_demos.mat'

# --------------------------- initilize ----------------------------------------
P = np.eye(1174) * 0.000000001
Q = np.eye(1174) * 0.
R = np.eye(17) * 0.0000000001

nnFactor = 2

system = OCIL.PolicyTuning(project, mode, case, dynsys, nnFactor, dir, demoFile, saveFlag)
system.set_iteration(2)
system.initialize_EKF(P, Q, R)

dt = 0.1
horizon = 35
ini_r_I = [-8, -6, 9.]
ini_v_I = [0.0, 0.0, 0.0]
ini_q = JinEnv.toQuaternion(0, [1, -1, 1])
ini_w = [0.0, 0.0, 0.0]
ini_state = ini_r_I + ini_v_I + ini_q + ini_w
system.generate_traj(dt, horizon, ini_state)

system.solve()
