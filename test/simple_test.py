import numpy as np
from casadi import *
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.getcwd() + '/src')
from dynamics_env import LTI, Pend, RobotArm, UAV, toQuaternion, Dot
from OptimalControl import OC
from Loss_function import Loss
    

##################### Set up #####################
dyn = 'Dot'
init = [2, 1]
target = [0, 0]  

weight_init = [1, 1, 1, 1]
   
############################### Perform optimal control ###############################    
T = 5 # number of control intervals
dt = 0.1
nT = int(T/0.1)

weight_oc = [2, 2, 1, 1]
OCsys = OC()
OCsys.getTraj(dyn, weight_oc, init, target, T)
# Ot
x_his_oc = OCsys.x_his
u_his_oc = OCsys.u_his

x_his_oc = np.concatenate(x_his_oc,axis = 1)
u_his_oc = np.concatenate(u_his_oc,axis = 1)

lossFun = Loss(len(x_his_oc), len(u_his_oc))
weight = weight_init

for idx in range(nT):
    # perform oc with current weight
    OCsys.getTraj(dyn, weight, init, target, T)
    x_his = OCsys.x_his
    u_his = OCsys.u_his
    x_his = np.concatenate(x_his,axis = 1)
    u_his = np.concatenate(u_his,axis = 1)
    
    # Traj at current t
    traj_oc_t = np.hstack((x_his_oc[:,idx], u_his_oc[:,idx]))
    traj_t = np.hstack((x_his[:,idx], u_his[:,idx]))

    # Loss function, dLdXi
    loss, dLdXi = lossFun.getLossValue(traj_t, traj_oc_t)

    # Gradient generator
    dXidTheta = 0

    # Chain rule
    dLdTheta = dLdXi * dXidTheta

    # Kalman gain
    K = 0

    # Update weight using EKF
    weight = weight


fig, axs = plt.subplots(2)

# axs[0].plot(x_his[0,:], label = 'x1')
# axs[0].plot(x_his[1,:], label = 'x2')
axs[0].plot(x_his_oc[0,:], label = 'x1_oc')
axs[0].plot(x_his_oc[1,:], label = 'x2_oc')
axs[0].legend()
axs[0].grid()
axs[0].set(ylabel = 'x')

# axs[1].plot(u_his[0,:], label = 'u1')
# axs[1].plot(u_his[1,:], label = 'u2')
axs[1].plot(u_his_oc[0,:], label = 'u1_oc')
axs[1].plot(u_his_oc[1,:], label = 'u2_oc')
axs[1].set(ylabel = 'u')
axs[1].legend()
axs[1].grid()

plt.xlabel('t')
plt.show()
    