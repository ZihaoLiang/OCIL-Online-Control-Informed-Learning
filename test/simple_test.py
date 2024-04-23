import numpy as np
from casadi import *
import matplotlib.pyplot as plt
from dynamics_env import LTI, Pend, RobotArm, UAV, toQuaternion
from OptimalControl import OC
from IOC import IOC
    

##################### Set up #####################
dyn = 'Dot'
init = [2, 1]
target = [0, 0]  
   
############################### Perform optimal control ###############################    
T = 5 # number of control intervals
dt = 0.1
nT = int(10/0.1)

OCsys = OC()
OCsys.getTraj(dyn, init, target, T)
# Ot
x_his = OCsys.x_his
u_his = OCsys.u_his


x_his = np.concatenate(x_his,axis = 1)
u_his = np.concatenate(u_his,axis = 1)

fig, axs = plt.subplots(2)

axs[0].plot(x_his[0,:], label = 'x1')
axs[0].plot(x_his[1,:], label = 'x2')
axs[0].legend()
axs[0].grid()
axs[0].set(ylabel = 'x')

axs[1].plot(u_his[0,:], label = 'u')
axs[1].set(ylabel = 'u')
axs[1].legend()
axs[1].grid()

plt.xlabel('t')
plt.show()
    