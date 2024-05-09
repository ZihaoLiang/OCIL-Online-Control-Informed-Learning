import numpy as np
from casadi import *
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/comparisons/Iterative IOC')
from dynamics_env import Pend, CartPole, UAV, Rocket, toQuaternion
from OptimalControl import OC
from IOC import IOC
    
dyn = 'Rocket' #choose model dynamics: 'Pendulum', 'CartPole', 'UAV', 'Rocket'

############################### pendulum ###############################
if dyn == 'Pendulum':
    demoFile = 'examples/ImitationLearning/pendulum/data/pendulum_demos.mat'

    omega = [10, 1, 0.001]

    data = sio.loadmat(demoFile)
    trajectories = data['trajectories']
    dt = data['dt']  
    demo_state_traj = trajectories[0, 0]['state_traj_opt'][0, 0]
    demo_control_traj = trajectories[0, 0]['control_traj_opt'][0, 0]
    demo_ini_state = demo_state_traj[0, :]
    demo_target_state = demo_state_traj[-1, :]
    demo_horizon = demo_control_traj.shape[0]

    start = 5


############################### CartPole ###############################    
elif dyn == 'CartPole': 

    demoFile = 'examples/ImitationLearning/cartpole/data/cartpole_demos.mat'

    omega = [1, 6, 1, 1, 0.1]

    data = sio.loadmat(demoFile)
    trajectories = data['trajectories']
    dt = data['dt']  
    demo_state_traj = trajectories[0, 0]['state_traj_opt'][0, 0]
    demo_control_traj = trajectories[0, 0]['control_traj_opt'][0, 0]
    demo_ini_state = demo_state_traj[0, :]
    demo_target_state = demo_state_traj[-1, :]
    demo_horizon = len(demo_control_traj[:,0])

    start = 8
   
############################### UAV ###############################    
elif dyn == 'UAV': 

    demoFile = 'examples/ImitationLearning/uav/data/uav_demos.mat'

    omega = [1, 1, 5, 1, 0.1]

    data = sio.loadmat(demoFile)
    trajectories = data['trajectories']
    dt = data['dt']  
    demo_state_traj = trajectories[0, 0]['state_traj_opt'][0, 0]
    demo_control_traj = trajectories[0, 0]['control_traj_opt'][0, 0]
    demo_ini_state = demo_state_traj[0, :]
    demo_target_state = demo_state_traj[-1, :]
    demo_horizon = len(demo_control_traj[:,0])

    start = 5

############################### Rocket ###############################    
elif dyn == 'Rocket': 

    demoFile = 'examples/ImitationLearning/rocket/data/rocket_demos.mat'

    omega = [1, 1, 50, 1, 1, 0.1]

    data = sio.loadmat(demoFile)
    trajectories = data['trajectories']
    dt = data['dt']  
    demo_state_traj = trajectories[0, 0]['state_traj_opt'][0, 0]
    demo_control_traj = trajectories[0, 0]['control_traj_opt'][0, 0]
    demo_ini_state = demo_state_traj[0, :]
    demo_target_state = demo_state_traj[-1, :]
    demo_horizon = len(demo_control_traj[:,0])

    start = 8

############################### Perform optimal control ###############################  

OCsys = OC()
OCsys.getTraj(dyn, demo_ini_state, demo_target_state, demo_horizon, dt, omega)

iter = [*range(len(demo_state_traj))]
fig, axs = plt.subplots(len(demo_state_traj[0]),1)
for idx in range(len(demo_state_traj[0])):
    axs[idx].plot(np.concatenate(OCsys.x_his,axis = 1)[idx,:], 'b')
    axs[idx].plot(demo_state_traj[:,idx], 'r')
    axs[idx].set_ylabel("x"+str(idx+1))
axs[-1].set_xlabel("Iteration")
axs[0].set_title("State Trajectory")

iter = [*range(len(demo_control_traj))]
if len(demo_control_traj[0]) == 1:
    fig, axs = plt.subplots()
    axs.plot(np.concatenate(OCsys.u_his,axis = 1), 'b')
    axs.plot(demo_control_traj, 'r')
    axs.set_ylabel("u")
    axs.set_xlabel("Iteration")
    axs.set_title("Control Trajectory")
else:
    fig, axs = plt.subplots(len(demo_control_traj[0]),1)
    for idx in range(len(demo_control_traj[0])):
        axs[idx].plot(np.concatenate(OCsys.x_his,axis = 1)[idx,:])
        axs[idx].plot(demo_control_traj[:,idx])
        axs[idx].set_ylabel("x"+str(idx+1))
    axs[-1].set_xlabel("Iteration")
    axs[0].set_title("Control Trajectory")
plt.show()

############################### IOC ###############################    
demo_state_traj = OCsys.x_his
demo_control_traj = OCsys.u_his


Loss = []
iter = []
for idx in range(demo_horizon):
    if idx < start:
        continue
    
    IOCsys = IOC()
    IOCsys.getdPhi(demo_state_traj, demo_control_traj, OCsys.PHI)
    IOCsys.getdf(demo_state_traj, demo_control_traj, OCsys.DYN)
    IOCsys.IOC_main(IOCsys.dfdx_his, IOCsys.dfdu_his, IOCsys.dphidx_his, IOCsys.dphidu_his, idx)
    omega = IOCsys.omega / min(IOCsys.omega) #normalize omega
    print('Omega =',omega)

    System = OC()
    System.getTraj(dyn, demo_ini_state, demo_target_state, demo_horizon, dt, omega)
    x_his = System.x_his
    u_his = System.u_his

    iter += [idx+1]
    loss = 0
    for jdx in range(demo_horizon):
        each_traj_t = np.vstack((x_his[jdx], u_his[jdx]))
        demo_traj_t = np.vstack((demo_state_traj[jdx], demo_control_traj[jdx]))
        lossNorm = norm_2(each_traj_t-demo_traj_t)**2
        loss += lossNorm

    Loss += [np.asarray(loss)[0,0]]



fig, axs = plt.subplots()
axs.plot(iter, Loss)
plt.yscale("log")
plt.show()
