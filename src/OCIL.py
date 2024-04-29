import numpy as np
from casadi import *
import scipy.io as sio
import matplotlib.pyplot as plt 
import os, shutil
import sys
sys.path.append(os.getcwd() + '../externals/Pontryagin-Differentiable-Programming')
from PDP import PDP
from JinEnv import JinEnv
from EKF import EKF
from Loss_function import Loss


class OCIL:
    def __init__(self, project="", mode="", dynsys=None, dir="", demoFile="", saveFlag=False):

        if not (mode == "Imitation Learning" or mode == "SysID"):
            print("Mode not defined!")
            sys.exit()

        self.dir = dir
        self.saveFlag = saveFlag
        self.plotTrajFlag = False
        if saveFlag:
            if os.path.exists(self.dir+"results/"):
                shutil.rmtree(self.dir+"results/")
            os.mkdir(self.dir+"results/")

        # ------------------------------ set up system ------------------------------
        self.project = project
        self.mode = mode
        self.dynsys = dynsys
        self.num_dyn_auxvar = dynsys.dyn_auxvar.shape[0]
        self.num_cost_auxvar = dynsys.cost_auxvar.shape[0]
        
        # ------------------------------ load demos data ------------------------------
        data = sio.loadmat(dir+demoFile)
        self.trajectories = data['trajectories']
        self.dt = data['dt']
        if mode == "Imitation Learning":
            self.true_theta = data['true_parameter'].flatten()
            self.true_theta = self.true_theta[len(self.true_theta)-self.num_cost_auxvar:]
        elif mode == "SysID":
            self.true_theta = data['true_parameter'].flatten()
            self.true_theta = self.true_theta[:self.num_dyn_auxvar]

        print(data['true_parameter'].flatten())
        print(self.true_theta)

        # ------------------------------ initialize Classes ------------------------------
        self.sysoc = PDP.OCSys()
        # sysoc.setAuxvarVariable(vertcat(dynsys.dyn_auxvar, dynsys.cost_auxvar)) #set which theta to learn
        if mode == "Imitation Learning":
            self.sysoc.setAuxvarVariable(self.dynsys.cost_auxvar)
        elif mode == "SysID":
            self.sysoc.setAuxvarVariable(self.dynsys.dyn_auxvar)
        self.sysoc.setControlVariable(self.dynsys.U)
        self.sysoc.setStateVariable(self.dynsys.X)
        self.dyn = self.dynsys.X + self.dt * self.dynsys.f
        self.sysoc.setDyn(self.dyn)
        self.sysoc.setPathCost(self.dynsys.path_cost)
        self.sysoc.setFinalCost(self.dynsys.final_cost)
        self.sysoc.diffPMP()
        self.lqr_solver = PDP.LQR()
        

        # ------------------------------ initilize tunable parameter ------------------------------
        self.sigma = 0.9
        self.initial_theta = self.true_theta + self.sigma * np.random.random(len(self.true_theta)) - self.sigma / 2
        self.theta = self.initial_theta
        print('theta = ', self.theta)

        self.loss = 0
        self.dp = np.zeros(self.theta.shape)
        self.demo_state_traj = self.trajectories[0, 0]['state_traj_opt'][0, 0]
        self.demo_control_traj = self.trajectories[0, 0]['control_traj_opt'][0, 0]
        self.demo_ini_state = self.demo_state_traj[0, :]
        self.demo_horizon = self.demo_control_traj.shape[0]

        # ------------------------------ other setup ------------------------------
        self.Loss_his = []
        self.theta_error = []
        self.iter_his = []

    def set_sigma(self, sigma):
        self.sigma = sigma
        self.initial_theta = self.true_theta + self.sigma * np.random.random(len(self.true_theta)) - self.sigma / 2

    def initialize_EKF(self, P, Q, R):
        self.P_prev = P
        self.Q_prev = Q
        self.R = R


    def solve(self):
        Loss = 0
        loss_his = []
        for idx in range(self.demo_horizon):
            # --------------------------- Trajectory based on current parameter guess ---------------------------------------- 
            traj = self.sysoc.ocSolver(ini_state=self.demo_ini_state, horizon=self.demo_horizon, auxvar_value = self.theta)

            # --------------------------- Gradient generator, dXidtheta ---------------------------------------- 
            aux_sys = self.sysoc.getAuxSys(state_traj_opt=traj['state_traj_opt'],
                                            control_traj_opt=traj['control_traj_opt'],
                                            costate_traj_opt=traj['costate_traj_opt'],
                                            auxvar_value = self.theta)
            self.lqr_solver.setDyn(dynF=aux_sys['dynF'], dynG=aux_sys['dynG'], dynE=aux_sys['dynE'])
            self.lqr_solver.setPathCost(Hxx=aux_sys['Hxx'], Huu=aux_sys['Huu'], Hxu=aux_sys['Hxu'], Hux=aux_sys['Hux'],
                                        Hxe=aux_sys['Hxe'], Hue=aux_sys['Hue'])
            self.lqr_solver.setFinalCost(hxx=aux_sys['hxx'], hxe=aux_sys['hxe'])
            aux_sol = self.lqr_solver.lqrSolver(numpy.zeros((self.sysoc.n_state, self.sysoc.n_auxvar)), self.demo_horizon)

            # take solution of the auxiliary control system
            dxdtheta_traj = aux_sol['state_traj_opt']
            dudtheta_traj = aux_sol['control_traj_opt']

            dxdtheta_t = dxdtheta_traj[idx]
            dudtheta_t = dudtheta_traj[idx]
            dxidtheta_t = np.vstack((dxdtheta_t, dudtheta_t))

            # --------------------------- Loss function, dLdXi ---------------------------------------- 
            state_traj = traj['state_traj_opt']
            control_traj = traj['control_traj_opt']

            xi = SX.sym("xi", self.dynsys.X.shape[0]+self.dynsys.U.shape[0])
            demo_traj = np.hstack((self.demo_state_traj[idx], self.demo_control_traj[idx]))
            current_traj = np.hstack((state_traj[idx], control_traj[idx]))

            loss = demo_traj - xi
            dLdXi = jacobian(loss, xi)
            lossFun = Function("lossFun", [xi], [loss])
            dLdXiFun = Function("dLdXiFun", [xi], [dLdXi])

            lossNow = lossFun(current_traj).full()
            dLdXiNow = dLdXiFun(current_traj).full()

            lossNorm = norm_2(lossNow)**2
            loss_his += [lossNorm]
            Loss += lossNorm

            # evaluate the loss
            dldx_traj = state_traj - self.demo_state_traj
            dldu_traj = control_traj - self.demo_control_traj
            
            # --------------------------- Chain rule ----------------------------------------
            dLdtheta = np.matmul(dLdXiNow, dxidtheta_t)[0]
            dp = dxidtheta_t

            # --------------------------- EKF ----------------------------------------
            updateTheta = EKF()
            updateTheta.predict(self.theta, self.P_prev, self.Q_prev)
            updateTheta.update(dp, self.R, lossNow)
            print('theta = ', updateTheta.theta)
            self.P_prev = updateTheta.P
            self.theta = updateTheta.theta
        
        if self.saveFlag:
            self.saveEach(idx+1, traj, loss_his)

    def solveAllLoss(self):
        for idx in range(self.demo_horizon):
            Loss = 0
            loss_his = []
            # --------------------------- Trajectory based on current parameter guess ---------------------------------------- 
            traj = self.sysoc.ocSolver(ini_state=self.demo_ini_state, horizon=self.demo_horizon, auxvar_value = self.theta)
            
            # --------------------------- Gradient generator, dXidtheta ---------------------------------------- 
            aux_sys = self.sysoc.getAuxSys(state_traj_opt=traj['state_traj_opt'],
                                            control_traj_opt=traj['control_traj_opt'],
                                            costate_traj_opt=traj['costate_traj_opt'],
                                            auxvar_value = self.theta)
            self.lqr_solver.setDyn(dynF=aux_sys['dynF'], dynG=aux_sys['dynG'], dynE=aux_sys['dynE'])
            self.lqr_solver.setPathCost(Hxx=aux_sys['Hxx'], Huu=aux_sys['Huu'], Hxu=aux_sys['Hxu'], Hux=aux_sys['Hux'],
                                        Hxe=aux_sys['Hxe'], Hue=aux_sys['Hue'])
            self.lqr_solver.setFinalCost(hxx=aux_sys['hxx'], hxe=aux_sys['hxe'])
            aux_sol = self.lqr_solver.lqrSolver(numpy.zeros((self.sysoc.n_state, self.sysoc.n_auxvar)), self.demo_horizon)

            # take solution of the auxiliary control system
            dxdtheta_traj = aux_sol['state_traj_opt']
            dudtheta_traj = aux_sol['control_traj_opt']

            dxdtheta_t = dxdtheta_traj[idx]
            dudtheta_t = dudtheta_traj[idx]
            dxidtheta_t = np.vstack((dxdtheta_t, dudtheta_t))

            # --------------------------- Loss function, dLdXi ---------------------------------------- 
            state_traj = traj['state_traj_opt']
            control_traj = traj['control_traj_opt']

            xi = SX.sym("xi", self.dynsys.X.shape[0]+self.dynsys.U.shape[0])
            demo_traj = np.hstack((self.demo_state_traj[idx], self.demo_control_traj[idx]))
            current_traj = np.hstack((state_traj[idx], control_traj[idx]))

            loss = demo_traj - xi
            dLdXi = jacobian(loss, xi)
            lossFun = Function("lossFun", [xi], [loss])
            dLdXiFun = Function("dLdXiFun", [xi], [dLdXi])

            lossNow = lossFun(current_traj).full()
            dLdXiNow = dLdXiFun(current_traj).full()

            for jdx in range(self.demo_horizon):
                each_traj_t = np.hstack((state_traj[jdx], control_traj[jdx]))
                demo_traj_t = np.hstack((self.demo_state_traj[jdx], self.demo_control_traj[jdx]))
                lossNorm = norm_2(each_traj_t-demo_traj_t)**2
                loss_his += [lossNorm]
                Loss += lossNorm
            
            self.Loss_his += [np.asarray(Loss)[0,0]]
            self.theta_error += [np.asarray(norm_2(self.theta-self.true_theta)**2)[0,0]]

            if self.saveFlag:
                self.saveEach(idx, traj, loss_his)

            if self.plotTrajFlag:
                self.plotTraj(state_traj, control_traj)

            # evaluate the loss
            dldx_traj = state_traj - self.demo_state_traj
            dldu_traj = control_traj - self.demo_control_traj
            
            # --------------------------- Chain rule ----------------------------------------
            dLdtheta = np.matmul(dLdXiNow, dxidtheta_t)[0]
            dp = dxidtheta_t

            # --------------------------- EKF ----------------------------------------
            updateTheta = EKF()
            updateTheta.predict(self.theta, self.P_prev, self.Q_prev)
            updateTheta.update(dp, self.R, lossNow)
            print('theta = ', updateTheta.theta)
            self.P_prev = updateTheta.P
            self.theta = updateTheta.theta

            self.iter_his += [idx]


        # --------------------------- learned full iter ---------------------------
        Loss = 0
        loss_his = []
        traj = self.sysoc.ocSolver(ini_state=self.demo_ini_state, horizon=self.demo_horizon, auxvar_value = self.theta)
        state_traj = traj['state_traj_opt']
        control_traj = traj['control_traj_opt']
        for jdx in range(self.demo_horizon):
            each_traj_t = np.hstack((state_traj[jdx], control_traj[jdx]))
            demo_traj_t = np.hstack((self.demo_state_traj[jdx], self.demo_control_traj[jdx]))
            lossNorm = norm_2(each_traj_t-demo_traj_t)**2
            loss_his += [lossNorm]
            Loss += lossNorm

        self.Loss_his += [np.asarray(Loss)[0,0]]
        self.theta_error += [np.asarray(norm_2(self.theta-self.true_theta)**2)[0,0]]
        self.iter_his += [idx+1]
        
        if self.saveFlag:
            self.saveEach(idx+1, traj, loss_his)

        # --------------------------- save all Loss ---------------------------
        if self.saveFlag:
            self.saveAll()
        
        
        self.plotLoss()

    def saveEach(self, idx, traj, loss_his):
        sio.savemat(self.dir+"results/iter_"+str(idx)+".mat", {'trajectories': traj,
                                                                'losses': loss_his,
                                                                'dt': self.dt,
                                                                'theta': self.theta})

    def saveAll(self):
        sio.savemat(self.dir+"results/Loss.mat", {'Loss': self.Loss_his,
                                                  'theta': self.theta_error})


    def load(self, dir):
        data = sio.loadmat(dir)


    def plotLoss(self):
        fig, axs = plt.subplots()
        axs.plot(self.iter_his, self.Loss_his)
        axs.set_xlabel("Learning Iteration")
        axs.set_ylabel("Loss")
        axs.set_title(self.mode + ": " + self.project)

        fig, axs = plt.subplots()
        axs.plot(self.iter_his, self.theta_error)
        axs.set_xlabel("Learning Iteration")
        axs.set_ylabel("Theta Error")
        axs.set_title(self.mode + ": " + self.project)
        plt.show()

    def plotTraj(self, state_traj, control_traj):

        iter = [*range(len(state_traj))]
        fig, axs = plt.subplots(len(state_traj[0]),1)
        for idx in range(len(state_traj[0])):
            axs[idx].plot(iter, state_traj[:,idx])
            axs[idx].plot(iter, self.demo_state_traj[:,idx])
            axs[idx].set_ylabel("x"+str(idx+1))
        axs[-1].set_xlabel("Iteration")
        axs[0].set_title("State Trajectory")

        iter = [*range(len(control_traj))]
        if len(control_traj[0]) == 1:
            fig, axs = plt.subplots()
            axs.plot(iter, control_traj)
            axs.plot(iter, self.demo_control_traj)
            axs.set_ylabel("u")
            axs.set_xlabel("Iteration")
            axs.set_title("Control Trajectory")
        else:
            fig, axs = plt.subplots(len(control_traj[0]),1)
            for idx in range(len(control_traj[0])):
                axs[idx].plot(iter, control_traj[:,idx])
                axs[idx].plot(iter, self.demo_control_traj[:,idx])
                axs[idx].set_ylabel("x"+str(idx+1))
            axs[-1].set_xlabel("Iteration")
            axs[0].set_title("Control Trajectory")
        plt.show()

