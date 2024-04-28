import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/src')
from plot import plot


# ------------------------------ Set up dynamic system ------------------------------
project = "Quadrotor"
mode = "Imitation Learning"
dir = 'examples/ImitationLearning/uav/data/test_run/Loss.mat'
myplot = plot(project, mode, dir)

myplot.plotLoss()