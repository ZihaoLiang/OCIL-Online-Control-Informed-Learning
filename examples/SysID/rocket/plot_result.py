import numpy as np
from casadi import *
import scipy.io as sio
import os
import sys
sys.path.append(os.getcwd() + '/src')
from plot import plot


# ------------------------------ Set up dynamic system ------------------------------
project = "Rocket"
mode = "SysID"
dir = 'examples/SysID/rocket/data/test_run/Loss.mat'
myplot = plot(project, mode, dir)

myplot.plotLoss()