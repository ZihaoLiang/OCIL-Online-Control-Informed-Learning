import matplotlib.pyplot as plt 
import scipy.io as sio

class plot():
    def __init__(self, project="", mode="", dir=""):
        self.mode = mode
        self.project = project
        self.dir = dir
        self.data = sio.loadmat(dir)

    def plotLoss(self):
        Loss = self.data['Loss'][0]
        iter = [*range(len(Loss))]

        fig, axs = plt.subplots()
        axs.plot(iter, Loss)
        axs.set_xlabel("Learning Iteration")
        axs.set_ylabel("Loss")
        axs.set_title(self.mode + ": " + self.project)
        plt.show()