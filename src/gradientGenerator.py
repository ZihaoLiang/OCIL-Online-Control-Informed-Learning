import numpy as np
from PDP import PDP

class gradientGenerator:

    def __init__(self):
        return
    
    def computeGradient(self, OCsystem, initialState, theta):
        self.OCsystem = OCsystem()
        xiTheta = self.OcSystem.solve(initialState, theta)
        lqrSystem = PDP.getLqrSystem(xiTheta, theta)
        resultLqr = PDP.solveLqr(lqrSystem)

        loss = PDP.lossFun(xiTheta["xi"], theta).full()[0, 0]
        dLdXi = PDP.dLdXiFun(xiTheta["xi"], theta)
        dXidTheta = np.vstack((np.concatenate(resultLqr["XTrajList"], axis=0),
            np.concatenate(resultLqr["UTrajList"], axis=0)))

        # this is full derivative
        dLdtheta = np.array(np.dot(dLdXi, dXidTheta)).flatten()

        return loss, dLdtheta
    
if __name__ == '__main__':


