
from models.kinematric_bicycle_model import Bicycle
import numpy as np
import copy
from scipy.optimize import minimize

class MpcController:
    def __init__(self):
        self.horiz = None
        self.R = np.diag([0.01, 0.01])                 # input cost matrix
        # input difference cost matrix
        self.Rd = np.diag([0.01, 0.01])
        self.Q = np.diag([1.0, 1.0])                   # state cost matrix
        self.Qf = self.Q                               # state final matrix

    def cost(
            self,
            u_k: np.ndarray,
            my_car: Bicycle,
            points: np.ndarray):
        mpc_car = copy.copy(my_car)
        u_k = u_k.reshape((self.horiz, 2)).T
        z_k = np.zeros((2, self.horiz + 1))

        desired_state = points.T
        cost = 0.0

        for i in range(self.horiz):
            mpc_car.step(u_k[0, i], u_k[1, i])

            z_k[0, i] = mpc_car.xc
            z_k[1, i] = mpc_car.yc
            cost += np.sum(self.R @ (u_k[:, i]**2))
            cost += np.sum(self.Q @ ((desired_state[:, i] - z_k[:, i])**2))
            if i < (self.horiz - 1):
                cost += np.sum(self.Rd @ ((u_k[:, i + 1] - u_k[:, i])**2))
        return cost

    def optimize(self, my_car: Bicycle, points: np.ndarray):
        self.horiz = points.shape[0]
        bnd = [(-5, 5), (-1.22, 1.22)] * self.horiz
        result = minimize(
            self.cost, args=(
                my_car, points), x0=np.zeros(
                (2 * self.horiz)), method='SLSQP', bounds=bnd)
        #print(result)
        return result.x[0], result.x[1]