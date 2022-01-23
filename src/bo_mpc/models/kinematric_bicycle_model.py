# https://github.com/daniel-s-ingram/self_driving_cars_specialization/blob/master/1_introduction_to_self_driving_cars/Kinematic_Bicycle_Model.ipynb

from math import sin, cos, tan, atan2
import numpy as np


class Bicycle:
    def __init__(self, lr=1.2, lf=0.8, dt=0.01):
        self.reset()

        self.L = lf + lr
        self.lr = lr
        self.a_max = 5
        self.w_max = 1.22

        self.dt = dt

    def reset(self):
        self.v = 0
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0

    def step(self, a, w):
        """
        param a:
        accelaration
        param w:
        steering change
        """
        v_dot = max(-self.a_max, min(a, self.a_max))
        xc_dot = self.v * cos(self.theta + self.beta)
        yc_dot = self.v * sin(self.theta + self.beta)
        theta_dot = self.v * cos(self.beta) * tan(self.delta) / self.L
        delta_dot = max(-self.w_max, min(self.w_max, w))
        self.v += v_dot * self.dt
        self.xc += xc_dot * self.dt
        self.yc += yc_dot * self.dt
        self.theta += theta_dot * self.dt
        self.delta += delta_dot * self.dt
        self.beta = atan2(self.lr * tan(self.delta), self.L)

    @property
    def state(self):
        return np.array([self.v, self.xc, self.yc, self.theta, self.delta, self.beta]).astype(float)

    @state.setter
    def state(self, value):
        self.v, self.xc, self.yc, self.theta, self.delta, self.beta = value
