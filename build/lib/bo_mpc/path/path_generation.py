from typing import Tuple
import numpy as np
from scipy import interpolate
from numpy import random


def generate_random_path(num_waypoints: int, bounds: Tuple[float], num_steps, s=0.):
    waypoints = random.random((num_waypoints, 2))
    waypoints *= (bounds[1] - bounds[0])
    waypoints += bounds[0]
    waypoints[-1, :] = waypoints[0, :]

    return generate_path(waypoints, num_steps, s)


def generate_path(waypoints: np.ndarray, num_steps: int, s=0.):
    tck, _ = interpolate.splprep([waypoints[:, 0], waypoints[:, 1]], s=s)
    x = np.linspace(0, 1, num=num_steps, endpoint=True)
    out = interpolate.splev(x, tck)

    return np.array(out).T

def generate_path(bike, num_steps: int):
    inputs = random.normal(size=(num_steps - 1, 2))
    path = np.zeros((num_steps, 2))
    bike.reset()
    for i in range(1, num_steps):
        bike.step(* inputs[i - 1, :])
        path[i, :] = [bike.xc, bike.yc]
    return path
