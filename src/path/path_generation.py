from typing import Tuple
import numpy as np
from scipy import interpolate
from numpy import random


def generate_random_path(num_waypoints: int, bounds: Tuple[float], num_steps):
    waypoints = random.random((num_waypoints, 2))
    waypoints *= (bounds[1] - bounds[0])
    waypoints += bounds[0]
    waypoints[-1, :] = waypoints[0, :]

    return generate_path(waypoints, num_steps)


def generate_path(waypoints: np.ndarray, num_steps: int):
    tck, _ = interpolate.splprep([waypoints[:, 0], waypoints[:, 1]], s=0)
    x = np.linspace(0, 1, num=num_steps, endpoint=True)
    out = interpolate.splev(x, tck)

    return np.array(out).T
