
from models.kinematric_bicycle_model import Bicycle
from controller import MpcController
from path.path_generation import generate_path
from matplotlib import pyplot as plt
import numpy as np
from numpy import random
import copy
import pathlib
import os
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

@dataclass(init=True)
class SimulationData:
    seed: int
    path : np.ndarray
    path_contoller : np.ndarray
    reference : np.ndarray
    inputs : np.ndarray
    measured_states : np.ndarray
    tlf : float = 0.
    tlr : float = 0.
    lf : float = 0.
    lr : float = 0.
    input_ratio : float = 0.


class Simulation:
    def __init__(
            self,
            measured_states=[
                0,
                1],
            tlr=1.2,
            tlf=0.8,
            track_num_points=200,
            track_seeds=[1, 2, 3],
            input_cost_ratio=0.001): #90297, 18806, 58798]):
        self._measured_states = measured_states
        self._tlr = tlr
        self._tlf = tlf
        self._track_num_points = track_num_points
        self._track_seeds = track_seeds
        self._input_cost_ratio = input_cost_ratio

        self.MPC_HORIZON = 10

    def run_seed(self, seed: int, lr=1.2, lf=0.8, no_files=False, file_path=None, no_gui=False) -> float:
        if not no_gui:
            plt.ion()
            plt.show()

        random.seed(seed)
        bike = Bicycle(dt=0.1, lr=self._tlr, lf=self._tlf)
        path = generate_path(bike, self._track_num_points)

        controller = MpcController(self._input_cost_ratio)
        bike = Bicycle(dt=0.1, lr=self._tlr, lf=self._tlf)
        bike.xc = path[0, 0]
        bike.yc = path[0, 1]
        bike_test = copy.copy(bike)

        mpc_bike = Bicycle(dt=0.1, lr=lr, lf=lf)
        # measure states
        mpc_bike.state = bike.state

        # record positions and inputs
        pos = np.zeros(path.shape)
        pos_c = np.zeros(path.shape)
        u = np.zeros(path.shape)

        # run MPC
        for i, _ in enumerate(path):
            a, w = controller.optimize(
                mpc_bike, path[i:i + self.MPC_HORIZON, :])
            bike.step(a, w)
            mpc_bike.step(a, w)

            u[i, :] += [a, w]
            #print(f"{a:.02f}, {w:.02f}, {inputs[i,0]:.02f}, {inputs[i,1]:.02f}")
            pos[i, :] = [bike.xc, bike.yc]
            pos_c[i, :] = [mpc_bike.xc, mpc_bike.yc]
            state = mpc_bike.state
            state[self._measured_states] = bike.state[self._measured_states]
            mpc_bike.state = state

            if not no_gui:
                # show control output
                plt.cla()
                plt.title(f"$l_r={lr :.2f}$, $l_f={lf :.2f}$ seed={seed}")
                plt.scatter(path[:, 0], path[:, 1], label="Reference")
                plt.scatter(pos_c[:i, 0], pos_c[:i, 1], label="Controller")
                plt.scatter(pos[:i, 0], pos[:i, 1], label="Bike")
                plt.legend()
                plt.draw()
                plt.pause(0.001)

        u = np.array(u)
        pos = np.array(pos)

        # calc cost
        controller.horiz = path.shape[0]
        cost = controller.cost(u, bike_test, path)

        # save files
        if no_files:
            return -cost

        if file_path is None:
            file_path = os.curdir

        pathlib.Path(os.path.join(file_path, f"{lr :.2f}_{lf :.2f}")).mkdir(
            parents=True, exist_ok=True)

        np.savez(
            os.path.join(file_path, f"{lr :.2f}_{lf :.2f}/sim{seed}.npz"),
            pos=pos,
            pos_c=pos_c,
            u=u)

        with open(os.path.join(file_path, f"{lr :.2f}_{lf :.2f}/log.txt"), 'a') as f:
            line = f"[{seed}, {cost}],"
            f.write(line + "\n")

        return -cost

    def run_sequential(self, lr=1.2, lf=0.8, no_files=False, file_path=None, no_gui=False) -> float:
        cost_sum = 0
        for seed in self._track_seeds:
            cost_sum += self.run_seed(seed, lr=lr, lf=lf, no_files=no_files, file_path=file_path, no_gui=no_gui)

        return cost_sum

    @classmethod
    def from_file(cls, file_name) -> "Simulation":
        #TODO fix this
        sim_file = np.load(file_name)
        sim = Simulation(sim_file["measured_states"])


if __name__ == "__main__":
    sim = Simulation(track_seeds=[50])
    print(sim.run_sequential(no_files=True))
