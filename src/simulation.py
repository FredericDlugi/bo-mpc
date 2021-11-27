
from models.kinematric_bicycle_model import Bicycle
from controller import MpcController
from path.path_generation import generate_random_path
from matplotlib import pyplot as plt
import numpy as np
from numpy import random
import copy
import pathlib
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

ref = [[25082, 20.532948320013894],
        [53816, 210.4562569750449],
        [76064, 55.96925996037113],
        [58549, 14.468911686067464],
        [427, 184.98888084734335],
        [16056, 60.090698631201114],
        [99010, 301.4617037514949],
        [88390, 6.390730425176852],
        [75542, 32.69409353462841],
        [30399, 81.33519708084226],
        [7108, 19.139958767371365],
        [3726, 14.942495462629457],
        [95374, 9.733050214245466],
        [50720, 38.02823104644416],
        [77323, 9.824426545346814],
        [6439, 74.75510398852445],
        [4875, 13.24150741589534],
        [3608, 400.3107905577367],
        [12763, 49.95084762012596],
        [41622, 98.4370625781701],
        [58798, 4.801656390614494],
        [35702, 312.3905652160304],
        [18806, 4.288301349816508],
        [36082, 49.36674991612697],
        [89159, 41.15665530305945],
        [2686, 31.61139954721623],
        [54004, 28.732657935459557],
        [50386, 333.87523188216505],
        [85580, 131.3118458004769],
        [77845, 370.71044711132976],
        [56685, 302.9659089976724],
        [5792, 5.755681452443642],
        [79745, 50.04276967431635],
        [93168, 32.22235967555776],
        [13279, 53.334707673612115],
        [89706, 618.0473584210496],
        [704, 18.677325007211916],
        [18736, 217.16298237934285],
        [87677, 15.615496317709164],
        [51454, 25.292586051428195],
        [96712, 25.823740309285],
        [90297, 4.843383774860714],
        [38163, 67.23338098787063],
        [41889, 59.88144258006616],
        [79491, 46.566620641858094],
        [36556, 88.76545573558323],
        [19014, 49.39164390714606],
        [25424, 12.838836167874437],
        [92508, 61.953038248897045],
        [41673, 8.70336298911908]]

def run_sim(lr=1.2, lf=0.8):
    train_tracks = [58798, 18806, 90297]#[58549, 25082, 53816, 76064, 427, 16056, 99010, 88390, 75542, 30399, 7108, 3726, 95374, 50720, 77323, 6439, 4875, 3608, 12763, 41622, 58798, 35702, 18806, 36082, 89159, 2686, 54004, 50386, 85580, 77845, 56685, 5792, 79745, 93168, 13279, 89706, 704, 18736, 87677, 51454, 96712, 90297, 38163, 41889, 79491, 36556, 19014, 25424, 92508, 41673]
    MPC_HORIZON = 10
    plt.ion()
    plt.show()

    cost_sum = 0
    for seed in train_tracks:
        random.seed(seed)
        path = generate_random_path(7, (0, 5), 200)

        controller = MpcController()
        bike = Bicycle(dt=0.1)
        bike.xc = path[0, 0]
        bike.yc = path[0, 1]
        bike_test = copy.copy(bike)

        mpc_bike = Bicycle(dt=0.1, lr = lr, lf = lf)
        mpc_bike.xc = path[0, 0]
        mpc_bike.yc = path[0, 1]


        # record positions and inputs
        pos = np.zeros(path.shape)
        pos_c = np.zeros(path.shape)
        u = np.zeros(path.shape)

        # run MPC

        for i, _ in enumerate(path):
            v, w = controller.optimize(mpc_bike, path[i:i + MPC_HORIZON,:])
            bike.step(v, w)
            mpc_bike.step(v, w)

            u[i,:] += [v, w]

            pos[i,:] = [bike.xc, bike.yc]
            pos_c[i,:] = [mpc_bike.xc, mpc_bike.yc]
            mpc_bike.xc = bike.xc
            mpc_bike.yc = bike.yc

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
        cost_sum += cost

        # save files
        pathlib.Path(f"{lr :.2f}_{lf :.2f}").mkdir(parents=True, exist_ok=True)

        np.savez(f"{lr :.2f}_{lf :.2f}/sim{seed}.npz", pos=pos, pos_c=pos_c, u=u)

        with open(f"{lr :.2f}_{lf :.2f}/log.txt", 'a') as f:
            line = f"[{seed}, {cost}],"
            f.write(line + "\n")



    return -cost_sum

if __name__ == "__main__":
    print(run_sim())
    input("test")
