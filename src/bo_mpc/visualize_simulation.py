from simulation import SimulationData
from visualize_gp import visualize_2d
import json
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import argparse
import glob
import os
import re
from path.path_generation import generate_path
from models.kinematric_bicycle_model import Bicycle
from numpy import random

def load_from_file(file) -> SimulationData:
    seed_re = "sim(\\d+).npz"
    seed = int(re.search(seed_re, file).group(1))
    if "l_estimation" in file:
        l_re = re.search("l_estimation_(\\d+\\.\\d+)_(\\d+\\.\\d+).*\\\\(\\d+\\.\\d+)_(\\d+\\.\\d+)", file)
        tlr = float(l_re.group(1))
        tlf = float(l_re.group(2))
        lr = float(l_re.group(4))
        lf = float(l_re.group(3))
        ratio = 0.002

    elif "input_ratio" in file:
        l_re = re.search("input_ratio_(\\d+\\.\\d+)_(\\d+\\.\\d+)\\\\(\\d+\\.\\d+)\\\\(\\d+\\.\\d+)_(\\d+\\.\\d+)", file)
        tlr = float(l_re.group(1))
        tlf = float(l_re.group(2))
        ratio = float(l_re.group(3))
        lr = tlr
        lf = tlf

    npf = np.load(file)
    if "ref" not in npf:
        bike = Bicycle(tlr,tlf,dt=0.1)
        random.seed(seed)
        ref = generate_path(bike, 200)
    else:
        ref = npf["ref"]
    sim = SimulationData(seed, npf["pos"], npf["pos_c"], ref, npf["u"], measured_states=[0, 1], tlf=tlf, tlr=tlr, lf=lf, lr=lr, input_ratio=ratio)
    return sim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to visualize past simulations.")
    parser.add_argument("glob_path")
    args = parser.parse_args()

    print(args.glob_path)
    files = glob.glob(args.glob_path)#, root_dir=os.curdir)
    print(files)

    next_i = 0
    class Index:
        ind = 0
        def __init__(self, files):
            self.ind = 0
            print(files[self.ind])
            sim = load_from_file(files[self.ind])

            self.title = plt.title(f"$l_r={sim.lr :.2f}$, $l_f={sim.lf :.2f}$ seed={sim.seed}")
            plt.axis("equal")
            plt.ylim([-10, 10])
            plt.xlim([-10, 10])
            self.path, = plt.plot(sim.path[:, 0], sim.path[:, 1], label="Bike")
            self.path_contoller, = plt.plot(sim.path_contoller[:, 0], sim.path_contoller[:, 1], label="Controller")
            self.reference, = plt.plot(sim.reference[:, 0], sim.reference[:, 1], label="Reference")

            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
            plt.legend()

        def next(self, event):
            self.ind += 1
            self.ind %= len(files)
            print(files[self.ind])
            sim = load_from_file(files[self.ind])
            self.path.set_data(sim.path[:, 0], sim.path[:, 1])
            self.path_contoller.set_data(sim.path_contoller[:, 0], sim.path_contoller[:, 1])
            self.reference.set_data(sim.reference[:, 0], sim.reference[:, 1])
            self.title.set_text(f"$l_r={sim.lr :.2f}$, $l_f={sim.lf :.2f}$ seed={sim.seed}")
            plt.draw()

        def prev(self, event):
            self.ind -= 1
            self.ind %= len(files)
            print(files[self.ind])
            sim = load_from_file(files[self.ind])
            self.path.set_data(sim.path[:, 0], sim.path[:, 1])
            self.path_contoller.set_data(sim.path_contoller[:, 0], sim.path_contoller[:, 1])
            self.reference.set_data(sim.reference[:, 0], sim.reference[:, 1])
            self.title.set_text(f"$l_r={sim.lr :.2f}$, $l_f={sim.lf :.2f}$ seed={sim.seed}")
            plt.draw()

    callback = Index(files)
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)



    plt.show()