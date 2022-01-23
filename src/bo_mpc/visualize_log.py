import numpy as np
import argparse
from os import path
from itertools import product

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

from matplotlib import pyplot as plt

if __name__ == "__main__":
    bounds = (0.5, 1.5)
    target = [1.2, 0.8]
    acquisition_function = "ei"
    xi = 0
    kappa = 2.576
    measured_states = [0, 1]
    num_references = 3
    alpha = 1e-6

    parser = argparse.ArgumentParser(prog="Log visualizer")
    parser.add_argument("path_in", type=str)

    args = parser.parse_args()

    log_path = path.join(args.path_in, "log.txt")

    log_lines = []
    with open(log_path, 'r') as log_file:
        log_lines = log_file.readlines()

    command = log_lines[0]
    log_lines = log_lines[3:]

    evaluations = []
    for line in log_lines:
        if line.startswith("|"):
            line_cells = line.split("|")
            iteration = int(line_cells[1])
            target = float(line_cells[2].replace("+0 ","3"))
            lf = float(line_cells[3])
            lr = float(line_cells[4])
            evaluations += [[iteration, target, lf, lr]]

    evaluations = np.array(evaluations)
    best_evaluation = np.argmax(evaluations[:, 1])

    gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1,
            normalize_y=True,
            n_restarts_optimizer=15,
            random_state=42
        )

    lr_space = np.linspace(bounds[0], bounds[1])
    lf_space = np.linspace(bounds[0], bounds[1])
    x1x2 = np.array(list(product(lr_space, lf_space)))

    gp.fit(evaluations[:, [2,3]], evaluations[:, 1])
    y_pred = gp.predict(x1x2)

    Zp = np.reshape(y_pred,(50,50))

    X0p, X1p = x1x2[:,0].reshape(50,50), x1x2[:,1].reshape(50,50)
    plt.title(command)
    plt.contourf(X0p, X1p, Zp, levels=32)
    plt.xlabel("$L_r$")
    plt.ylabel("$L_f$")
    plt.scatter(evaluations[:, 2], evaluations[:, 3], label="Evaluations")
    plt.scatter(target[0], target[1], label="True Maximum")
    plt.scatter(evaluations[best_evaluation, 2], evaluations[best_evaluation, 3], label="Best Evaluation")
    plt.colorbar()
    plt.show()