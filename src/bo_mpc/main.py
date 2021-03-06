from simulation import Simulation
from bayes_opt import BayesianOptimization
from matplotlib import pyplot as plt
import numpy as np
import argparse


def plot_bo(f, bo):
    x = np.linspace(-2, 10, 10000)
    mean, sigma = bo._gp.predict(x.reshape(-1, 1), return_std=True)

    plt.figure(figsize=(16, 9))
    plt.plot(x, f(x))
    plt.plot(x, mean)
    plt.fill_between(x, mean + sigma, mean - sigma, alpha=0.1)
    plt.scatter(
        bo.space.params.flatten(),
        bo.space.target,
        c="red",
        s=50,
        zorder=10)
    plt.show()


parser = argparse.ArgumentParser(
    description="Run BO on an MPC problem to find L_f and L_r params.")

parser.add_argument(
    "--target_lf",
    "-tlf",
    help="Target value for L_f",
    default=1.2,
    type=float)

parser.add_argument(
    "--target_lr",
    "-tlr",
    help="Target value for L_r",
    default=0.8,
    type=float)

parser.add_argument(
    "--bounds",
    "-b",
    nargs=2,
    help="Search bounds for L_r and L_f",
    default=[
        0.5,
        1.5],
    type=float)

parser.add_argument(
    "--n_iter",
    help="Number of iterations of BO",
    default=50,
    type=int)

parser.add_argument(
    "--acq",
    help="Acquisition function of BO (ei, ucb or poi)",
    default="ei",
    type=str)
parser.add_argument(
    "--acq_xi",
    help="Xi value for acquisition function of BO (ei or poi)",
    default=0,
    type=float)
parser.add_argument(
    "--acq_kappa",
    help="Kappa value for acquisition function of BO (ucb)",
    default=2.576,
    type=float)

parser.add_argument(
    "--measured_states",
    help="Measure theses states from actual controller (0-4)",
    nargs="*",
    default=[0, 1],
    type=int)

parser.add_argument(
    "--track_waypoints",
    help="Number of waypoints per track",
    default=7,
    type=int)

parser.add_argument(
    "--track_bounds",
    nargs=2,
    help="Track bounds",
    default=[
        0.,
        5.],
    type=float)

parser.add_argument(
    "--track_num_points",
    help="Number of points per track",
    default=200,
    type=int)

parser.add_argument(
    "--track_seeds",
    help="Seeds of points per track",
    nargs="+",
    default=[18806, 58798, 90297],
    type=int)

parser.add_argument(
    "--track_num",
    help="Number of tracks per evaluation (ignored if track_seeds != default)",
    default=3,
    type=int)

parser.add_argument(
    "--gp_alpha",
    help="Alpha value for GP",
    default=1e-6,
    type=float)

args = parser.parse_args()

bounds = (args.bounds[0], args.bounds[1])

measured_states = args.measured_states

sim = Simulation(
    measured_states=args.measured_states,
    tlr=args.target_lr,
    tlf=args.target_lf,
    track_waypoints=args.track_waypoints,
    track_bounds=args.track_bounds,
    track_num_points=args.track_num_points,
    track_seeds=args.track_seeds)

def f(lr, lf): return sim.run(lr, lf)

bo = BayesianOptimization(
    f=f,
    pbounds={
        "lr": bounds,
        "lf": bounds},
    random_state=1)

bo.set_gp_params(alpha=args.gp_alpha)
bo.maximize(
    n_iter=args.n_iter,
    acq=args.acq,
    xi=args.acq_xi,
    kappa=args.acq_kappa)
