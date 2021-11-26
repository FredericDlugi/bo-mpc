from simulation import run_sim
from bayes_opt import BayesianOptimization
from matplotlib import pyplot as plt
import numpy as np

def plot_bo(f, bo):
    x = np.linspace(-2, 10, 10000)
    mean, sigma = bo._gp.predict(x.reshape(-1, 1), return_std=True)

    plt.figure(figsize=(16, 9))
    plt.plot(x, f(x))
    plt.plot(x, mean)
    plt.fill_between(x, mean + sigma, mean - sigma, alpha=0.1)
    plt.scatter(bo.space.params.flatten(), bo.space.target, c="red", s=50, zorder=10)
    plt.show()

bo = BayesianOptimization(f=run_sim, pbounds={"lr":(0.5,1.5),"lf":(0.5,1.5)}, random_state=1)
print(bo.maximize(n_iter=50))
