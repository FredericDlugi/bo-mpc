import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

def visualize_1d(X, y, bounds, color='b', show_observations=True, show_legend=True):

    gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha= 0.1,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=1,
        )

    gp.fit(X, y)

    x_plot = np.linspace(*bounds, num=500).reshape(-1, 1)

    mean, std = gp.predict(x_plot, return_std=True)

    plt.plot(x_plot, mean, label="Mean", color=color)
    plt.fill_between(x_plot.ravel(), mean - std, mean + std, alpha=0.2, label="$1 \sigma$ confidence interval", color=color)

    if show_observations:
        plt.scatter(X, y, marker="x",label="Observations", color=color)
        i_max = np.argmax(y)
        plt.scatter(X[i_max], y[i_max], marker="^", label="Optimum", color=color)

    if show_legend:
        plt.legend()


def visualize_2d(X, y, bounds):

    gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-1,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=1,
        )

    gp.fit(X, y)

    x0_plot = np.linspace(*bounds[0], num=500)
    x1_plot = np.linspace(*bounds[1], num=500)

    x_plot = np.array(np.meshgrid(x0_plot, x1_plot)).T.reshape(-1, 2)

    mean, std = gp.predict(x_plot, return_std=True)

    img_mean = mean.reshape((500, 500))

    plt.contourf(img_mean, levels=20, extent=[*bounds[0], *bounds[1]])
    plt.scatter(X[:,0], X[:,1], label="Observations", marker='x', color='r')
    i_max = np.argmax(y)
    plt.scatter(X[i_max, 0], X[i_max, 1], label="Found optimum")
    plt.legend()
    plt.colorbar()


