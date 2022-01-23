from simulation import Simulation
import numpy as np
import logging as log
from pathos.multiprocessing import Pool
from bayes_opt import BayesianOptimization, JSONLogger, Events

def bo_cost_function(r):
    log.basicConfig(filename="test.log",format='%(asctime)s - %(message)s', level=log.INFO)
    lr = 1.2
    lf = 0.8
    tested_seeds = np.arange(40)

    worker_func = lambda seed: worker_function(seed, r, lr, lf)
    with Pool(12) as pool:
        res = pool.map(worker_func, tested_seeds)

    res = np.sum(res)

    log.info(f"Ratio {r:.4f} resulted in {res:.2f} error")
    return res

def worker_function(seed, r, lr, lf):
    sim = Simulation(input_cost_ratio=r, tlr=lr, tlf=lf)
    res = sim.run_seed(seed, no_gui=True, file_path=f"{r:.7f}/", lf=lf, lr=lr)
    print(seed)
    return res


def main():
    log.basicConfig(filename="test.log",format='%(asctime)s - %(message)s', level=log.INFO)
    log.info("This log will log optimal ratio search for MPC")
    bo = BayesianOptimization(
        f=bo_cost_function,
        pbounds={
            "r": (0.0, 1.0)},
        random_state=1)
    logger = JSONLogger(path="./logs.json")
    bo.subscribe(Events.OPTIMIZATION_STEP, logger)

    res = bo.maximize()
    print(res)


if __name__ == "__main__":
    main()