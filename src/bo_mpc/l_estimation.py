from json.decoder import JSONDecoder
from simulation import Simulation
import numpy as np
import logging as log
from pathos.multiprocessing import Pool
from bayes_opt import BayesianOptimization, JSONLogger, Events
import cli_interface
import json

def bo_cost_function(lf, lr, tlr, tlf, track_num):
    log.basicConfig(filename="test.log",format='%(asctime)s - %(message)s', level=log.INFO)

    tested_seeds = np.arange(track_num)

    worker_func = lambda seed: worker_function(seed, tlr, tlf, lr, lf)
    with Pool(12) as pool:
        res = pool.map(worker_func, tested_seeds)

    res = np.sum(res)

    log.info(f"L_f {lf:.4f} and L_r {lr:.4f} resulted in {res:.2f} error")
    return res

def worker_function(seed, tlr, tlf, lr, lf):
    sim = Simulation(input_cost_ratio=0.002, tlr=tlr, tlf=tlf)
    res = sim.run_seed(seed, no_gui=True, file_path=f"{lf:.3f}_{lr:.3f}/", lf=lf, lr=lr)
    print(seed)
    return res


def main():
    cli_interface.cli_parser.add_argument("--from_file", action="store_true")
    args = cli_interface.cli_parser.parse_args()

    log.basicConfig(filename="test.log",format='%(asctime)s - %(message)s', level=log.INFO)
    log.info("This log will log optimal lf, lr search for MPC")
    log.info(f"Arguments: {args}")
    f = lambda lf, lr : bo_cost_function(lf=lf, lr=lr, tlf=args.target_lf, tlr=args.target_lr, track_num=args.track_num)
    bo = BayesianOptimization(
        f=f,
        pbounds={
            "lr": args.bounds,
            "lf": args.bounds},
        random_state=1)

    if args.from_file:
        json_dec = json.load(open("logs.json"))
    logger = JSONLogger(path="./logs.json")
    bo.subscribe(Events.OPTIMIZATION_STEP, logger)

    bo.maximize(acq=args.acq, xi=args.acq_xi, kappa=args.acq_kappa, n_iter=args.n_iter)


if __name__ == "__main__":
    main()