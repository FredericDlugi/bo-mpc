from argparse import ArgumentParser

cli_parser = ArgumentParser(
    description="CLI interface of BO on an MPC problem to find L_f and L_r params.")

cli_parser.add_argument(
    "--target_lf",
    "-tlf",
    help="Target value for L_f",
    default=1.2,
    type=float)

cli_parser.add_argument(
    "--target_lr",
    "-tlr",
    help="Target value for L_r",
    default=0.8,
    type=float)

cli_parser.add_argument(
    "--bounds",
    "-b",
    nargs=2,
    help="Search bounds for L_r and L_f",
    default=[
        0.5,
        1.5],
    type=float)

cli_parser.add_argument(
    "--n_iter",
    help="Number of iterations of BO",
    default=25,
    type=int)

cli_parser.add_argument(
    "--acq",
    help="Acquisition function of BO (ei, ucb or poi)",
    default="ei",
    type=str)

cli_parser.add_argument(
    "--acq_xi",
    help="Xi value for acquisition function of BO (ei or poi)",
    default=0,
    type=float)

cli_parser.add_argument(
    "--acq_kappa",
    help="Kappa value for acquisition function of BO (ucb)",
    default=2.576,
    type=float)

cli_parser.add_argument(
    "--measured_states",
    help="Measure theses states from actual controller (0-4)",
    nargs="*",
    default=[0, 1],
    type=int)

cli_parser.add_argument(
    "--track_num_points",
    help="Number of points per track",
    default=200,
    type=int)

cli_parser.add_argument(
    "--track_num",
    help="Number of tracks per evaluation",
    default=40,
    type=int)

cli_parser.add_argument(
    "--gp_alpha",
    help="Alpha value for GP",
    default=1e-6,
    type=float)

if __name__ == "__main__":
    print(cli_parser.parse_args())