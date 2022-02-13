from visualize_gp import visualize_2d
import json
from matplotlib import pyplot as plt
import numpy as np
import argparse
import os

def directory(path):
    if not os.path.isdir(path) and os.path.isfile(os.path.join(path, "logs.json")):
        raise argparse.ArgumentTypeError(f"{path} is not a valid directory")
    return path

def main():
    parser = argparse.ArgumentParser(prog="Visualize Cost Ratio Experiments")
    parser.add_argument("path", type=directory)
    args = parser.parse_args()

    json_file = open(os.path.join(args.path, "logs.json"))

    x = []
    y = []
    for line in json_file.readlines():
        json_l = json.loads(line)
        x += [[json_l["params"]["lf"], json_l["params"]["lr"]]]
        y += [json_l["target"]]

    x = np.array(x)
    y = np.array(y)

    visualize_2d(x, y, [(x.min(), x.max()), (x.min(), x.max())])
    plt.xlabel("$L_f$")
    plt.ylabel("$L_r$")
    plt.show()

if __name__ == "__main__":
    main()
