from visualize_gp import visualize_1d
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
    parser.add_argument("paths", type=directory, nargs='+')
    args = parser.parse_args()

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    plt.figure(figsize=(8,5))

    for i, path in enumerate(args.paths):
        json_file = open(os.path.join(path, "logs.json"))

        x = []
        y = []
        for line in json_file.readlines():
            json_l = json.loads(line)
            x += [json_l["params"]["r"]]
            y += [json_l["target"]]

        x = np.array(x)
        x = np.reshape(x, (-1, 1))
        y = np.array(y)

        visualize_1d(x, y, (x.min(), x.max()), colors[i], len(args.paths) == 1, len(args.paths) == 1)
    plt.xlabel("Ratio (input/output cost)")
    plt.ylabel("Accumulated cost")
    plt.show()

if __name__ == "__main__":
    main()
