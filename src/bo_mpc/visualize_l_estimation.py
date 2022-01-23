from visualize_gp import visualize_2d
import json
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    json_file = open("logs.json")

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
