from visualize_gp import visualize_1d
import json
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    json_file = open("logs.json")

    x = []
    y = []
    for line in json_file.readlines():
        json_l = json.loads(line)
        x += [json_l["params"]["r"]]
        y += [json_l["target"]]

    x = np.array(x)
    x = np.reshape(x, (-1, 1))
    y = np.array(y)

    print(x.shape, y.shape)
    visualize_1d(x, y, (x.min(), x.max()))
    plt.xlabel("Ratio (input/output cost)")
    plt.ylabel("Accumulated cost")
    plt.show()
