from cProfile import run
import os
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

ROOT_FOLDER = os.path.join(os.getcwd(), "robustness_check")
CSV_PATH = os.path.join(ROOT_FOLDER, "robustness_check.csv")


def plot3d(x, y, z, save_path):
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection="3d")
    ax.scatter(x, y, z, cmap="Reds")
    ax.set_xlabel("Angle")
    ax.set_ylabel("Total shift")
    ax.set_zlabel("Avg. Mean IoU")
    plt.savefig(os.path.join(save_path, "robustess_3d.png"))


def plot2d(x, y, save_path):
    fig = plt.figure()
    ax = plt.axes()
    ax.scatter(x, y)
    ax.set_xlabel("Angles")
    ax.set_ylabel("Avg. Mean IoU")

    plt.savefig(os.path.join(save_path, "robustess_2d.png"))


def main():
    df = pd.read_csv(CSV_PATH)

    runs = np.arange(0, 7**3, step=1)
    angles = df["Angle"].to_numpy()
    shift_x = df["Shift X"].to_numpy()
    shift_y = df["Shift Y"].to_numpy()
    avg_iou = df["Avg. Mean IoU"].to_numpy()

    total_shift = np.add(shift_x, shift_y)

    plot3d(angles, total_shift, avg_iou, ROOT_FOLDER)
    plot2d(angles, avg_iou, ROOT_FOLDER)

    print(df.to_string())
    print("Done")


if __name__ == '__main__':
    main()