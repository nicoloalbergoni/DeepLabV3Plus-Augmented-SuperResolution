from cProfile import run
import os
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

DATA_DIR = os.path.join(os.getcwd(), "data")
ROOT_FOLDER = os.path.join(DATA_DIR, "robustness_check")
CSV_PATH = os.path.join(ROOT_FOLDER, "robustness_check.csv")


def plot3d(x, y, z, save_path):
    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection="3d")
    ax.scatter(x, y, z, c=z, cmap="jet")
    ax.set_xlabel("Angle")
    ax.set_ylabel("Total shift")
    ax.set_zlabel("Avg. Mean IoU")
    plt.savefig(os.path.join(save_path, "robustess_small_3d.png"))


def plot2d(x, y, save_path, z=None):
    plt.figure()
    plt.axes()
    if z is not None:
        sct = plt.scatter(x, y, c=z, cmap="jet")
        plt.xticks(np.unique(x))
        clb = plt.colorbar()
        clb.ax.set_title("Total Shift")
        plt.axhline(np.max(y))
        plt.text(0.5, np.max(y), np.max(y))
    else:
        plt.scatter(x, y, cmap="jet")
    plt.xlabel("Angles")
    plt.ylabel("Avg. Mean IoU")

    plt.savefig(os.path.join(save_path, "robustess_small_2d.png"))


def main():
    df = pd.read_csv(CSV_PATH)

    runs = np.arange(0, 7**3, step=1)
    angles = df["Angle"].to_numpy()
    shift_x = df["Shift X"].to_numpy()
    shift_y = df["Shift Y"].to_numpy()
    avg_iou = df["Avg. Mean IoU"].to_numpy()

    total_shift = np.add(shift_x, shift_y)

    plot3d(angles, total_shift, avg_iou, ROOT_FOLDER)
    plot2d(angles, avg_iou, ROOT_FOLDER, z=total_shift)

    print(df.to_string())
    print("Done")


if __name__ == '__main__':
    main()
