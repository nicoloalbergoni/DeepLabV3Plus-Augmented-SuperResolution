import os
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

DATA_DIR = os.path.join(os.getcwd(), "data")
ROOT_FOLDER = os.path.join(DATA_DIR, "robustness_check")
CSV_PATH = os.path.join(ROOT_FOLDER, "robustness_small_350.csv")


def plot3d(x, y, z, save_path):
    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection="3d")
    ax.scatter(x, y, z, c=z, cmap="jet")
    ax.set_xlabel("Angle")
    ax.set_ylabel("Total shift")
    ax.set_zlabel("Avg. Mean IoU")
    plt.savefig(os.path.join(save_path, "robustess_350_small_3d.png"))


def plot2d(x, y, save_path, z=None, hline_text=None):
    plt.figure(figsize=(22, 8))
    plt.axes()
    if z is not None:
        sct = plt.scatter(x, y, c=z, cmap="jet")
        plt.xticks(np.unique(x), fontsize=8)
        clb = plt.colorbar()
        clb.ax.set_title("Total Shift")
        plt.axhline(np.max(y))
        plt.text(0.5, np.max(y), hline_text)
        plt.axvline(0.0)
    else:
        plt.scatter(x, y, cmap="jet")
    plt.xlabel("Angles")
    plt.ylabel("Avg. Mean IoU")

    plt.savefig(os.path.join(save_path, "robustess_350_small_2d.png"))


def main():
    df = pd.read_csv(CSV_PATH)

    runs = np.arange(0, 7**3, step=1)
    angles = df["Angle"].to_numpy()
    shift_x = df["Shift_X"].to_numpy()
    shift_y = df["Shift_Y"].to_numpy()
    avg_iou = df["IoU"].to_numpy()

    total_shift = np.add(shift_x, shift_y)
    max_row = df.iloc[df['IoU'].idxmax()]
    hline_str = f"IoU: {max_row['IoU']}, Shift: {max_row['Shift_X']},{max_row['Shift_Y']}"

    plot3d(angles, total_shift, avg_iou, ROOT_FOLDER)
    plot2d(angles, avg_iou, ROOT_FOLDER, z=total_shift, hline_text=hline_str)

    print(df.to_string())
    print(f"Best record: {max_row}")
    print("Done")


if __name__ == '__main__':
    main()
