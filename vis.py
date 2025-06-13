import numpy as np
from typing import List
import os
from matplotlib import pyplot as plt


def visualise_point_clouds(
    point_clouds: List[np.ndarray],
    frame_id: int,
    save_frame: bool = False,
    save_dir: str = "",
):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3D Liver Registration")

    ax.scatter(point_clouds[0][:, 0], point_clouds[0][:, 1], point_clouds[0][:, 2], marker="o", c="red", s=1)
    ax.scatter(point_clouds[1][:, 0], point_clouds[1][:, 1], point_clouds[1][:, 2], marker="^", c="blue", s=1)

    if save_frame:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{frame_id}.png"), bbox_inches="tight")
    else:
        plt.show()

    plt.close("all")
