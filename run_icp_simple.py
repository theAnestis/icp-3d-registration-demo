import orjson
from time import perf_counter

import numpy as np
import open3d as o3d
import rich
from sklearn.neighbors import NearestNeighbors

import transforms
from vis import visualise_point_clouds


def estimate_rigid_transform(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """Estimate the rigid transform than can make points_a
    match with points_b.

    Args:
        points_a (np.ndarray): The first point set.
        points_b (np.ndarray): The second point set.

    Returns:
        np.ndarray: The estimated transform.
    """

    # Calculate centres.
    a_centre = np.mean(points_a, axis=0)
    b_centre = np.mean(points_b, axis=0)

    # Centre the point sets to the origin.
    points_a_cntrd = points_a - a_centre
    points_b_cntrd = points_b - b_centre

    # Get their covariance.
    covariance = points_a_cntrd.T @ points_b_cntrd

    # Get the USV vectors from SVD.
    u, s, Vt = np.linalg.svd(covariance)

    # Extract rotation and translation vectors, and
    # combine them to a homogenous matrix.
    rotation = Vt.T @ u.T
    translation = b_centre - rotation @ a_centre

    trans_mat = np.eye(4)
    trans_mat[:3, :3] = rotation
    trans_mat[:3, 3] = translation

    return trans_mat


def my_icp(
    fixed_cloud: np.ndarray,
    moving_cloud: np.ndarray,
    initial_guess: np.ndarray,
    iterations: int = 100,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Perform the ICP algorithm to match points to points.

    Args:
        fixed_cloud (np.ndarray): The fixed point cloud.
        moving_cloud (np.ndarray): The moving point cloud.
        initial_guess (np.ndarray): The initial guess for the transformation.
        iterations (int, optional): The number of iteration for the ICP algorithm. Defaults to 100.
        tolerance (float, optional): The error tolerance. Defaults to 1e-6.

    Returns:
        np.ndarray: The optimal transform.
    """
    rigid_transform = initial_guess
    fixed_cloud_3d = fixed_cloud[:, :3]
    neighbours = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(fixed_cloud_3d)
    prev_error = float("inf")
    info = {"iteration": [], "rmse": [], "time": []}

    for i in range(iterations):
        t = perf_counter()

        # Apply transformation.
        moved_cloud = (rigid_transform @ moving_cloud.T).T
        moved_cloud_3d = moved_cloud[:, :3]

        # Get nearest points.
        distances, indices = neighbours.kneighbors(moved_cloud_3d)

        # Calculate RMSE error.
        rmse = np.sqrt(np.mean(distances**2))

        # Estimate the rigid transform between moving and fixed point clouds after the transformation has been applied.
        rigid_transform = estimate_rigid_transform(moved_cloud_3d, fixed_cloud_3d[indices.ravel()]) @ rigid_transform
        t = perf_counter() - t

        info["iteration"].append(i)
        info["rmse"].append(rmse)
        info["time"].append(t)

        visualise_point_clouds([moved_cloud_3d, fixed_cloud_3d], save_dir="vis_simple", save_frame=True, frame_id=i)
        # Reporting.
        rich.print(f"Iteration {i + 1}\t |\t RMSE = {rmse} \t|\t Time: {int(t * 1000)}ms")
        if abs(prev_error - rmse) < tolerance:
            rich.print(f"Converged at iteration {i} with error {rmse} at {int(np.mean(info['time']) * 1000)}ms per iteration.")
            open("./res_simple.json", "wb").write(orjson.dumps(info))
            return rigid_transform
        prev_error = rmse

    rich.print(f"Did not converge - Last error: {prev_error} at {int(np.mean(info['time']) * 1000)}ms per iteration")
    open("./res_simple.json", "wb").write(orjson.dumps(info))
    return rigid_transform


if __name__ == "__main__":
    liver_mesh = o3d.io.read_triangle_mesh("./hepatitis-liver/hepatitis liver.obj")

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = liver_mesh.vertices
    point_cloud = np.asarray(point_cloud.points)

    trans1 = transforms.rotate_and_translate_mat(
        rot_parms={"axis": "x", "angle": 42},
        trans_params={"x": 2, "y": 1, "z": 2},
    )
    trans2 = transforms.rotate_and_translate_mat(
        rot_parms={"axis": "y", "angle": 21},
        trans_params={"x": 0, "y": 0, "z": 0},
    )
    trans3 = transforms.rotate_and_translate_mat(
        rot_parms={"axis": "z", "angle": 270},
        trans_params={"x": 9, "y": 3, "z": 3},
    )

    point_cloud_hom = np.hstack([point_cloud, np.ones((point_cloud.shape[0], 1))])
    point_cloud_chngd = (trans3 @ trans2 @ trans1 @ point_cloud_hom.T).T
    point_cloud_fixed = point_cloud_hom.copy()
    point_cloud_moving = point_cloud_chngd.copy()
    t_total = perf_counter()
    initial_guess = np.array(
        [
            [np.cos(30), -np.sin(30), 0, 10],
            [np.sin(30), np.cos(30), 0, 20],
            [0, 0, 1, 30],
            [0, 0, 0, 1],
        ]
    )

    my_icp(
        fixed_cloud=point_cloud_fixed,
        moving_cloud=point_cloud_moving,
        initial_guess=initial_guess,
        iterations=1000,
    )

    print(f"Total duration: {perf_counter() - t_total:.4f} seconds.")
