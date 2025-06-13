from typing import Optional
import orjson
import rich
import transforms
import open3d as o3d
from tqdm import tqdm
from time import perf_counter
from sklearn.neighbors import NearestNeighbors
import numpy as np
from vis import visualise_point_clouds


class OctreeNode:
    """
    The node of an octree
    """

    def __init__(self, center: np.ndarray, size: int, depth: int):
        self.center = center
        self.size = size
        self.depth = depth
        self.points = []
        self.children = [None] * 8
        self.is_leaf = True


class Octree:
    """
    An Octree
    Each node has 8 children, which children are also nodes.
    Edge nodes are called leafs.
    If a point is added to a node, and that node has more than 8 children

    """

    def __init__(self, point_cloud: np.ndarray):
        self.point_cloud = point_cloud

        min_coords = np.min(point_cloud, axis=0)
        max_coords = np.max(point_cloud, axis=0)
        self.root_size = np.max(max_coords - min_coords)
        self.root_center = (min_coords + max_coords) / 2.0
        self.max_depth = 0

        # Initialize the root of the tree with a node representing the entire point cloud.
        self.root = OctreeNode(self.root_center, self.root_size, 0)

    def add_point_to_node(self, node: OctreeNode, point: np.ndarray) -> int:
        """
        Add a new point to a specific node

        Args:
            node (OctreeNode): The node to add the point to.
            point (np.ndarray): The point to add.

        Returns:
            int: THe index of the new point to the node.
        """
        # Get the position the point will hold in the node (0-7).
        index = self.assign_point_to_branch_index(point, node.center)

        # If the node has this spot available create a new node, which will take the place
        # of the initially intended index
        if node.children[index] is None:
            # Cut the size into half since we are going one level down.
            new_size = node.size / 2.0
            offset = new_size / 2.0

            # Calculate the centre of the new node.
            # Binary representation of positions to make the division of the axes more understandable.
            new_center = np.array(
                [
                    node.center[0] + offset if (index & 0b100) else node.center[0] - offset,
                    node.center[1] + offset if (index & 0b010) else node.center[1] - offset,
                    node.center[2] + offset if (index & 0b001) else node.center[2] - offset,
                ]
            )
            # Update octree depth.
            self.max_depth = node.depth + 1 if node.depth + 1 > self.max_depth else self.max_depth

            # Add the new node.
            node.children[index] = OctreeNode(center=new_center, size=new_size, depth=node.depth + 1)
        return index

    def build(self):
        """
        Builds the octree by inserting all points iteratively.
        """
        for point in tqdm(self.point_cloud, desc="Building octree representation."):
            # We begin by the root of the tree.
            current_node = self.root

            while True:
                # If we are on a leaf (edge node), we can attempt to add the point to it
                if current_node.is_leaf:
                    current_node.points.append(point)

                    # If the node's capacity is exceeded...
                    if len(current_node.points) > 8:
                        current_node.is_leaf = False
                        points_to_redistribute = current_node.points.copy()
                        current_node.points = []

                        for p in points_to_redistribute:
                            self.redistribute(current_node, p)

                    break

                else:
                    index = self.add_point_to_node(node=current_node, point=point)
                    current_node = current_node.children[index]

    def redistribute(self, node: OctreeNode, point: np.ndarray):
        """
        A standard recursive insert, used only for redistributing points
        from a node that has just been subdivided.
        While recursion is generally a bad idea, this cannot get out of bounds
        since we will be always managing 8 points in total.
        """
        # If the node is a leaf, just add the point to the existing.
        if node.is_leaf:
            node.points.append(point)
            return

        # Otherwise, find its position in the current node,
        index = self.add_point_to_node(node=node, point=point)
        self.redistribute(node.children[index], point)

    def assign_point_to_branch_index(self, point: np.ndarray, node_centre: np.ndarray) -> int:
        """Assign a new point to the correct index/branch of the nodem depending on its
        position in relation to the node centre.
        For this assignment we want to see whether the point is in front of / behind
        the node centre in the x/y/z axis. This gives us 2**3 combinations, hence octree.
        Binary representation of positions to make the division of the axes more understandable.

        Args:
            point (np.ndarray): The point.
            node_centre (np.ndarray): the node centre.

        Returns:
            int:  The index of the point in the node's children.
        """
        index = 0b000
        if point[0] >= node_centre[0]:
            index |= 0b100
        if point[1] >= node_centre[1]:
            index |= 0b010
        if point[2] >= node_centre[2]:
            index |= 0b001
        return index

    def subsample_point_cloud(self) -> np.ndarray:
        """Subsample the point cloud by going down the octree
        and keeping the mean point position at its lowest depths.

        Returns:
            np.ndarray: The subsampled point cloud.
        """
        centres = []

        def traverse(node: Optional[OctreeNode]):
            """Helper function to tranverse the octree.

            Args:
                node (Optional[OctreeNode]): The node from which to traverse
            """
            if node is None:
                return
            if node.is_leaf:
                if node.points:
                    centres.append(np.mean(node.points, axis=0))
                return
            for child in node.children:
                traverse(child)

        # Begin from the root, and go down.
        traverse(self.root)
        return np.array(centres)


def estimate_normals(point_cloud: np.ndarray, radius: float = 0.1) -> np.ndarray:
    """Estimates normals for each point in the cloud bsed on its neighbours.

    Args:
        point_cloud (np.ndarray): The point cloud.
        radius (float, optional): The radius for neighbour calculation. Defaults to 0.1.

    Returns:
        np.ndarray: The normal vectors for each point in the point cloud.
    """
    normals = np.zeros_like(point_cloud)

    # Use KNN to find neighbours for each point
    nn = NearestNeighbors(radius=radius).fit(point_cloud)
    neighbourhoods = nn.radius_neighbors(point_cloud, return_distance=False)

    for i, indices in enumerate(tqdm(neighbourhoods, desc="Estimating Normals")):
        # Get the neighboring points
        neighbours = point_cloud[indices]
        if len(neighbours) < 3:
            # Not enough points to define a plane, result in a zero vector
            normals[i] = np.array([0, 0, 0])
            continue

        # Fit a plane to the neighbours using SVD
        # The normal is the last singular vector (the one with the smallest singular value).
        centered_neighbours = neighbours - np.mean(neighbours, axis=0)
        _, _, vh = np.linalg.svd(centered_neighbours)
        normals[i] = vh[2, :]  # The normal vector

    return normals


def compute_spfh(point_cloud: np.ndarray, normals: np.ndarray, radius: float) -> np.ndarray:
    """
    Computes Simple Point Feature Histograms (SPFH) for each point.
    The results is a feature vector describing each point in the cloud.
    The feature vector

    Args:
        point_cloud (np.ndarray): The (N, 3) downsampled point cloud.
        normals (np.ndarray): The (N, 3) normals for the point cloud.
        radius (float): The search radius for defining a neighborhood.

    Returns:
        np.ndarray: An (N, 33) array where each row is the SPFH descriptor.
    """
    # Use KNN to find neighbors for each point efficiently
    nn = NearestNeighbors(radius=radius).fit(point_cloud)
    neighborhoods = nn.radius_neighbors(point_cloud, return_distance=False)

    # 3 histograms each with 11 bins = 33.
    num_bins = 11
    all_features = np.zeros((len(point_cloud), num_bins * 3))

    for i in tqdm(range(len(point_cloud)), desc="Computing SPFH"):
        p_i = point_cloud[i]
        n_i = normals[i]

        # Get the neighbors for the current point p_i
        neighbor_indices = neighborhoods[i]
        # Skip the point itself if it's included in its own neighborhood
        neighbor_indices = neighbor_indices[neighbor_indices != i]

        if len(neighbor_indices) == 0:
            continue

        # Define the local coordinate system at p_i
        u = n_i
        # Create a vector v that is not parallel to u
        v_candidate = np.random.randn(3)
        v_candidate -= v_candidate.dot(u) * u  # Make it orthogonal to u
        v = v_candidate / np.linalg.norm(v_candidate)
        w = np.cross(u, v)

        # Initialize histograms for the current point
        hist_alpha = np.zeros(num_bins)
        hist_phi = np.zeros(num_bins)
        hist_theta = np.zeros(num_bins)

        # Iterate over neighbors to compute features
        for j in neighbor_indices:
            p_j = point_cloud[j]
            n_j = normals[j]

            # Vector from p_i to p_j
            diff = p_j - p_i
            diff_norm = np.linalg.norm(diff)
            if diff_norm == 0:
                continue

            # Calculate the three angular features
            #   - alpha is the angle between n_j and the local frame's v
            #   - phi is the angle between the difference vector and normal n_i's perpendicular (v)
            #   - theta is the angle between the normal vectors of the two points
            alpha = v.dot(n_j)
            phi = v.dot(diff) / diff_norm
            theta = np.arctan2(w.dot(n_j), u.dot(n_j))

            # Bin the features
            # Convert feature values from [-1, 1] or [-pi, pi] to a bin index [0, num_bins-1]
            alpha_bin = int((alpha + 1.0) / 2.0 * (num_bins - 1))
            phi_bin = int((phi + 1.0) / 2.0 * (num_bins - 1))
            theta_bin = int((theta + np.pi) / (2 * np.pi) * (num_bins - 1))

            hist_alpha[alpha_bin] += 1
            hist_phi[phi_bin] += 1
            hist_theta[theta_bin] += 1

        # Normalize the histograms
        hist_alpha /= np.sum(hist_alpha) if np.sum(hist_alpha) > 0 else 1
        hist_phi /= np.sum(hist_phi) if np.sum(hist_phi) > 0 else 1
        hist_theta /= np.sum(hist_theta) if np.sum(hist_theta) > 0 else 1

        # Concatenate and store the final feature descriptor
        all_features[i] = np.concatenate([hist_alpha, hist_phi, hist_theta])

    return all_features


def find_feature_correspondences(features_a, features_b):
    """
    Finds potential correspondences by matching feature descriptors.
    For each feature in A, it finds the closest feature in B.

    Returns:
        np.ndarray: An array of indices [index_in_A, index_in_B]
    """
    # Use KNN to find the nearest neighbour in FEATURE space
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(features_b)
    distances, indices_b = nn.kneighbors(features_a)

    # Create an array of corresponding indices
    indices_a = np.arange(len(features_a))
    correspondences = np.vstack((indices_a, indices_b.ravel())).T
    return correspondences


def run_ransac(
    moving_points: np.ndarray,
    fixed_points: np.ndarray,
    moving_features: np.ndarray,
    fixed_features: np.ndarray,
    num_iterations: int = 1000,
    sample_size: int = 3,
    distance_threshold: float = 0.5,
):
    """
    Performs RANSAC to find the best rigid transformation matching a moving point cloud to a fixed one.
    This will be used as the initial guess instead of a random transform in the ICP algorithm.
    It matches the features between the clouds, and then randomly selects some points (assumed correct)
    and uses them to calculate a transform.
    The transform resulting in the greatest number of close-enough (threshold based) points,
    is kept as the best one.

    Args:
        moving_points (np.ndarray): The moving point cloud.
        fixed_points (np.ndarray): The fixed point cloud.
        moving_features (np.ndarray): The features for the moving cloud.
        fixed_features (np.ndarray): The features for the fixed cloud.
        num_iterations (int): How many random samples to try.
        sample_size (int): The number of points to use for each hypothesis (usually 3).
        distance_threshold (float): The max distance to count a correspondence as an inlier.

    Returns:
        np.ndarray: The best 4x4 transformation matrix found.
    """
    rich.print("Starting RANSAC to find initial alignment...")

    best_transform = np.eye(4)
    max_inliers = 0

    # Find all potential correspondences based on feature similarity
    all_correspondences = find_feature_correspondences(moving_features, fixed_features)

    for i in tqdm(range(num_iterations), desc="Running RANSAC"):
        # Select a random sample of correspondences
        sample_indices = np.random.choice(len(all_correspondences), sample_size, replace=False)
        sample_corr = all_correspondences[sample_indices]

        # Get the actual 3D points for our sample
        sample_points_a = moving_points[sample_corr[:, 0]]
        sample_points_b = fixed_points[sample_corr[:, 1]]

        # Compute a candidate transform from this small sample
        transform_candidate = estimate_rigid_transform(sample_points_a, sample_points_b)

        # Apply the transform to ALL source points
        points_a_transformed = (transform_candidate @ np.hstack([moving_points, np.ones((len(moving_points), 1))]).T).T[:, :3]

        # Get the corresponding points in cloud B based on our initial feature matching
        corresponding_points_b = fixed_points[all_correspondences[:, 1]]

        # Calculate the distance between the transformed points and their correspondences
        distances_sq = np.sum((points_a_transformed[all_correspondences[:, 0]] - corresponding_points_b) ** 2, axis=1)

        # Count the inliers (numbers of close enough sample pairs)
        inlier_count = np.sum(distances_sq < distance_threshold**2)

        # Keep the best transform
        if inlier_count > max_inliers:
            max_inliers = inlier_count
            best_transform = transform_candidate

    rich.print(f"[bold green]RANSAC finished. Best alignment found with {max_inliers} inliers.[/bold green]")
    return best_transform


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


def solve_point_to_plane_least_squares(moving_points, fixed_points, fixed_normals):
    """
    Calculates the small rigid transformation that minimizes the point-to-plane
    error using a linear least-squares approach.

    Args:
        moving_points (np.ndarray): The (N, 3) moving points (p_i).
        fixed_points (np.ndarray): The (N, 3) corresponding fixed points (q_i).
        fixed_normals (np.ndarray): The (N, 3) normals at the fixed points (n_i).

    Returns:
        np.ndarray: A 4x4 incremental transformation matrix.
    """

    # We want to solve Ax=b
    # A is the
    A = np.hstack([np.cross(moving_points, fixed_normals), fixed_normals])
    b = np.sum((fixed_points - moving_points) * fixed_normals, axis=1)

    # Solve the least-squares system Ax = b
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # The solution vector x contains our 6 motion parameters
    alpha, beta, gamma, tx, ty, tz = x

    # Convert the small rotation angles into a 4x4 transformation matrix
    incremental_transform = np.eye(4)

    # Rotation part
    angle = np.linalg.norm([alpha, beta, gamma])
    if angle > 1e-6:  # Avoid division by zero
        K = np.array([[0, -gamma, beta], [gamma, 0, -alpha], [-beta, alpha, 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    else:
        R = np.eye(3)
    incremental_transform[:3, :3] = R

    # Translation part
    incremental_transform[:3, 3] = [tx, ty, tz]

    return incremental_transform


def my_icp_point_to_plane(
    fixed_cloud: np.ndarray,
    fixed_normals: np.ndarray,
    moving_cloud: np.ndarray,
    initial_guess: np.ndarray,
    iterations: int = 100,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Perform the ICP algorithm to match points to planes, instead of the conventional
    points to points. This ensures greater smoothness.

    Args:
        fixed_cloud (np.ndarray): The fixed point cloud.
        fixed_normals (np.ndarray): The normal vectors of the points of the fixed point cloud.
        moving_cloud (np.ndarray): The moving point cloud.
        initial_guess (np.ndarray): The initial guess for the transformation.
        iterations (int, optional): The number of iteration for the ICP algorithm. Defaults to 100.
        tolerance (float, optional): The error tolerance. Defaults to 1e-6.

    Returns:
        np.ndarray: The optimal transform.
    """
    fixed_cloud_3d = fixed_cloud[:, :3]
    info = {"iteration": [], "rmse": [], "time": []}
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(fixed_cloud_3d)

    rigid_transform = initial_guess
    prev_error = float("inf")

    for i in range(iterations):
        t = perf_counter()
        moving_cloud_hom = np.hstack([moving_cloud[:, :3], np.ones((len(moving_cloud), 1))])
        moved_cloud_3d = (rigid_transform @ moving_cloud_hom.T).T[:, :3]

        _, indices = nn.kneighbors(moved_cloud_3d)

        corresponding_points = fixed_cloud_3d[indices.ravel()]
        corresponding_normals = fixed_normals[indices.ravel()]

        rmse = np.sqrt(np.mean(np.sum((moved_cloud_3d - corresponding_points) * corresponding_normals, axis=1) ** 2))

        correction_transform = solve_point_to_plane_least_squares(
            moved_cloud_3d,
            corresponding_points,
            corresponding_normals,
        )

        rigid_transform = correction_transform @ rigid_transform

        t = perf_counter() - t

        info["iteration"].append(i)
        info["rmse"].append(rmse)
        info["time"].append(t)

        visualise_point_clouds([moved_cloud_3d, fixed_cloud_3d], save_dir="vis_optim", save_frame=True, frame_id=i)
        rich.print(f"Iteration {i + 1}\t |\t RMSE = {rmse} \t|\t Time: {int(t * 1000)}ms")
        if abs(prev_error - rmse) < tolerance:
            rich.print(f"[bold green]Converged at iteration {i} with error {rmse} at {int(np.mean(info['time']) * 1000)}ms per iteration.[/bold green]")
            open("./res_simple.json", "wb").write(orjson.dumps(info))
            return rigid_transform
        prev_error = rmse

    rich.print(f"[bold yellow]Did not converge - Last error: {prev_error} at {int(np.mean(info['time']) * 1000)}ms per iteration/bold yellow]")
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
    point_cloud_fixed = point_cloud.copy()
    point_cloud_moving = point_cloud_chngd[:, :-1].copy()

    t_total = perf_counter()

    # Define a radius based on the size of the object
    cloud_span = np.max(np.max(point_cloud_fixed, axis=0) - np.min(point_cloud_fixed, axis=0))
    radius_normal = cloud_span * 0.05
    radius_feature = cloud_span * 0.1

    octree = Octree(point_cloud=point_cloud_fixed)
    octree.build()
    optim_point_cloud_fixed = octree.subsample_point_cloud().copy()
    normals_fixed = estimate_normals(optim_point_cloud_fixed, radius=0.1)
    features_fixed = compute_spfh(optim_point_cloud_fixed, normals_fixed, radius=0.1)
    rich.print("Computed SPFH features. Shape:", features_fixed.shape)

    octree = Octree(point_cloud=point_cloud_moving)
    octree.build()
    optim_point_cloud_moving = octree.subsample_point_cloud().copy()
    normals_moving = estimate_normals(optim_point_cloud_moving, radius=0.1)
    features_moving = compute_spfh(optim_point_cloud_moving, normals_moving, radius=0.1)
    rich.print("Computed SPFH features. Shape:", features_moving.shape)

    del octree

    initial_guess = run_ransac(
        moving_points=optim_point_cloud_moving,
        fixed_points=optim_point_cloud_fixed,
        moving_features=features_moving,
        fixed_features=features_fixed,
        distance_threshold=cloud_span * 0.01,
        num_iterations=5000,
    )

    my_icp_point_to_plane(
        fixed_cloud=np.hstack([optim_point_cloud_fixed, np.ones((optim_point_cloud_fixed.shape[0], 1))]),
        fixed_normals=normals_fixed,
        moving_cloud=np.hstack([optim_point_cloud_moving, np.ones((optim_point_cloud_moving.shape[0], 1))]),
        initial_guess=initial_guess,
        iterations=1000,
        tolerance=1e-6,
    )
    print(f"Total duration: {perf_counter() - t_total:.4f} seconds.")
