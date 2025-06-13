import numpy as np
from typing import Dict, Literal, TypedDict


class TranslationTransform(TypedDict):
    x: float = 0
    y: float = 0
    z: float = 0


class RotationTransform(TypedDict):
    axis: Literal["x", "y", "z"] = "x"
    angle: float = 0


class ScaleTransform(TypedDict):
    x: float = 0
    y: float = 0
    z: float = 0


class ShearTranform(TypedDict):
    axis: Literal["x", "y", "z"] = "x"
    params: Dict[Literal["x", "y", "z"], float] = {"y": 0, "z": 0}


def rotate_and_translate_mat(
    rot_parms: RotationTransform,
    trans_params: TranslationTransform,
):
    tranform_matrix = np.eye(4, dtype=np.float32)

    cos = np.cos(rot_parms["angle"] * np.pi / 180)
    sin = np.sin(rot_parms["angle"] * np.pi / 180)

    rot_mat = np.eye(3)
    match rot_parms["axis"]:
        case "x":
            rot_mat[1, 1] = cos
            rot_mat[2, 2] = cos
            rot_mat[1, 2] = -sin
            rot_mat[2, 1] = sin
        case "y":
            rot_mat[0, 0] = cos
            rot_mat[2, 2] = cos
            rot_mat[2, 0] = -sin
            rot_mat[0, 2] = sin
        case "z":
            rot_mat[0, 0] = cos
            rot_mat[1, 1] = cos
            rot_mat[0, 1] = -sin
            rot_mat[1, 0] = sin

    tranform_matrix[:3, :3] = rot_mat
    tranform_matrix[:3, -1] = np.array(list(trans_params.values()))

    return tranform_matrix


def scale_mat(
    scale_params: ScaleTransform,
):
    tranform_matrix = np.eye(4)
    for i in range(len(scale_params)):
        tranform_matrix[i, i] = list(scale_params.values())[i]

    return tranform_matrix


def shear_mat(
    shear_params: ShearTranform,
):
    assert shear_params["axis"] not in shear_params["params"].keys(), ValueError("You cannot modify the axis on which you base your shear operations.")
    tranform_matrix = np.eye(4)
    shear_mat = np.eye(3)
    match shear_params["axis"]:
        case "x":
            shear_mat[1, 0] = shear_params["params"]["y"]
            shear_mat[2, 0] = shear_params["params"]["z"]
        case "y":
            shear_mat[0, 1] = shear_params["params"]["x"]
            shear_mat[2, 1] = shear_params["params"]["z"]
        case "z":
            shear_mat[0, 2] = shear_params["params"]["x"]
            shear_mat[1, 2] = shear_params["params"]["y"]

    tranform_matrix[:3, :3] = shear_mat
    return tranform_matrix
