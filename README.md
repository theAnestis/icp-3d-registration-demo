# ICP 3D Registration Demo

This repository contains the code for a demo on the ICP algorithm for 3D Registration.

The 3D model used for the purposes of this demo, is a liver model downloaded from [here](https://free3d.com/3d-model/hepatitis-liver-217237.html). Apparently, it is from a hepatitis patient. Extract the contents, and place them in a directory called `hepatitis-liver` inside the project.

## Setup
To recreate the demo, you need to reproduce the python environment. For this you will need UV:
1. Install UV
    - For Linux/MacOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
    - For Windows: `winget install astral-sh.uv`
2. Navigate to the project directory.
3. Run `uv sync`.
4. The environment will be in the `.venv` directory. 
    - You can activate it via `.venv/Scripts/activate`.
    - You can directly use the python environment via `.venv/Scripts/python.exe`.

## Demo

Two related scripts are present:
1. `run_icp_simple.py` which uses a simple version of the ICP algorithm, beginning with a transformation guess and then iteratively looking for the best transformation for the point-to-point task.
2. `run_icp_optim.py` which is an optimised version, that reduces the number of points via an octree, forms a better initial guess for the transformation using RANSAC with SPFH, and performs ICP for a point-to-plane task.

Logs for demo run can be seen in `res_simple.txt` and `res_optim.txt`. Images and videos for each iteration of each implementation can be seen in the `visualisations` directory.

> **IMPORTANT**: Since there is an inherent randomness to the algorithms, the results may not be 100% reproducible.

Two additional files, `transforms.py` and `vis.py` provide helper functions for transformations and visualisations respectively.

You can run the demonstrations **after** downloading the liver 3d model, and setting up the python environment by running:

```bash
.venv/Scripts/python.exe run_icp_simple.py  # For simple ICP
.venv/Scripts/python.exe run_icp_optim.py  # For optimised ICP
```