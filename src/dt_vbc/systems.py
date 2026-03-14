import numpy as np

def system1(x):
    x1, x2 = x
    return np.array([
        0.72 * x1 + 0.10 * x2 - 0.12 * x1**3,
        -0.08 * x1 + 0.68 * x2 - 0.08 * x2**3,
    ], dtype=float)

SYSTEM1 = {
    "name": "S1",
    "label": "baseline polynomial benchmark",
    "f": system1,
    "domain": (-1.30, 1.30, -1.30, 1.30),
    "X0": (-0.20, 0.20, -0.20, 0.20),
    "Xu": (0.95, 1.25, 0.95, 1.25),
    "description": "Stylized polynomial benchmark on which all formulations are feasible with low-degree templates.",
}

def system2(x):
    x1, x2 = x
    return np.array([
        0.60 * x1 + 0.05 * x2,
        -0.02 * x1**3 + 0.70 * x2,
    ], dtype=float)

SYSTEM2 = {
    "name": "S2",
    "label": "backward-favorable polynomial benchmark",
    "f": system2,
    "domain": (-1.20, 1.20, -1.20, 1.20),
    "X0": (-0.15, 0.15, -0.15, 0.15),
    "Xu": (0.50, 1.10, 0.50, 1.10),
    "description": "Polynomial benchmark for which backward formulations achieve larger feasible margins.",
}

def encoding_scalar_map(x):
    return 0.5 * x

def encoding_frames(x):
    x = np.asarray(x)
    b0 = 0.20 - x**2
    b1 = 0.16 - x**2
    b2 = 0.12 - x**2
    return np.stack([b0, b1, b2], axis=-1)
