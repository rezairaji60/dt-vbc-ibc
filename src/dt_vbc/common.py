"""
Author: Reza Iraji
Date:   March 2026
"""
import numpy as np

def make_grid(bounds, n):
    xs = np.linspace(bounds[0], bounds[1], n)
    ys = np.linspace(bounds[2], bounds[3], n)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    return X, Y, pts

def sample_box(box, n):
    xs = np.linspace(box[0], box[1], n)
    ys = np.linspace(box[2], box[3], n)
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X.ravel(), Y.ravel()], axis=1)

def phi2(x):
    x = np.asarray(x)
    x1, x2 = x[..., 0], x[..., 1]
    return np.stack([np.ones_like(x1), x1**2, x2**2], axis=-1)

def simulate(f, x0, steps):
    x = np.asarray(x0, dtype=float)
    traj = [x.copy()]
    for _ in range(steps):
        x = np.asarray(f(x), dtype=float)
        traj.append(x.copy())
    return np.array(traj)

def rollout_box(f, box, count, steps, rng):
    lo = np.array([box[0], box[2]], dtype=float)
    hi = np.array([box[1], box[3]], dtype=float)
    pts = rng.uniform(lo, hi, size=(count, 2))
    return [simulate(f, p, steps) for p in pts]
