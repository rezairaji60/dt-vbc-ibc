import time
import numpy as np
from scipy.optimize import linprog
from .common import phi2

def _solve_lp(c, Aub, bub, bounds):
    return linprog(c, A_ub=np.asarray(Aub), b_ub=np.asarray(bub), bounds=bounds, method="highs")

def synth_forward_dt_vbc(f, X0_pts, Xu_pts, D_pts, A, m=2, sep_idx=0, coeff_bound=5.0, eps_bound=2.0):
    t0 = time.time()
    nv = 3 * m + 1
    ie = nv - 1
    Aub, bub = [], []

    P0 = phi2(X0_pts)
    Pu = phi2(Xu_pts)
    PD = phi2(D_pts)
    PF = phi2(np.array([f(x) for x in D_pts]))

    for p in P0:
        for i in range(m):
            row = np.zeros(nv)
            row[3*i:3*i+3] = p
            row[ie] = 1.0
            Aub.append(row); bub.append(0.0)

    for p in Pu:
        row = np.zeros(nv)
        row[3*sep_idx:3*sep_idx+3] = -p
        row[ie] = 1.0
        Aub.append(row); bub.append(0.0)

    for p, pf in zip(PD, PF):
        for i in range(m):
            row = np.zeros(nv)
            row[3*i:3*i+3] += pf
            for j in range(m):
                row[3*j:3*j+3] += -A[i, j] * p
            row[ie] = 1.0
            Aub.append(row); bub.append(0.0)

    bounds = [(-coeff_bound, coeff_bound)] * (nv - 1) + [(0.0, eps_bound)]
    c = np.zeros(nv); c[ie] = -1.0
    res = _solve_lp(c, Aub, bub, bounds)
    return {
        "success": bool(res.success),
        "epsilon": float(res.x[ie]) if res.success else None,
        "coeffs": res.x[:-1].reshape(m, 3).tolist() if res.success else None,
        "A": np.asarray(A).tolist(),
        "runtime_sec": time.time() - t0,
        "message": res.message,
        "formulation": "Forward DT-VBC",
        "degree": 2,
        "num_functions": m,
    }

def synth_backward_dt_vbc(f, X0_pts, Xu_pts, D_pts, A, m=2, sep_idx=0, coeff_bound=5.0, eps_bound=2.0):
    t0 = time.time()
    nv = 3 * m + 1
    ie = nv - 1
    Aub, bub = [], []

    P0 = phi2(X0_pts)
    Pu = phi2(Xu_pts)
    PD = phi2(D_pts)
    PF = phi2(np.array([f(x) for x in D_pts]))

    for p in Pu:
        for i in range(m):
            row = np.zeros(nv)
            row[3*i:3*i+3] = p
            row[ie] = 1.0
            Aub.append(row); bub.append(0.0)

    for p in P0:
        row = np.zeros(nv)
        row[3*sep_idx:3*sep_idx+3] = -p
        row[ie] = 1.0
        Aub.append(row); bub.append(0.0)

    for p, pf in zip(PD, PF):
        for i in range(m):
            row = np.zeros(nv)
            row[3*i:3*i+3] += p
            for j in range(m):
                row[3*j:3*j+3] += -A[i, j] * pf
            row[ie] = 1.0
            Aub.append(row); bub.append(0.0)

    bounds = [(-coeff_bound, coeff_bound)] * (nv - 1) + [(0.0, eps_bound)]
    c = np.zeros(nv); c[ie] = -1.0
    res = _solve_lp(c, Aub, bub, bounds)
    return {
        "success": bool(res.success),
        "epsilon": float(res.x[ie]) if res.success else None,
        "coeffs": res.x[:-1].reshape(m, 3).tolist() if res.success else None,
        "A": np.asarray(A).tolist(),
        "runtime_sec": time.time() - t0,
        "message": res.message,
        "formulation": "Backward DT-VBC",
        "degree": 2,
        "num_functions": m,
    }

def synth_forward_ibc(f, X0_pts, Xu_pts, D_pts, lambdas, k=2, coeff_bound=5.0, eps_bound=2.0):
    t0 = time.time()
    m = k + 1
    nv = 3 * m + 1
    ie = nv - 1
    Aub, bub = [], []

    P0 = phi2(X0_pts)
    Pu = phi2(Xu_pts)
    PD = phi2(D_pts)
    PF = phi2(np.array([f(x) for x in D_pts]))

    for p in P0:
        row = np.zeros(nv)
        row[0:3] = p
        row[ie] = 1.0
        Aub.append(row); bub.append(0.0)

    for p in Pu:
        for i in range(m):
            row = np.zeros(nv)
            row[3*i:3*i+3] = -p
            row[ie] = 1.0
            Aub.append(row); bub.append(0.0)

    for p, pf in zip(PD, PF):
        for i in range(k):
            row = np.zeros(nv)
            row[3*(i+1):3*(i+1)+3] += lambdas[i] * pf
            row[3*i:3*i+3] += -p
            row[ie] = 1.0
            Aub.append(row); bub.append(0.0)
        row = np.zeros(nv)
        row[3*k:3*k+3] += lambdas[k] * pf
        row[3*k:3*k+3] += -p
        row[ie] = 1.0
        Aub.append(row); bub.append(0.0)

    bounds = [(-coeff_bound, coeff_bound)] * (nv - 1) + [(0.0, eps_bound)]
    c = np.zeros(nv); c[ie] = -1.0
    res = _solve_lp(c, Aub, bub, bounds)
    return {
        "success": bool(res.success),
        "epsilon": float(res.x[ie]) if res.success else None,
        "coeffs": res.x[:-1].reshape(m, 3).tolist() if res.success else None,
        "lambdas": list(map(float, lambdas)),
        "runtime_sec": time.time() - t0,
        "message": res.message,
        "formulation": "Forward IBC",
        "degree": 2,
        "num_functions": m,
    }

def synth_backward_ibc(f, X0_pts, Xu_pts, D_pts, lambdas, k=2, coeff_bound=5.0, eps_bound=2.0):
    t0 = time.time()
    m = k + 1
    nv = 3 * m + 1
    ie = nv - 1
    Aub, bub = [], []

    P0 = phi2(X0_pts)
    Pu = phi2(Xu_pts)
    PD = phi2(D_pts)
    PF = phi2(np.array([f(x) for x in D_pts]))

    for p in Pu:
        row = np.zeros(nv)
        row[0:3] = p
        row[ie] = 1.0
        Aub.append(row); bub.append(0.0)

    for p in P0:
        for i in range(m):
            row = np.zeros(nv)
            row[3*i:3*i+3] = -p
            row[ie] = 1.0
            Aub.append(row); bub.append(0.0)

    for p, pf in zip(PD, PF):
        for i in range(k):
            row = np.zeros(nv)
            row[3*(i+1):3*(i+1)+3] += p
            row[3*i:3*i+3] += -lambdas[i] * pf
            row[ie] = 1.0
            Aub.append(row); bub.append(0.0)
        row = np.zeros(nv)
        row[3*k:3*k+3] += p
        row[3*k:3*k+3] += -lambdas[k] * pf
        row[ie] = 1.0
        Aub.append(row); bub.append(0.0)

    bounds = [(-coeff_bound, coeff_bound)] * (nv - 1) + [(0.0, eps_bound)]
    c = np.zeros(nv); c[ie] = -1.0
    res = _solve_lp(c, Aub, bub, bounds)
    return {
        "success": bool(res.success),
        "epsilon": float(res.x[ie]) if res.success else None,
        "coeffs": res.x[:-1].reshape(m, 3).tolist() if res.success else None,
        "lambdas": list(map(float, lambdas)),
        "runtime_sec": time.time() - t0,
        "message": res.message,
        "formulation": "Backward IBC",
        "degree": 2,
        "num_functions": m,
    }
