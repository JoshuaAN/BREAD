import numpy as np
from solver.ldlt import LDLT
from math import *

def projected_cg(H, c, A, delta, x0):
    # Profiling showed majority of time is spent in LDLT solves

    n_constraints = A.shape[0]
    n_vars = x0.shape[0]
    x = x0.astype('float64')
    # [I  Aᵀ]
    # [A  0 ]
    lhs = np.vstack((
        np.hstack((np.identity(n_vars), A.T)), 
        np.hstack((A, np.zeros((n_constraints, n_constraints))))
    ))

    # [r]
    # [0]
    rhs = lambda r: np.vstack((r, np.zeros((n_constraints, 1))))
    y = lambda r: np.linalg.solve(lhs, (rhs(r)))[n_vars:(n_vars + n_constraints)]
    r = H @ x + c
    r -= A.T @ y(r)
    g = np.linalg.solve(lhs, (rhs(r)))[0:n_vars]
    p = -g

    iterations = 0

    while ((r.T @ g)[0, 0] > 1e-24 and iterations < 2 * (n_vars - n_constraints)): #
        print("c norm: ", np.linalg.norm(A @ x))
        tmp = (p.T @ H @ p)[0, 0]
        absOld = (r.T @ g)[0, 0]
        # Check for negative curvature
        if tmp < 0:
            # Solve τ to satisfy 
            # 
            #   |x + τp|² = Δ²
            # 
            #   (x + τp)ᵀ(x + τp) = Δ²
            #   τ²pᵀp + τ(2xᵀp) + xᵀx = Δ²
            # 
            # This is a quadratic problem
            # 
            #   A = pᵀp
            #   B = 2xᵀp
            #   C = xᵀx - Δ²
            print("Projected CG - Negative curvature end condition")
            _A = (p.T @ p)[0, 0]
            _B = 2 * (x.T @ p)[0, 0]
            _C = (x.T @ x)[0, 0] - delta * delta
            tau = (-_B + sqrt(_B * _B - 4 * _A * _C)) / (2 * _A)
            return x + tau * p
        alpha = absOld / tmp
        if (np.linalg.norm(x + alpha * p) >= delta):
            print("Projected CG - Exceeded trust region end condition")
            # Same quadratic formulation as negative curvature end condition
            _A = (p.T @ p)[0, 0]
            _B = 2 * (x.T @ p)[0, 0]
            _C = (x.T @ x)[0, 0] - delta * delta
            tau = (-_B + sqrt(_B * _B - 4 * _A * _C)) / (2 * _A)
            return x + tau * p
        x += alpha * p
        r += alpha * H @ p
        sol = np.linalg.solve(lhs, (rhs(r)))
        g = sol[0:n_vars]
        beta = (r.T @ g)[0, 0] / absOld
        p = -g + beta * p
        r -= A.T @ sol[n_vars:(n_vars + n_constraints)]

        iterations += 1

    print("c norm: ", np.linalg.norm(A @ x))

    print("Projected CG - Reached tolerance end condition")

    return x