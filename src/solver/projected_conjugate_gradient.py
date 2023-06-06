import numpy as np
from scipy.linalg import cho_factor, cho_solve
from solver.ldlt import LDLT

def projected_cg(H, g, A, delta, x0):
    # Profiling show majority of time is spent in LDLT solves

    n_constraints = A.shape[0]
    n_vars = x0.shape[0]
    x = x0.astype('float64')
    # [I  Aᵀ]
    # [A  0 ]
    lhs = LDLT(np.vstack((
        np.hstack((np.identity(n_vars), A.T)), 
        np.hstack((A, 1e-12 * np.identity(n_constraints)))
    )))
    # [r]
    # [0]
    rhs = lambda r: np.vstack((r, np.zeros((n_constraints, 1))))
    y = lambda r: lhs.solve(rhs(r))[n_vars:(n_vars + n_constraints)]
    r = H @ x + g
    r -= A.T @ y(r)
    g = lhs.solve(rhs(r))[0:n_vars]
    p = -g

    iterations = 0

    while ((r.T @ g)[0, 0] > 1e-12 and iterations < (n_vars - n_constraints)):
        tmp = (r.T @ g)[0, 0]
        alpha = tmp / (p.T @ H @ p)[0, 0]
        x += alpha * p
        r += alpha * H @ p
        r -= A.T @ y(r)
        g = lhs.solve(rhs(r))[0:n_vars]
        beta = (r.T @ g)[0, 0] / tmp
        p = -g + beta * p

        iterations += 1

    return x