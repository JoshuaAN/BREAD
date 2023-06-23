from solver.projected_conjugate_gradient import projected_cg
import numpy as np
import time
from solver.ldlt import LDLT
import cProfile
from pstats import Stats, SortKey
import random


def positive_definite_no_trust_region():
    print("POSITIVE DEFINITE NO TRUST REGION ------------")

    N = 1000

    H = np.random.rand(N, N)
    H = H.T @ H
    g = np.random.rand(N, 1)
    A = np.random.rand(int(N / 2), N)
    x0 = np.random.rand(N, 1)
    c = A @ x0

    n_constraints = A.shape[0]
    n_vars = x0.shape[0]

    lhs = np.vstack(
        (np.hstack((H, A.T)), np.hstack((A, np.zeros((n_constraints, n_constraints)))))
    )
    rhs = np.vstack((-g, c))

    t0 = time.time()
    x = projected_cg(H, g, A, float("inf"), x0)
    t1 = time.time()

    t2 = time.time()
    sol = LDLT(lhs).solve(rhs)[0:n_vars]
    t3 = time.time()

    print("CG time: \t", (t1 - t0) * 1000, "ms")
    print("Exact time: \t", (t3 - t2) * 1000, "ms")
    print("CG cost: \t", (0.5 * x.T @ H @ x + x.T @ g)[0, 0])
    print("Exact cost: \t", (0.5 * sol.T @ H @ sol + sol.T @ g)[0, 0])


def indefinite_trust_region():
    print("INDEFINITE WITH TRUST REGION ------------")

    H = np.array([[-1, 5], [5, -1]])
    g = np.array([[1], [1]])
    A = np.array([[1, 1]])
    x0 = np.array([[-0.1], [0.2]])
    c = A @ x0

    n_constraints = A.shape[0]
    n_vars = x0.shape[0]

    lhs = np.vstack(
        (np.hstack((H, A.T)), np.hstack((A, np.zeros((n_constraints, n_constraints)))))
    )
    rhs = np.vstack((-g, c))

    t0 = time.time()
    x = projected_cg(H, g, A, 0.5, x0)
    t1 = time.time()

    t2 = time.time()
    sol = LDLT(lhs).solve(rhs)[0:n_vars]
    t3 = time.time()

    print("CG time: \t", (t1 - t0) * 1000, "ms")
    print("Exact time: \t", (t3 - t2) * 1000, "ms")
    print("CG cost: \t", (0.5 * x.T @ H @ x + x.T @ g)[0, 0])
    print("Exact cost: \t", (0.5 * sol.T @ H @ sol + sol.T @ g)[0, 0])
    print("CG grad: \t", (H @ x + g)[0, 0])
    print("Exact grad: \t", (H @ sol + g)[0, 0])
    print("CG x: \n", x)
    print("Exact x: \n", sol)


positive_definite_no_trust_region()
# indefinite_trust_region()
