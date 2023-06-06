from solver.projected_conjugate_gradient import projected_cg
import numpy as np
import time
from solver.ldlt import LDLT
import cProfile
from pstats import Stats, SortKey

N = 1000

H = np.random.rand(N, N)
H = H.T @ H + np.identity(N)
g = np.random.rand(N, 1)
A = np.random.rand(500, N)
x0 = np.random.rand(N, 1)
c = A @ x0

n_constraints = A.shape[0]
n_vars = x0.shape[0]

lhs = np.vstack((
    np.hstack((H, A.T)), 
    np.hstack((A, np.zeros((n_constraints, n_constraints))))
))
rhs = np.vstack((-g, c))

t0 = time.time()
x = projected_cg(H, g, A, 1, x0)
t1 = time.time()
print("CG time: ", (t1 - t0) * 1000, "ms")

t2 = time.time()
sol = np.linalg.solve(lhs, rhs)[0:n_vars]
t3 = time.time()
print("Exact time: ", (t3 - t2) * 1000, "ms")

print("CG cost: ", 0.5 * x.T @ H @ x + x.T @ g)
print("Exact cost: ", 0.5 * sol.T @ H @ sol + sol.T @ g)