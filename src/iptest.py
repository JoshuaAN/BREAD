import numpy as np
from solver.qp_solver import solve_qp

H = np.array([[3, 1], [1, 3]])
g = np.array([[1], [1]])
A_e = np.array([[3, 1]])
c_e = np.array([[1]])
A_i = np.array([[1, 0]])
c_i = np.array([[2]])

print(solve_qp(H, g, A_e, A_i, c_e, c_i, np.ones((2, 1)))[0])