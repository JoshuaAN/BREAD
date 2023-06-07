import numpy as np
from src.solver.ldlt import LDLT

# LDLT solver and numpy.linalg.solve are returning different results

A = np.array([[1., 0., 1.],
              [0., 1., 3.],
              [1., 3., 0.]])

b = np.array([[0.        ],
              [9.02023858],
              [0.        ]])

ldlt = LDLT(A)

print("LDLT: \n", ldlt.solve(b))
print("np: \n", np.linalg.solve(A, b))