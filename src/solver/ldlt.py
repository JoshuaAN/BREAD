import numpy as np
from scipy.linalg import ldl, solve_triangular

class LDLT:
    def __init__(self, mat):
        self.LT, self.D, self.perm = ldl(mat)
    
    def solve(self, rhs):
        # Ax = b
        # LDLᵀx = b
        #
        # Ly = b
        # Dz = y
        # Lᵀx = z
        y = solve_triangular(self.LT, rhs, lower=True)
        z = np.zeros((rhs.shape[0], 1))
        for row in range(rhs.shape[0]):
            z[row, 0] = y[row, 0] / self.D[row, row]
        x = solve_triangular(self.LT.T, z, lower=False)

        return x