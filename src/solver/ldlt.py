import numpy as np
from scipy.linalg import ldl, solve_triangular

class LDLT:
    def __init__(self, mat):
        self.L, self.D, self.perm = ldl(mat)
        self.L = self.L[self.perm,:]
        self.perm_inv = list(range(len(self.perm)))
        for row in range(len(self.perm)):
            self.perm_inv[self.perm[row]] = row
    
    def solve(self, rhs):
        P = np.identity(len(self.perm))[self.perm,:]

        # Ax = b
        # LDLᵀx = b
        #
        # Ly = b
        # Dz = y
        # Lᵀx = z
        y = solve_triangular(self.L, rhs[self.perm_inv,:], lower=True)
        z = np.zeros((rhs.shape[0], 1))
        for row in range(rhs.shape[0]):
            z[row, 0] = y[row, 0] / self.D[row, row]
        x = solve_triangular(self.L.T, z, lower=False)

        return x[self.perm,:]