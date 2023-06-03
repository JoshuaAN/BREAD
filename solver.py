from sympy import *
import numpy as np
from qp_solver import *
import scipy

class Solver:
    def __init__(self):
        self.wrt = []
        self.f = None
        self.equality_constraints = []
        self.inequality_constraints = []
        self.varCount = 0
    
    def private_variable(self):
        self.varCount += 1
        return symbols(str(self.varCount))

    def variable(self):
        self.wrt.append(self.private_variable())
        return self.wrt[-1]
    
    def minimize(self, cost):
        self.f = cost

    def subject_to_equality(self, constraint):
        self.equality_constraints.append(constraint)

    def subject_to_inequality(self, constraint):
        self.inequality_constraints.append(constraint)

    def dict_to_matrix(self, dict):
        vals = []
        for var in self.wrt:
            vals.append(dict[var])
        return np.matrix(vals).reshape(-1, 1)
    
    def matrix_to_dict(self, keys, matrix):
        dict = {}
        for row in range(len(keys)):
            val = matrix[row, 0]
            dict[keys[row]] = val
        return dict

    def solve(self, dict):
        print("x rows: ", len(self.wrt))
        print("e rows: ", len(self.equality_constraints))
        print("i rows: ", len(self.inequality_constraints))

        yAD = []
        for i in range(len(self.equality_constraints)):
            yAD.append(self.private_variable())
        y = np.ones((len(yAD), 1))

        zAD = []
        for i in range(len(self.inequality_constraints)):
            zAD.append(self.private_variable())
        z = np.ones((len(zAD), 1))

        L = self.f
        for multiplier, constraint in zip(yAD, self.equality_constraints):
            L -= multiplier * constraint
        for multiplier, constraint in zip(zAD, self.inequality_constraints):
            L -= multiplier * constraint

        x = self.dict_to_matrix(dict).astype('float64')

        def gradient_f(x):
            vars = self.matrix_to_dict(self.wrt, x)
            g = np.matrix([self.f.diff(var).evalf(subs=vars) for var in self.wrt])
            return g.reshape(-1, 1).astype('float64') # Reshape and convert to float64
        
        def hessian_L(x, y, z):
            vars = self.matrix_to_dict(self.wrt, x) | self.matrix_to_dict(yAD, y) | self.matrix_to_dict(zAD, z)
            return np.array(hessian(L, self.wrt).evalf(subs=vars)).astype('float64')
        
        def equality(x):
            if len(self.equality_constraints) == 0:
                return None
            vars = self.matrix_to_dict(self.wrt, x)
            return np.array(Matrix(self.equality_constraints).evalf(subs=vars)).astype('float64')

        def inequality(x):
            if len(self.inequality_constraints) == 0:
                return None
            vars = self.matrix_to_dict(self.wrt, x)
            return np.array(Matrix(self.inequality_constraints).evalf(subs=vars)).astype('float64')

        def jacobian_equality(x):
            if len(self.equality_constraints) == 0:
                return None
            vars = self.matrix_to_dict(self.wrt, x)
            symbolic_jacobian = Matrix(self.equality_constraints).jacobian(self.wrt)
            return np.array(symbolic_jacobian.evalf(subs=vars)).astype('float64')

        def jacobian_inequality(x):
            if len(self.inequality_constraints) == 0:
                return None
            vars = self.matrix_to_dict(self.wrt, x)
            symbolic_jacobian = Matrix(self.inequality_constraints).jacobian(self.wrt)
            return np.array(symbolic_jacobian.evalf(subs=vars)).astype('float64')
        
        iteration = 0
        
        while True:
            c_e = equality(x)
            c_i = inequality(x)

            A_e = jacobian_equality(x)
            A_i = jacobian_inequality(x)

            g = gradient_f(x)
            H = hessian_L(x, y, z)

            # Convergence test
            if (
              max(
                infinity_norm(g - A_e.T @ y - A_i.T @ z),
                infinity_norm(c_e)
              ) < 1e-6):
                break

            # Regularize Hessian if not positive definite.
            beta = 1e-3
            tau = None
            min_dia = min(np.diag(H))
            if (min_dia > 0):
                tau = 0
            else:
                tau = -min_dia + beta
            while True:
                B = H + tau * np.identity(rows(x))
                if (min(np.linalg.eig(B)[0]) > 0):
                    break
                else:
                    print(tau)
                    tau = max(10 * tau, beta)
            H = B

            # Solve quadratic subproblem
            # 
            #        min ½xᵀHx + xᵀg
            # subject to Aₑx + cₑ = 0
            #            Aᵢx + cᵢ > 0

            # print(A_e.shape)
            # print("----------------------------------------")
            # print(A_e @ np.linalg.lstsq(A_e, np.zeros((16, 1)))[0])
            # print("----------------------------------------")
            
            p, y, z, = solve_qp(H, g, A_e, A_i, c_e, c_i, np.zeros((rows(x), 1)))

            x += p

            print("ITERATION ", iteration)
            print("x: \n", x)
            # print("H: \n", hessian_L(x, y, z))
            # print("Eigenvalues: \n", np.linalg.eig(hessian_L(x, y, z))[0])

            iteration += 1