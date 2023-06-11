from sympy import *
import numpy as np
from math import *
from solver.qp_solver import solve_qp

def rows(mat):
    return mat.shape[0]

def infinity_norm(vec):
    norm = 0
    for row in range(vec.shape[0]):
        norm = max(norm, abs(vec[row, 0]))
    return norm

def fraction_to_boundary(p_x, tau):
    alpha = 1.0
    for row in range(rows(p_x)):
        if (p_x[row, 0] < -tau):
            alpha = min(alpha, -tau / p_x[row, 0])
    return alpha

def dia(vec):
    mat = np.identity(rows(vec))
    for row in range(rows(vec)):
        mat[row, row] = vec[row, 0]
    return mat

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
        # Solves problem of the form
        # 
        #          min f(x)
        #   subject to cₑ(x) = 0
        #              cᵢ(x) ≥ 0
        # 
        # The interior point method is used to transform the inequality 
        # constraints into equality constraints.
        # 
        #          min f(x) - μlog(s)
        #   subject to cₑ(x) = 0
        #              cᵢ(x) - s = 0
        # 
        # The iterate step is defined as
        # 
        #   p = [pₓ] = [  dˣ ]
        #       [pₛ]   [S⁻¹dˢ]
        # 
        #   L(x, s, y, z) = φ(z) − λᵀc(z)
        # 
        # Approximate the objective function and constraints as the quadratic
        # programming problem shown in equation 19.33 of [1].
        #
        #          min ½pᵀWp + pᵀΦ                             (1a)
        #   subject to Aₑvₓ + cₑ = 0                           (1b)
        #              Aᵢv - Svₛ + cᵢ - s = 0                  (1c)
        #              ‖p‖ < Δ                                 (1d)
        #              pˢ > −τe                                (1e)
        #
        # An inexact solution to the subproblem is computed in two stages.
        # A normal step v is computed which attempts to minimize constraint
        # violation within the trust region.
        #
        #          min ‖Aₑvₓ + cₑ‖² + ‖Aᵢv - Svₛ + cᵢ - s‖²    (2a)
        #   subject to ‖vₓ, vₛ‖ < ξΔ                           (2b)
        #              vₛ > −ξτe                               (2c)
        # 
        # The total step p is computed by solving a modified version of (1):
        # 
        #          min ½pᵀWp + pᵀΦ                             (3a)
        #   subject to Aₑpₓ = Aₑvₓ                             (3b)
        #              Aᵢpₓ - Spₛ = Aᵢvₓ - Svₛ                 (3c)
        #              ‖pₓ, pₛ‖ < Δ                            (3d)
        #              pˢ > −τe                                (3e)
        #
        # The constraints (1e), (2c), and (3e) are equivalent to the "fraction to the
        # boundary" rule, and are applied by backtracking the solution vector.
        # 
        # https://link.springer.com/content/pdf/10.1007/PL00011391.pdf?pdf=button
        yAD = []
        for i in range(len(self.equality_constraints)):
            yAD.append(self.private_variable())

        zAD = []
        for i in range(len(self.inequality_constraints)):
            zAD.append(self.private_variable())

        L = self.f
        for multiplier, constraint in zip(yAD, self.equality_constraints):
            L -= multiplier * constraint
        for multiplier, constraint in zip(zAD, self.inequality_constraints):
            L -= multiplier * constraint

        x = self.dict_to_matrix(dict).astype('float64')
        y = np.ones((len(self.equality_constraints), 1))
        z = np.ones((len(self.inequality_constraints), 1))

        def f(x):
            return self.f.evalf(subs=self.matrix_to_dict(self.wrt, x))

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
        
        def merit_function(x, s, v, mu):
            merit = f(x) + v * hypot(np.linalg.norm(equality(x)), np.linalg.norm(inequality(x) - s))
            for row in range(rows(s)):
                merit -= mu * ln(s[row, 0])
            return merit
        
        iteration = 0

        while True:
            print("ITERATION ", iteration)

            c_e = equality(x)
            c_i = inequality(x)

            A_e = jacobian_equality(x)
            A_i = jacobian_inequality(x)

            g = gradient_f(x)
            H = hessian_L(x, y, z)

            # Convergence test
            inequality_error = 0
            for row in range(len(self.inequality_constraints)):
                inequality_error = max(inequality_error, -c_i[row, 0])
            E = max(
                infinity_norm(g - A_e.T @ y - A_i.T @ z),
                infinity_norm(c_e),
                inequality_error
            )
            if (E < 1e-8):
                break

            # Regularize Hessian if not positive definite.
            beta = 1e-3
            tau = None
            min_dia = min(np.diag(H))
            B = None
            if (min_dia > 0):
                tau = 0
            else:
                tau = -min_dia + beta
            while True:
                B = H + tau * np.identity(rows(x))
                if (min(np.linalg.eig(B)[0]) > 0):
                    break
                else:
                    tau = max(10 * tau, beta)

            p_x, y, z = solve_qp(B, g, A_e, A_i, c_e, c_i, np.zeros((rows(x), 1)))

            x += p_x

            # Diagnostics
            print("x: \n", x)

            iteration += 1