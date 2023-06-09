from sympy import *
import numpy as np
from math import *
from solver.projected_conjugate_gradient import projected_cg
from solver.dogleg_method import dogleg

def rows(mat):
    return mat.shape[0]

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
        s = np.ones((len(self.inequality_constraints), 1))
        e = np.ones((len(self.inequality_constraints), 1))

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

        # Barrier parameter μ
        mu = 0.00000001

        # Trust region size Δ
        delta = 1.0

        # Trust region scaling factor ξ
        xi = 0.8

        # Trust region step acceptance parameter η
        eta = 1e-8

        # Penalty parameter v
        v = 1.0

        # 
        tau = 0.995

        for ite in range(50):
            print("ITERATION ", iteration)

            c_e = equality(x)
            c_i = inequality(x)

            A_e = jacobian_equality(x)
            A_i = jacobian_inequality(x)

            # A = [Aₑ  0]
            #     [Aᵢ -S]
            A = np.vstack((
                np.hstack((A_e, np.zeros((A_e.shape[0], A_i.shape[0])))),
                np.hstack((A_i, -dia(s)))
            ))

            g = gradient_f(x)

            # ϕ = [ ∇f]
            #     [-μe]
            phi = np.vstack((g, -mu * e))

            # AAᵀ[y] = Aϕ
            #    [z]
            multipliers = np.linalg.solve(A @ A.T, A @ phi)
            y = multipliers[0:rows(c_e)]
            z = multipliers[rows(c_e):(rows(c_e) + rows(c_i))]
            
            # The multiplier estimates z obtained in this manner may not
            # always be positive; to enforce positivity, we may redefine them as
            # 
            #   zᵢ = min(10⁻³, μ/sᵢ)
            for row in range(rows(z)):
                if (z[row, 0] <= 0):
                    z[row, 0] = min(10e-3, mu / s[row, 0])

            # End if first order optimality conditions are met
            # if (max(
            #     np.linalg.norm(),
            #     np.linalg.norm(c_e)
            # ) < 1e-8): break

            H = hessian_L(x, y, z)

            # c = [  cₑ  ]
            #     [cᵢ - s]
            c = np.vstack((c_e, c_i - s))

            print(c)

            # Solve (2)
            dogleg_step = dogleg(A, c, xi * delta)
            v_s = dogleg_step[rows(x):(rows(x) + rows(s))]
            dogleg_step *= fraction_to_boundary(v_s, xi * tau)

            print("dogleg norm: ", np.linalg.norm(dogleg_step))

            # W = [H   0]
            #     [0  SZ]
            W = np.vstack((
                np.hstack((H, np.zeros((rows(x), rows(s))))),
                np.hstack((np.zeros((rows(s), rows(x))), dia(s) @ dia(z)))
            ))

            # Solve (3)
            p = projected_cg(W, phi, A, delta, dogleg_step)
            p_s = p[rows(x):(rows(x) + rows(s))]

            p *= fraction_to_boundary(p_s, tau)

            p_x = p[0:rows(x)]
            p_s = p[rows(x):(rows(x) + rows(s))]

            # Update penalty parameter
            # 
            # TODO: Add math docs... this all seems kinda sketch
            rho = 0.1
            v_lhs = -(phi.T @ p + 0.5 * p.T @ W @ p)[0, 0]
            v_rhs = (rho - 1.0) * (np.linalg.norm(c) - np.linalg.norm(A @ p + c))
            v = max(v, v_lhs / v_rhs)
            
            ared = merit_function(x, s, v, mu) - merit_function(x + p_x, s + dia(s) @ p_s, v, mu)
            pred = -(phi.T @ p + 0.5 * p.T @ W @ p - v * (np.linalg.norm(c) - np.linalg.norm(A @ p + c)))[0, 0]

            # print("ared: ", ared)
            # print("pred: ", pred)

            print("delta: ", delta)

            # Accept or reject step and update trust region size.
            reduction = ared / pred
            print(reduction)
            if (reduction < 0.25):
                delta *= 0.25
            else:
                if reduction > 0.75 and abs(np.linalg.norm(p) - delta) < 1e-12:
                    delta *= 2.0
            if (ared > eta * pred):
                x += p_x
                s += dia(s) @ p_s

            # Diagnostics
            print("x: \n", x)
            print("s: \n", s)

            iteration += 1