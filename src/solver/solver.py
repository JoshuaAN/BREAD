from sympy import *
import numpy as np
from math import *
from solver.projected_conjugate_gradient import projected_cg

def rows(mat):
    return mat.shape[0]

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
        # For simplicity of presentation we define
        # 
        #   z = [x]
        #       [s]
        # 
        #   φ(z) = f(x) - μlog(s)
        # 
        #   c(z) = [  cₑ(x)  ]
        #          [cᵢ(x) - s]
        # 
        # The barrier problem is rewritten as
        # 
        #          min φ(z)
        #   subject to c(z) = 0
        # 
        # The iterate step is defined as
        # 
        #   p = [  dˣ ]
        #       [S⁻¹dˢ]
        # 
        # Constraint Jacobian
        # 
        #   A = [Aₑ   0]
        #       [Aᵢ  −S]
        # 
        # Approximate the objective function and constraints as the quadratic
        # programming problem shown in equation 19.33 of [1].
        #
        #          min ½pᵀWp + pᵀΦ             (1a)
        #   subject to Ap + c = 0              (1b)
        #              ‖p‖ < Δ                 (1c)
        #              pˢ > −τe                (1d)
        #
        # An inexact solution to the subproblem is computed in two stages.
        # A normal step v is computed which attempts to minimize constraint
        # violation within the trust region.
        #
        #          min ‖Av + c‖                (2a)
        #   subject to ‖v‖ < ξΔ                (2b)
        #              vₛ > −ξτe               (2c)
        # 
        # The total step p is computed by solving a modified version of (1):
        # 
        #          min ½pᵀWp + pᵀΦ             (3a)
        #   subject to Ap = Av              (3b)
        #              ‖p‖ < Δ                 (3c)
        #              pˢ > −τe                (3d)
        #
        # The constraints (1d) and (2c) are equivelent to the "fraction to the
        # boundary" rule, and are applied by backtracking the solution vector.
        # 
        # https://link.springer.com/content/pdf/10.1007/PL00011391.pdf?pdf=button
        print("x rows: ", len(self.wrt))
        print("e rows: ", len(self.equality_constraints))
        print("i rows: ", len(self.inequality_constraints))

        yAD = []
        for i in range(len(self.equality_constraints)):
            yAD.append(self.private_variable())
        y = 0 * np.ones((len(yAD), 1))

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
        
        def merit_function(x, v):
            return f(x) + v * np.linalg.norm(equality(x))
        
        iteration = 0

        delta = 1.0

        mu = 1.0
        
        while True:
            c_e = equality(x)
            c_i = inequality(x)

            A_e = jacobian_equality(x)
            A_i = jacobian_inequality(x)

            g = gradient_f(x)

            # Compute lagrange multipliers to minimize
            # 
            #   |g - Aᵀy|₂
            #   gᵀg - 2yᵀAg + yᵀAAᵀy
            #   yᵀAAᵀy - 2yᵀAg
            # 
            #   AAᵀy = Ag
            y = np.linalg.solve(A_e @ A_e.T, A_e @ g)

            # End if first order optimality conditions are met
            if (max(
                np.linalg.norm(g - A_e.T @ y),
                np.linalg.norm(c_e)
            ) < 1e-8): break

            print("ITERATION ", iteration)

            H = hessian_L(x, y, z)

            # Solve trust region subproblem
            # 
            #          min ½pᵀHp + pᵀg
            #           p
            #   subject to Ap + c = 0
            #              |p|₂ < Δ

            # Solve with Powell's dog leg method. 
            # 
            #          min |Av + c|₂²
            #           v
            #   subject to |p|₂ < ηΔ
            # 
            # https://en.wikipedia.org/wiki/Powell%27s_dog_leg_method

            # Trust region scaling factor η
            eta = 0.8

            # Add regularization matrix to resolve rank deficiency in A.
            delta_gn = np.linalg.solve(1e-12 * np.identity(rows(x)) + A_e.T @ A_e, -A_e.T @ c_e)
            delta_sd = -A_e.T @ c_e
            t = np.linalg.norm(delta_sd, ord=2) / np.linalg.norm(A_e @ delta_sd, ord=2)
            v = None

            # If Gauss-Newton step is within the trust region, accept it.
            if np.linalg.norm(delta_gn) < eta * delta:
                v = delta_gn
            elif np.linalg.norm(t * delta_sd) > eta * delta:
                v = eta * delta * delta_sd / np.linalg.norm(delta_sd)
            else:
                # Dogleg step
                # 
                #   |t𝛿_sd + s(𝛿_gn - t𝛿_sd)|₂ = ηΔ
                #   |t𝛿_sd + s(𝛿_gn - t𝛿_sd)|₂² = (ηΔ)²
                #   (t𝛿_sd + s(𝛿_gn - t𝛿_sd))ᵀ(t𝛿_sd + s(𝛿_gn - t𝛿_sd)) = (ηΔ)²
                #   s²|𝛿_gn - t𝛿_sd|₂² + 2s(t𝛿_sd)ᵀ(𝛿_gn - t𝛿_sd) + |t𝛿_sd|₂² = (ηΔ)²
                # 
                # This is a quadratic function
                # 
                #   As² + Bs + C = 0
                #   A = |𝛿_gn - t𝛿_sd|₂²
                #   B = 2(t𝛿_sd)ᵀ(𝛿_gn - t𝛿_sd)
                #   C = |t𝛿_sd|₂² - (ηΔ)²
                A = np.linalg.norm(delta_gn - t * delta_sd, ord=2)
                B = 2 * ((t * delta_sd).T @ (delta_gn - t * delta_sd))[0, 0]
                C = np.linalg.norm(t * delta_sd, ord=2) - (eta * delta) ** 2
                s = (-B + sqrt(B * B - 4 * A * C)) / (2 * A)
                v = delta_sd + s * (delta_gn - t * delta_sd)

            p_x = projected_cg(H, g, A_e, delta, v)

            # Update penalty parameter
            # 
            # TODO: Add math docs... this all seems kinda sketch
            rho = 0.1
            mu_lhs = -(g.T @ p_x + 0.5 * p_x.T @ H @ p_x)[0, 0]
            mu_rhs = (rho - 1.0) * (np.linalg.norm(c_e) - np.linalg.norm(A_e @ p_x + c_e))
            mu = max(mu, mu_lhs / mu_rhs)
            
            ared = merit_function(x, mu) - merit_function(x + p_x, mu)
            pred = -(g.T @ p_x + 0.5 * p_x.T @ H @ p_x - mu * (np.linalg.norm(c_e) - np.linalg.norm(A_e @ p_x + c_e)))[0, 0]

            # Accept or reject step and update trust region size.
            psi = 1e-8
            reduction = ared / pred
            if (reduction < 0.25):
                delta *= 0.25
            else:
                if reduction > 0.75 and abs(np.linalg.norm(p_x) - delta) < 1e-12:
                    delta *= 2.0
            if (ared > psi * pred):
                x += p_x

            print("x: \n", x)

            iteration += 1