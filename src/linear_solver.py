import numpy as np

def rows(mat):
    return mat.shape[0]

def diag(vector):
    mat = np.zeros((rows(vector), rows(vector)))
    for row in range(rows(vector)):
        mat[row, row] = vector[row][0]
    return mat

def infinity_norm(vec):
    norm = 0
    for row in range(rows(vec)):
        norm = max(norm, abs(vec[row, 0]))
    return norm

def fraction_to_boundary(v, p_v, tau):
    alpha = 1.0
    for row in range(rows(v)):
        delta = p_v[row, 0]
        value = v[row, 0]
        if (delta > 0):
            alpha = min(alpha, tau * value / delta)
        elif (delta < 0):
            alpha = min(alpha, -tau * value / delta)
    return alpha

# Solve problem of the form
# 
#          min cᵀx 
#           x
#   subject to Aᵢx + cᵢ > 0
def solve_linear_problem(c, A, b, initial_guess):
    # Add slack variables
    # 
    #          min cᵀx 
    #           x
    #   subject to Ax + b - s = 0
    #              s > 0
    # 
    # Peturbed KKT conditions
    # 
    #   c - Aᵀz = 0
    #   SZe - σμe = 0
    #   Ax + b - s = 0
    # 
    # Apply Newton-Rhapson method to the system
    # 
    #   [     c - Aᵀz    ]
    #   [    SZe - σμe   ] = 0
    #   [    Ax + b - s  ]
    # 
    #   [0    0   -Aᵀ][pˣ]    [  c - Aᵀz ]
    #   [0    Z    S ][pˢ] = -[ SZe - μe ]
    #   [A   -I    0 ][pᶻ]    [Ax + b - s]
    # 
    # Take second equation
    #   
    #   Zpˢ + Spᶻ = -SZe + σμe
    #   Zpˢ = -Spᶻ - SZe + σμe
    #   pˢ = Z⁻¹(-Spᶻ + σμe) - s
    # 
    # Substitute into third row
    # 
    #   Apˣ - pˢ = -Ax - b + s
    #   Apˣ - (Z⁻¹(-Spᶻ + σμe) - s) = -Ax - b + s
    #   Apˣ - Z⁻¹(-Spᶻ + σμe) = -Ax - b
    #   Apˣ + SZ⁻¹pᶻ = -Ax - b + σμZ⁻¹e
    # 
    # Substitute new second and third rows into system
    # 
    #   [0    0   -Aᵀ ][pˣ]    [      c - Aᵀz      ]
    #   [0    I    0  ][pˢ] = -[Z⁻¹(-Spᶻ + σμe) - s]
    #   [A    0   SZ⁻¹][pᶻ]    [  Ax + b - σμZ⁻¹e  ]
    # 
    # Eliminate the second row and column
    # 
    #   [0   -Aᵀ ][pˣ] = -[    c - Aᵀz    ]
    #   [A   SZ⁻¹][pᶻ]    [Ax + b - σμZ⁻¹e]
    # 
    # Take second equation
    # 
    #   Apˣ + SZ⁻¹pᶻ = -Ax - b + σμZ⁻¹e
    #   SZ⁻¹pᶻ = -Ax - Apˣ - b + σμZ⁻¹e
    #   pᶻ = -S⁻¹ZAx - S⁻¹ZApˣ - S⁻¹Zb + σμS⁻¹e
    #   pᶻ = -S⁻¹(ZAx + ZApˣ + Zb - σμe)
    # 
    # Substitute into first row
    # 
    #   -Aᵀpᶻ = -c + Aᵀz
    #   AᵀS⁻¹(ZAx + ZApˣ + Zb - σμe) = -c + Aᵀz
    #   AᵀS⁻¹ZApˣ = -c + Aᵀ(S⁻¹(ZAx + Zb - σμe) - z)
    # 
    # Substitute into reduced system
    # 
    #   [AᵀS⁻¹ZApˣ  0][pˣ] = -[c + Aᵀ(S⁻¹(ZAx + Zb - σμe) - z)]
    #   [    0      I][pᶻ]    [   S⁻¹(ZAx + ZApˣ + Zb - σμe)  ]
    # 
    # Eliminate second row and column
    # 
    #   AᵀS⁻¹ZApˣ = -c - Aᵀ(S⁻¹(ZAx + Zb - σμe) - z)
    # 
    # The reduced system gives the iterate pˣ with the iterates pᶻ and pˢ given by
    # 
    #   pᶻ = -S⁻¹(ZA(x + pˣ) + Zb - σμe)
    #   pˢ = Z⁻¹(-Spᶻ + σμe) - s
    #   
    tau = 0.995

    x = initial_guess
    s = np.ones((rows(A), 1))
    z = np.ones((rows(A), 1))

    e = np.ones((rows(A), 1))

    print("Lagrangian       Infeasibility        Error")

    while True:
        S = diag(s)
        Z = diag(z)

        inverse_S = diag(np.reciprocal(s))
        inverse_Z = diag(np.reciprocal(z))

        # Diagnostics
        print(" {}           {}         {}".format(
            str(round(np.linalg.norm(c - A.T @ z), 5)).center(8, " "),
            str(round(np.linalg.norm(A @ x + b), 5)).center(8, " "),
            str(round((s.T @ z)[0, 0], 5)).center(8, " ")))

        # Affine step with σ = 0
        p_x = np.linalg.solve(A.T @ inverse_S @ Z @ A, 
                              -c - A.T @ (inverse_S @ (Z @ A @ x + Z @ b) - z))
        p_z = -inverse_S @ (Z @ A @ (x + p_x) + Z @ b)
        p_s = -inverse_Z @ S @ p_z - s
        
        alpha_primal = fraction_to_boundary(s, p_s, tau)
        alpha_dual = fraction_to_boundary(z, p_z, tau)

        mu = (s.T @ z / rows(s))[0, 0]
        mu_affine = ((s + alpha_primal * p_s).T @ (z + alpha_dual * p_z) / rows(s))[0, 0]

        sigma = ((mu_affine / mu) ** 3)
        print(sigma)

        # sigma = 0.1
        # mu = 1.0

        error = max(
            infinity_norm(c - A.T @ z),
            infinity_norm(S @ Z @ e - mu * sigma * e)
            # infinity_norm(A @ x + b - s)
        )

        # print(c - A.T @ z)
        # print(S @ Z @ e - mu * sigma * e)
        # print(x[0][0])

        if (error < 1e-6):
            break
        
        # Normal step
        p_x = np.linalg.solve(A.T @ inverse_S @ Z @ A, 
                              -c - A.T @ (inverse_S @ (Z @ (A @ x + b) - sigma * mu * e) - z)) 
        p_z = -inverse_S @ (Z @ A @ (x + p_x) + Z @ b - sigma * mu * e)
        p_s = inverse_Z @ (-S @ p_z + sigma * mu * e) - s

        alpha_primal = fraction_to_boundary(s, p_s, tau)
        alpha_dual = fraction_to_boundary(z, p_z, tau)

        x += alpha_primal * p_x
        s += alpha_primal * p_s
        z += alpha_dual * p_z

    print(x)
        

c = np.array([[-1]])
A = np.array([[1], [-1]])
b = np.array([[5], [5]])

solve_linear_problem(c, A, b, np.array([[0.0]]))