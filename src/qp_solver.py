import numpy as np
import time

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

# Solve quadratic programming problem
# 
#          min ½xᵀHx + xᵀg
#   subject to Aₑx + cₑ = 0
#              Aᵢx + cᵢ ≥ 0
def solve_qp(H, g, A_e, A_i, c_e, c_i, initial_guess):
    x = initial_guess
    s = np.ones((rows(c_i), 1))
    y = np.ones((rows(c_e), 1))
    z = np.ones((rows(c_i), 1))

    e = np.ones((rows(c_i), 1))

    tau = 0.995
    
    error = float('inf')

    # print("Lagrangian       Infeasibility        Error")

    while error > 1e-6:
        S = diag(s)
        Z = diag(z)

        mu = max((s.T @ z)[0, 0] / rows(s), 1e-9)

        # [H     0    -Aₑᵀ  -Aᵢᵀ]
        # [Aₑ    0     0     0  ]
        # [Aᵢ   -I     0     0  ]
        # [0     Z     0     S  ]
        lhs = np.vstack((
            np.hstack((H, np.zeros((rows(x), rows(s))), -np.transpose(A_e), -np.transpose(A_i))),
            np.hstack((A_e, np.zeros((rows(y), rows(s))), np.zeros((rows(y), rows(y))), np.zeros((rows(y), rows(s))))),
            np.hstack((A_i, -np.identity(rows(s)), np.zeros((rows(s), rows(y))), np.zeros((rows(s), rows(z))))),
            np.hstack((np.zeros((rows(s), rows(x))), Z, np.zeros((rows(s), rows(y))), S))
        ))

        # [Hx + g - Aₑᵀy - Aᵢᵀz]
        # [      Aₑx + cₑ      ]
        # [    Aᵢx + cᵢ - s    ]
        # [         SZe        ]
        rhs = -np.vstack((
            H @ x + g - A_e.T @ y - A_i.T @ z,
            A_e @ x + c_e,
            A_i @ x + c_i - s,
            S @ Z @ e
        ))

        # Affine step
        affine_step = np.linalg.solve(lhs, rhs)

        # Compute sigma from affine step
        p_s_affine = affine_step[len(x):(len(x) + len(s))]
        p_z_affine = affine_step[(len(x) + len(s) + len(y)):(len(x) + len(s) + len(y) + len(z))]
        s_affine = s + fraction_to_boundary(s, p_s_affine, tau) * p_s_affine
        z_affine = z + fraction_to_boundary(z, p_z_affine, tau) * p_z_affine

        mu_affine = (s_affine.T @ z_affine)[0, 0] / rows(s)

        # sigma = (mu_affine / mu) ** 3
        sigma = 0.1

        # [Hx + g - Aₑᵀy - Aᵢᵀz]
        # [      Aₑx + cₑ      ]
        # [    Aᵢx + cᵢ - s    ]
        # [      SZe - σμe     ]
        rhs = -np.vstack((
            H @ x + g - A_e.T @ y - A_i.T @ z,
            A_e @ x + c_e,
            A_i @ x + c_i - s,
            S @ Z @ e - mu * sigma * e
        ))

        step = np.linalg.solve(lhs, rhs)

        p_x = step[0:len(x)]
        p_s = step[len(x):(len(x) + len(s))]
        p_y = step[(len(x) + len(s)):(len(x) + len(s) + len(y))]
        p_z = step[(len(x) + len(s) + len(y)):(len(x) + len(s) + len(y) + len(z))]

        error = max(
            infinity_norm(H @ x + g - A_e.T @ y - A_i.T @ z),
            infinity_norm(S @ Z @ e - mu * sigma * e),
            infinity_norm(A_e @ x + c_e),
            infinity_norm(A_i @ x + c_i - s)
        )

        print("----------------------")
        # print(error)
        # print(A_e)
        # print((c_e))
        # print(x)
        print(infinity_norm(lhs @ step - rhs))
        print(np.linalg.matrix_rank(lhs))
        print(lhs.shape)
        print(np.linalg.matrix_rank(A_e))
        print(A_e.shape)

        # Diagnostics
        # print(" {}           {}         {}".format(
        #     str(round(np.linalg.norm(H @ x + g - A_e.T @ y - A_i.T @ z), 3)).center(8, " "),
        #     str(round(np.linalg.norm(A_e @ x + c_e), 3)).center(8, " "),
        #     str(round((s.T @ z)[0, 0], 3)).center(8, " ")))

        # Apply fraction to boundary rule to avoid overshooting bounding variables.
        alpha_primal = fraction_to_boundary(s, p_s, tau)
        alpha_dual = fraction_to_boundary(z, p_z, tau)

        x += alpha_primal * p_x
        s += alpha_primal * p_s
        y += alpha_dual * p_y
        z += alpha_dual * p_z

        time.sleep(1)

    return x, y, z

# Solve quadratic programming problem
# 
#          min ½xᵀHx + xᵀg
#   subject to Aₑx + cₑ = 0
#              Aᵢx + cᵢ ≥ 0
def solve_qp_lagrangian(H, g, A_e, A_i, c_e, c_i, initial_guess):
    # Hx + g - y(Aₑᵀ) - z()
    # Aₑx + cₑ = 0
    # Aᵢx + cᵢ - sᵀs = 0

    x = initial_guess
    s = np.ones((rows(c_i), 1))
    y = np.ones((rows(c_e), 1))
    z = np.ones((rows(c_i), 1))

    e = np.ones((rows(c_i), 1))

    tau = 0.995
    
    error = float('inf')

    print("Lagrangian       Infeasibility        Error")

    while error > 1e-6:
        S = diag(s)
        Z = diag(z)

        mu = max((s.T @ z)[0, 0] / rows(s), 1e-9)

        # [H     0    -Aₑᵀ  -Aᵢᵀ]
        # [Aₑ    0     0     0  ]
        # [Aᵢ   -I     0     0  ]
        # [0     Z     0     S  ]
        lhs = np.vstack((
            np.hstack((H, np.zeros((rows(x), rows(s))), -np.transpose(A_e), -np.transpose(A_i))),
            np.hstack((A_e, np.zeros((rows(y), rows(s))), np.zeros((rows(y), rows(y))), np.zeros((rows(y), rows(s))))),
            np.hstack((A_i, -np.identity(rows(s)), np.zeros((rows(s), rows(y))), np.zeros((rows(s), rows(z))))),
            np.hstack((np.zeros((rows(s), rows(x))), Z, np.zeros((rows(s), rows(y))), S))
        ))

        # [Hx + g - Aₑᵀy - Aᵢᵀz]
        # [      Aₑx + cₑ      ]
        # [    Aᵢx + cᵢ - s    ]
        # [         SZe        ]
        rhs = -np.vstack((
            H @ x + g - A_e.T @ y - A_i.T @ z,
            A_e @ x + c_e,
            A_i @ x + c_i - s,
            S @ Z @ e
        ))
        
        step = np.linalg.solve(lhs, rhs)

        p_x = step[0:len(x)]
        p_s = step[len(x):(len(x) + len(s))]
        p_y = step[(len(x) + len(s)):(len(x) + len(s) + len(y))]
        p_z = step[(len(x) + len(s) + len(y)):(len(x) + len(s) + len(y) + len(z))]

        error = max(
            infinity_norm(H @ x + g - A_e.T @ y - A_i.T @ z),
            infinity_norm(S @ Z @ e - mu * sigma * e),
            infinity_norm(A_e @ x + c_e),
            infinity_norm(A_i @ x + c_i - s)
        )

        print("----------------------")
        # print(error)
        # print(A_e)
        # print((c_e))
        # print(x)
        print(infinity_norm(lhs @ step - rhs))
        print(np.linalg.matrix_rank(lhs))
        print(lhs.shape)
        print(np.linalg.matrix_rank(A_e))
        print(A_e.shape)

        # Diagnostics
        # print(" {}           {}         {}".format(
        #     str(round(np.linalg.norm(H @ x + g - A_e.T @ y - A_i.T @ z), 3)).center(8, " "),
        #     str(round(np.linalg.norm(A_e @ x + c_e), 3)).center(8, " "),
        #     str(round((s.T @ z)[0, 0], 3)).center(8, " ")))

        # Apply fraction to boundary rule to avoid overshooting bounding variables.
        alpha_primal = fraction_to_boundary(s, p_s, tau)
        alpha_dual = fraction_to_boundary(z, p_z, tau)

        x += alpha_primal * p_x
        s += alpha_primal * p_s
        y += alpha_dual * p_y
        z += alpha_dual * p_z

        time.sleep(1)

    return x, y, z
