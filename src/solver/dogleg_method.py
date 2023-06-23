import math

import numpy as np


def squaredNorm(vec):
    return np.linalg.norm(vec) * np.linalg.norm(vec)


# Solves least squares problem subject to trust region constraint
#
#          min ‖Ax + b‖
#   subject to ‖x‖ < Δ
#
# https://en.wikipedia.org/wiki/Powell%27s_dog_leg_method
def dogleg(A, b, delta):
    # Add small multiple of the identity to ensure the existence of a solution.
    delta_gn = np.linalg.solve(1e-12 * np.identity(A.shape[1]) + A.T @ A, -A.T @ b)
    delta_sd = -A.T @ b
    t = squaredNorm(delta_sd) / squaredNorm(A @ delta_sd)

    # If Gauss-Newton step is within the trust region, accept it.
    if np.linalg.norm(delta_gn) < delta:
        return delta_gn
    elif np.linalg.norm(t * delta_sd) > delta:
        print("steepest descent")
        return delta * delta_sd / np.linalg.norm(delta_sd)
    else:
        # Dogleg step
        #
        #   ‖t𝛿_sd + s(𝛿_gn - t𝛿_sd)‖₂ = Δ
        #   ‖t𝛿_sd + s(𝛿_gn - t𝛿_sd)‖₂² = Δ²
        #   (t𝛿_sd + s(𝛿_gn - t𝛿_sd))ᵀ(t𝛿_sd + s(𝛿_gn - t𝛿_sd)) = Δ²
        #   s²‖𝛿_gn - t𝛿_sd‖₂² + 2s(t𝛿_sd)ᵀ(𝛿_gn - t𝛿_sd) + ‖t𝛿_sd‖₂² = Δ²
        #
        # This is a quadratic function
        #
        #   As² + Bs + C = 0
        #   A = ‖𝛿_gn - t𝛿_sd‖₂²
        #   B = 2(t𝛿_sd)ᵀ(𝛿_gn - t𝛿_sd)
        #   C = ‖t𝛿_sd‖₂² - Δ²
        _A = squaredNorm(delta_gn - t * delta_sd)
        _B = 2 * ((t * delta_sd).T @ (delta_gn - t * delta_sd))[0, 0]
        _C = squaredNorm(t * delta_sd) - delta * delta
        s = (-_B + math.sqrt(_B * _B - 4 * _A * _C)) / (2 * _A)
        x = t * delta_sd + s * (delta_gn - t * delta_sd)
        return x
