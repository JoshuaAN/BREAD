import numpy as np
from math import *

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
    t = np.linalg.norm(delta_sd, ord=2) / np.linalg.norm(A @ delta_sd, ord=2)

    # If Gauss-Newton step is within the trust region, accept it.
    if np.linalg.norm(delta_gn) < delta:
        return delta_gn
    elif np.linalg.norm(t * delta_sd) > delta:
        return delta * delta_sd / np.linalg.norm(delta_sd)
    else:
        # Dogleg step
        # 
        #   ‖t𝛿_sd + s(𝛿_gn - t𝛿_sd)‖₂ = ηΔ
        #   ‖t𝛿_sd + s(𝛿_gn - t𝛿_sd)‖₂² = (ηΔ)²
        #   (t𝛿_sd + s(𝛿_gn - t𝛿_sd))ᵀ(t𝛿_sd + s(𝛿_gn - t𝛿_sd)) = (ηΔ)²
        #   s²‖𝛿_gn - t𝛿_sd‖₂² + 2s(t𝛿_sd)ᵀ(𝛿_gn - t𝛿_sd) + ‖t𝛿_sd‖₂² = (ηΔ)²
        # 
        # This is a quadratic function
        # 
        #   As² + Bs + C = 0
        #   A = ‖𝛿_gn - t𝛿_sd‖₂²
        #   B = 2(t𝛿_sd)ᵀ(𝛿_gn - t𝛿_sd)
        #   C = ‖t𝛿_sd‖₂² - (ηΔ)²
        _A = np.linalg.norm(delta_gn - t * delta_sd, ord=2)
        _B = 2 * ((t * delta_sd).T @ (delta_gn - t * delta_sd))[0, 0]
        _C = np.linalg.norm(t * delta_sd, ord=2) - delta ** 2
        s = (-_B + sqrt(_B * _B - 4 * _A * _C)) / (2 * _A)
        return delta_sd + s * (delta_gn - t * delta_sd)