import numpy as np

def RK4(f, x, u, h):
    k1 = f(x, u)
    k2 = f(x + h * 0.5 * k1, u)
    k3 = f(x + h * 0.5 * k2, u)
    k4 = f(x + h * k3, u)

    return x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def cart_pole_dynamics(x, u):
    # https://underactuated.mit.edu/acrobot.html#cart_pole

    # θ is CCW+ measured from negative y-axis.

    # q = [x, θ]ᵀ
    # q̇ = [ẋ, θ̇]ᵀ
    # u = f_x

    # M(q)q̈ + C(q, q̇)q̇ = τ_g(q) + Bu
    # M(q)q̈ = τ_g(q) − C(q, q̇)q̇ + Bu
    # q̈ = M⁻¹(q)(τ_g(q) − C(q, q̇)q̇ + Bu)

    #        [ m_c + m_p  m_p l cosθ]
    # M(q) = [m_p l cosθ    m_p l²  ]

    #           [0  −m_p lθ̇ sinθ]
    # C(q, q̇) = [0       0      ]

    #          [     0      ]
    # τ_g(q) = [-m_p gl sinθ]

    #     [1]
    # B = [0]
    m_c = 5
    m_p = 0.5
    l = 0.5
    g = 9.806

    q = 