import numpy as np
from solver import *

def main():
    N = 2

    solver = Solver()

    x = []
    y = []
    theta = []
    vl = []
    vr = []
    al = []
    ar = []

    T = solver.variable()
    dt = T / N
    solver.subject_to_inequality(T)

    dict = {T: 0}

    for i in range(N + 1):
        x.append(solver.variable())
        y.append(solver.variable())
        theta.append(solver.variable())
        vl.append(solver.variable())
        vr.append(solver.variable())

        dict[x[-1]] = 0
        dict[y[-1]] = 0
        dict[theta[-1]] = 0
        dict[vl[-1]] = 0
        dict[vr[-1]] = 0

    for i in range(N):
        al.append(solver.variable())
        ar.append(solver.variable())

        solver.subject_to_inequality(al[-1] + 5)
        solver.subject_to_inequality(ar[-1] + 5)
        solver.subject_to_inequality(-al[-1] + 5)
        solver.subject_to_inequality(-ar[-1] + 5)

        dict[al[-1]] = 0
        dict[ar[-1]] = 0

    for i in range(N):
        solver.subject_to_equality(vl[i + 1] - (vl[i] + dt * al[i]))
        solver.subject_to_equality(vr[i + 1] - (vr[i] + dt * ar[i]))
        solver.subject_to_equality(x[i + 1] - (x[i] + dt * cos(theta[i]) * (vl[i] + vr[i]) / 2.0))
        solver.subject_to_equality(y[i + 1] - (y[i] + dt * sin(theta[i]) * (vl[i] + vr[i]) / 2.0))
        solver.subject_to_equality(theta[i + 1] - (theta[i] + dt * (vr[i] - vl[i])))

    solver.subject_to_equality(x[0])
    solver.subject_to_equality(y[0])
    solver.subject_to_equality(theta[0])

    solver.subject_to_equality(x[-1] - 1)
    solver.subject_to_equality(y[-1])
    solver.subject_to_equality(theta[-1])

    solver.minimize(T)

    solver.solve(dict)

main()