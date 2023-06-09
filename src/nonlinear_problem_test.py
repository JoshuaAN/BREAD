from solver.solver import Solver
from sympy import *

solver = Solver()

x = solver.variable()
y = solver.variable()
f = solver.variable()
z = (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x) + f * f

solver.subject_to_inequality(-((x - 1) * (x - 1) * (x - 1) - y + 1))
solver.subject_to_inequality(-(x + y - 2))
# solver.subject_to_inequality(-(x * x + y * y - 2))
solver.subject_to_equality(f)
# solver.subject_to_inequality(f + 1)
# 
dict = {x: 1.1, y: 1.5, f: 0}

solver.minimize(z)
solver.solve(dict)