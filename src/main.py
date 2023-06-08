from solver.solver import Solver
from sympy import *

solver = Solver()

x = solver.variable()
y = solver.variable()
# z = 2.5 * x * x + 2.5 * y * y
solver.subject_to_equality(x + y - 1)
z = x * x + y * y
# solver.subject_to_equality(x + 3 * y - 36)

solver.subject_to_inequality(x - 10)
# 
dict = {x: 10.0, y: 3.0}

solver.minimize(z)
solver.solve(dict)