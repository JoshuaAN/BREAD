from solver.solver import Solver
from sympy import *

solver = Solver()

x = solver.variable()
y = solver.variable()
# z = 2.5 * x * x + 2.5 * y * y
solver.subject_to_equality(x * x + y - 1)
z = x * x + y * y
# solver.subject_to_equality(x + 3 * y - 36)

# solver.subject_to_inequality(x * x - 1)

dict = {x: 10.0, y: 3.0}

solver.minimize(z)
solver.solve(dict)