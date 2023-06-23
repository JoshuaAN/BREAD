from solver.solver import Solver
from sympy import *

solver = Solver()

x = solver.variable()
y = solver.variable()
solver.subject_to_equality(x * x + y - 1)
z = x * x + y * y

solver.subject_to_inequality(y + 5)
#
dict = {x: 11.0, y: 3.0}

solver.minimize(z)
solver.solve(dict)
