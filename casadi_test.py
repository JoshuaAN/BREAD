from casadi import *

solver = Opti()

x = solver.variable()
y = solver.variable()

solver.subject_to(x * x + y * y < 2)
solver.minimize((1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x))

solver.solver('ipopt')
sol = solver.solve()

print(sol.value(x))
print(sol.value(y))