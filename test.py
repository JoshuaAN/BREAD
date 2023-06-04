import numpy as np

a = np.random.rand(3, 1)
b = np.random.rand(3, 3)
c = np.random.rand(3, 1)

print(a.T @ b @ c)
print(c.T @ b.T @ a)