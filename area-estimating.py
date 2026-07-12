import numpy as np

a = 1  # side length of square
N = 100_000  # number of samples
mask = np.ones(N, dtype=int)
corners = [
    (0, 0),
    (0, a),
    (a, 0),
    (a, a)
]

X = np.random.uniform(0, a, size=N)
Y = np.random.uniform(0, a, size=N)

for x, y in corners:
    R = np.sqrt(np.square(X - x) + np.square(Y - y))
    mask &= (R <= a)

area = (1 + np.pi/3 - np.sqrt(3)) * a**2
estimated_area = np.mean(mask)

print('Exact Area:', area)
print('Estimated Area:', estimated_area)
