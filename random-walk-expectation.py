import numpy as np

n = 25  # steps in each sample
k = 10_000  # number of samples

T = np.random.random((k, n)) * 2*np.pi
X = np.cos(T).sum(axis=1)
Y = np.sin(T).sum(axis=1)
R = np.sqrt(X**2 + Y**2)

print("Expected:", np.sqrt(n*np.pi/4))
print("Observed:", np.mean(R))
