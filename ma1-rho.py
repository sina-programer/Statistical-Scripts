from matplotlib import pyplot as plt
import numpy as np

# MA(1) -> Moving average with one backward:
# X(t) = theta * Z(t-1) + Z(t)

a, b = -10, 10
theta = np.arange(a, b, 0.01)
rho = theta / (1 + theta**2)  # auto-corr for time_delta=1

formula = r"$ \rho = \frac{\theta}{1 + \theta^2} $"
plt.plot(theta, rho, label=formula)
plt.title(r'Auto Correlation for MA(1) based on $\theta ; \Delta t=1$')
plt.xticks(list(range(int(a), int(b+1))))
plt.xlabel('$ \\theta $')
plt.ylabel('$ \\rho $')
plt.tight_layout()
plt.legend()
plt.grid()
plt.show()
