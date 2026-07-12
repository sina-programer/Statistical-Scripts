from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA

from matplotlib import pyplot as plt
import numpy as np

def simulate_arma(ar: list[float], ma: list[float], n=100):
    arma = ArmaProcess(np.array(ar), np.array(ma))
    return arma.generate_sample(nsample=n)


if __name__ == '__main__':
    series = simulate_arma([1, 1], [1])

    print(ARIMA(series, order=(1, 0, 0)).fit().summary())

    plt.plot(series)
    plot_acf(series)
    plot_pacf(series)
    plt.show()
