def f(n: int) -> int:
    if n%2 == 0:
        return n//2
    return 3*n + 1

def path(n: int):
    history = []
    while n > 1:
        history.append(n)
        n = f(n)
    return history


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    K = 10
    L = {}

    for k in range(3, K+3):
        p = path(k)
        plt.plot(p, label=str(k))
        L[k] = len(p)
    plt.title('Cycles')
    plt.legend()

    plt.figure()
    plt.plot(L.keys(), L.values())
    plt.title('Path Length')

    plt.show()
