# https://en.wikipedia.org/wiki/Upside_potential_ratio

def upside_potential_ratio(returns, probablities, minimal):
    '''
    The upside-potential ratio is a measure of a return of
    an investment asset relative to the minimal acceptable return (minimal).
    if `inf` is returned it means all returns are greater than minimal (no risk)
    '''

    fum = 0  # first upper moment
    slm = 0  # second lower moment

    for r, p in zip(returns, probablities):
        diff = r - minimal
        if diff > 0:
            fum += diff * p
        else:
            slm += (diff ** 2) * p

    if slm == 0:
        return float('inf')
    return fum / (slm ** 0.5)


if __name__ == '__main__':
    M = 2  # minimal acceptable return
    R = [1, 2, 3, 4]
    P1 = [0.2, 0.3, 0.3, 0.2]
    P2 = [0.3, 0.3, 0.2, 0.2]
    print('Returns:', R)
    print('Minimal Acceptable Return:', M)
    print('First Probabilities (P1):', P1)
    print('Second Probabilities (P2):', P2)

    u1 = upside_potential_ratio(R, P1, M)
    u2 = upside_potential_ratio(R, P2, M)
    print('Upside Potential Ratio for P1:', u1)
    print('Upside Potential Ratio for P2:', u2)
