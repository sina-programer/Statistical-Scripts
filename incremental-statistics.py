from copy import deepcopy
import math

number = int | float

class Welford:
    '''
    Welford Incremental Method for computing central moments and mean
    Notice: all values are not unbiased if used for samples from population (unnormalized)
    '''

    copy = deepcopy

    def __init__(self, series=None):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # ss - s**2/n
        self.M3 = 0.0  # third central moment
        self.M4 = 0.0

        if series is not None:
            assert hasattr(series, '__iter__')
            for x in series:
                self.add(x)

    @property
    def n(self):
        return self.count

    @property
    def sum(self):
        return self.count * self.mean

    @property
    def var(self):
        return self.M2 / self.count

    @property
    def std(self):
        return math.sqrt(self.var)

    @property
    def skew(self):
        return self.M3 * math.sqrt(self.count) / (self.M2**1.5)

    @property
    def kurt(self):
        return self.count * self.M4 / (self.M2**2) - 3

    def add(self, x: number):
        '''updates current instance; simplified from extend with {x}'''
        assert isinstance(x, number)
        delta = x - self.mean
        n = self.count + 1
        self.count = n
        self.mean += delta / n
        self.M4 += delta**4 * (n-1)*(n**2-3*n+3)/n**3 + 6*delta**2*self.M2/n**2 - 4*delta*self.M3/n
        self.M3 += delta**3 * (n-1)*(n-2)/(n**2) - 3*delta*self.M2/n
        self.M2 += delta**2 * (n-1)/n

    def extend(self, other):
        assert isinstance(other, type(self))
        delta = other.mean - self.mean
        cc = self.count * other.count

        new = Welford()
        new.count = self.count + other.count
        new.mean = (self.sum + other.sum) / new.count
        new.M2 = self.M2 + other.M2 + delta**2 * cc / new.count

        m3_term1 = delta**3 * cc * (self.count - other.count) / new.count**2
        m3_term2 = 3*delta * (self.count*other.M2 - other.count*self.M2) / new.count
        new.M3 = self.M3 + other.M3 + m3_term1 + m3_term2

        m4_term1 = delta**4 * cc * (self.count**2 - cc + other.count**2) / new.count**3
        m4_term2 = 6*delta**2 * (self.count**2 * other.M2 + other.count**2 * self.M2) / new.count**2
        m4_term3 = 4*delta * (self.count*other.M3 - other.count*self.M3) / new.count
        new.M4 = self.M4 + other.M4 + m4_term1 + m4_term2 + m4_term3
        return new

    def __add__(self, other):
        if isinstance(other, type(self)):
            return self.extend(other)
        elif isinstance(other, number):
            return self.extend(Welford([other]))
        raise TypeError(f"type <{type(other)}> is not supported for addition with Welford object")

    __radd__ = __add__

    def __iter__(self):
        return iter((self.count, self.mean, self.M2, self.M3, self.M4))

    def __hash__(self):
        return hash(tuple(self))

    def __repr__(self):
        return "<{}: {} at {}>".format(type(self).__name__, hash(self), hex(id(self)))

    def __str__(self):
        params = ', '.join(map(str, self))
        return "{}({})".format(type(self).__name__, params)


if __name__ == '__main__':
    A = [1, 2, 3, 4]
    B = [1, 3, 3, 4]

    a = Welford(A)
    b = Welford(B)
    c = Welford(A+B)
    d = a + b

    print(a)
    print(b)
    print(c)
    print(d)
