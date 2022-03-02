import math
import numpy as np
from scipy import integrate
import matplotlib.pyplot as mp


# Умножение матриц
def find_x(first: np.ndarray, second: np.ndarray):
    return np.dot(np.linalg.inv(first), second)


def f(x, y, z):
    return x + y + z


def hy(x):
    return 1 - x


def hz(x, y):
    return 1 - x - y


def zero():
    return 0


def show_functions():
    # Creating figure 8x5 inches with dpi = 100, without frame
    mp.figure(figsize=(8, 5), dpi=100, facecolor='white',)
    ax = mp.subplot(111, frameon=False)

    # Deleting right ant top spines
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))

    # Our functions C and L
    X = np.linspace(-2 * np.pi, 2 * np.pi, 256, endpoint=True)
    C, L = 2*np.cos(X - math.pi/4), X + 3

    mp.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="Cos Function")
    mp.plot(X, L, color="green", linewidth=2.5, linestyle="-", label="Lin Function")

    mp.xlim(X.min(initial=None) * 1.1, X.max(initial=None) * 1.1)

    mp.xticks([-2 * np.pi, -3 * np.pi / 2, -np.pi, -np.pi / 2, 0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
              [r'$-2\pi$', r'$-3\pi/2$', r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$', r'$+3 \pi/2$',
               r'$+ 2\pi$'])

    mp.ylim(C.min() * 1.1, C.max() * 1.1)
    mp.yticks([-2, -1, +1, +2],
              [r'$-2$', r'$-1$', r'$+1$', r'$+2$'])

    mp.legend(loc='upper left', frameon=False)
    mp.grid()
    mp.show()


if __name__ == '__main__':
    ones_matrix = np.ones((5, 5), dtype=int)
    big_ones_matrix = np.ones((50, 50), dtype=int)
    print(f'5х5 matrix:\n {ones_matrix}\n')
    print(f'5х5 matrix:\n {big_ones_matrix}\n')

    det_matrix = np.array([[3, -1, 2, 3, 2],
                           [1, 2, -3, 3, 4],
                           [2, -3, 4, 2, 1],
                           [3, 0, 0, 5, 0],
                           [2, 0, 0, 4, 0]], dtype=int)
    print(f'det of det_matrix: \n{round(np.linalg.det(det_matrix))}')

    A = np.random.randint(0, 6, (4, 4))
    B = np.random.randint(0, 6, (4, 1))
    print(f'A matrix:\n {A}\n')
    print(f'B matrix:\n {B}\n')

    MTX = find_x(A, B)
    print(f'Solution of AX=B matrix:\n {MTX}\n')

    first_i = integrate.quad(lambda x: math.cosh(3 * x), 0, 1 / 3)
    print(f'Solution of first integral:\n {first_i[0]}\n')

    second_i = integrate.tplquad(f, 0, 1, lambda x: 0, hy, lambda x, y: 0, hz)
    print(f'Solution of second integral:\n {second_i[0]}\n')

    show_functions()
