import numpy as np
from matplotlib import pyplot as plt

from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.termination import NoTermination
from pymoo.optimize import minimize
import pymoo.gradient.toolbox as anp
from pymoo.visualization.scatter import Scatter


# Code copied from: pymoo.util.dominator
def calc_domination_matrix(F, _F=None, epsilon=0.0):

    if _F is None:
        _F = F

    # look at the obj for dom
    n = F.shape[0]
    m = _F.shape[0]

    L = np.repeat(F, m, axis=0)
    R = np.tile(_F, (n, 1))

    smaller = np.reshape(np.any(L + epsilon < R, axis=1), (n, m))
    larger = np.reshape(np.any(L > R + epsilon, axis=1), (n, m))

    M = np.logical_and(smaller, np.logical_not(larger)) * 1 \
        + np.logical_and(larger, np.logical_not(smaller)) * -1
    # M1 = smaller * 1 + larger * -1 # Marche aussi

    # if cv equal then look at dom
    # M = constr + (constr == 0) * dom

    return M


#M = calc_domination_matrix(np.array([[0, 0], [0, 2], [2, 0]], dtype=float))
#print(M)


# Tests d'appartenance de points aux circles...
# def circ_dist(F, center=[0, 0])
def calc_1(F):
    M = np.sum(F**2, axis=1)
    T1 = (M < 1)
    T2 = (M >= 1) & (M < 4)
    T3 = (M >= 4) & (M < 9)
    T4 = (M >= 9) & (M < 16)
    T5 = (M >= 16)
    X1 = [k for k in range(len(M)) if T1[k]]
    X2 = [k for k in range(len(M)) if T2[k]]
    X3 = [k for k in range(len(M)) if T3[k]]
    X4 = [k for k in range(len(M)) if T4[k]]
    X5 = [k for k in range(len(M)) if T5[k]]
    return T2


#calc_1(np.array([[1, 1], [1, 2], [2, 1]], dtype=float))


# Distances entre un point donné à d'autres points F
def calc_distances(F, center=None):
    """
    Calcule les distances entre chaque point de F à un autre point center.
    :param F:
    :param center:
    :return: array
    """
    if center is None:
        center = [0, 0]
    if isinstance(center, list):
        center = np.array(center)
    df = F - center
    S = df**2
    #print(S)
    M = np.sum(S, axis=1)
    Q = np.sqrt(M)
    #print([F[:2, :], S[:2, :], M[:2], Q[:2]])
    #print(M)
    return Q


#calc_distances(np.array([[1, 1], [1, 2], [2, 1], [2, 2]], dtype=float), [1, 1])
#quit()


# Traçe 04 rectangles
def rect4(s1=(0, 0.4, 0.2, 1), s2=(0, 1.25, 0.5, 3)):
    x1, l1, dx, xu = s1
    y1, h1, dy, yu = s2

    a, b = x1, x1 + l1
    a1 = b + dx
    c, d = y1, y1+h1
    c1 = d + dy
    r1 = np.column_stack(([a, b, b, a, a], [c, c, d, d, c]))
    r2 = np.column_stack(([a1, xu, xu, a1, a1], [c, c, d, d, c]))
    r3 = np.column_stack(([a1, xu, xu, a1, a1], [c1, c1, yu, yu, c1]))
    r4 = np.column_stack(([a, b, b, a, a], [c1, c1, yu, yu, c1]))

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(r1[:, 0], r1[:, 1])
    ax.plot(r2[:, 0], r2[:, 1])
    ax.plot(r3[:, 0], r3[:, 1])
    ax.plot(r4[:, 0], r4[:, 1])
    # plt.show()
    return ax


# rect4()
# plt.show()
# quit()


def gen_points(n=5, xl=[0, 0], xu=[1, 3]):
    # Reproduction d'une partie de FloatRandomSampling pour générer des points
    r = np.random.random((n, 2))
    xl = np.array(xl)
    xu = np.array(xu)
    normalized = xl + r * (xu - xl)  # Voir: pymoo.util.normalization
    # Remarquer qu'on n'aura jamais un point du genre (., xu)
    # print(normalized)
    return normalized


#gen_points()

# TODO: Les points qui se trouvent dans un rectangle...


def calc_2(): #(F, P1, P2):
    ax = rect4()
    F = gen_points(15)
    ax.scatter (F[:, 0], F[:, 1], c="black", marker="o")
    plt.show()
    pass


def _plotF(F, problem, nr=3, nc=2, fig=None, i=0, title=""):
    #fig, ax = plt.subplots()
    if fig is None:
        fig, axs = plt.subplots(nr, nc)
    ax = fig.axes[i]
    ax.scatter(F[:, 0], F[:, 1], c="black", marker="o")
    # ax.plot(F[:, 0], F[:, 1], c="blue")
    pf = problem.pareto_front()
    ax.plot(pf[:, 0], pf[:, 1], color="blue", alpha=0.7)
    ax.set_xlabel('$f_1$')
    ax.set_ylabel('$f_2$')
    ax.set_title(title)
    #plt.show()
    return fig


def _plotX(X, nr=3, nc=2, fig=None, i=0, title=""):
    if fig is None:
        fig, axs = plt.subplots(nr, nc)
    ax = fig.axes[i]
    ax.scatter(X[:, 0], X[:, 1], c="blue", marker="+")
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(title)
    ax.grid(True)
    return fig


#calc_2()
#quit()

# Copied from: pymoo.operators.sampling.rnd.py
# Voir aussi: pymoo.core.sampling.py
class BinaryRandomSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        val = np.random.random((n_samples, problem.n_var))
        return (val < 0.5).astype(bool)


# Idée: Générer des points à distance égale...
# Par exemple, dans [0,1] : 0, 0.2, 0.4, 0.6, 0,8, 1.0.
class SpacingSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        print("TODO")
        np.linspace(problem.xl, problem.xu, n_samples, axis=1)
        #return (val < 0.5).astype(bool)

