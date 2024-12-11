import math
from scipy.integrate import quad
import scipy
import numpy as np
import matplotlib.pyplot as plt
from TracyWidom import TracyWidom  # type: ignore


def mpDensity(ndf, pdim, var=1):
    gamma = ndf / pdim
    inv_gamma_sqrt = math.sqrt(1 / gamma)
    a = var * (1 - inv_gamma_sqrt) ** 2
    b = var * (1 + inv_gamma_sqrt) ** 2
    return a, b


def dmp(x, ndf, pdim, var=1, log=False):
    gamma = ndf / pdim

    a, b = mpDensity(ndf, pdim, var)

    if not log:
        # we have to handle +/- zero carefully when gamma=1
        if gamma == 1 and x == 0 and 1 / x > 0:
            d = math.inf
        elif x <= a and x >= b:
            d = 0
        else:
            d = gamma / (2 * math.pi * var * x) * math.sqrt((x - a) * (b - x))
    else:
        if gamma == 1 and x == 0 and 1 / x > 0:
            d = math.inf
        elif x <= a and x >= b:
            d = -math.inf
        else:
            d = (
                math.log(gamma)
                - (math.log(2) + math.log(math.pi) + math.log(var) + math.log(x))
                + 0.5 * math.log(x - a)
                + 0.5 * math.log(b - x)
            )

    return d


def pmp(q, ndf, pdim, var=1, lower_tail=True, log_p=False):
    gamma = ndf / pdim
    a, b = mpDensity(ndf, pdim, var)
    f = lambda x: dmp(x, ndf, pdim, var)
    if lower_tail:
        if q <= a:
            p = 0
        elif q >= b:
            p = 1
        else:
            p = quad(f, a, q)[0]
        if gamma < 1 and q >= 0:
            p += 1 - gamma
    else:
        if q <= a:
            p = min(1, gamma)
        elif q >= b:
            p = 0
        else:
            p = quad(f, q, b)[0]
        if gamma < 1 and q <= 0:
            p += 1 - gamma
    if log_p:
        res = math.log(p)
    else:
        res = p
    return res


def qmp(p, ndf, pdim, var=1, lower_tail=True, log_p=False):
    svr = ndf / pdim
    if lower_tail:
        p = p
    else:
        p = 1 - p
    if log_p:
        p = math.exp(p)
    a, b = mpDensity(ndf, pdim, var)
    q = None
    if p <= 0:
        if svr <= 1:
            q = -0
        else:
            q = a
    else:
        if p >= 1:
            q = b
    if svr < 1:
        if p < 1 - svr:
            q = -0
        else:
            if p == 1 - svr:
                q = 0
    if q is None:
        F = lambda x: pmp(x, ndf, pdim, var) - p
        q = scipy.optimize.brentq(F, a, b)
    return q


# bema_inside is where the BEMA algorithm is calculated
# use bema_mat_wrapper instead
def bema_inside(pdim, ndf, eigs, alpha, beta):
    pTilde = min(pdim, ndf)
    gamma = pdim / ndf
    ev = np.sort(eigs)
    ind = list(range(int(alpha * pTilde), int((1 - alpha) * pTilde)))
    num = 0
    q = [qmp(i / pTilde, ndf, pdim, 1) for i in ind]
    lamda = [ev[i] for i in ind]
    num = np.dot(q, lamda)
    denum = np.dot(q, q)
    sigma_sq = num / denum
    tw1 = TracyWidom(beta=1)
    t_b = tw1.cdfinv(1 - beta)
    lamda_plus = sigma_sq * (
        (
            (1 + np.sqrt(gamma)) ** 2
            + t_b
            * ndf ** (-2 / 3)
            * (gamma) ** (-1 / 6)
            * (1 + np.sqrt(gamma)) ** 4
            / 3
        )
    )
    l2 = sigma_sq * (1 + np.sqrt(gamma)) ** 2
    return sigma_sq, lamda_plus, l2


# use this function to compute bema
def bema_mat_wrapper(
    matrix, pReal, nReal, alpha, beta, goodnessOfFitCutoff, show=False
):
    # this block uses the fact that eigenvalues are invariant under transposition
    # and hence without loss of generality our input matrix is p x n where
    # p <= n. This is used to ensure that our matrix has all positive singular values
    if pReal <= nReal:
        p = pReal
        n = nReal
        matrix_norm = np.matmul(matrix, matrix.transpose()) / nReal
    else:
        p = nReal
        n = pReal
        matrix_norm = np.matmul(matrix.transpose(), matrix) / nReal

    v = np.linalg.eigvalsh(matrix_norm)
    sigma_sq, lamda_plus, l2 = bema_inside(p, n, v, alpha, beta)
    pTilde = min(p, n)
    LinfError = error(v, alpha, pTilde, p / n, sigma_sq)
    gamma = p / n
    goodFit = True if LinfError < goodnessOfFitCutoff else False
    if show:
        print("error", LinfError)
        plt.hist(
            v[-min(p, n) :],
            bins=100,
            color="black",
            label="Empirical Density",
            density=True,
        )
        # plt.axvline(x=lamda_plus, label = "Predicted Lambda Plus", color = "blue")
        Z = v[-min(p, n) :]
        for t in range(len(Z)):
            if Z[t] > lamda_plus:
                Z = Z[:t]
                break
        Y = MP_Density_Wrapper(gamma, sigma_sq, Z)
        # plt.plot(Z,Y, color = "orange", label = "Predicted Density")
        plt.axvline(x=lamda_plus, label="Lambda Plus", color="red")
        plt.legend()
        plt.title("Empirical Distribution Density")
        plt.show()

        eigsTruncated = [i for i in v[-min(p, n) :] if i < lamda_plus]
        plt.hist(
            eigsTruncated,
            bins=100,
            color="black",
            label="Truncated Empirical Density",
            density=True,
        )
        plt.plot(Z, Y, color="orange", label="Predicted Density")
        plt.legend()
        plt.title("Density Comparison Zoomed")
        plt.show()

    return v, p / n, sigma_sq, lamda_plus, goodFit


# helper MP density function evaluated at x
def MP_Density_Inner(gamma, sigma_sq, x):
    lp = sigma_sq * pow(1 + math.sqrt(gamma), 2)
    lm = sigma_sq * pow(1 - math.sqrt(gamma), 2)
    dv = math.sqrt((lp - x) * (x - lm)) / (gamma * x * 2 * math.pi * sigma_sq)
    return dv


# at the sampled points, compute the MP distribution density
def MP_Density_Wrapper(gamma, sigma_sq, samplePoints):
    lp = sigma_sq * pow(1 + math.sqrt(gamma), 2)
    lm = sigma_sq * pow(1 - math.sqrt(gamma), 2)

    y = []
    for i in samplePoints:
        if lm <= i and i <= lp:
            y.append(MP_Density_Inner(gamma, sigma_sq, i))
        else:
            y.append(0)
    return np.array(y)


# helper function to compute MP CDF
def MP_CDF_inner(gamma, sigma_sq, x):
    lp = sigma_sq * pow(1 + math.sqrt(gamma), 2)
    lm = sigma_sq * pow(1 - math.sqrt(gamma), 2)
    r = math.sqrt((lp - x) / (x - lm))

    F = math.pi * gamma + (1 / sigma_sq) * math.sqrt((lp - x) * (x - lm))
    F += -(1 + gamma) * math.atan((r * r - 1) / (2 * r))
    if gamma != 1:
        F += (1 - gamma) * math.atan(
            (lm * r * r - lp) / (2 * sigma_sq * (1 - gamma) * r)
        )
    F /= 2 * math.pi * gamma
    return F


# at the sample points compute the theoretical MP CDF
def MP_CDF(gamma, sigma_sq, samplePoints):
    lp = sigma_sq * pow(1 + math.sqrt(gamma), 2)
    lm = sigma_sq * pow(1 - math.sqrt(gamma), 2)

    output = []
    for x in samplePoints:
        if gamma <= 1:
            if x < lm:
                output.append(0)
            elif x >= lp:
                output.append(0)
            else:
                output.append(MP_CDF_inner(gamma, sigma_sq, x))
        else:
            if x < lm:
                output.append((gamma - 1) / gamma)
            elif x >= lp:
                output.append(1)
            else:
                output.append(
                    (gamma - 1) / (2 * gamma) + MP_CDF_inner(gamma, sigma_sq, x)
                )
    return np.array(output)


def empiricalCDF(S):
    return np.array([(i) / len(S) for i in range(len(S))])


def error(singular_values, alpha, pTilde, gamma, sigma_sq, show=False):
    pTilde = len(singular_values)
    ind = np.arange(int(alpha * pTilde), int((1 - alpha) * pTilde))
    prunedSingularValues = singular_values[ind]
    theoretical = MP_CDF(gamma, sigma_sq, prunedSingularValues)
    empirical = alpha + (1 - 2 * alpha) * empiricalCDF(prunedSingularValues)
    difference = theoretical - empirical
    if show:
        plt.hist(difference, label="Difference histogram")
        plt.legend()
        plt.show()
        x = np.arange(len(empirical))
        plt.plot(x, empirical, label="empirical")
        plt.plot(x, theoretical, label="theoretical")
        plt.legend()
        plt.show()
    return np.linalg.norm(difference, np.inf)
