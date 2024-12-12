import math
from scipy.integrate import quad
import scipy
import numpy as np
import matplotlib.pyplot as plt
from TracyWidom import TracyWidom


def mpDensity(ndf, pdim, var=1):
    """
    Calculate the Marcenko-Pastur density bounds.

    Parameters:
    ndf (int): Number of degrees of freedom.
    pdim (int): Dimensionality of the data.
    var (float): Variance, default is 1.

    Returns:
    tuple: Lower and upper bounds of the Marcenko-Pastur density.
    """
    gamma = ndf / pdim
    inv_gamma_sqrt = math.sqrt(1 / gamma)
    a = var * (1 - inv_gamma_sqrt) ** 2
    b = var * (1 + inv_gamma_sqrt) ** 2
    return a, b


def dmp(x, ndf, pdim, var=1, log=False):
    """
    Calculate the Marcenko-Pastur density.

    Parameters:
    x (float): Point at which to evaluate the density.
    ndf (int): Number of degrees of freedom.
    pdim (int): Dimensionality of the data.
    var (float): Variance, default is 1.
    log (bool): If True, return the log density, default is False.

    Returns:
    float: The Marcenko-Pastur density at point x.
    """
    gamma = ndf / pdim
    a, b = mpDensity(ndf, pdim, var)
    if not log:
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
    """
    Calculate the cumulative distribution function of the Marcenko-Pastur distribution.

    Parameters:
    q (float): Quantile to evaluate.
    ndf (int): Number of degrees of freedom.
    pdim (int): Dimensionality of the data.
    var (float): Variance, default is 1.
    lower_tail (bool): If True, return the lower tail probability, default is True.
    log_p (bool): If True, return the log probability, default is False.

    Returns:
    float: The cumulative probability at quantile q.
    """
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
    """
    Calculate the quantile function of the Marcenko-Pastur distribution.

    Parameters:
    p (float): Probability to evaluate.
    ndf (int): Number of degrees of freedom.
    pdim (int): Dimensionality of the data.
    var (float): Variance, default is 1.
    lower_tail (bool): If True, return the lower tail quantile, default is True.
    log_p (bool): If True, p is given as log(p), default is False.

    Returns:
    float: The quantile corresponding to probability p.
    """
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


def bema_inside(pdim, ndf, eigs, alpha, beta):
    """
    BEMA algorithm calculation.

    Parameters:
    pdim (int): Dimensionality of the data.
    ndf (int): Number of degrees of freedom.
    eigs (array): Eigenvalues.
    alpha (float): Alpha parameter.
    beta (float): Beta parameter.

    Returns:
    tuple: sigma_sq, lamda_plus, l2
    """
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


def MP_Density_Inner(gamma, sigma_sq, x):
    """
    Helper function to compute MP density.

    Parameters:
    gamma (float): Ratio of dimensions.
    sigma_sq (float): Variance.
    x (float): Point at which to evaluate the density.

    Returns:
    float: The MP density at point x.
    """
    lp = sigma_sq * pow(1 + math.sqrt(gamma), 2)
    lm = sigma_sq * pow(1 - math.sqrt(gamma), 2)
    dv = math.sqrt((lp - x) * (x - lm)) / (gamma * x * 2 * math.pi * sigma_sq)
    return dv


def MP_Density_Wrapper(gamma, sigma_sq, samplePoints):
    """
    Compute MP density at sample points.

    Parameters:
    gamma (float): Ratio of dimensions.
    sigma_sq (float): Variance.
    samplePoints (array): Points at which to evaluate the density.

    Returns:
    array: MP density at sample points.
    """
    lp = sigma_sq * pow(1 + math.sqrt(gamma), 2)
    lm = sigma_sq * pow(1 - math.sqrt(gamma), 2)
    y = []
    for i in samplePoints:
        if lm <= i and i <= lp:
            y.append(MP_Density_Inner(gamma, sigma_sq, i))
        else:
            y.append(0)
    return np.array(y)


def MP_CDF_inner(gamma, sigma_sq, x):
    """
    Helper function to compute MP CDF.

    Parameters:
    gamma (float): Ratio of dimensions.
    sigma_sq (float): Variance.
    x (float): Point at which to evaluate the CDF.

    Returns:
    float: The MP CDF at point x.
    """
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


def MP_CDF(gamma, sigma_sq, samplePoints):
    """
    Compute MP CDF at sample points.

    Parameters:
    gamma (float): Ratio of dimensions.
    sigma_sq (float): Variance.
    samplePoints (array): Points at which to evaluate the CDF.

    Returns:
    array: MP CDF at sample points.
    """
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
    """
    Compute empirical CDF.

    Parameters:
    S (array): Sample points.

    Returns:
    array: Empirical CDF values.
    """
    return np.array([(i) / len(S) for i in range(len(S))])


def error(singular_values, alpha, pTilde, gamma, sigma_sq, show=False):
    """
    Compute error between theoretical and empirical CDFs.

    Parameters:
    singular_values (array): Singular values.
    alpha (float): Alpha parameter.
    pTilde (int): Number of pruned singular values.
    gamma (float): Ratio of dimensions.
    sigma_sq (float): Variance.
    show (bool): If True, display plots, default is False.

    Returns:
    float: The error between theoretical and empirical CDFs.
    """
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
