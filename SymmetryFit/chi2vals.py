#!/usr/bin/python
import scipy.stats
import math

# stand deviations to calculate
sigma = [
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
]

# confidence intervals these sigmas represent:
conf_int = [scipy.stats.chi2.cdf(s**2, 1) for s in sigma]

# degrees of freedom to calculate
dof = list(range(1, 5)) + [6.7, 8.7] + [13.4, 17.4]

print("sigma     \t" + "\t".join(["%1.2f" % (s) for s in sigma]))
print("conf_int  \t" + "\t".join(["%1.2f%%" % (100 * ci) for ci in conf_int]))
print("p-value   \t" + "\t".join(["%1.5f" % (1 - ci) for ci in conf_int]))

for d in dof:
    chi_squared = [scipy.stats.chi2.ppf(ci, d) for ci in conf_int]
    print("chi2(k=%1.2f)\t" % d + ", ".join(["%1.2f" % c for c in chi_squared]))
