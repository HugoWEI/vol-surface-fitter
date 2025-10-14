import numpy as np


# Raw SVI total variance: w(k) = a + b*( rho*(k-m) + sqrt((k-m)^2 + sigma^2) )
# Vectorized evaluation + simple box constraints for stability.


def svi_total_var(k, a, b, rho, m, sigma):
return a + b * (rho*(k-m) + np.sqrt((k-m)**2 + sigma**2))


BOUNDS = {
"a": (1e-6, 5.0),
"b": (1e-6, 10.0),
"rho": (-0.999, 0.999),
"m": (-2.0, +2.0),
"sigma": (1e-4, 5.0),
}
