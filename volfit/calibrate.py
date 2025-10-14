import numpy as np
from dataclasses import dataclass
from .svi import svi_total_var, BOUNDS
from scipy.optimize import minimize


@dataclass
class SVIParams:
a: float; b: float; rho: float; m: float; sigma: float


_DEF = SVIParams(0.05, 0.5, -0.3, 0.0, 0.3)


def _box(p: SVIParams):
lo = [BOUNDS[k][0] for k in ("a","b","rho","m","sigma")]
hi = [BOUNDS[k][1] for k in ("a","b","rho","m","sigma")]
return list(zip(lo,hi))


def calibrate_slice(k, iv, ttm, p0: SVIParams=_DEF):
# minimize squared error on total variance
w_obs = (iv**2)*ttm
x0 = np.array([p0.a, p0.b, p0.rho, p0.m, p0.sigma])
def loss(x):
w = svi_total_var(k, *x)
return np.mean((w - w_obs)**2)
res = minimize(loss, x0, bounds=_box(p0), method='L-BFGS-B')
return SVIParams(*res.x), res.fun
