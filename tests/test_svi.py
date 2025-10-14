import numpy as np
from volfit.calibrate import calibrate_slice


def test_calibrate_recovers_params_noiseless():
rng = np.random.default_rng(7)
k = np.linspace(-1.5, 1.5, 21)
true_iv = 0.3 + 0.1*(k**2)
iv = true_iv
p, loss = calibrate_slice(k, iv, ttm=0.5)
assert loss < 1e-2
