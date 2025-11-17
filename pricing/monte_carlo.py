import numpy as np
from math import sqrt, exp

def mc_paths_gbm(S0, T, r, sigma, n_paths, n_steps, antithetic=True,
seed=None):
    
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    shape = (n_paths, n_steps)

# Variance reduction with antithetic variates

    if antithetic:
        half_paths = n_paths // 2
        z_half = rng.standard_normal((half_paths, n_steps))
        z = np.vstack((z_half, -z_half))
        if n_paths % 2 == 1:
            z_extra = rng.standard_normal((1, n_steps))
            z = np.vstack((z, z_extra))
        else:
            z = np.vstack([z_half, -z_half, rng.standard_normal((1, n_steps))])
    else:
        z = rng.standard_normal(shape)

    increments = (r - 0.5 * sigma ** 2) * dt + sigma * sqrt(dt) * z
    log_returns = np.cumsum(increments, axis=1)
    log_prices = np.hstack([np.zeros((log_returns.shape[0], 1)), log_returns])
    S = S0 * np.exp(log_prices)
    return S

def mc_price_call(S_paths, K, r, T, control_variate=False):
    ST = S_paths[:, -1]
    payoffs = np.maximum(ST - K, 0.0)
    disc_payoffs = exp(-r * T) * payoffs
    price = disc_payoffs.mean()

# Variance reduction with control variates

    if control_variate:
        S0_exp = np.mean(ST) 
        theoretical_ST = S0_exp * exp(r * T)
        cov = np.cov(disc_payoffs, ST, bias=True)[0,1]
        var_ST = np.var(ST)
        if var_ST > 0:
            b = cov / var_ST
            adjusted = disc_payoffs - b * (ST - theoretical_ST)
            price = adjusted.mean()
    return price, payoffs

if __name__ == "__main__":
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    n_paths, n_steps = 10000, 252
    S_paths = mc_paths_gbm(S0, T, r, sigma, n_paths, n_steps,
                           antithetic=True, seed=None)
    price, _ = mc_price_call(S_paths, K, r, T, control_variate=True)
    print("MC Call Price:", price)