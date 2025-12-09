import numpy as np
from math import sqrt, exp

def mc_paths_gbm(S0, T, r, sigma, n_paths, n_steps, antithetic=True, seed=None):

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    shape = (n_paths, n_steps)

    # Antithetic variates 
    if antithetic:
        half_paths = n_paths // 2
        z_half = rng.standard_normal((half_paths, n_steps))
        z = np.vstack((z_half, -z_half))

        # si nombre impair : ajoute juste 1 set normal
        if n_paths % 2 == 1:
            z_extra = rng.standard_normal((1, n_steps))
            z = np.vstack((z, z_extra))

    else:
        z = rng.standard_normal(shape)

    # Generate paths 
    increments = (r - 0.5 * sigma**2) * dt + sigma * sqrt(dt) * z
    log_returns = np.cumsum(increments, axis=1)
    log_prices = np.hstack([np.zeros((log_returns.shape[0], 1)), log_returns])
    S = S0 * np.exp(log_prices)
    return S


def mc_price_call(S_paths, K, r, T, control_variate=False):
    ST = S_paths[:, -1]
    payoffs = np.maximum(ST - K, 0.0)
    disc_payoffs = np.exp(-r * T) * payoffs
    price = disc_payoffs.mean()

    if not control_variate:
        return price

    # Control variate basé sur E[ST] analytique
    S0 = S_paths[0, 0]
    theoretical_ST = S0 * exp(r * T)

    cov = np.cov(disc_payoffs, ST, bias=True)[0, 1]
    var_ST = np.var(ST)

    if var_ST > 0:
        b = cov / var_ST
        adjusted = disc_payoffs - b * (ST - theoretical_ST)
        price = adjusted.mean()

    return price


def mc_price_put(S_paths, K, r, T, control_variate=False):
    ST = S_paths[:, -1]
    payoffs = np.maximum(K - ST, 0.0)
    disc_payoffs = np.exp(-r * T) * payoffs
    price = disc_payoffs.mean()

    if not control_variate:
        return price

    # Control variate basé sur E[ST] analytique
    S0 = S_paths[0, 0]
    theoretical_ST = S0 * exp(r * T)

    cov = np.cov(disc_payoffs, ST, bias=True)[0, 1]
    var_ST = np.var(ST)

    if var_ST > 0:
        b = cov / var_ST
        adjusted = disc_payoffs - b * (ST - theoretical_ST)
        price = adjusted.mean()

    return price

def mc_price_put_parity (S_paths, K, r, T, control_variate=True):

    price_call = mc_price_call(S_paths, K, r, T, control_variate=control_variate) # Put-call parity : p = c - S0 + K * exp(-rT)
    S0 = S_paths[0, 0]
    price_put_parity = price_call - S0 + K * exp(-r * T)
    return price_put_parity


if __name__ == "__main__":
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    n_paths, n_steps = 10000, 252

    S_paths = mc_paths_gbm(S0, T, r, sigma, n_paths, n_steps,
                           antithetic=True, seed=None)

    price_call = mc_price_call(S_paths, K, r, T, control_variate=True)
    price_put = mc_price_put(S_paths, K, r, T, control_variate=True)
    price_put_parity = mc_price_put_parity(S_paths, K, r, T, control_variate=True)

    print("MC Call Price:", price_call)
    print("MC Put Price:", price_put)
    print("MC Put Price (Put-Call Parity):", price_put_parity)