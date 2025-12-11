import numpy as np
from math import sqrt, exp
from scipy.stats import norm

#BS Merton avec dividendes

def bs_call_price(S, K, T, r, sigma, q=0.0):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def bs_call_delta(S, K, T, r, sigma, q=0.0):
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    return exp(-q * T) * norm.cdf(d1)

# Put-Call parity 

def bs_put_price(S, K, T, r, sigma, q=0.0):
    C = bs_call_price(S, K, T, r, sigma, q)
    S_adj = S * exp(-q * T) 
    K_disc = K * exp(-r * T)
    return float(C - S_adj + K_disc)

def bs_put_delta(S, K, T, r, sigma, q=0.0):
    return float(bs_call_delta(S, K, T, r, sigma, q) - exp(-q * T))

# Numerical test

if __name__ == "__main__":
    S0, K, T, r, sigma, q = 100, 100, 0.5, 0.01, 0.2, 0.03
    print("BS call price:", bs_call_price(S0, K, T, r, sigma, q))
    print("BS call delta:", bs_call_delta(S0, K, T, r, sigma, q))
    print("BS put price:", bs_put_price(S0, K, T, r, sigma, q))
    print("BS put delta:", bs_put_delta(S0, K, T, r, sigma, q))