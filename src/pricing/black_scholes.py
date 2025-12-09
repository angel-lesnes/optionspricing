import numpy as np
from math import sqrt, exp
from scipy.stats import norm


def bs_call_price(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def bs_call_delta(S, K, T, r, sigma):
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    return float(norm.cdf(d1))

# Put-Call parity 

def bs_put_price(S, K, T, r, sigma):
    C = bs_call_price(S, K, T, r, sigma)
    return float(C - float(S) + K * exp(-r * T))

def bs_put_delta(S, K, T, r, sigma):
    return float(bs_call_delta(S, K, T, r, sigma) - 1.0)

# Numerical test

if __name__ == "__main__":
    S0, K, T, r, sigma = 100, 100, 0.5, 0.01, 0.2
    print("BS call price:", bs_call_price(S0, K, T, r, sigma))
    print("BS call delta:", bs_call_delta(S0, K, T, r, sigma))
    print("BS put price:", bs_put_price(S0, K, T, r, sigma))
    print("BS put delta:", bs_put_delta(S0, K, T, r, sigma))