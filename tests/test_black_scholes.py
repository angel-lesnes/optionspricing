import pytest
from pricing.black_scholes import bs_call_price, bs_call_delta, bs_put_price
from math import exp

def test_bs_price_positive():
    p = bs_call_price(100, 90, 0.5, 0.01, 0.2)
    assert p > 0

def test_bs_delta():
    delta = bs_call_delta(100, 100, 0.5, 0.01, 0.2)
    assert 0.0 <= delta <= 1.0

def test_put_call_parity():
    S, K, T, r, sigma = 100.0, 95.0, 0.75, 0.02, 0.25
    C = bs_call_price(S, K, T, r, sigma)
    P = bs_put_price(S, K, T, r, sigma)
    expected = S - K * exp(-r * T)
    assert C - P == pytest.approx(expected, rel=1e-9)