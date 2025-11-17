from pricing.black_scholes import bs_call_price, bs_call_delta

def test_bs_price_positive():
    p = bs_call_price(100, 90, 0.5, 0.01, 0.2)
    assert p > 0

def test_bs_delta_bounds():
    delta = bs_call_delta(100, 100, 0.5, 0.01, 0.2)
    assert 0.0 <= delta <= 1.0