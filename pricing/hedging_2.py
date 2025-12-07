import numpy as np
from black_scholes import bs_call_price, bs_call_delta

def simulate_delta_hedge_paths(S_paths, K, r, sigma, T,
                               option_type="call",
                               tc_frac=0.0,
                               option_price_override=None):
    """
    Simule le delta-hedge discret pour chaque trajectoire S_paths.
    - S_paths: ndarray (n_paths, n_steps+1)
    - option_type: "call" or "put"
    - tc_frac: fraction of trade value paid as transaction cost
    - option_price_override: if provided (float), this is the premium received at t0
      (otherwise we compute the price via BS at t0)
    Retour: pnl array (n_paths,) -- pnl for the seller (short).
    """
    n_paths, n_steps_p1 = S_paths.shape
    n_steps = n_steps_p1 - 1
    dt = T / n_steps

    pnl = np.zeros(n_paths)

    for i in range(n_paths):
        S = S_paths[i]  # price path: S[0], ..., S[-1]

        # compute initial delta and option price (use override if provided)
        if option_price_override is None:
            if option_type == "call":
                option_price = bs_call_price(S[0], K, T, r, sigma)
                delta = bs_call_delta(S[0], K, T, r, sigma)
            else:  # put
                # put price via parity and put delta = call_delta - 1
                call_price = bs_call_price(S[0], K, T, r, sigma)
                option_price = call_price - S[0] + K * np.exp(-r * T)
                delta = bs_call_delta(S[0], K, T, r, sigma) - 1.0
        else:
            # option_price_override given: compute delta using BS (same sigma assumed)
            option_price = float(option_price_override)
            if option_type == "call":
                delta = bs_call_delta(S[0], K, T, r, sigma)
            else:
                delta = bs_call_delta(S[0], K, T, r, sigma) - 1.0

        # Initial replication: buy 'delta' shares, rest in cash (seller received premium)
        shares = delta
        cash = option_price - shares * S[0]

        # Rebalancing loop
        for t_idx in range(1, n_steps + 1):
            t = (t_idx - 1) * dt
            tau = T - t

            # compute new delta
            if tau <= 0:
                if option_type == "call":
                    new_delta = 1.0 if S[t_idx] > K else 0.0
                else:
                    new_delta = -1.0 if S[t_idx] < K else 0.0
            else:
                if option_type == "call":
                    new_delta = bs_call_delta(S[t_idx], K, tau, r, sigma)
                else:
                    new_delta = bs_call_delta(S[t_idx], K, tau, r, sigma) - 1.0

            trade = new_delta - shares
            trade_cost = abs(trade) * tc_frac * S[t_idx]
            cash -= trade * S[t_idx] + trade_cost
            shares = new_delta
            cash *= np.exp(r * dt)

        # payoff and pnl for seller
        if option_type == "call":
            payoff = max(S[-1] - K, 0.0)
        else:
            payoff = max(K - S[-1], 0.0)

        pnl_seller = -payoff + shares * S[-1] + cash
        pnl[i] = pnl_seller

    return pnl


if __name__ == "__main__":
    # paramètres
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    n_paths, n_steps = 5000, 252

    # génère S_paths (utilise ta fonction mc_paths_gbm)
    from monte_carlo import mc_paths_gbm, mc_price_call, mc_price_put_parity

    S_paths = mc_paths_gbm(S0, T, r, sigma, n_paths, n_steps, antithetic=True, seed=42)

    # calculer prix MC (call) - prime reçue
    price_mc_call = mc_price_call(S_paths, K, r, T, control_variate=True)
    print("MC call price (prime):", price_mc_call)

    # simuler hedging en passant price_mc comme prime initiale
    pnl_call = simulate_delta_hedge_paths(S_paths, K, r, sigma, T,
                                          option_type="call",
                                          tc_frac=0.001,
                                          option_price_override=price_mc_call)

    print("Mean PnL (seller, using MC price as premium):", pnl_call.mean())
    print("Std PnL:", pnl_call.std())

    # calculer prix MC (put) - prime reçue
    price_mc_put = mc_price_put_parity(S_paths, K, r, T, control_variate=True)
    print("MC put price (prime):", price_mc_put)

    # simuler hedging en passant price_mc comme prime initiale
    pnl_put = simulate_delta_hedge_paths(S_paths, K, r, sigma, T,
                                          option_type="put",
                                          tc_frac=0.001,
                                          option_price_override=price_mc_put)

    print("Mean PnL Put(seller, using MC price as premium):", pnl_put.mean())
    print("Std PnL Put:", pnl_put.std())