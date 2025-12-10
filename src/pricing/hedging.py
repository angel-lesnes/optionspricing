import numpy as np
from pricing.black_scholes import bs_call_price, bs_call_delta

def simulate_delta_hedge_paths(S_paths, K, r, sigma, T, option_type="call", tc_frac=0.0):
    """
    Simule la couverture delta pour le vendeur (short) d'une option européenne.
    Retourne : pnl array pour chaque chemin (n_paths,)
    """

    n_paths, n_steps_p1 = S_paths.shape
    n_steps = n_steps_p1 - 1
    dt = T / n_steps

    pnl = np.zeros(n_paths)

    for i in range(n_paths):
        S = S_paths[i]  # shape (n_steps+1,)

        # initial time t=0
        if option_type == "call":
            option_price = bs_call_price(S[0], K, T, r, sigma)
            delta = bs_call_delta(S[0], K, T, r, sigma)
        elif option_type == "put":
            # put delta = call_delta - 1
            call_delta = bs_call_delta(S[0], K, T, r, sigma)
            delta = call_delta - 1.0
            option_price = bs_call_price(S[0], K, T, r, sigma) - S[0] + K * np.exp(-r * T)  # put-call parity
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        # seller (short) replication initialisation:
        # seller receives option_price as premium, buys `delta` shares, remainder is cash
        shares = delta
        cash = option_price - shares * S[0]

        # iterate through rebalancing dates
        for t_idx in range(1, n_steps + 1):
            # current time just before rebalancing step: t = (t_idx - 1) * dt
            t = (t_idx - 1) * dt
            tau = T - t  # time to maturity before rebalancing

            # compute new delta at current spot S[t_idx]
            if tau <= 0:
                # at maturity, delta is 1 for call if ITM else 0 ; for put  -1 if ITM else 0
                if option_type == "call":
                    new_delta = 1.0 if S[t_idx] > K else 0.0
                else:
                    # put delta at maturity = -1 if ITM (K > S) else 0
                    new_delta = -1.0 if S[t_idx] < K else 0.0
            else:
                if option_type == "call":
                    new_delta = bs_call_delta(S[t_idx], K, tau, r, sigma)
                else:
                    # put delta = call_delta - 1
                    new_delta = bs_call_delta(S[t_idx], K, tau, r, sigma) - 1.0

            # how many shares to trade
            trade = new_delta - shares

            # transaction cost (proportionnel sur la valeur échangée)
            trade_cost = abs(trade) * tc_frac * S[t_idx]

            # update cash (buy = negative cash; sell = positive cash)
            cash -= trade * S[t_idx] + trade_cost

            # update share position
            shares = new_delta

            # cash accrues at risk-free rate over the interval dt
            cash *= np.exp(r * dt)

        # At maturity
        if option_type == "call":
            payoff = max(S[-1] - K, 0.0)
        else:
            payoff = max(K - S[-1], 0.0)

        # P&L for the seller (short)
        pnl_seller = -payoff + shares * S[-1] + cash
        pnl[i] = pnl_seller

    return pnl


if __name__ == "__main__":
    from pricing.monte_carlo import mc_paths_gbm

    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    n_paths, n_steps = 5000, 252

    S_paths = mc_paths_gbm(S0, T, r, sigma, n_paths, n_steps)

    pnl_call = simulate_delta_hedge_paths(
        S_paths, K, r, sigma, T,
        option_type="call",
        tc_frac=0.001
    )

    pnl_put = simulate_delta_hedge_paths(
        S_paths, K, r, sigma, T,
        option_type="put",
        tc_frac=0.001
    )

    print("Mean PnL (CALL short):", pnl_call.mean())
    print("Mean PnL (PUT short):", pnl_put.mean())