import streamlit as st
import numpy as np
import plotly.graph_objs as go
from pricing.monte_carlo import mc_paths_gbm, mc_price_call, mc_price_put_parity, mc_price_put

def render_mc():
    st.header("Monte Carlo Pricing")

    # inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        S0 = st.number_input("Prix spot S₀", value=100.0)
        K = st.number_input("Strike K", value=100.0)
    with col2:
        T = st.number_input("Maturité T (années)", value=1.0)
        r = st.number_input("Taux sans risque r", value=0.01)
    with col3:
        sigma = st.number_input("Volatilité σ", value=0.2)
        option_type = st.selectbox("Type d'option", ["Call", "Put"], index=0)

    n_paths = st.slider("Nombre de chemins", 1000, 30000, 5000)
    n_steps = st.slider("Pas par an", 50, 365, 252)

    colA, colB = st.columns(2)
    with colA:
        antithetic = st.checkbox("Antithetic variates", value=True)
    with colB:
        control = st.checkbox("Control variate (E[S_T])", value=True)

    # compute price of chosen option


    if st.button("Lancer la simulation"):
        S_paths = mc_paths_gbm(S0, T, r, sigma, n_paths, n_steps, antithetic=antithetic, seed=42)

        if option_type == "Call":
            price = mc_price_call(S_paths, K, r, T, control_variate=control)
        else:
            price = mc_price_put_parity(S_paths, K, r, T, control_variate=control)

        st.success(f"MC price ({option_type}): {price:.4f}")

        # plot trajectories
        N = min(50, n_paths)
        fig = go.Figure()
        for i in range(min(30, N)):
            fig.add_trace(go.Scatter(x=np.arange(n_steps+1), y=S_paths[i], mode="lines",
                                     line=dict(width=1), name=f"path_{i}", showlegend=False))
        fig.update_layout(title="Quelques trajectoires simulées", xaxis_title="time step", yaxis_title="S")
        st.plotly_chart(fig, width="stretch")

        # histogram of terminal prices
        ST = S_paths[:, -1]
        hist = np.histogram(ST, bins=50)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=hist[1][:-1], y=hist[0]))
        fig2.update_layout(title="Distribution de S(T)", xaxis_title="S(T)", yaxis_title="count", template="plotly_white")
        st.plotly_chart(fig2, width="stretch")

        # payoff histogram for chosen option
        if option_type == "Call":
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=payoffs, nbinsx=50))
        fig3.update_layout(title="Distribution des payoffs", xaxis_title="payoff", yaxis_title="count")
        st.plotly_chart(fig3, width="stretch")

        # convergence check : mean payoff as function of number of paths
        sample_sizes = np.linspace(100, n_paths, 20, dtype=int)
        estimates = [np.exp(-r*T)*np.maximum(S_paths[:m, -1]-K,0).mean() if option_type=='Call' else np.exp(-r*T)*np.maximum(K-S_paths[:m, -1],0).mean() for m in sample_sizes]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=sample_sizes, y=estimates, mode="lines+markers"))
        fig4.update_layout(title="Convergence approximative", xaxis_title="n paths", yaxis_title="price estimate")
        st.plotly_chart(fig4, width="stretch")