import streamlit as st
import numpy as np
import plotly.graph_objs as go
from pricing.monte_carlo import mc_paths_gbm, mc_price_call, mc_price_put_parity, mc_price_put

def render_mc():
    st.header("🎲 Monte Carlo Simulation")

    # --- INPUTS ---
    col1, col2, col3 = st.columns(3)
    with col1:
        S0 = st.number_input("Spot Price ($S_0$)", value=100.0)
        K = st.number_input("Strike Price ($K$)", value=100.0)
    with col2:
        T = st.number_input("Maturity ($T$)", value=1.0, help="Time to maturity in years")
        r = st.number_input("Risk-free Rate ($r$)", value=0.03, help="Annualized risk-free interest rate (e.g., OIS or Treasury yield)")
    with col3:
        sigma = st.number_input("Volatility ($\sigma$)", value=0.20)
        option_type = st.selectbox("Option Type", ["Call", "Put"], index=0)

    n_paths = st.slider("Number of Simulations (Paths)", 1000, 50000, 5000, step=1000)
    n_steps = st.slider("Time Steps per Year", 50, 365, 252)

    st.info("ℹ️ **Variance Reduction Techniques:** Select options below to improve precision without increasing calculation time.")

    # --- VARIANCE REDUCTION ---
    colA, colB = st.columns(2)
    with colA:
        antithetic = st.checkbox("Antithetic Variates", value=True, 
                                 help="Generates paths in negatively correlated pairs. For every random shock $Z$, the engine also generates $-Z$. This ensures that if one path deviates upward, its pair likely deviates downward, significantly reducing sampling error.")
    with colB:
        control = st.checkbox("Control Variate", value=True, 
                              help=r"Uses the known analytical expected value of the asset $\mathbb{E}[S_T] = S_0 e^{rT}$ to adjust the simulation result. If the simulated mean of $S_T$ is naturally too high/low compared to theory, the option price is corrected accordingly.")

    # --- SIMULATION TRIGGER ---
    if st.button("Run Simulation", type="primary"):
        
        # CALCULATIONS
        with st.spinner("Simulating asset paths and pricing..."):
            S_paths = mc_paths_gbm(S0, T, r, sigma, n_paths, n_steps, antithetic=antithetic, seed=None)

            if option_type == "Call":
                price = mc_price_call(S_paths, K, r, T, control_variate=control)
            else:
                # Using Put-Call Parity is often more numerically stable for Monte Carlo if Call is well priced
                price = mc_price_put_parity(S_paths, K, r, T, control_variate=control)

        # DISPLAY RESULT
        st.success(f"### Estimated {option_type} Price: **{price:.4f}**")

        # VISUALIZATION (Tabs)
        with st.spinner("Generating visualizations..."):
            
            # Création des onglets
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Trajectories", 
                "📊 Terminal Distribution", 
                "💰 Payoffs", 
                "🎯 Convergence"
            ])

            # --- Tab 1: Trajectories ---
            with tab1:
                st.caption("Visualizing the first 50 simulated paths of the underlying asset.")
                N_plot = min(50, n_paths)
                fig = go.Figure()
                # On trace les lignes
                for i in range(min(30, N_plot)):
                    fig.add_trace(go.Scatter(x=np.arange(n_steps+1), y=S_paths[i], mode="lines",
                                             line=dict(width=1, opacity=0.6), name=f"Path {i}", showlegend=False))
                
                # Ajout de la moyenne théorique 
                time_axis = np.linspace(0, T, n_steps+1)
                expected_path = S0 * np.exp(r * time_axis)
                fig.add_trace(go.Scatter(x=np.arange(n_steps+1), y=expected_path, mode="lines",
                                         line=dict(color='black', width=2, dash='dash'), name="Theoretical Expectation"))

                fig.update_layout(title="Simulated Asset Price Paths (Geometric Brownian Motion)",
                                  xaxis_title="Time Steps", yaxis_title="Price ($S_t$)",
                                  template="plotly_white", hovermode="x")
                st.plotly_chart(fig, use_container_width=True)

            # --- Tab 2: Terminal Distribution ---
            with tab2:
                st.caption(f"Distribution of the asset prices at maturity T={T} years.")
                ST = S_paths[:, -1]
                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(x=ST, nbinsx=100, name="S(T)", marker_color='#1f77b4', opacity=0.75))
                
                # Ajout ligne verticale pour le Strike
                fig2.add_vline(x=K, line_width=3, line_dash="dash", line_color="red", annotation_text="Strike K")
                
                fig2.update_layout(title="Distribution of Terminal Prices $S_T$", 
                                   xaxis_title="Price at Maturity $S_T$", yaxis_title="Frequency", 
                                   template="plotly_white")
                st.plotly_chart(fig2, use_container_width=True)

            # --- Tab 3: Payoff Distribution ---
            with tab3:
                st.caption(f"Distribution of the {option_type} option payoffs at maturity.")
                if option_type == "Call":
                    payoffs = np.maximum(ST - K, 0)
                else:
                    payoffs = np.maximum(K - ST, 0)
                
                fig3 = go.Figure()
                fig3.add_trace(go.Histogram(x=payoffs, nbinsx=100, marker_color='#2ca02c', opacity=0.75))
                fig3.update_layout(title=f"Distribution of {option_type} Payoffs", 
                                   xaxis_title="Payoff value", yaxis_title="Frequency", 
                                   template="plotly_white")
                st.plotly_chart(fig3, use_container_width=True)

            # --- Tab 4: Convergence ---
            with tab4:
                st.caption("How the price estimate stabilizes as we increase the number of simulations.")
                

                sample_points = np.linspace(100, n_paths, 50, dtype=int)
                estimates = []
                

                discount_factor = np.exp(-r * T)
                
                for m in sample_points:
                    current_payoffs = payoffs[:m]
                    estimates.append(discount_factor * current_payoffs.mean())

                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=sample_points, y=estimates, mode="lines+markers", name="Estimate"))
                
                fig4.add_hline(y=price, line_width=1, line_color="red", line_dash="dot", annotation_text="Final Price")

                fig4.update_layout(title="Monte Carlo Convergence", 
                                   xaxis_title="Number of Paths", yaxis_title="Estimated Option Price",
                                   template="plotly_white")
                st.plotly_chart(fig4, use_container_width=True)