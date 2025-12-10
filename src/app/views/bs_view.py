import streamlit as st
import numpy as np
import plotly.graph_objs as go
from pricing.black_scholes import bs_call_price, bs_put_price

def render_bs():

    st.header("Black-Scholes Pricing")
    
   # inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        S0 = st.number_input("Prix spot S₀", value=100.0)
        K = st.number_input("Strike K", value=100.0)

    with col2:
        T = st.number_input("Maturité T (années)", value=1.0)
        r = st.number_input("Taux sans risque r", value=0.05)

    with col3:
        sigma = st.number_input("Volatilité σ", value=0.2, step=0.01)
        option_type = st.selectbox("Type d’option", ["Call", "Put"])

    st.markdown("---")

    # bouton 
    if st.button("Calculer"):
        if option_type == "Call":
            price = bs_call_price(S0, K, T, r, sigma)
        else:
            price = bs_put_price(S0, K, T, r, sigma)

        # Sauvegarde dans session state = sauvegarde les datas
        st.session_state["bs_result"] = {
            "S0": S0,
            "K": K,
            "T": T,
            "r": r,
            "sigma": sigma,
            "option_type": option_type,
            "price": price,
        }

    if "bs_result" not in st.session_state:
        return

    res = st.session_state["bs_result"]
    price = res["price"]

    st.success(f"Prix {res['option_type']} : **{price:.4f}**")

    choice = st.radio("Afficher :", ("Payoff", "P&L (buyer)", "P&L (seller)")) # choix du graph

    S_range = np.linspace(0.5 * res["S0"], 1.5 * res["S0"], 400)

    #Payoff
    if res["option_type"] == "Call":
        payoff = np.maximum(S_range - res["K"], 0)
    else:
        payoff = np.maximum(res["K"] - S_range, 0)

    #Choix du graph
    if choice == "Payoff":
        y = payoff
        y_title = "Payoff"
    elif choice == "P&L (buyer)":
        y = payoff - price
        y_title = "P&L Acheteur"
    else:
        y = price - payoff
        y_title = "P&L Vendeur"

    # Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=y, mode="lines", line=dict(width=2)))
    fig.update_layout(
        title=y_title,
        xaxis_title="S(T)",
        yaxis_title=y_title,
        template="plotly_white",
    )

    st.plotly_chart(fig, width="stretch")

    # Stats P&L
    if choice != "Payoff":
        st.write("## Statistiques")
        st.write(f"Min: {y.min():.4f} | Max: {y.max():.4f} | Mean: {y.mean():.4f}")