import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from pricing.black_scholes import (
    bs_call_price, bs_put_price,
    bs_call_delta, bs_put_delta
)

st.title("📘 Pricing Black-Scholes")

st.markdown("### Paramètres du modèle")

col1, col2, col3 = st.columns(3)

with col1:
    S = st.number_input("Prix spot S₀", value=100.0)
    K = st.number_input("Strike K", value=100.0)

with col2:
    T = st.number_input("Maturité T (années)", value=1.0)
    r = st.number_input("Taux sans risque r", value=0.05)

with col3:
    sigma = st.number_input("Volatilité σ", value=0.2)
    option_type = st.selectbox("Type d’option", ["Call", "Put"])

st.markdown("---")

if st.button("🔍 Calculer le prix"):
    if option_type == "Call":
        price = bs_call_price(S, K, T, r, sigma)
        delta = bs_call_delta(S, K, T, r, sigma)
    else:
        price = bs_put_price(S, K, T, r, sigma)
        delta = bs_put_delta(S, K, T, r, sigma)

    st.success(f"**Prix {option_type} : {price:.4f}**")
    st.info(f"**Delta : {delta:.4f}**")

    # ------------ Graphique de payoff ------------
    st.markdown("### 📈 Payoff à maturité")

    S_range = np.linspace(0.5 * S, 1.5 * S, 200)

    if option_type == "Call":
        payoff = np.maximum(S_range - K, 0)
    else:
        payoff = np.maximum(K - S_range, 0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(S_range, payoff)
    ax.set_xlabel("Prix de l’actif à maturité Sₜ")
    ax.set_ylabel("Payoff")
    ax.grid(True)

    st.pyplot(fig)