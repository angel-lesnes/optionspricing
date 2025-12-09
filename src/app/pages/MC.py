import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from pricing.monte_carlo import (
    mc_paths_gbm, mc_price_call, mc_price_put
)

st.title("🎲 Pricing Monte Carlo")

st.markdown("### Paramètres du modèle")

col1, col2, col3 = st.columns(3)

with col1:
    S0 = st.number_input("Prix spot S₀", value=100.0)
    K = st.number_input("Strike K", value=100.0)

with col2:
    T = st.number_input("Maturité T (années)", value=1.0)
    r = st.number_input("Taux sans risque r", value=0.05)

with col3:
    sigma = st.number_input("Volatilité σ", value=0.2)
    option_type = st.selectbox("Type d’option", ["Call", "Put"])

n_paths = st.slider("Nombre de chemins Monte Carlo", 1000, 50000, 5000)
n_steps = st.slider("Pas de temps (par an)", 50, 365, 252)

st.markdown("### Options avancées")

colA, colB = st.columns(2)

with colA:
    antithetic = st.checkbox("Antithetic variates", value=True,
         help="Réduit la variance en utilisant des chemins opposés")

with colB:
    control = st.checkbox("Control variate", value=True,
         help="Réduit la variance via E[S_T] analytique")

st.markdown("---")

if st.button("🚀 Lancer la simulation"):
    S_paths = mc_paths_gbm(S0, T, r, sigma, n_paths, n_steps, antithetic)

    if option_type == "Call":
        price = mc_price_call(S_paths, K, r, T, control)
    else:
        price = mc_price_put(S_paths, K, r, T, control)

    st.success(f"**Prix {option_type} : {price:.4f}**")

    # ---- Graph des 30 premières trajectoires ----    
    st.markdown("### 📉 Quelques trajectoires simulées")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(S_paths[:30].T)
    ax.set_xlabel("Temps")
    ax.set_ylabel("Prix simulé")
    ax.grid(True)

    st.pyplot(fig)