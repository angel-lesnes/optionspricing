import streamlit as st

st.set_page_config(
    page_title="Option Pricing",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Option Pricing Simulator")

st.markdown("""
Bienvenue dans ton outil de pricing d’options !  
Choisis une méthode dans le menu à gauche :
- **Black-Scholes**
- **Monte Carlo**

🌟 Le site est en développement, plus de fonctionnalités arrivent bientôt !
""")