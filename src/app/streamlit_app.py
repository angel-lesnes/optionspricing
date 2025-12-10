import streamlit as st

# config
st.set_page_config(page_title="Option Pricing Simulator", page_icon="📈", layout="wide")

# style léger (optionnel) - petit padding
st.markdown(
    """
    <style>
      .main { padding-left: 2rem; padding-right: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----- SIDEBAR -----
with st.sidebar:
    #st.image="lien"
    st.header("Method Selection")
    method = st.selectbox("Choose a pricing method:", ["Black-Scholes", "Monte Carlo"])
    st.markdown("---")

    #disclaimer
    st.caption("This application is for educational and informational purposes only. "
               "It does not constitute financial, investment, or trading advice.")

# ----- MAIN -----
st.title("Option Pricing Simulator")

st.write(
    """
    Bienvenue dans l'outil de pricing d'options — sélectionne une méthode dans la colonne de gauche.
    """
)

if method == "Black-Scholes":
    from app.views.bs_view import render_bs 
    render_bs()
elif method == "Monte Carlo":
    from app.views.mc_view import render_mc
    render_mc()