import streamlit as st

##################################################
    ########## CONFIG DE LA PAGE ###########
##################################################

st.set_page_config(
    page_title="Option Pricing Simulator",
    page_icon="src/app/images/Icon.png", 
    layout="wide",
    initial_sidebar_state="expanded"
)

##################################################
    ########## PRESENTATION CSS ###########
##################################################

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 1rem; text-align: center; width: 100%;}
    .sub-header {font-size: 1.5rem; font-weight: 600; color: #4B5563; margin-bottom: 1.5rem; text-align: center; width: 100%;}
    .info-text {font-size: 1.1rem; color: #374151; line-height: 1.6;}
    .link-button {
        display: inline-block;
        padding: 0.5em 1em;
        color: #FFFFFF;
        background-color: #2563EB;
        border-radius: 5px;
        text-decoration: none;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }
    .link-button:hover {background-color: #1D4ED8;}
    .github-icon {color: #333; text-align: center; width: 100%;}
    .linkedin-icon {color: #0077b5;}
    .disclaimer-box {background-color: #FEF3C7; padding: 1rem; border-radius: 8px; border-left: 5px solid #F59E0B; border-right: 5px solid #F59E0B; margin: 2rem 0;}
</style>
""", unsafe_allow_html=True)

########################################
    ########## SIDEBAR ###########
########################################

with st.sidebar:
    st.image("src/app/images/3D volat surface.png", width='stretch')
    st.markdown("---")
    st.header("Navigation")
    # Le selectbox est le seul élément de navigation
    selected_page = st.selectbox(
        "Choose a section:",
        ["🏠 Home", "———————————", "📜 Black-Scholes", "🌳 Binomial Tree", "🎲 Monte Carlo"],
        index=0
    )

#####################################
    ########## PAGES ###########
#####################################

########## ACCUEIL ##########

if selected_page == "🏠 Home" or selected_page == "---":
    st.markdown('<div class="main-header"> Option Pricing Simulator</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header">Created by Lesnes Angel</div>', unsafe_allow_html=True)
    
### Réseaux ###

    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; gap: 20px; margin-bottom: 20px;">
        <a href="https://www.linkedin.com/in/angel-lesnes-7714b6386" target="_blank" style="text-decoration: none;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" width="36" height="36" alt="LinkedIn"/>
        </a>
        <a href="https://github.com/angel-lesnes" target="_blank" style="text-decoration: none;">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="50" height="50" alt="GitHub"/>
        </a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

### Contenu ###
    col_main, col_side = st.columns([3, 2])

    with col_main:
        st.markdown('<div class="info-text">Welcome to this interactive tool designed to understand, simulate and compare different option pricing models against real market data.</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="disclaimer-box">
            <strong>⚠️ Educational purpose only :</strong><br>
            This application is developed for educational and informational purposes.<br>
            The data, calculations, and models provided <strong>do not constitute financial, investment, or trading advice.</strong><br>
            Always do your own research or consult a certified financial professional before making investment decisions.
        </div>
        """, unsafe_allow_html=True)

        st.info("👈 **To get started, please select a pricing method from the menu in the sidebar.**")

    with col_side:
        st.markdown("### 👨‍💻 Future Developments")
        st.markdown("""
        This project is actively evolving. Planned features include :
        * 🕹️ **Monte Carlo Simulations :** For pricing path-dependent exotic options and visualizing price scenarios.
        * 🛡️ **Hedging Simulator :** A module to simulate the P&L of a dynamic hedging strategy over time.
        * 📊 **Advanced Volatility Analysis :** Implied Volatility Surface visualization and local volatility modeling.
        * ⚡ **Performance Optimization :** Using Numba/Cython for faster calculations.
        """)

    st.markdown("---")
    st.caption("© 2025 Lesnes Angel. All rights reserved.")


### Chargement des pages ###

elif selected_page == "📜 Black-Scholes":
    with st.spinner("Please wait, the engine is loading..."):
        from app.views.bs_view import render_bs 
        render_bs()
elif selected_page == "🌳 Binomial Tree":
    with st.spinner("Please wait, the engine is loading..."):
        from app.views.bino_view import render_american
        render_american()
elif selected_page == "🎲 Monte Carlo":
    with st.spinner("Please wait, the engine is loading..."):
        try:
            from app.views.mc_view import render_mc
            render_mc()
        except ImportError:
            st.error("🚧 The Monte Carlo module is currently under construction. Please check back later !")
            #si j'ai des modifications à faire qui prennent du temps : afficher ce message avant de mettre version adaptée à l'utilisateur