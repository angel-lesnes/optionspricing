import streamlit as st
import numpy as np
import plotly.graph_objs as go
from pricing.black_scholes import bs_call_price, bs_put_price
from app.data_fetcher import get_market_data, get_chain_for_expiration

def render_bs():
    st.header("Données de marché du sous-jacent")

###############################################
    ########## CHOIX DU TICKER ###########
###############################################

    col_search, col_info = st.columns([1, 2])
    with col_search:
        ticker_input = st.text_input("Ticker (ex: AAPL, NVDA,^SPX)", value="AAPL").upper()
        if st.button("Charger données"):
            with st.spinner('Récupération des données marché...'):
                data = get_market_data(ticker_input)
                if data:
                    st.session_state['market_data'] = data
                    st.session_state['current_ticker'] = ticker_input
                    st.rerun() # Recharger la page pour afficher la suite
                else:
                    st.error("Ticker introuvable.")

    # Si pas de données chargées, on s'arrête là
    if 'market_data' not in st.session_state:
        st.info("Entrez un ticker pour commencer.")
        return

    data = st.session_state['market_data']

    # Affichage Infos
    with col_info:
        st.metric("Spot :", f"{data['S0']:.2f} {data['currency']}")
        st.metric("Taux :", f"{data['r']:.2%}", help="Taux de rendement annualisé des bons du trésor à 10 ans")
        st.metric("Dividendes :", f"{data['q']:.2%}")

    st.markdown("---")

    st.subheader("Options côtées")

###################################################################
    ########## SÉLECTION OPTION (Maturité & Strike) ###########
###################################################################

    col_params1, col_params2, col_params3 = st.columns(3)
    
    with col_params1: #Maturité
        exp_dates = data['expirations']
        selected_date = st.selectbox("Maturité (Expiration)", exp_dates)
        # Warning si pas de données d'option
        if not exp_dates or len(exp_dates) == 0:
            st.warning(f"⚠️ Aucune donnée d'option disponible pour le ticker **{st.session_state['current_ticker']}** sur Yahoo Finance.")
            st.info("Essayez un ticker d'action liquide américaine (ex: AAPL, MSFT, TSLA) ou européenne (ex: AIR.PA).")
            return
        else :
        #Calcul du T en années
            from datetime import datetime
            days = (datetime.strptime(selected_date, '%Y-%m-%d') - datetime.now()).days
            T_market = max(days / 365.0, 1e-4)

    calls, puts = get_chain_for_expiration(data['ticker_obj'], selected_date) #récupération de la chaine
    
    with col_params2:
        option_type = st.selectbox("Type", ["Call", "Put"])
        chain_df = calls if option_type == "Call" else puts
        strikes = chain_df['strike'].values

        idx_closest = (np.abs(strikes - data['S0'])).argmin() # strike le plus proche du spot
        selected_strike = st.selectbox("Strike (K)", strikes, index=int(idx_closest))

    # récupération de la IV du marché (colonne 'impliedVolatility' de yfinance)) et affiche le prix marché
    row = chain_df[chain_df['strike'] == selected_strike].iloc[0]
    sigma_market = row['impliedVolatility']
    market_price = row['lastPrice']

    with col_params3:
        st.metric("Prix Marché (Last)", f"{market_price:.2f}")
        st.metric("Volatilité Implicite (Market)", f"{sigma_market:.2%}")

    st.subheader("Paramètres du Modèle")
    st.caption("Vous pouvez modifier les valeurs ci-dessous pour simuler des scénarios.")

######################################################
    ########## PARAMÈTRES MODIFIABLES ###########
######################################################

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        S = st.number_input("Spot S₀", value=float(data['S0']))
    with c2:
        K = st.number_input("Strike K", value=float(selected_strike))
    with c3:
        T = st.number_input("Maturité T (ans)", value=float(T_market), format="%.4f")
    with c4:
        r = st.number_input("Taux r", value=float(data['r']), format="%.4f")
    with c5:
        sigma = st.number_input("Volatilité σ", value=float(sigma_market), format="%.4f")


#####################################
    ########## CALCUL ###########
#####################################

    if st.button("Pricer avec ces paramètres"):
        price_theo = bs_call_price(S, K, T, r, sigma) if option_type == "Call" else bs_put_price(S, K, T, r, sigma)
        st.write(f"## Prix Théorique : {price_theo:.4f} {data['currency']}")
        

        diff = price_theo - market_price #diff avec marché
        st.write(f"Écart vs Marché : {diff:.4f} ({(diff/market_price)*100:.1f}%)")  
        diff_percent = (price_theo - market_price) / market_price * 100

        st.write("### 💡 Interprétation de l'écart") #message d'interprétation

        if abs(diff_percent) < 5:
             st.success("Votre modèle est très proche du marché ! La volatilité utilisée est cohérente.") #vert
        elif diff_percent > 0:
            st.warning(f"Votre modèle est plus cher que le marché (+{diff_percent:.1f}%). "
               f"Cela suggère que la volatilité implicite réelle pour ce strike est inférieure à {sigma:.2%}, " #jaune
               "ou que le marché anticipe moins de dividendes/risques.")
        else:
             st.error(f"Votre modèle est moins cher que le marché ({diff_percent:.1f}%). "
                f"Le marché 'price' une volatilité plus forte (Smile de volatilité) ou un risque d'événement.") #rouge

    st.markdown("---")
    st.subheader("📊 Analyse Visuelle : Théorie vs Marché")

    chain_df = calls if option_type == "Call" else puts #df des options
    subset = chain_df[
        (chain_df['strike'] > data['S0'] * 0.6) & 
        (chain_df['strike'] < data['S0'] * 1.4)
    ].copy() #filtrage du strike pour éviter valeurs aberrantes

    # Calcul des prix BS pour chaque strike
    subset['BS_Price'] = subset['strike'].apply(
        lambda k: bs_call_price(data['S0'], k, T, r, sigma) if option_type == "Call" 
        else bs_put_price(data['S0'], k, T, r, sigma)
    )
    # Graphique Interactif
    fig = go.Figure()

    # 1. --> Courbe Marché
    fig.add_trace(go.Scatter(
        x=subset['strike'], y=subset['lastPrice'],
        mode='lines+markers', name='Prix Marché',
        line=dict(color='blue')
    ))

    # 2. --> Courbe BS
    fig.add_trace(go.Scatter(
        x=subset['strike'], y=subset['BS_Price'],
        mode='lines', name='Prix Black-Scholes (Simulé)',
        line=dict(color='red', dash='dash')
    ))
    fig.add_vline(x=data['S0'], line_dash="dot", annotation_text="Spot Actuel", annotation_position="top left")

    fig.update_layout(
        title=f"Comparaison Prix {option_type} : Modèle vs Réalité (Maturité {selected_date})",
        xaxis_title="Strike (K)",
        yaxis_title="Prix de l'Option",
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(fig, width='stretch')