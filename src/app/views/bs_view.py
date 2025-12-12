import streamlit as st
import numpy as np
import plotly.graph_objs as go
from pricing.black_scholes import bs_call_price, bs_put_price, implied_volatility
from app.data_fetcher import get_market_data, get_chain_for_expiration
import pandas as pd

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
                    st.rerun() #rechargement de la page pour affichage
                else:
                    st.error("Ticker introuvable.")

    if 'market_data' not in st.session_state:
        st.info("Entrez un ticker pour commencer.")
        return

    data = st.session_state['market_data']

##########################################################
    ########## AFFICHAGE DONNEES SS JACENT ###########
##########################################################
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
    
########## Maturité ##########

    with col_params1: 
        exp_dates = data['expirations']
        selected_date = st.selectbox("Maturité (Expiration)", exp_dates)
        #si absence de données d'option :
        if not exp_dates or len(exp_dates) == 0:
            st.warning(f"⚠️ Aucune donnée d'option disponible pour le ticker **{st.session_state['current_ticker']}** sur Yahoo Finance.")
            st.info("Essayez un ticker d'action liquide américaine (ex: AAPL, MSFT, TSLA) ou européenne (ex: AIR.PA).")
            return
        else :
        #calcul du t en années
            from datetime import datetime
            days = (datetime.strptime(selected_date, '%Y-%m-%d') - datetime.now()).days
            T_market = max(days / 365.0, 1e-4)

    calls, puts = get_chain_for_expiration(data['ticker_obj'], selected_date) #récupération de la chaine

########## Strike ##########
    
    with col_params2:
        option_type = st.selectbox("Type", ["Call", "Put"])
        chain_df = calls if option_type == "Call" else puts
        strikes = chain_df['strike'].values

        idx_closest = (np.abs(strikes - data['S0'])).argmin() # strike le plus proche du spot
        selected_strike = st.selectbox("Strike (K)", strikes, index=int(idx_closest))

#########################################################################################
    ########## IV : Récupération si >0 sinon calcul implicite avec brentq ###########
#########################################################################################

    # fixation à la ligne du strike sélectionné
    row = chain_df[chain_df['strike'] == selected_strike].iloc[0]

    # prix de référence pour IV : Mid-Price si >0 sinon lastPrice
    price_reference = row['Mid_Price'] 
    if price_reference <= 0:
        price_reference = row['lastPrice'] # Fallback

    # récupération IV
    sigma_market = row.get('impliedVolatility', 0.0)

    if sigma_market < 0.01 or pd.isna(sigma_market):
        with st.spinner("Recalcul de la volatilité implicite (données Yahoo invalides)..."):
            calculated_iv = implied_volatility(
                S=data['S0'],
                K=selected_strike,
                T=T_market,
                r=data['r'],
                price=price_reference,
                call_put=option_type.lower(),
                q=data.get('q', 0.0)
            )
            
            if not np.isnan(calculated_iv):
                sigma_market = calculated_iv
                st.info(f"💡 Volatilité recalculée basée sur le Mid-Price ({price_reference:.2f}) : {sigma_market:.2%}")
            else:
                st.warning("⚠️ Impossible de calculer la volatilité implicite (Prix incohérent avec les bornes d'arbitrage).")
                sigma_market = 0.2 # Valeur par défaut de secours

    market_price = row['lastPrice']

########## Affichage de l'IV ##########

    with col_params3:
        st.metric("Prix Marché (Last)", f"{market_price:.2f}")
        st.metric("Volatilité Implicite (Market)", f"{sigma_market:.2%}")

######################################################
    ########## PARAMÈTRES MODIFIABLES ###########
######################################################

    st.subheader("Paramètres du Modèle")
    st.caption("Vous pouvez modifier les valeurs ci-dessous pour simuler des scénarios.")

    c1, c2, c3, c4, c5, c6 = st.columns(6) 
    with c1:
        S = st.number_input("Spot S₀", value=float(data['S0']))
    with c2:
        K = st.number_input("Strike K", value=float(selected_strike))
    with c3:
        T = st.number_input("Maturité T (ans)", value=float(T_market), format="%.4f")
    with c4:
        r = st.number_input("Taux r", value=float(data['r']), format="%.4f", help="Taux sans risque")
    with c5:
        default_q = float(data.get('q', 0.0)) #prendre q des datas sinon 0
        q = st.number_input("Dividende q", value=default_q, format="%.4f", help="Rendement du dividende annualisé (ex: 0.03 pour 3%)")
    with c6:
        sigma = st.number_input("Volatilité σ", value=float(sigma_market), format="%.4f")


#####################################
    ########## CALCUL ###########
#####################################

    if st.button("Pricer"):
        if option_type == "Call":
            price_theo = bs_call_price(S, K, T, r, sigma, q)
        else:
            price_theo = bs_put_price(S, K, T, r, sigma, q)
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

#####################################
    ########## GRAPHS ###########
#####################################


    st.subheader("📊 Analyse Visuelle : Théorie vs Marché")

    chain_df = calls if option_type == "Call" else puts #df des options
    subset = chain_df[
        (chain_df['strike'] > data['S0'] * 0.6) & 
        (chain_df['strike'] < data['S0'] * 1.4)&
        (chain_df['lastPrice'] > 0.01)
    ].copy() #filtrage du strike pour éviter valeurs aberrantes

########## Calculs ##########

    # Calcul des prix BS pour chaque strike
    subset['BS_Price'] = subset['strike'].apply(
        lambda k: bs_call_price(data['S0'], k, T, r, sigma, q) if option_type == "Call" 
        else bs_put_price(data['S0'], k, T, r, sigma, q)
    )

    # (model - market) / market
    subset['Diff_Pct'] = (subset['BS_Price'] - subset['lastPrice']) / subset['lastPrice']

########## Affichage ##########

    tab1, tab2, tab3 = st.tabs(["Écart de Prix", "Écart de Prix (%)", "Smile de Volatilité"])

    with tab1:

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

    with tab2:
        st.caption("Ce graphique montre de combien (%) votre modèle est plus cher ou moins cher que le marché.")
        
        fig_diff = go.Figure()
        fig_diff.add_trace(go.Bar(
            x=subset['strike'], 
            y=subset['Diff_Pct'],
            marker_color=subset['Diff_Pct'].apply(lambda x: 'red' if x < 0 else 'green'),
            name='Écart %'
        ))
        
        fig_diff.add_vline(x=S, line_dash="dot", annotation_text="Spot", line_color="black")
        
        fig_diff.update_layout(
            title="Sur/Sous-évaluation du Modèle par Strike",
            xaxis_title="Strike (K)",
            yaxis_title="Écart relatif (Model vs Market)",
            yaxis_tickformat=".1%",
            template="plotly_white"
        )
        st.plotly_chart(fig_diff, width='stretch')

    with tab3:
        st.caption("Comparez votre volatilité constante (ligne rouge) à la réalité du marché (points bleus).")
        
        if 'impliedVolatility' in subset.columns:
            fig_vol = go.Figure()
            
            #IV marché
            fig_vol.add_trace(go.Scatter(
                x=subset['strike'], 
                y=subset['impliedVolatility'],
                mode='lines+markers',
                name='Volatilité Implicite (Marché)',
                line=dict(shape='spline', smoothing=1.3) # Lisser un peu la courbe
            ))
            #volatilité utilisateur
            fig_vol.add_trace(go.Scatter(
                x=subset['strike'], 
                y=[sigma] * len(subset),
                mode='lines',
                name='Votre Volatilité (Input)',
                line=dict(color='red', dash='dash')
            ))
            
            fig_vol.add_vline(x=S, line_dash="dot", annotation_text="Spot", line_color="black")

            fig_vol.update_layout(
                title="Smile de Volatilité",
                xaxis_title="Strike (K)",
                yaxis_title="Volatilité (σ)",
                yaxis_tickformat=".1%",
                template="plotly_white"
            )
            st.plotly_chart(fig_vol, width='stretch')
        else:
            st.warning("Données de volatilité implicite non disponibles pour ce ticker.")

   