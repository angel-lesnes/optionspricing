import streamlit as st
import numpy as np
import plotly.graph_objs as go
from pricing.black_scholes import bs_call_price, bs_put_price, implied_volatility
from app.data_fetcher import get_market_data, get_chain_for_expiration
import pandas as pd
from datetime import datetime

def render_bs():
    st.header("Underlying market data")

###############################################
    ########## CHOIX DU TICKER ###########
###############################################

    col_search, col_info = st.columns([1, 2])
    with col_search:
        ticker_input = st.text_input("Ticker (ex: AAPL, NVDA,^SPX...)", value="AAPL").upper()
        if st.button("Load data"):
            with st.spinner('Market data retrieval...'):
                data = get_market_data(ticker_input)
                if data:
                    st.session_state['market_data'] = data
                    st.session_state['current_ticker'] = ticker_input
                    st.rerun() #rechargement de la page pour affichage
                else:
                    st.error("Ticker not found. Please try another one.")

    if 'market_data' not in st.session_state:
        st.info("Enter a ticker to begin.")
        return

    data = st.session_state['market_data']

##########################################################
    ########## AFFICHAGE DONNEES SS JACENT ###########
##########################################################
    with col_info:
        st.metric("Spot Price :", f"{data['S0']:.2f} {data['currency']}")
        st.metric("Risk-free rate :", f"{data['r']:.2%}", help="Annualized yield of US Treasury bonds (10-year)")
        st.metric("Dividends :", f"{data['q']:.2%}")

    st.markdown("---")

####################################################################
    ########## SÉLECTION OPTION (T, K & option type) ###########
####################################################################

    st.subheader("Listed options")

    col_params1, col_params2, col_params3 = st.columns(3)
    
########## Maturité ##########

    with col_params1: #Maturité
        exp_dates = data['expirations']
        if not exp_dates:
            st.warning("⚠️ No option data available.")
            st.info("Try a more liquid stock ticker (e.g., AAPL, MSFT, TSLA).")
            return
            
        selected_date = st.selectbox("Maturity (expiration date)", exp_dates)
        
        #calcul du t en années
        days = (datetime.strptime(selected_date, '%Y-%m-%d') - datetime.now()).days
        T_market = max(days / 365.0, 1e-4)

########## Option type & strike ##########

    calls, puts = get_chain_for_expiration(data['ticker_obj'], selected_date) 

    with col_params2:
        option_type = st.selectbox("Type", ["Call", "Put"])
        chain_df = calls if option_type == "Call" else puts
        
        if chain_df.empty:
            st.warning("⚠️ No option data available.")
            st.info("Try another expiration date.")
            return

        strikes = chain_df['strike'].values
        idx_closest = (np.abs(strikes - data['S0'])).argmin() # strike le plus proche du spot
        selected_strike = st.selectbox("Strike (K)", strikes, index=int(idx_closest))

##########################################################
    ########## CALCUL IV (logique cascade) ###########
##########################################################

#Logique : calcul IV avec brentq par mid-price -> calcul IV par last price -> IV yfinance -> IV fallback
    
    row = chain_df[chain_df['strike'] == selected_strike].iloc[0]
    
########## Initialisation ##########

    #BS paramètres
    S0_val = data['S0']
    r_val = data['r']
    q_val = data['q']
    
    # résultats finaux
    final_sigma = np.nan
    final_price_ref = 0.0
    status_msg = ""
    source_type = ""

    # data
    mid_price = row.get('Mid_Price', 0.0)
    last_price = row.get('lastPrice', 0.0)
    yahoo_iv = row.get('impliedVolatility', 0.0)
    
########## Etape 1 : mid price calcul ##########
    if mid_price > 0:
        with st.spinner("Computing implied volatility (Mid-Price)..."):
            iv_mid = implied_volatility(
                S=S0_val, K=selected_strike, T=T_market, r=r_val, 
                price=mid_price, call_put=option_type.lower(), q=q_val
            )
        
        if not np.isnan(iv_mid):
            final_sigma = iv_mid
            final_price_ref = mid_price
            source_type = "Mid-Price : (Bid - Ask) / 2"
            status_msg = "✅ Implied volatility extracted computed (Brent Method) from Mid-Price."

########## Etape 2 : last price calcul ##########
    if np.isnan(final_sigma) and last_price > 0:
        with st.spinner("Computing implied volatility (Last Price)..."):
            iv_last = implied_volatility(
                S=S0_val, K=selected_strike, T=T_market, r=r_val, 
                price=last_price, call_put=option_type.lower(), q=q_val
            )
        
        if not np.isnan(iv_last):
            final_sigma = iv_last
            final_price_ref = last_price
            source_type = "Last Price (Yahoo Finance)"
            status_msg = "⚠️ Mid-Price invalid/missing. Implied volatility computed (Brent Method) from Last Price."
########## Etape 3 : import yfinance ##########
    if np.isnan(final_sigma):
        if yahoo_iv > 0.01 and not pd.isna(yahoo_iv):
            final_sigma = yahoo_iv
            final_price_ref = last_price 
            source_type = "Last Price (Yahoo Finance)"
            status_msg = "⚠️ Volatility compute failed (arbitrage conditions not met). Using implied volatility from Yahoo Finance."

########## Etape 4 : fallback ##########
    if np.isnan(final_sigma):
        final_sigma = 0.25 
        final_price_ref = last_price if last_price > 0 else 0.01 # éviter div par 0
        source_type = "Last Price (Yahoo Finance)"
        status_msg = "❌ Market data unusable. Arbitrary volatility (25%) used."

    # attention à bien assigner les valeurs finales
    sigma_market = final_sigma
    market_price = final_price_ref 
    
##################################################
    ########## AFFICHAGE PRIX & IV ###########
##################################################

    with col_params3:
        st.metric("Market Price", f"{market_price:.2f}")
        with st.expander("ℹ️ Price details"):
            st.markdown(f"{source_type}")
        st.metric("Implied Volatility", f"{sigma_market:.2%}")
        with st.expander("ℹ️ Volatility details"):
            st.markdown(f"{status_msg}")

######################################################
    ########## PARAMÈTRES MODIFIABLES ###########
######################################################

    st.subheader("Black-Scholes Model Parameters")
    st.caption("You can modify the values below to simulate different scenarios.")

    c1, c2, c3, c4, c5, c6 = st.columns(6) 
    with c1:
        S = st.number_input("Spot (S₀)", value=float(data['S0']))
    with c2:
        K = st.number_input("Strike (K)", value=float(selected_strike))
    with c3:
        T = st.number_input("Maturity in years (T)", value=float(T_market), format="%.4f")
    with c4:
        r = st.number_input("Risk-free rate (r)", value=float(data['r']), format="%.4f", help="The common practice is to use OIS rates")
    with c5:
        q = st.number_input("Dividend yield (q)", value=float(data['q']), format="%.4f", help="Annualized dividend yield of the underlying")
    with c6:
        sigma = st.number_input("Volatility (σ)", value=float(sigma_market), format="%.4f")

#####################################
    ########## PRICING ###########
#####################################

    if st.button("Click for pricing"):
        if option_type == "Call":
            price_theo = bs_call_price(S, K, T, r, sigma, q)
        else:
            price_theo = bs_put_price(S, K, T, r, sigma, q)
        
        st.write(f"## Theoretical price : {price_theo:.4f} {data['currency']}")
        
        diff = price_theo - market_price
        diff_percent = (price_theo - market_price) / market_price * 100 if market_price > 0.01 else 0
        st.write(f"Gap (Theory / Market) : {diff:.4f} ({diff_percent:.1f}%)") 

        st.write("### 💡 Interpretation")
        if abs(diff_percent) < 5:
             st.success(
                 f"🎯 **Perfect Match!** You are aligned with the market consensus.\n\n"
                 f"The market prices this option with a volatility of **{sigma_market:.2%}**, and you used **{sigma:.2%}**."
             ) #vert
        elif diff_percent > 0:
             st.warning(
                 f"📉 **Bearish on Volatility?** Your price is higher than the market.\n\n"
                 f"You assume a volatility of **{sigma:.2%}**, but the market is only pricing in **{sigma_market:.2%}**.\n"
                 f"*Meaning: You think the stock will move MORE than what the market expects.*"
             ) #jaune
        else:
             st.error(
                 f"📈 **Bullish on Volatility?** Your price is lower than the market.\n\n"
                 f"You assume a volatility of **{sigma:.2%}**, but the market is pricing in **{sigma_market:.2%}** (Risk Premium).\n"
                 f"*Meaning: The market is protecting itself against a bigger move than you anticipate.*"
             ) #rouge

    st.markdown("---")

#####################################
    ########## GRAPHS ###########
#####################################

    st.subheader("📊 Visual analysis")

########## data ##########
    subset = chain_df[
        (chain_df['strike'] > data['S0'] * 0.5) & 
        (chain_df['strike'] < data['S0'] * 1.5)
    ].copy() #filtrage du strike : 50% autour du spot

########## Fonction IV pour plot ##########
    def get_robust_iv_for_plot(row):
        #même système de calcul IV, mais pour chaque ligne du subset
        mid = row.get('Mid_Price', 0.0)
        last = row.get('lastPrice', 0.0)
        yahoo_iv = row.get('impliedVolatility', 0.0)
        
        if mid > 0:
            iv_mid = implied_volatility(S=S0_val, K=row['strike'], T=T_market, r=r_val, price=mid, call_put=option_type.lower(), q=q_val)
            if not np.isnan(iv_mid): return iv_mid
        if last > 0:
            iv_last = implied_volatility(S=S0_val, K=row['strike'], T=T_market, r=r_val, price=last, call_put=option_type.lower(), q=q_val)
            if not np.isnan(iv_last): return iv_last
        if yahoo_iv > 0.01 and not pd.isna(yahoo_iv):
            return yahoo_iv
        return np.nan

########## Intégration IV au df ##########
    with st.spinner("Generation of the volatility Smile (Data cleaning)..."):
        subset['Computed_IV'] = subset.apply(get_robust_iv_for_plot, axis=1)

#si valeur manquante pour le plot, interpolation linéaire
    subset['Computed_IV'] = subset['Computed_IV'].interpolate(method='linear', limit_direction='both')
    subset['Computed_IV'] = subset['Computed_IV'].fillna(0.25) # Valeur par défaut si tout est vide

########## Intégration prix BS au df ##########
    subset['BS_Price_Input'] = subset['strike'].apply(
        lambda k: bs_call_price(data['S0'], k, T, r, sigma, q) if option_type == "Call" 
        else bs_put_price(data['S0'], k, T, r, sigma, q)
    )

########## AFFICHAGE DES GRAPHS ##########
    tab1, tab2, tab3, tab4 = st.tabs(["Price gap", "Price gap (%)", "Volatility Smile", "Price and Spread Comparison"])

    # GRAPH 1 : PRIX 
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=subset['strike'], y=subset['Mid_Price'], mode='lines+markers', name='Market Price', marker=dict(color='blue', opacity=0.5)))
        fig.add_trace(go.Scatter(x=subset['strike'], y=subset['BS_Price_Input'], mode='lines', name='Black-Scholes Price', line=dict(color='red', dash='dash')))
        fig.add_vline(x=data['S0'], line_dash="dot", annotation_text="Spot")
        fig.update_layout(title=f"Comparison of the price of a {option_type} by Strike", xaxis_title="Strike", yaxis_title="Option Price")
        st.plotly_chart(fig, width='stretch')

    # GRAPH 2 : DIFF %
    with tab2:
        subset['Diff_Pct'] = (subset['BS_Price_Input'] - subset['Mid_Price']) / subset['Mid_Price']
        fig_diff = go.Figure()
        fig_diff.add_trace(go.Bar(x=subset['strike'], y=subset['Diff_Pct'], marker_color=subset['Diff_Pct'].apply(lambda x: 'red' if x < 0 else 'green'), name='Gap (%)'))
        fig_diff.add_vline(x=data['S0'], line_dash="dot", annotation_text="Spot")
        fig_diff.update_layout(yaxis_tickformat=".1%", title="Over/Underestimation of the Black-Scholes Model by Strike", xaxis_title="Strike", yaxis_title="Relative Gap (%)")
        st.plotly_chart(fig_diff, width='stretch')

    # GRAPH 3 : IV
    with tab3:
        fig_vol = go.Figure()
        
        # bleu = IV
        fig_vol.add_trace(go.Scatter(
            x=subset['strike'], 
            y=subset['Computed_IV'], 
            mode='lines+markers',
            name='Implied Volatility',
            line=dict(color='blue', shape='spline')
        ))
        
        # rouge = modèle input
        fig_vol.add_trace(go.Scatter(
            x=subset['strike'], 
            y=[sigma] * len(subset),
            mode='lines',
            name='Volatility chosen in the model',
            line=dict(color='red', dash='dash')
        ))
        fig_vol.add_vline(x=data['S0'], line_dash="dot", annotation_text="Spot")
        fig_vol.update_layout(yaxis_tickformat=".1%", title="Volatility Smile compared to the Strike", xaxis_title="Strike", yaxis_title="Implied Volatility")
        st.plotly_chart(fig_vol, width='stretch')

    with tab4:
        st.caption("Are you within the spread? The gray area represents the Bid-Ask spread (Liquidity). If your Red Line is inside, your price is realistic.")
        
        fig = go.Figure()

        # 1. Zone Bid-Ask (Le Tunnel)
        # On doit gérer les Bid/Ask à 0 qui cassent le graph
        subset['ask_clean'] = subset['ask'].replace(0, np.nan).fillna(subset['lastPrice'])
        subset['bid_clean'] = subset['bid'].replace(0, np.nan).fillna(subset['lastPrice'])

        fig.add_trace(go.Scatter(
            x=subset['strike'], y=subset['ask_clean'],
            mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=subset['strike'], y=subset['bid_clean'],
            mode='lines', fill='tonexty', fillcolor='rgba(0, 100, 255, 0.2)', line=dict(width=0),
            name='Bid-Ask Spread'
        ))

        # 2. Mid Price
        fig.add_trace(go.Scatter(
            x=subset['strike'], y=subset['Mid_Price'],
            mode='markers', name='Market Mid-Price', marker=dict(color='blue', size=4)
        ))

        # 3. Ton Modèle
        fig.add_trace(go.Scatter(
            x=subset['strike'], y=subset['BS_Price_Input'],
            mode='lines', name=f'Your Model (σ={sigma:.1%})', line=dict(color='red', width=2)
        ))

        fig.add_vline(x=data['S0'], line_dash="dot", annotation_text="Spot")
        fig.update_layout(title="Can you trade this price?", xaxis_title="Strike", yaxis_title="Option Price")
        st.plotly_chart(fig, width='stretch')