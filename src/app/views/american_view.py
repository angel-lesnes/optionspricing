import streamlit as st
import numpy as np
import plotly.graph_objs as go
from app.data_fetcher import get_market_data, get_chain_for_expiration
# On importe le nouveau pricer
from pricing.binomial import binomial_option_pricing
# On garde BS pour comparer ou pour l'IV (car l'IV est toujours calculée via BS par convention)
from pricing.black_scholes import implied_volatility, bs_call_price, bs_put_price 
import pandas as pd
from datetime import datetime

def render_american():
    st.header("US Option Pricing (Binomial Tree)")

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

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Binomial Tree Settings")
    steps_N = st.sidebar.slider("Number of Steps (N)", min_value=10, max_value=2000, value=200, step=10, 
                                help="More steps = Higher precision but slower calculation.")
    
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
        st.metric("Market Price", f"{market_price:.2f} $")
        with st.expander("ℹ️ Price details"):
            st.markdown(f"{source_type}")
        st.metric("Implied Volatility", f"{sigma_market:.2%}")
        with st.expander("ℹ️ Volatility details"):
            st.markdown(f"{status_msg}")

######################################################
    ########## PARAMÈTRES MODIFIABLES ###########
######################################################

    st.subheader("Binomial tree Parameters")
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
        
        # 1. Calcul du Prix Américain (Binomial)
        with st.spinner(f"Computing Binomial Tree with N={steps_N}..."):
            price_american = binomial_option_pricing(
                S, K, T, r, sigma, q, 
                N=steps_N, 
                option_type=option_type.lower(), 
                american=True
            )

        # 2. Calcul du Prix Européen (BS) pour comparer (Valeur pédagogique !)
        if option_type == 'Call':
            price_european = bs_call_price(S, K, T, r, sigma, q)
        else:
            price_european = bs_put_price(S, K, T, r, sigma, q)

        # Affichage
        st.write(f"## 🇺🇸 American Price : {price_american:.4f} {data['currency']}")
        
        # Comparaison intéressante
        premium = price_american - price_european
        if premium > 0.005:
            st.info(f"💡 **Early Exercise Premium:** This option is worth **{premium:.4f}** more than its European equivalent. "
                    f"This confirms that the possibility of early exercise has value here (typically for Puts or Calls with high dividends).")
        else:
            st.info("💡 **No Early Exercise Premium:** The American price is identical to the European price. "
                    "Optimally, you would likely hold this option until maturity.")

        # Calcul de l'écart avec le marché
        diff = price_american - market_price
        diff_percent = (price_american - market_price) / market_price * 100 if market_price > 0.01 else 0
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("Price Gap (Model - Market)", f"{diff:+.2f} $ ({diff_percent:+.1f}%)")

        with col_res2:
             sigma_diff = sigma - sigma_market
             st.metric("Volatility Gap", f"{sigma_diff*100:+.2f} %")

        with st.expander("💡 Click to analyze the gap"):

            # Détection précise des changements
            params_changed = []
            if abs(data['S0'] - S) > 0.01: params_changed.append("Spot Price")
            if abs(data['r'] - r) > 0.001: params_changed.append("Risk-free Rate")
            if abs(data['q'] - q) > 0.001: params_changed.append("Dividend Yield")
            if abs(selected_strike - K) > 0.01: params_changed.append("Strike") 
            if abs(T_market - T) > 0.001: params_changed.append("Maturity")
            if abs(sigma_diff) > 0.001: params_changed.append("Volatility")

            if params_changed:
                st.info(f"ℹ️ **Simulation Mode :** You have modified: **{', '.join(params_changed)}**.")
                st.write(f"You are comparing a **theoretical option** with custom parameters against the **real market option**.")
                
                if diff > 0:
                    st.write(f"👉 Your simulation results in a price **${diff:.2f} higher** than the current market price.")
                else:
                    st.write(f"👉 Your simulation results in a price **${abs(diff):.2f} lower** than the current market price.")
            
            if abs(sigma_diff) > 0.01 :
                st.info(f"ℹ️ **Volatility Analysis :** You are using your own volatility input.")
                if sigma_diff > 0:
                     st.warning(
                         f"📈 **You are bullish on volatility** - You assume a higher volatility than the market.\n\n"
                         f"*Meaning : You think the stock will move MORE than what the market expects.*"
                     )
                else:
                     st.error(
                         f"📉 **You are bearish on volatility** - You assume a lower volatility than the market.\n\n"
                         f"*Meaning : The market is protecting itself against a bigger move than you anticipate.*"
                     )
            if not params_changed and abs(sigma_diff) < 0.001:
                st.info(f"ℹ️ You are using the exact same parameters as the market option. Your theoretical price matches perfectly the market price.")
    st.markdown("---")