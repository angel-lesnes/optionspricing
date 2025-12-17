import streamlit as st
import numpy as np
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime
from pricing.black_scholes import bs_call_price, bs_put_price, implied_volatility
from app.data_fetcher import get_market_data, get_chain_for_expiration
from pricing.greeks import calculate_greeks

def render_bs():
    st.header("Underlying market data")

###############################################
    ########## CHOIX DU TICKER ###########
###############################################

    col_search, col_info = st.columns([1, 2])
    with col_search:
        ticker_input = st.text_input("Ticker (ex : AAPL, NVDA,^SPX...)", value="AAPL").upper()
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

    with col_params1:
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

    # /!\ assigner les valeurs finales (à retenir pr prochains codes) /!\
    sigma_market = final_sigma
    market_price = final_price_ref 
    
##################################################
    ########## AFFICHAGE PRIX & IV ###########
##################################################

    with col_params3:
        st.metric("Market Price", f"{market_price:.4f} $")
        with st.expander("ℹ️ Price details"):
            st.markdown(f"{source_type}")
        st.metric("Implied Volatility", f"{sigma_market:.2%}")
        with st.expander("ℹ️ Volatility details"):
            st.markdown(f"{status_msg}")

######################################################
    ########## PARAMÈTRES MODIFIABLES ###########
######################################################

    st.subheader("⚙️ Black-Scholes Model Parameters")
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

########## Bouton + affichage du prix ##########

    if st.button("Click for pricing"):
        if option_type == "Call":
            price_theo = bs_call_price(S, K, T, r, sigma, q)
        else:
            price_theo = bs_put_price(S, K, T, r, sigma, q)
        
        st.write(f"## Theoretical price : {price_theo:.4f} {data['currency']}")

########## Affichage gap prix & volat ##########
        
        diff = price_theo - market_price
        diff_percent = (price_theo - market_price) / market_price * 100 if market_price > 0.01 else 0
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("Price Gap", f"{diff:+.4f} $ ({diff_percent:+.2f}%)")
        with col_res2:
             sigma_diff = sigma - sigma_market
             st.metric("Volatility Gap", f"{sigma_diff*100:+.2f} %")

########## Détection des changements ##########

        with st.expander("💡 Click to analyze the gap"):

            params_changed = []
            if abs(data['S0'] - S) > 0.01: params_changed.append("Spot Price")
            if abs(data['r'] - r) > 0.001: params_changed.append("Risk-free Rate")
            if abs(data['q'] - q) > 0.001: params_changed.append("Dividend Yield")
            if abs(selected_strike - K) > 0.01: params_changed.append("Strike") 
            if abs(T_market - T) > 0.001: params_changed.append("Maturity")
            if abs(sigma_diff) > 0.001: params_changed.append("Volatility")

########## Interprétation ##########

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

################################################################
    ########## OUTILS D'ANALYSE (Graphs, Greeks) ###########
################################################################

    st.subheader("📊 Analysis tools")
    st.caption(
    f"⚠️ Volatility used to compute **model price** for each strike = **your input**.\n\n "
    f"⚠️ Volatility used to compute **market price** = **implied volatility**.")

########## création df avec strikes ##########
    subset = chain_df[
        (chain_df['strike'] > data['S0'] * 0.5) & 
        (chain_df['strike'] < data['S0'] * 1.5)
    ].copy() #filtrage : 50% autour du spot

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
    subset['Computed_IV'] = subset.apply(get_robust_iv_for_plot, axis=1)

#si valeur manquante pour le plot, interpolation linéaire ou fallback de 0.25
    subset['Computed_IV'] = subset['Computed_IV'].interpolate(method='linear', limit_direction='both').fillna(0.25) 

########## Intégration prix BS au df ##########
    subset['BS_Price_Graph'] = subset['strike'].apply(
        lambda k: bs_call_price(S, k, T, r, sigma, q) if option_type == "Call" 
        else bs_put_price(S, k, T, r, sigma, q)
    )
                                    ##########################################
                                    ########## AFFICHAGE DES GRAPHS ##########
                                    ##########################################
    tab1, tab2, tab3 = st.tabs(["Price gap", "Volatility Smile", "Greeks"])

                                ########## Onglet 1 : Analyse prix ##########
    with tab1:
        
### Graph 1 : comparaison prix BS et marché ###
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=subset['strike'], y=subset['Mid_Price'], 
            mode='lines+markers', name='Market Price', 
            marker=dict(color='blue', opacity=0.6)
        ))
        fig_price.add_trace(go.Scatter(
            x=subset['strike'], y=subset['BS_Price_Graph'], 
            mode='lines', name='Black-Scholes (Model)', 
            line=dict(color='red', dash='dash')
        ))
        fig_price.add_vline(x=S, line_dash="dot", annotation_text="Spot", annotation_position="top left")
        fig_price.update_layout(title=f"Price Curve: Market vs Model", xaxis_title="Strike", yaxis_title="Price ($)")
        st.plotly_chart(fig_price, width='stretch')

### Outil sélection $/% pour graph 2 ###

        gap_unit = st.radio(
            "Select Gap Unit:", 
            ["Absolute Value ($)", "Percentage (%)"], 
            horizontal=True,
            key="bs_gap_unit" # Clé unique pour éviter les conflits
        )

        # Calcul de l'écart selon le choix
        if gap_unit == "Absolute Value ($)":
            subset['Gap_Display'] = subset['BS_Price_Graph'] - subset['Mid_Price']
            y_title = "Gap ($)"
            hover_template = "%{y:.2f} $"
        else:
            subset['Gap_Display'] = (subset['BS_Price_Graph'] - subset['Mid_Price']) / subset['Mid_Price'] * 100
            subset['Gap_Display'] = subset['Gap_Display'].fillna(0)
            y_title = "Gap (%)"
            hover_template = "%{y:.2f} %"

### Graph 2 : Histogramme price gap ###

        fig_gap = go.Figure()
        fig_gap.add_trace(go.Bar(
            x=subset['strike'], y=subset['Gap_Display'],
            marker_color=subset['Gap_Display'].apply(lambda x: 'red' if x < 0 else 'green'),
            name='Gap',
            hovertemplate=f"Strike: %{{x}}<br>Gap: {hover_template}"
        ))
        fig_gap.add_vline(x=S, line_dash="dot", annotation_text="Spot")
        fig_gap.update_layout(title=f"Discrepancy: Model - Market [{gap_unit}]", xaxis_title="Strike", yaxis_title=y_title)
        st.plotly_chart(fig_gap, width='stretch')

                                ########## Onglet 2 : IV ##########

    with tab2:
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
            name='Volatility BS (your input)',
            line=dict(color='red', dash='dash')
        ))
        fig_vol.add_vline(x=data['S0'], line_dash="dot", annotation_text="Spot")
        fig_vol.update_layout(yaxis_tickformat=".1%", title="Volatility Smile according to the strike", xaxis_title="Strike", yaxis_title="Implied Volatility")
        st.plotly_chart(fig_vol, width='stretch')

                                ########## Onglet 3 : Greeks ##########

    with tab3:

### Explications ###
        
        st.subheader("Sensitivities / Risk Analysis : Greeks")
        st.expander("❓  **What are the Greeks ?**", expanded=False).markdown(r"""
            The Greeks are measures of risk and sensitivity of an option price to changes in the underlying parameters of the Black-Scholes model.                                                              
            *Here ( $V$ ) represents the option price.*
            
            * **Delta $\Delta$** : **Underlying Price Sensitivity** ($\frac{\partial V}{\partial S}$).
                * *Definition :* Option price variation if the underlying price changes by $1.
                * *Example :* $\Delta$ = 0.5 ➜ If Spot +1\$, Option Price +0.5\$.
            * **Gamma $\Gamma$** : **Delta Sensitivity to Price** ($\frac{\partial^2 V}{\partial S^2}$).
                * *Definition :* Measures the rate of change in Delta for a $1 change in the underlying price.                               
                                 High Gamma means Delta changes rapidly, requiring more frequent hedging.
                * *Example :* $\Gamma$ = 0.05 ➜ If Spot +1\$, $\Delta$ +0.05.
            * **Theta $\Theta$** : **Time Decay** ($\frac{\partial V}{\partial t}$).
                * *Definition :* Amount of value the option loses *per unity of time*, all else being equal.   
                                 It is the daily cost for the option holder.
                * *Example :* $\Theta$ (Daily) = -0.10\$ ➜ The option loses 0.10\$ in value tomorrow.
            * **Vega $\mathcal{V}$** : **Volatility Sensitivity** ($\frac{\partial V}{\partial \sigma}$).
                * *Definition :* Option price variation if Volatility changes by *1 point (i.e., 1% = 0.01)*.
                * *Example :* $\mathcal{V}$ = 0.20 ➜ If Volatility +1%, Option Price +0.20\$.
            * **Rho $\rho$** : **Risk-Free Rate Sensitivity** ($\frac{\partial V}{\partial r}$).
                * *Definition :* Option price variation if the risk-free rate ($r$) changes by *1 point*.
                * *Example :* $\rho$ = 0.15 ➜ If $r$ +1%, Option Price +0.15\$.
            """)

### Calcul ###
        
        # Greeks Modèle (avec inputs utilisateur)
        greeks_model = calculate_greeks(S, K, T, r, sigma, q, option_type)
        # Greeks Marché (avec paramètres réels/implicites)
        greeks_market = calculate_greeks(S0_val, selected_strike, T_market, data['r'], sigma_market, data['q'], option_type)
        #Rappel : ces variables sont tirées de yfinance et initialisées en début de script

### Création du df + affichage ###

        col_model, col_market = st.columns(2)

        # Fonction pour générer le df
        def create_greeks_df(greeks_data, title):
            data_dict = {
                "Greek": ["$\Delta$", "$\Gamma$", "$\Theta$ (Daily)", "$\mathcal{V}$", "$ρ$"],
                "Value": [
                    f"{greeks_data['delta']:.4f}",
                    f"{greeks_data['gamma']:.4f}",
                    f"{greeks_data['theta_day']:.4f}",
                    f"{greeks_data['vega']/100:.4f}",
                    f"{greeks_data['rho']/100:.4f}"
                ]
            }
            df = pd.DataFrame(data_dict).set_index("Greek")
            return df
        
        # Affichage greeks BS 
        with col_model:
            st.markdown(f"### Black-Scholes model (your assumptions)")
            st.caption(f"Based on your inputs in the model")
            df_model = create_greeks_df(greeks_model, "BS Model")
            st.table(df_model)

        # Affichage greeks marché
        with col_market:
            st.markdown(f"### Market model (implied parameters)")
            st.caption(f"Based on the selected option market data")
            df_market = create_greeks_df(greeks_market, "Market model")
            st.table(df_market)

        st.markdown("---")
            
### Interprétation greeks gaps ###
        
        delta_gap = greeks_model['delta'] - greeks_market['delta']
        gamma_gap = greeks_model['gamma'] - greeks_market['gamma']
        theta_gap = greeks_model['theta_day'] - greeks_market['theta_day']
        vega_gap = greeks_model['vega']/100 - greeks_market['vega']/100
        rho_gap = greeks_model['rho']/100 - greeks_market['rho']/100
        
        st.subheader("💡 Greeks gaps interpretation", help = "The gap corresponds to : *(model - market)*")
        
        st.markdown("#### $\Delta$")
        if abs(delta_gap) < 0.001:
            st.success("✅ Your sensitivity to the underlying price is consistent with the market.")
        elif delta_gap > 0:
            st.warning(f"🔺 **Gap :** `{delta_gap:+.4f}` ➜ Your model **overestimates** *Delta*.\n\n"
                       f"**Consequence :** Your option is priced as **more sensitive to Spot movements** than the market anticipates.\n\n"
                       f"**Interpretation :** Your *Delta* hedge will be **more aggressive** than the market consensus.")
        else:
            st.error(f"🔻 **Gap :** `{delta_gap:+.4f}` ➜ Your model **underestimates** *Delta*.\n\n"
                     f"**Consequence :** Your option is priced as **less sensitive to Spot movements** than the market anticipates.\n\n"
                     f"**Interpretation :** Your *Delta* hedge will be **more conservative (weaker)** than the market consensus.")
        st.markdown("#### $\Gamma$")
        if abs(gamma_gap) < 0.001:
            st.success("✅ The stability of your *Delta* hedge is consistent with the market.")
        elif gamma_gap > 0:
            st.warning(f"🔺 **Gap :** `{gamma_gap:+.4f}` ➜ Your model overestimates *Gamma*.\n\n"
                       f"**Consequence :** Your *Delta* is **less stable** and changes rapidly with Spot movements.\n\n"
                       f"**Interpretation :** You anticipate needing to re-adjust your hedge position more frequently and significantly.")
        else:
            st.error(f"🔻 **Gap :** `{gamma_gap:+.4f}` ➜ Your model underestimates *Gamma*.\n\n"
                     f"**Consequence :** Your *Delta* is **insufficiently dynamic** and understates the change in directional risk.\n\n"
                     f"**Interpretation :** You may be underestimating the need to re-adjust your hedge position as the underlying moves.")

        st.markdown("#### $\Theta$ (daily)")
        if abs(theta_gap) < 0.001:
            st.success("✅ Your daily time decay estimation is consistent with the market.")
        elif theta_gap > 0:
            st.error(f"🔺 **Gap :** `{theta_gap:+.4f}` ➜ Your model underestimates time decay (*Theta* is less negative).\n\n"
                     f"**Consequence :** Your option is expected to lose value **slower** than the market anticipates.\n\n"
                     f"**Interpretation :** You are overestimating the remaining time value of the option.")
        else:
            st.warning(f"🔻 **Gap :** `{theta_gap:+.4f}` ➜ Your model overestimates time decay (*Theta* is more negative).\n\n"
                       f"**Consequence :** Your option is expected to lose value **faster** than the market anticipates.\n\n"
                       f"**Interpretation :** You are underestimating the remaining time value of the option.")
            
        st.markdown("#### $\mathcal{V}$")
        if abs(vega_gap) < 0.001:
            st.success("✅ Your exposure to changes in volatility is consistent with the market.")
        elif vega_gap > 0:
            st.warning(f"🔺 **Gap :** `{vega_gap:+.4f}` ➜ Your model overestimates *Vega*.\n\n"
                       f"**Consequence :** Your option is **more exposed to volatility risk** than the market consensus.\n\n"
                       f"**Interpretation :** If volatility rises by 1%, the positive impact on your price is stronger than the market consensus.")
        else:
            st.error(f"🔻 **Gap :** `{vega_gap:+.4f}` ➜ Your model underestimates *Vega*.\n\n"
                     f"**Consequence :** Your option is **less exposed to volatility risk** than the market consensus.\n\n"
                     f"**Interpretation :** If volatility changes, the resulting P&L impact on your position will be smaller than the market consensus.")

        st.markdown("#### $ρ$")

        if abs(rho_gap) < 0.001:
            st.success("✅ Your sensitivity to the risk-free rate is consistent with the market.")
        elif rho_gap > 0:
            if option_type == "Call":
                st.warning(f"🔺 **Gap :** `{rho_gap:+.4f}$` ➜ Your model overestimates *Rho*.\n\n"
                           f"**Consequence :** Your option is **more sensitive** to changes in the risk-free rate ($r$).\n\n"
                           f"**Interpretation :** A rate increase will cause a **stronger positive price impact** than the market consensus.")
            else: 
                st.warning(f"🔺 **Gap :** `{rho_gap:+.4f}$` ➜ Your model overestimates *Rho*.\n\n"
                           f"**Consequence :** Your option is **less sensitive** to changes in the risk-free rate ($r$).\n\n"
                           f"**Interpretation :** A rate increase will cause a **weaker negative price impact** than the market consensus.")
        else: # rho_gap < 0
            if option_type == "Call":
                st.error(f"🔻 **Gap :** `{rho_gap:+.4f}$` ➜ Your model underestimates *Rho*.\n\n"
                         f"**Consequence :** Your option is **less sensitive** to changes in the risk-free rate ($r$).\n\n"
                         f"**Interpretation :** A rate increase will cause a **weaker positive price impact** than the market consensus.")
            else: # Put
                st.error(f"🔻 **Gap :** `{rho_gap:+.4f}$` ➜ Your model underestimates *Rho*.\n\n"
                         f"**Consequence :** Your option is **more sensitive** to changes in the risk-free rate ($r$).\n\n"
                         f"**Interpretation :** A rate increase will cause a **stronger negative price impact** than the market consensus.")
            