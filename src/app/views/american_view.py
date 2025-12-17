import streamlit as st
import numpy as np
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime
from app.data_fetcher import get_market_data, get_chain_for_expiration
from pricing.binomial import binomial_option_pricing
from pricing.black_scholes import implied_volatility, bs_call_price, bs_put_price 
from pricing.greeks import calculate_binomial_greeks, calculate_greeks

def render_american():
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
            st.info("Try a more liquid stock ticker (e.g. : AAPL, MSFT, TSLA).")
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

    st.subheader("⚙️ Binomial Tree Parameters")
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
    
    steps_N = st.slider("Number of Steps (N)", min_value=10, max_value=2000, value=200, step=10, 
                                help="More steps ➜ Higher precision but slower calculation.")

#####################################
    ########## PRICING ###########
#####################################

    if st.button("Click for pricing"):
        
########## Calcul avec binomial ##########
        with st.spinner(f"Computing Binomial Tree with N={steps_N}..."):
            price_american = binomial_option_pricing(
                S, K, T, r, sigma, q, 
                N=steps_N, 
                option_type=option_type.lower(), 
                american=True
            )

########## Calcul BS (pour comparaison) ##########
        if option_type == 'Call':
            price_european = bs_call_price(S, K, T, r, sigma, q)
        else:
            price_european = bs_put_price(S, K, T, r, sigma, q)

########## Affichage prix + comparaison BS ##########
        st.write(f"## US Option Price : {price_american:.4f} {data['currency']}")
        
        premium = price_american - price_european
        diff_premium = (price_american - price_european) / price_european * 100
        if premium > 0.005:
            st.info(f"💡 **Early Exercise Premium:** This option is worth **{premium:.4f} ({diff_premium :.2f})** more than its European equivalent. "
                    f"This confirms that the possibility of early exercise has value here (typically for underlying with high dividends).")
        else:
            st.info("💡 **No Early Exercise Premium:** The American price is identical to the European price. "
                    "Optimally, you would likely hold this option until maturity.")

########## Affichage gap prix & volat ##########
        diff = price_american - market_price
        diff_percent = (price_american - market_price) / market_price * 100 if market_price > 0.01 else 0
        
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
            else :
                if diff > 0.05 and premium > 0.05:
                    st.success(
                        f"**US option Premium Detected:** You are using the exact same parameters as the market.\n\n"
                        f"However, the Model Price is **${diff:.2f} higher**."
                        f"It represents the **early exercise right** premium that the Binomial model captures."
                    )
                elif abs(diff) < 0.05:
                     st.info(f"✅ **Perfect Match:** Your theoretical American price matches the Market price perfectly.")
                else:
                    st.warning(f"⚠️ **Market Inefficiency / Spread:** The model valuation differs by **${diff:.2f}** despite identical parameters.")
                st.info(f"ℹ️ You are using the exact same parameters as the market option. Your theoretical price matches perfectly the market price.")

################################################################
    ########## OUTILS D'ANALYSE (Graphs, Greeks) ###########
################################################################

    st.markdown("---")
    st.subheader("📊 Binomial Analysis")
    st.caption(
    f"⚠️ Volatility used to compute **model price** for each strike = **your input**.\n\n "
    f"⚠️ Volatility used to compute **market price** = **implied volatility**.")

########## création df avec strikes ##########
    subset = chain_df[
        (chain_df['strike'] > data['S0'] * 0.5) & 
        (chain_df['strike'] < data['S0'] * 1.5)
    ].copy() #filtrage du strike : 50% autour du spot

######### Fonction IV pour plot ##########
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
    subset['Computed_IV'] = subset['Computed_IV'].interpolate(method='linear', limit_direction='both').fillna(0.25) 

########## Fonction convergence ##########

    def plot_convergence(S, K, T, r, sigma, q, option_type):
        N_values = list(range(10, 2001, 10))
        prices = []
        
        for N in N_values:
            price = binomial_option_pricing(S, K, T, r, sigma, q, N=N, option_type=option_type, american=True)
            prices.append(price)
        
        # Prix BS pour référence
        if option_type == 'call':
            european_price = bs_call_price(S, K, T, r, sigma, q)
        else:
            european_price = bs_put_price(S, K, T, r, sigma, q)
        
        # Création du graphique
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=N_values, y=prices, mode='lines+markers', name='American Price (Binomial)', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=N_values, y=[european_price]*len(N_values), mode='lines', name='European Price (Black-Scholes)', line=dict(color='red', dash='dash')))
        
        fig.update_layout(
            title="Convergence of American Option Price with Binomial Steps (N)",
            xaxis_title="Number of Steps (N)",
            yaxis_title="Option Price",
            legend_title="Legend",
            template="plotly_white"
        )
        
        return fig
    
                                    ##########################################
                                    ########## AFFICHAGE DES GRAPHS ##########
                                    ##########################################

    with st.spinner(f"Creating graphs and analysis..."):
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Price gap", "Convergence", "🌳 Tree Preview", "Volatility Smile","📊 Greeks"])

                                ########## Onglet 1 : Analyse prix ##########

        with tab1:
            
            # Mm chose que bs_view (BS price dans df pour mettre courbe BS et marché)
            subset['BS_Price_Graph'] = subset['strike'].apply(
                lambda k: bs_call_price(S, k, T, r, sigma, q) if option_type == 'Call' 
                else bs_put_price(S, k, T, r, sigma, q)
            )

### Graph 1 : comparaison prix BS, binomial et marché ###
# /!\ binomial seuleement sur le strike sélectionné car sinon trop long à calculer 
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=subset['strike'], y=subset['Mid_Price'], 
                mode='lines+markers', name='Market Price', 
                marker=dict(color='blue', opacity=0.3)
            ))
            fig.add_trace(go.Scatter(
                x=subset['strike'], y=subset['BS_Price_Graph'], 
                mode='lines', name='Black-Scholes (EU)', 
                line=dict(color='red', dash='dot', width=1)
            ))

            #Point unique : prix binomial (ssi l'utilisateur a bien cliqué sur pricing)
            if 'price_american' in locals():
                fig.add_trace(go.Scatter(
                    x=[K], y=[price_american], 
                    mode='markers', name='Binomial (US)', 
                    marker=dict(color='green', size=15, symbol='star'),
                    text=[f"US Price: {price_american:.2f} $"],
                    hoverinfo='text'
                ))
                
                fig.add_vline(x=K, line_dash="dash", line_color="green", opacity=0.5, annotation_text="Selected Strike")

            fig.add_vline(x=S, line_dash="dot", annotation_text="Spot")
            fig.update_layout(title="Price Comparison : Market / BS / Binomial", xaxis_title="Strike", yaxis_title="Price")
            st.plotly_chart(fig, width="stretch")

### Outil sélection $/% pour graph 2 ### --> A voir si je garde ou si manque de pertinence car calcul d'un seul point du binomial

            gap_unit_us = st.radio(
                "Gap Display Unit:", 
                ["Absolute ($)", "Relative (%)"], 
                horizontal=True,
                key="us_gap_unit"
            )

            if gap_unit_us == "Absolute ($)":
                subset['Gap_Display'] = subset['BS_Price_Graph'] - subset['Mid_Price']
                y_gap_label = "Gap ($)"
            else:
                subset['Gap_Display'] = (subset['BS_Price_Graph'] - subset['Mid_Price']) / subset['Mid_Price'] * 100
                y_gap_label = "Gap (%)"

### Graph 2 : Histogramme price gap ###
            fig_gap = go.Figure()
            fig_gap.add_trace(go.Bar(
                x=subset['strike'], y=subset['Gap_Display'],
                marker_color=subset['Gap_Display'].apply(lambda x: 'red' if x < 0 else 'green'),
                name='Gap (BS - Market)'
            ))
            
            # indicateur spécifique pour le binomial (si calculé)
            if 'price_american' in locals():
                gap_bin = price_american - market_price
                if gap_unit_us == "Relative (%)":
                    gap_bin = (gap_bin / market_price) * 100 if market_price > 0 else 0
                    
                fig_gap.add_trace(go.Scatter(
                    x=[K], y=[gap_bin],
                    mode='markers+text',
                    marker=dict(color='black', size=10, symbol='diamond'),
                    name='Gap (Binomial - Market)',
                    text=[f"{gap_bin:.2f}"],
                    textposition="top center"
                ))

            fig_gap.update_layout(title=f"Gap Analysis ({y_gap_label})", xaxis_title="Strike", yaxis_title=y_gap_label)
            st.plotly_chart(fig_gap, width="stretch")

                                ########## Onglet 2 : Convergence ##########

            with tab2:
                st.caption(f"Convergence is typically achieved after N=200 steps. The red dashed line is the theoretical price if the option were European.")
                try:
                    fig_conv = plot_convergence(S, K, T, r, sigma, q, option_type.lower())
                    st.plotly_chart(fig_conv, width='stretch')
                except Exception as e:
                    st.error(f"Error generating convergence plot : {e}")

                                ########## Onglet 3 : Données + visualisation de l'arbre ##########

        with tab3:
            st.markdown("### 🌳 Binomial Tree Visualization (Preview)")
            st.caption("This interactive graph shows the **first 4 steps** of the pricing tree. Hover over nodes to see the underlying price.")

### Données de l'arbre ###
            dt = T / steps_N
            u = np.exp(sigma * np.sqrt(dt))
            d = 1 / u
            p_up = (np.exp((r - q) * dt) - d) / (u - d)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Up Factor (u)", f"{u:.4f}", help=r"Multiplicator of underlying price going up : $e^{\sigma \sqrt{dt}}$")
            c2.metric("Down Factor (d)", f"{d:.4f}", help=r"Multiplicator of underlying price going down : $1/u$")
            c3.metric("Risk-Neutral Prob (p)", f"{p_up:.2%}", help=r"Theoretical probability of an up move (u) in the risk-neutral world : $\frac{e^{(r-q)dt} - d}{u - d}$")
            c4.metric("Time Step (dt)", f"{dt*365:.2f} days", help="Duration of one step in the tree")

### Graph : preview de l'arbre ###
            vis_steps = 4  #seulement 4 pas pour visibilité
            dt_vis = T / steps_N
            u_vis = np.exp(sigma * np.sqrt(dt_vis))
            d_vis = 1 / u_vis
            
            fig_tree = go.Figure()

            # Génération des noeuds et des liens
            for i in range(vis_steps + 1):
                for j in range(i + 1):
                    # Calcul du S (prix ss jacent) à ce noeud (i=temps / j=nombre de down)
                    # Formule = S * u^(i-j) * d^j
                    S_node = S * (u_vis ** (i - j)) * (d_vis ** j)
                    
                    # Coordonnées : x = pas de temps / y = S
                    x_pos = i
                    y_pos = S_node
                    
                    # Création des noeuds
                    fig_tree.add_trace(go.Scatter(
                        x=[x_pos], y=[y_pos],
                        mode='markers+text',
                        marker=dict(size=25, color='#D4AC0D', line=dict(color='black', width=1)),
                        text=[f"{S_node:.1f}"], # Affiche le prix dans le rond
                        textposition="middle center",
                        textfont=dict(color='black', size=10),
                        hoverinfo='text',
                        hovertext=f"Step: {i}<br>Downs: {j}<br>Price: {S_node:.4f}",
                        showlegend=False
                    ))
                    
                    # Création des lignes vers les noeuds
                    if i < vis_steps:
                        # Si up (i+1, j) --> meme nombre de down
                        next_S_up = S_node * u_vis
                        fig_tree.add_trace(go.Scatter(
                            x=[i, i+1], y=[S_node, next_S_up],
                            mode='lines', line=dict(color='gray', width=1), hoverinfo='skip', showlegend=False
                        ))
                        
                        # Si down (i+1, j+1) --> on ajoute un down dans l'affichage
                        next_S_down = S_node * d_vis
                        fig_tree.add_trace(go.Scatter(
                            x=[i, i+1], y=[S_node, next_S_down],
                            mode='lines', line=dict(color='gray', width=1), hoverinfo='skip', showlegend=False
                        ))

            fig_tree.update_layout(
                title=f"Tree Structure (First {vis_steps} steps of N={steps_N})",
                xaxis=dict(title="Time Steps", showgrid=False, zeroline=False, showticklabels=True),
                yaxis=dict(title="Underlying Price ($)", showgrid=False),
                plot_bgcolor='white',
                height=500
            )
            
            st.plotly_chart(fig_tree, width="stretch")
            
            st.info(f"💡 **Note :** The actual calculation runs on **N = {steps_N}** steps. This tree is a simplified zoom on the start of the process.")

                                ########## Onglet 4 : IV (comme bs_view) ##########

        with tab4:
            fig_vol = go.Figure()
            
            fig_vol.add_trace(go.Scatter(
                x=subset['strike'], 
                y=subset['Computed_IV'], 
                mode='lines+markers',
                name='Implied Volatility',
                line=dict(color='blue', shape='spline')
            ))
            
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

                                ########## Onglet 5 : Greeks ##########

        with tab5:
            st.subheader("🇺🇸 Sensitivities Analysis (American Greeks)")
            st.info("ℹ️ These Greeks are calculated using **Finite Differences** on the Binomial Tree, providing a more accurate sensitivity analysis for American options than standard Black-Scholes Greeks.")

### Explications ###

            st.expander("❓ **What are the Greeks ?**", expanded=False).markdown(r"""
                The Greeks are measures of risk and sensitivity of an option price to changes in the underlying parameters.
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

### Calcul ### (binomial pour le modèle, BS pour le marché car IV est BS)
            
            with st.spinner("Computing American Greeks..."):
                # Greeks binomial : N=100 pour que ce soit rapide à l'affichage
                greeks_model = calculate_binomial_greeks(S, K, T, r, sigma, q, option_type.lower(), N=100)
                
                # Greeks Marché (BS avec IV marché) : comparaison avec BS pour voir sensibilité "américaine"
                greeks_market = calculate_greeks(S0_val, selected_strike, T_market, data['r'], sigma_market, data['q'], option_type)

### Création du df + affichage ###

            col_model, col_market = st.columns(2)

            def create_greeks_df(greeks_data):
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
                return pd.DataFrame(data_dict).set_index("Greek")
            
            with col_model:
                st.markdown(f"### Binomial Model (User)")
                st.caption(f"Sensitivities of your American pricing")
                st.table(create_greeks_df(greeks_model))

            with col_market:
                st.markdown(f"### Market Baseline (BS)")
                st.caption(f"Standard sensitivities using Market IV")
                st.table(create_greeks_df(greeks_market))
                
            st.markdown("---")
            
### Interprétation greeks gaps ###

            delta_gap = greeks_model['delta'] - greeks_market['delta']
            gamma_gap = greeks_model['gamma'] - greeks_market['gamma']
            theta_gap = greeks_model['theta_day'] - greeks_market['theta_day']
            vega_gap = greeks_model['vega']/100 - greeks_market['vega']/100
            rho_gap = greeks_model['rho']/100 - greeks_market['rho']/100

            st.subheader("Greeks gaps interpretation", help = "The gap corresponds to : *(model - market)*")
            
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
                else: # Put
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
