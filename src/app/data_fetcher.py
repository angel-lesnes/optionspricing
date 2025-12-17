import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

###########################################################################
    ########## Données de marché ss jacent + taux ss risque ###########
###########################################################################

@st.cache_data(ttl=3600, show_spinner=False)

def get_market_data(ticker_symbol): # données de marché pour un ticker donné
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        # prix spot
        S0 = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose') #plusieurs fallback
        
        if S0 is None:
            return None # si ticker invalide ou pas de données
            
        #devise et r (taux ss risque)
        currency = info.get('currency', 'USD')
        r = get_risk_free_rate_by_currency(currency)
        
        # rendement dividende
        q_yfinance = info.get('dividendYield', 0.0) 
        if q_yfinance != 0.0:
            q = q_yfinance / 100.0
        else:
            q = 0.0


        # dates d'expirations dispos
        expirations = ticker.options
        
        return {
            "S0": S0,
            "currency": currency,
            "r": r,
            "q": q,
            "expirations": expirations,
        }
    except Exception as e:
        print(f"Erreur data: {e}")
        return None

def get_risk_free_rate_by_currency(currency): #taux par rapport à la devise
    if currency == 'USD':
        #^TNX = bons du trésor US 10 ans / ^IRX = 13 semaines --> prix du ticker = rendement (4 = 4%)
        try:
            tnx = yf.Ticker("^TNX")
            rate = tnx.info.get('regularMarketPrice') or tnx.info.get('previousClose')
            return rate / 100.0 if rate else 0.045
        except:
            return 0.045 #fallback
    elif currency == 'EUR':
        return 0.03 # taux bce moyen car pas de ticker sur yfinance (à modifier si j'ai une meilleure API plus tard)
    else:
        return 0.045
    
#####################################################################
    ########## Mid price (market price si données OK) ###########
#####################################################################
    
def process_option_chain(calls, puts):
    def enrich_df(df):
        #moyenne bid-ask si dispo, sinon lastPrice
        df['Mid_Price'] = df.apply(
            lambda x: (x['bid'] + x['ask']) / 2 if (x['bid'] > 0 and x['ask'] > 0) else x['lastPrice'], 
            axis=1
        )
        return df

    return enrich_df(calls), enrich_df(puts)

###############################################
    ########## chaînes d'option ###########
###############################################

@st.cache_data(ttl=3600, show_spinner=False)
def get_chain_for_expiration(ticker_symbol, expiration_date):
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        chain = ticker_obj.option_chain(expiration_date)
        calls = chain.calls
        puts = chain.puts
        return process_option_chain(calls, puts) # On retourne les DFs enrichis
    except:
        return None, None    
