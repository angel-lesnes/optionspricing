import yfinance as yf
import pandas as pd
from datetime import datetime

def get_market_data(ticker_symbol): # données de marché pour un ticker donné
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        #1. --> Spot Price (S0)
        S0 = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose') #plusieurs fallback
        
        if S0 is None:
            return None # ticker invalide ou pas de données
            
        #2. --> Devise et Taux sans risque (r)
        currency = info.get('currency', 'USD')
        r = get_risk_free_rate_by_currency(currency)
        
        #3. --> Dividend Yield (q)
        q = info.get('dividendYield', 0.0) 
        if q is None: q = 0.0

        #4. --> Dates d'expiration
        expirations = ticker.options
        
        return {
            "S0": S0,
            "currency": currency,
            "r": r,
            "q": q,
            "expirations": expirations,
            "ticker_obj": ticker #on met le ticker ici pour récupérer les chaînes d'options plus tard
        }
    except Exception as e:
        print(f"Erreur data: {e}")
        return None

def get_risk_free_rate_by_currency(currency): #taux par rapport à la devise
    if currency == 'USD':
        #^ TNX = bons du trésor US 10 ans / ^IRX = 13 semaines
        try:
            tnx = yf.Ticker("^TNX")
            # Le prix est le rendement (ex: 4.25 pour 4.25%)
            rate = tnx.info.get('regularMarketPrice') or tnx.info.get('previousClose')
            return rate / 100.0 if rate else 0.045
        except:
            return 0.045 #fallback
    elif currency == 'EUR':
        return 0.03 # taux bce moyen car pas de ticker sur yfinance
    else:
        return 0.05 #taux par défaut pour les autres devises
    
def process_option_chain(calls, puts):
    """
    Nettoie les données et ajoute le Mid-Price.
    """
    # Fonction interne pour traiter un DF
    def enrich_df(df):
        # 1. Calcul du Mid-Price
        # Si bid ou ask est 0 ou vide, on garde le lastPrice par défaut, sinon on prend la moyenne
        df['Mid_Price'] = df.apply(
            lambda x: (x['bid'] + x['ask']) / 2 if (x['bid'] > 0 and x['ask'] > 0) else x['lastPrice'], 
            axis=1
        )
        return df

    return enrich_df(calls), enrich_df(puts)

# Mettez à jour votre fonction get_chain_for_expiration pour utiliser ceci :
def get_chain_for_expiration(ticker_obj, expiration_date):
    try:
        chain = ticker_obj.option_chain(expiration_date)
        calls = chain.calls
        puts = chain.puts
        return process_option_chain(calls, puts) # On retourne les DFs enrichis
    except:
        return None, None    
