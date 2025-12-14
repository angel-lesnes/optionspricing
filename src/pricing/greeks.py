import numpy as np
from scipy.stats import norm

def d1_function(S, K, T, r, sigma, q=0):
    if sigma < 1e-6 or T < 1e-6:
        return 0
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2_function(d1, sigma, T):
    return d1 - sigma * np.sqrt(T)

# --- Greeks pour un Call ---

def delta_call(S, K, T, r, sigma, q=0):
    d1 = d1_function(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.cdf(d1)

def gamma(S, K, T, r, sigma, q=0): #pareil call et put
    d1 = d1_function(S, K, T, r, sigma, q)
    denominator = S * sigma * np.sqrt(T)
    if denominator < 1e-6:
        return 0
    return np.exp(-q * T) * norm.pdf(d1) / denominator

def theta_call(S, K, T, r, sigma, q=0):
    d1 = d1_function(S, K, T, r, sigma, q)
    d2 = d2_function(d1, sigma, T)
    term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
    term3 = q * S * np.exp(-q * T) * norm.cdf(d1)
    return term1 + term2 + term3

def vega(S, K, T, r, sigma, q=0): #Pareil call et put
    d1 = d1_function(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

def rho_call(S, K, T, r, sigma, q=0):
    d2 = d2_function(d1_function(S, K, T, r, sigma, q), sigma, T)
    return K * T * np.exp(-r * T) * norm.cdf(d2)

# --- Greeks pour un Put ---

def delta_put(S, K, T, r, sigma, q=0):
    d1 = d1_function(S, K, T, r, sigma, q)
    return delta_call(S, K, T, r, sigma, q) - np.exp(-q * T)

def theta_put(S, K, T, r, sigma, q=0):
    d1 = d1_function(S, K, T, r, sigma, q)
    d2 = d2_function(d1, sigma, T)
    term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    term3 = q * S * np.exp(-q * T) * norm.cdf(-d1)
    return term1 - term2 + term3

def rho_put(S, K, T, r, sigma, q=0):
    d2 = d2_function(d1_function(S, K, T, r, sigma, q), sigma, T)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)

# Fonction principale pour calculer tous les greeks
def calculate_greeks(S, K, T, r, sigma, q, option_type):
#fallbacks pour éviter div par 0
    if T <= 1e-6:
        T = 1e-6 
    if sigma <= 1e-6:
        sigma = 1e-6
        
    greeks = {}
    greeks['gamma'] = gamma(S, K, T, r, sigma, q)
    greeks['vega'] = vega(S, K, T, r, sigma, q)
    
    if option_type == "Call":
        greeks['delta'] = delta_call(S, K, T, r, sigma, q)
        greeks['theta'] = theta_call(S, K, T, r, sigma, q)
        greeks['rho'] = rho_call(S, K, T, r, sigma, q)
    else:
        greeks['delta'] = delta_put(S, K, T, r, sigma, q)
        greeks['theta'] = theta_put(S, K, T, r, sigma, q)
        greeks['rho'] = rho_put(S, K, T, r, sigma, q)
        
    #theta annualisé en theta journalier
    greeks['theta_day'] = greeks['theta'] / 365 
    
    return greeks