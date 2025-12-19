import numpy as np
from scipy.stats import norm
from src.pricing.binomial import binomial_option_pricing

#######################################################
    ########## Calcul Greeks BS ###########
#######################################################

def d1_function(S, K, T, r, sigma, q=0):
    if sigma < 1e-6 or T < 1e-6:
        return 0
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2_function(d1, sigma, T):
    return d1 - sigma * np.sqrt(T)

########## Greeks call ##########

def delta_call(S, K, T, r, sigma, q=0):
    d1 = d1_function(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.cdf(d1)

def theta_call(S, K, T, r, sigma, q=0):
    d1 = d1_function(S, K, T, r, sigma, q)
    d2 = d2_function(d1, sigma, T)
    term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
    term3 = q * S * np.exp(-q * T) * norm.cdf(d1)
    return term1 + term2 + term3

def rho_call(S, K, T, r, sigma, q=0):
    d2 = d2_function(d1_function(S, K, T, r, sigma, q), sigma, T)
    return K * T * np.exp(-r * T) * norm.cdf(d2)

########## Greeks put ##########

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

########## Greeks égaux pour call et put ##########

def vega(S, K, T, r, sigma, q=0): #Pareil call et put
    d1 = d1_function(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

def gamma(S, K, T, r, sigma, q=0): #pareil call et put
    d1 = d1_function(S, K, T, r, sigma, q)
    denominator = S * sigma * np.sqrt(T)
    if denominator < 1e-6:
        return 0
    return np.exp(-q * T) * norm.pdf(d1) / denominator

########## Fonction principale pour calculer tous les greeks ##########

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

###########################################################################
    ########## Calcul Greeks différences finies --> binomial ###########
###########################################################################

def calculate_binomial_greeks(S, K, T, r, sigma, q, option_type, N=100):
    # Prix central
    P0 = binomial_option_pricing(S, K, T, r, sigma, q, N, option_type, american=True)
    
    # Delta & Gamma (choc S + dS)
    dS = S * 0.01
    P_up = binomial_option_pricing(S + dS, K, T, r, sigma, q, N, option_type, american=True)
    P_down = binomial_option_pricing(S - dS, K, T, r, sigma, q, N, option_type, american=True)
    delta = (P_up - P_down) / (2 * dS)
    gamma = (P_up - 2*P0 + P_down) / (dS ** 2)
    
    # Theta (choc T - 1 jour)
    dt_day = 1/365
    if T > dt_day:
        P_th = binomial_option_pricing(S, K, T - dt_day, r, sigma, q, N, option_type, american=True)
        theta = (P_th - P0)
    else: theta = 0

    # Vega (choc Sigma + 1%)
    P_vega = binomial_option_pricing(S, K, T, r, sigma + 0.01, q, N, option_type, american=True)
    vega = (P_vega - P0) * 100

    # Rho (choc r + 1%)
    P_rho = binomial_option_pricing(S, K, T, r + 0.01, sigma, q, N, option_type, american=True)
    rho = (P_rho - P0) * 100
    
    return {"delta": delta, "gamma": gamma, "theta_day": theta, "vega": vega, "rho": rho}