import numpy as np
from math import sqrt, exp
from scipy.stats import norm
from scipy.optimize import brentq

#BS Merton avec dividendes

def bs_call_price(S, K, T, r, sigma, q=0.0):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def bs_call_delta(S, K, T, r, sigma, q=0.0):
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    return exp(-q * T) * norm.cdf(d1)

# Put-Call parity 

def bs_put_price(S, K, T, r, sigma, q=0.0):
    C = bs_call_price(S, K, T, r, sigma, q)
    S_adj = S * exp(-q * T) 
    K_disc = K * exp(-r * T)
    return float(C - S_adj + K_disc)

def bs_put_delta(S, K, T, r, sigma, q=0.0):
    return float(bs_call_delta(S, K, T, r, sigma, q) - exp(-q * T))

def bs_vega(S, K, T, r, sigma, q=0.0): # <-- AJOUTER q ici
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T)) # <-- UTILISER q
    return S * exp(-q * T) * norm.pdf(d1) * sqrt(T) # <-- UTILISER q

def implied_volatility(S, K, T, r, price, call_put='call', q=0.0, method='brent'):
    
    # --- 1. Vérifications préliminaires (bornes d'arbitrage BSM) ---
    
    # Valeur actuelle du sous-jacent ajustée par dividendes
    PV_S = S * exp(-q * T)
    # Valeur actuelle du Strike
    PV_K = K * exp(-r * T)
    
    if call_put == 'call':
        intrinsic_value = max(PV_S - PV_K, 0)
        if price < intrinsic_value:
            return np.nan # Prix d'arbitrage : IV impossible
    elif call_put == 'put':
        intrinsic_value = max(PV_K - PV_S, 0)
        if price < intrinsic_value:
            return np.nan # Prix d'arbitrage : IV impossible
    
    # --- 2. Définir la fonction d'erreur ---
    def error_func(sigma):
        if call_put == 'call':
            # ON PASSE q AU PRICING
            return bs_call_price(S, K, T, r, sigma, q) - price
        else:
            # ON PASSE q AU PRICING
            return bs_put_price(S, K, T, r, sigma, q) - price

    # --- 3. Résolution ---
    
    if method == 'brent':
        # La méthode de Brent (robuste)
        try:
            # Plage de recherche (très large)
            iv = brentq(error_func, 1e-6, 5.0) 
            return iv
        except ValueError:
            return np.nan 

    elif method == 'newton':
        # La méthode de Newton-Raphson (rapide)
        # Nécessite une fonction qui retourne la dérivée (Vega)
        def vega_func(sigma):
            return bs_vega(S, K, T, r, sigma, q) # On passe q au Vega
        
        # Initial guess
        sigma_guess = 0.20 
        tolerance = 1e-6
        max_iterations = 100
        
        for i in range(max_iterations):
            f = error_func(sigma_guess)
            f_prime = vega_func(sigma_guess) # Vega
            
            # Éviter la division par zéro (si Vega est proche de zéro)
            if abs(f_prime) < 1e-10:
                break 

            sigma_new = sigma_guess - f / f_prime
            
            # Condition d'arrêt et de positivité
            if abs(sigma_new - sigma_guess) < tolerance:
                # S'assurer que la Vol est positive
                return max(sigma_new, 1e-6) 
            
            sigma_guess = sigma_new
        
        # Si la boucle n'a pas convergé
        return np.nan 
        
    return np.nan # Si la méthode n'est pas reconnue

# Numerical test

if __name__ == "__main__":
    S0, K, T, r, sigma, q = 100, 90, 0.5, 0.01, 0.2, 0.03
    print("BS call price:", bs_call_price(S0, K, T, r, sigma, q))
    print("BS call delta:", bs_call_delta(S0, K, T, r, sigma, q))
    print("BS put price:", bs_put_price(S0, K, T, r, sigma, q))
    print("BS put delta:", bs_put_delta(S0, K, T, r, sigma, q))
    print("Implied Volatility (call):", implied_volatility(S0, K, T, r, bs_call_price(S0, K, T, r, sigma, q), 'call', q))