import numpy as np

def binomial_option_pricing(S, K, T, r, sigma, q, N, option_type='call', american=True):
    """
    Price une option (Américaine ou Européenne) avec la méthode de l'Arbre Binomial (CRR).
    
    Paramètres:
    S : Prix Spot
    K : Strike
    T : Maturité (années)
    r : Taux sans risque
    sigma : Volatilité
    q : Rendement du dividende
    N : Nombre de pas de temps (ex: 200 pour la vitesse, 1000 pour la précision)
    option_type : 'call' ou 'put'
    american : True pour Américaine, False pour Européenne
    """
    
    # 1. Paramètres de l'arbre (Cox-Ross-Rubinstein)
    dt = T / N  # Durée d'un pas
    u = np.exp(sigma * np.sqrt(dt))  # Facteur de montée
    d = 1 / u                        # Facteur de descente
    
    # Probabilité risque-neutre (ajustée du dividende q)
    # p est la probabilité que le prix monte
    p = (np.exp((r - q) * dt) - d) / (u - d)
    
    # Facteur d'actualisation pour un pas de temps
    disc = np.exp(-r * dt)

    # 2. Initialisation des prix du sous-jacent à maturité (Dernière étape de l'arbre)
    # À l'étape N, le prix peut être S * u^j * d^(N-j) pour j allant de 0 à N
    # On génère tous les prix finaux possibles d'un coup grâce à numpy
    j = np.arange(0, N + 1)
    ST = S * (u ** (N - j)) * (d ** j)

    # 3. Calcul du Payoff à maturité
    if option_type == 'call':
        C = np.maximum(ST - K, 0)
    else:
        C = np.maximum(K - ST, 0)

    # 4. Backward Induction (On remonte l'arbre de la fin vers le début)
    for i in range(N - 1, -1, -1):
        # C contient les valeurs de l'option à l'étape i+1.
        # On calcule la "Valeur de Continuation" (si on garde l'option) :
        # C_continuation = disc * (p * C_up + (1-p) * C_down)
        # Avec numpy, C[:-1] correspond aux nœuds "up" et C[1:] aux nœuds "down" dans notre vecteur ordonné
        
        C_continuation = disc * (p * C[:-1] + (1 - p) * C[1:])
        
        # Si Option Américaine : On vérifie l'exercice anticipé
        if american:
            # On doit recalculer le prix du sous-jacent à l'étape i
            # S_t à l'étape i a (i+1) nœuds
            j = np.arange(0, i + 1)
            S_t = S * (u ** (i - j)) * (d ** j)
            
            # Valeur Intrinsèque (Gain immédiat si exercice)
            if option_type == 'call':
                intrinsic = np.maximum(S_t - K, 0)
            else:
                intrinsic = np.maximum(K - S_t, 0)
            
            # La valeur de l'option est le max entre garder et exercer
            C = np.maximum(C_continuation, intrinsic)
        else:
            # Si Européenne, on ne peut pas exercer, on garde juste la valeur de continuation
            C = C_continuation

    # À la fin de la boucle (i=0), C contient une seule valeur : le prix à t=0
    return C[0]