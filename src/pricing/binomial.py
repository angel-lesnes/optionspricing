import numpy as np

def binomial_option_pricing(S, K, T, r, sigma, q, N, option_type='call', american=True):
    """
    Prix d'une option Européenne ou Américaine via un modèle binomial Cox-Ross-Rubinstein.
    N : Nombre de pas de temps (ex: 200 pour la vitesse, 1000 pour la précision)
    american : True pour Américaine, False pour Européenne
    """
    
##########################################################################
    ########## Paramètres de l'arbre (Cox-Ross-Rubinstein) ###########
##########################################################################

    dt = T / N  #= durée d'un step
    u = np.exp(sigma * np.sqrt(dt))  # = facteur de hausse 
    d = 1 / u                        # = facteur de baisse
    
    # Proba risque neutre (ajustée du dividende) : p = proba hausse du prix
    p = (np.exp((r - q) * dt) - d) / (u - d)

    disc = np.exp(-r * dt) # facteur d'actu d'un step

###########################################################################################################
    ########## Initialisation des prix du sous-jacent à maturité = dernière étape de l'arbre ###########
###########################################################################################################

    # À l'étape N : le prix peut être S * u^j * d^(N-j) pour j allant de 0 à N
    # Génération de tous les prix finaux possibles d'un coup grâce à numpy
    j = np.arange(0, N + 1)
    ST = S * (u ** (N - j)) * (d ** j)

###############################################
    ########## Payoff à maturité ###########
###############################################

    if option_type == 'call':
        C = np.maximum(ST - K, 0)
    else:
        C = np.maximum(K - ST, 0)

#####################################################################
    ########## Backward Induction (remontée de l'arbre) ###########
#####################################################################

    for i in range(N - 1, -1, -1):
        # C contient les valeurs de l'option à l'étape i+1.
        # Calcul de la valeur de continuation (= si on garde l'option) : C_continuation = disc * (p * C_up + (1-p) * C_down)
        # Avec numpy : C[:-1] = noeuds "up" / C[1:] = noeuds "down" dans le vecteur ordonné
        
        C_continuation = disc * (p * C[:-1] + (1 - p) * C[1:])
        
        # Si option US => Vérification de l'exercice anticipé
        if american:
            #Recalcul du prix du ss-jacent à l'étape i
            # S_t à l'étape i contient : (i+1) noeuds
            j = np.arange(0, i + 1)
            S_t = S * (u ** (i - j)) * (d ** j)
            
            # Valeur intrinsèque => Gain immédiat si exercice
            if option_type == 'call':
                intrinsic = np.maximum(S_t - K, 0)
            else:
                intrinsic = np.maximum(K - S_t, 0)
            
            # Valeur de l'option = max entre conserver & exercer
            C = np.maximum(C_continuation, intrinsic)
        else:
            # Si option EU : on ne peut pas exercer => on garde juste la valeur de continuation
            C = C_continuation

    # À la fin de la boucle (i=0) : C contient une seule valeur (prix à t=0)
    return C[0]