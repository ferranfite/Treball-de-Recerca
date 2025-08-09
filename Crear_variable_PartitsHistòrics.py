# De moment guardo això aqui prq ja ho afegiré
# La funció ha d'anar a la cela 3
# Després s'ha d'incorporar a Xfeatures

def crear_historic_directe(df_partits):
    """
    Crea features basades en l'històric directe entre equips
    """
    print("📊 Creant històric directe entre equips...")
    
    # Crear diccionari per emmagatzemar l'històric
    historic = {}
    
    # Analitzar tots els partits per crear l'històric
    for _, partit in df_partits.iterrows():
        home_team = partit['Home Team']
        away_team = partit['Away Team']
        home_goals = partit['Home Goal']
        away_goals = partit['Away Goal']
        
        # Crear clau per la parella d'equips
        parella = tuple(sorted([home_team, away_team]))
        
        if parella not in historic:
            historic[parella] = {'partits': 0, 'victories_home': 0, 'victories_away': 0, 'empats': 0}
        
        historic[parella]['partits'] += 1
        
        if home_goals > away_goals:
            historic[parella]['victories_home'] += 1
        elif away_goals > home_goals:
            historic[parella]['victories_away'] += 1
        else:
            historic[parella]['empats'] += 1
    
    return historic

# Crear l'històric
historic_directe = crear_historic_directe(df_partits)
print(f"✅ Històric directe creat per {len(historic_directe)} parelles d'equips")



    # FEATURES D'HISTÒRIC DIRECTE
    print("�� Afegint features d'històric directe...")
    
    # Crear l'històric directe
    historic_directe = crear_historic_directe(df_partits)
    
    # Afegir features d'històric a cada partit
    df_nou['Historic_Home_Wins'] = 0
    df_nou['Historic_Away_Wins'] = 0
    df_nou['Historic_Draws'] = 0
    df_nou['Historic_Total_Matches'] = 0
    df_nou['Historic_Home_Win_Rate'] = 0.0
    df_nou['Historic_Away_Win_Rate'] = 0.0
    
    for idx, partit in df_nou.iterrows():
        home_team = partit['Home Team']
        away_team = partit['Away Team']
        parella = tuple(sorted([home_team, away_team]))
        
        if parella in historic_directe:
            hist = historic_directe[parella]
            df_nou.loc[idx, 'Historic_Total_Matches'] = hist['partits']
            df_nou.loc[idx, 'Historic_Home_Wins'] = hist['victories_home']
            df_nou.loc[idx, 'Historic_Away_Wins'] = hist['victories_away']
            df_nou.loc[idx, 'Historic_Draws'] = hist['empats']
            
            if hist['partits'] > 0:
                df_nou.loc[idx, 'Historic_Home_Win_Rate'] = hist['victories_home'] / hist['partits']
                df_nou.loc[idx, 'Historic_Away_Win_Rate'] = hist['victories_away'] / hist['partits']
    
    print(f"✅ Features d'històric directe afegides")
    
    return df_nou


