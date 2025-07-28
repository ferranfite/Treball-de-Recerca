#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model de Predicció d'Enfrontaments - La Liga 2023-2024
=======================================================

Aquest script crea un model de predicció per a partits individuals basat en les estadístiques dels equips.

Objectius:
- Predir victòries locals
- Predir gols totals del partit  
- Predir nombre de corners
- Predir resultat complet (victòria/empat/derrota)

Autor: Treball de Recerca
Data: 2024
"""

# =============================================================================
# 1. IMPORTACIÓ I CONFIGURACIÓ
# =============================================================================

print("🚀 Iniciant Model de Predicció d'Enfrontaments...")
print("=" * 60)

# Llibreries bàsiques
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

# Visualització
import matplotlib.pyplot as plt
import seaborn as sns

# Configuració
plt.style.use('default')
sns.set_palette("Set2")
pd.set_option('display.max_columns', None)

print("✅ Llibreries importades correctament")

# =============================================================================
# 2. CARREGAMENT DE DADES
# =============================================================================

print("\n📊 Carregant dades...")

# Dades de classificació general dels equips
df_classificacio = pd.read_excel('BDD_EntrenamentModel_Estadístiques-La_Liga-2023-2024.xlsx', 
                                sheet_name='ClassificacióGeneral')

# Dades detallades per partit
df_partits = pd.read_excel('BDD_EntrenamentModel_Estadístiques-La_Liga-2023-2024.xlsx', 
                           sheet_name='StatsPartit')

print(f"📈 Dades de classificació: {df_classificacio.shape}")
print(f"⚽ Dades de partits: {df_partits.shape}")
print(f"🏆 Equips disponibles: {len(df_classificacio['Team'].unique())}")
print(f"🎯 Partits analitzats: {len(df_partits)}")

# =============================================================================
# 3. ANÀLISI EXPLORATÒRIA
# =============================================================================

print("\n🔍 ANÀLISI EXPLORATÒRIA")
print("=" * 50)

# Estadístiques bàsiques dels partits
print("\n📊 Estadístiques dels partits:")
print(f"- Mitjana de gols per partit: {df_partits['Home Goal'].mean() + df_partits['Away Goal'].mean():.2f}")
print(f"- Mitjana de corners per partit: {df_partits['wonCorners'].mean():.1f}")
print(f"- Mitjana de xuts per partit: {df_partits['totalShots'].mean():.1f}")
print(f"- Mitjana de possessió: {df_partits['possessionPct'].mean():.1f}%")

# Anàlisi de correlacions
print("\n🔗 CORRELACIONS PRINCIPALS")
print("=" * 50)

# Seleccionem columnes numèriques
numeric_columns = df_partits.select_dtypes(include=[np.number]).columns

# Correlacions amb gols marcats
correlacions_gols = df_partits[numeric_columns].corr()['Home Goal'].sort_values(ascending=False)
print("\n🏆 Top 5 correlacions amb gols marcats:")
for i, (variable, correlacio) in enumerate(correlacions_gols.head(6).items()):
    if variable != 'Home Goal':
        print(f"  {i}. {variable}: {correlacio:.3f}")

# Correlacions amb victòria
df_partits['Victory'] = (df_partits['Home Goal'] > df_partits['Away Goal']).astype(int)
correlacions_victoria = df_partits[numeric_columns].corr()['Victory'].sort_values(ascending=False)
print("\n🎯 Top 5 correlacions amb victòria local:")
for i, (variable, correlacio) in enumerate(correlacions_victoria.head(6).items()):
    if variable != 'Victory':
        print(f"  {i}. {variable}: {correlacio:.3f}")

# =============================================================================
# 4. PREPARACIÓ DE DADES
# =============================================================================

print("\n🔧 PREPARACIÓ DE DADES")
print("=" * 50)

# Funció per obtenir estadístiques d'un equip
def obtenir_stats_equip(equip_nom, df_classificacio):
    """Obtenim les estadístiques d'un equip específic"""
    if equip_nom in df_classificacio['Team'].values:
        return df_classificacio[df_classificacio['Team'] == equip_nom].iloc[0]
    else:
        # Si no trobem l'equip, retornem valors mitjans
        return df_classificacio.mean()

# Funció per crear features dels equips
def crear_features_equips(df_partits, df_classificacio):
    """Creem features combinant estadístiques dels equips amb dades de partits"""
    df_nou = df_partits.copy()
    
    # Columnes de classificació (excloent 'Team')
    stats_columns = [col for col in df_classificacio.columns if col != 'Team']
    
    # Afegim estadístiques de l'equip local
    for col in stats_columns:
        df_nou[f'Home_{col}'] = df_nou['Home Team'].map(
            df_classificacio.set_index('Team')[col]
        ).fillna(df_classificacio[col].mean())
    
    # Afegim estadístiques de l'equip visitant
    for col in stats_columns:
        df_nou[f'Away_{col}'] = df_nou['Away Team'].map(
            df_classificacio.set_index('Team')[col]
        ).fillna(df_classificacio[col].mean())
    
    return df_nou

print("🔧 Preparant dades...")

# Apliquem la funció
df_complet = crear_features_equips(df_partits, df_classificacio)

print(f"✅ Dataset complet creat: {df_complet.shape}")
print(f"📊 Features dels equips: {len([col for col in df_complet.columns if col.startswith('Home_') or col.startswith('Away_')])}")

# Creació de variables objectiu
print("\n🎯 Creant variables objectiu...")

# Variables objectiu
df_complet['Victory'] = (df_complet['Home Goal'] > df_complet['Away Goal']).astype(int)
df_complet['Draw'] = (df_complet['Home Goal'] == df_complet['Away Goal']).astype(int)
df_complet['Total_Goals'] = df_complet['Home Goal'] + df_complet['Away Goal']
df_complet['Goal_Difference'] = df_complet['Home Goal'] - df_complet['Away Goal']

# Resultat categòric
def crear_resultat_categoric(row):
    if row['Home Goal'] > row['Away Goal']:
        return 'Home_Win'
    elif row['Home Goal'] < row['Away Goal']:
        return 'Away_Win'
    else:
        return 'Draw'

df_complet['Result'] = df_complet.apply(crear_resultat_categoric, axis=1)

# Mostrem estadístiques
print("\n📈 Estadístiques de les variables objectiu:")
print(f"- Victòries locals: {df_complet['Victory'].sum()} ({df_complet['Victory'].mean():.1%})")
print(f"- Empats: {df_complet['Draw'].sum()} ({df_complet['Draw'].mean():.1%})")
print(f"- Gols totals (mitjana): {df_complet['Total_Goals'].mean():.2f}")
print(f"- Corners (mitjana): {df_complet['wonCorners'].mean():.1f}")

print("\n🏆 Distribució de resultats:")
for resultat, count in df_complet['Result'].value_counts().items():
    print(f"  {resultat}: {count} ({count/len(df_complet):.1%})")

# Selecció de features per al model
print("\n🔍 Seleccionant features...")

# Features del partit
features_partit = [
    'foulsCommitted', 'yellowCards', 'redCards', 'offsides', 'wonCorners',
    'saves', 'possessionPct', 'totalShots', 'shotsOnTarget', 'shotPct',
    'penaltyKickGoals', 'penaltyKickShots', 'accuratePasses', 'totalPasses',
    'passPct', 'accurateCrosses', 'totalCrosses', 'crossPct', 'accurateLongBalls',
    'totalLongBalls', 'longballPct', 'blockedShots', 'effectiveTackles',
    'totalTackles', 'tacklePct', 'interceptions', 'effectiveClearance', 'totalClearance'
]

# Features dels equips
features_equips = [col for col in df_complet.columns if col.startswith('Home_') or col.startswith('Away_')]

# Combinem totes les features
X_features = features_partit + features_equips

# Dataset final per al model
df_model = df_complet[X_features + ['Victory', 'Total_Goals', 'wonCorners', 'Result']].dropna()

X = df_model[X_features]
y_victory = df_model['Victory']
y_goals = df_model['Total_Goals']
y_corners = df_model['wonCorners']
y_result = df_model['Result']

print(f"✅ Dataset final: {df_model.shape}")
print(f"📊 Features utilitzades: {len(X_features)}")
print(f"🎯 Variables objectiu: Victòria, Gols Totals, Corners, Resultat")

# =============================================================================
# 5. ENTRENAMENT DE MODELS
# =============================================================================

print("\n🤖 ENTRENAMENT DE MODELS")
print("=" * 50)

# Divisió de dades
print("🔄 Dividint dades en entrenament i test...")

X_train, X_test, y_victory_train, y_victory_test = train_test_split(
    X, y_victory, test_size=0.2, random_state=42, stratify=y_victory
)

_, _, y_goals_train, y_goals_test = train_test_split(
    X, y_goals, test_size=0.2, random_state=42
)

_, _, y_corners_train, y_corners_test = train_test_split(
    X, y_corners, test_size=0.2, random_state=42
)

_, _, y_result_train, y_result_test = train_test_split(
    X, y_result, test_size=0.2, random_state=42, stratify=y_result
)

print(f"📚 Dades d'entrenament: {X_train.shape}")
print(f"🧪 Dades de test: {X_test.shape}")

# Entrenament dels models
print("\n🤖 ENTRENAMENT DELS MODELS")
print("=" * 50)

# Model 1: Predicció de victòria
print("\n1️⃣ Model de Predicció de Victòria")
model_victory = RandomForestClassifier(n_estimators=100, random_state=42)
model_victory.fit(X_train, y_victory_train)
victory_pred = model_victory.predict(X_test)
victory_accuracy = accuracy_score(y_victory_test, victory_pred)
print(f"   ✅ Precisió: {victory_accuracy:.3f}")

# Model 2: Predicció de gols totals
print("\n2️⃣ Model de Predicció de Gols Totals")
model_goals = RandomForestRegressor(n_estimators=100, random_state=42)
model_goals.fit(X_train, y_goals_train)
goals_pred = model_goals.predict(X_test)
goals_mse = mean_squared_error(y_goals_test, goals_pred)
print(f"   ✅ RMSE: {np.sqrt(goals_mse):.3f}")

# Model 3: Predicció de corners
print("\n3️⃣ Model de Predicció de Corners")
model_corners = RandomForestRegressor(n_estimators=100, random_state=42)
model_corners.fit(X_train, y_corners_train)
corners_pred = model_corners.predict(X_test)
corners_mse = mean_squared_error(y_corners_test, corners_pred)
print(f"   ✅ RMSE: {np.sqrt(corners_mse):.3f}")

# Model 4: Predicció de resultat complet
print("\n4️⃣ Model de Predicció de Resultat")
model_result = RandomForestClassifier(n_estimators=100, random_state=42)
model_result.fit(X_train, y_result_train)
result_pred = model_result.predict(X_test)
result_accuracy = accuracy_score(y_result_test, result_pred)
print(f"   ✅ Precisió: {result_accuracy:.3f}")

print("\n📊 Report de classificació (Resultat):")
print(classification_report(y_result_test, result_pred))

# =============================================================================
# 6. ANÀLISI DE RESULTATS
# =============================================================================

print("\n🔍 ANÀLISI D'IMPORTÀNCIA DE FEATURES")
print("=" * 50)

# Obtenim importància per a cada model
feature_importance_victory = pd.DataFrame({
    'feature': X_features,
    'importance': model_victory.feature_importances_
}).sort_values('importance', ascending=False)

feature_importance_goals = pd.DataFrame({
    'feature': X_features,
    'importance': model_goals.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🏆 Top 10 features per a predicció de victòria:")
for i, row in feature_importance_victory.head(10).iterrows():
    print(f"  {i+1:2d}. {row['feature']:<25} {row['importance']:.4f}")

print("\n⚽ Top 10 features per a predicció de gols:")
for i, row in feature_importance_goals.head(10).iterrows():
    print(f"  {i+1:2d}. {row['feature']:<25} {row['importance']:.4f}")

# =============================================================================
# 7. PREDICCIONS
# =============================================================================

print("\n🎯 FUNCIÓ DE PREDICCIÓ")
print("=" * 50)

# Funció per fer prediccions
def predicir_partit(equip_local, equip_visitant, df_classificacio, df_model, X_features, models):
    """
    Prediu el resultat d'un partit específic
    
    Args:
        equip_local (str): Nom de l'equip local
        equip_visitant (str): Nom de l'equip visitant
        df_classificacio: DataFrame amb estadístiques dels equips
        df_model: DataFrame del model entrenat
        X_features: Llista de features utilitzades
        models: Diccionari amb els models entrenats
    
    Returns:
        dict: Prediccions del partit
    """
    
    # Obtenim estadístiques dels equips
    stats_local = df_classificacio[df_classificacio['Team'] == equip_local]
    stats_visitant = df_classificacio[df_classificacio['Team'] == equip_visitant]
    
    if stats_local.empty or stats_visitant.empty:
        print(f"⚠️  No s'han trobat les dades per a {equip_local} o {equip_visitant}")
        return None
    
    # Creem vector de features
    features_partit = {}
    
    # Estadístiques dels equips
    for col in df_classificacio.columns:
        if col != 'Team':
            features_partit[f'Home_{col}'] = stats_local[col].iloc[0]
            features_partit[f'Away_{col}'] = stats_visitant[col].iloc[0]
    
    # Valors mitjans per a estadístiques del partit
    for col in X_features:
        if col not in features_partit:
            features_partit[col] = df_model[col].mean()
    
    # Convertim a DataFrame
    df_prediccio = pd.DataFrame([features_partit])
    X_prediccio = df_prediccio[X_features]
    
    # Fem prediccions
    prob_victory = models['victory'].predict_proba(X_prediccio)[0][1]
    pred_goals = models['goals'].predict(X_prediccio)[0]
    pred_corners = models['corners'].predict(X_prediccio)[0]
    pred_result = models['result'].predict(X_prediccio)[0]
    
    return {
        'equip_local': equip_local,
        'equip_visitant': equip_visitant,
        'probabilitat_victoria_local': f"{prob_victory:.1%}",
        'gols_totals_previstos': f"{pred_goals:.1f}",
        'corners_previstos': f"{pred_corners:.1f}",
        'resultat_previst': pred_result
    }

# Diccionari amb els models
models = {
    'victory': model_victory,
    'goals': model_goals,
    'corners': model_corners,
    'result': model_result
}

print("🎯 EXEMPLES DE PREDICCIÓ")
print("=" * 50)

# Exemple 1
print("\n🏆 Real Madrid vs Barcelona:")
prediccio1 = predicir_partit('Real Madrid', 'Barcelona', df_classificacio, df_model, X_features, models)
if prediccio1:
    for key, value in prediccio1.items():
        print(f"  {key}: {value}")

# Exemple 2
print("\n⚽ Girona vs Atlético Madrid:")
prediccio2 = predicir_partit('Girona', 'Atlético Madrid', df_classificacio, df_model, X_features, models)
if prediccio2:
    for key, value in prediccio2.items():
        print(f"  {key}: {value}")

# Exemple 3
print("\n🔵 Barcelona vs Real Madrid:")
prediccio3 = predicir_partit('Barcelona', 'Real Madrid', df_classificacio, df_model, X_features, models)
if prediccio3:
    for key, value in prediccio3.items():
        print(f"  {key}: {value}")

# =============================================================================
# 8. RESUM I CONCLUSIONS
# =============================================================================

print("\n📋 RESUM DEL MODEL")
print("=" * 50)

print(f"\n📊 Dades utilitzades:")
print(f"  - Partits analitzats: {len(df_partits)}")
print(f"  - Equips disponibles: {len(df_classificacio)}")
print(f"  - Features utilitzades: {len(X_features)}")

print(f"\n🤖 Models entrenats:")
print(f"  - Predicció de victòria (Classificació)")
print(f"  - Predicció de gols totals (Regressió)")
print(f"  - Predicció de corners (Regressió)")
print(f"  - Predicció de resultat (Classificació multiclasse)")

print(f"\n📈 Rendiment dels models:")
print(f"  - Precisió victòria: {victory_accuracy:.1%}")
print(f"  - RMSE gols: {np.sqrt(goals_mse):.2f}")
print(f"  - RMSE corners: {np.sqrt(corners_mse):.2f}")
print(f"  - Precisió resultat: {result_accuracy:.1%}")

print(f"\n🎯 Variables més importants:")
print(f"  - Per victòria: {feature_importance_victory.iloc[0]['feature']}")
print(f"  - Per gols: {feature_importance_goals.iloc[0]['feature']}")

print(f"\n✅ El model està llest per a fer prediccions!")

# =============================================================================
# 9. FUNCIÓ PER FER PREDICCIONS PERSONALITZADES
# =============================================================================

print("\n🎯 FUNCIÓ PER FER PREDICCIONS PERSONALITZADES")
print("=" * 60)

def fer_prediccio_personalitzada():
    """
    Funció interactiva per fer prediccions personalitzades
    """
    print("\n🎮 Predicció Personalitzada")
    print("-" * 30)
    
    # Mostrem equips disponibles
    print("🏆 Equips disponibles:")
    for i, equip in enumerate(df_classificacio['Team'].values, 1):
        print(f"  {i:2d}. {equip}")
    
    # Demanem equips
    try:
        equip_local = input("\n🏠 Equip local: ")
        equip_visitant = input("✈️  Equip visitant: ")
        
        # Fem la predicció
        prediccio = predicir_partit(equip_local, equip_visitant, df_classificacio, df_model, X_features, models)
        
        if prediccio:
            print(f"\n🎯 PREDICCIÓ: {equip_local} vs {equip_visitant}")
            print("=" * 40)
            for key, value in prediccio.items():
                print(f"  {key}: {value}")
        else:
            print("❌ No s'ha pogut fer la predicció. Verifica els noms dels equips.")
            
    except KeyboardInterrupt:
        print("\n👋 Fi del programa")
    except Exception as e:
        print(f"❌ Error: {e}")

# Comentem aquesta línia si no volem execució interactiva
# fer_prediccio_personalitzada()

print("\n🚀 Script completat amb èxit!")
print("💡 Per fer prediccions personalitzades, descomenta la línia 'fer_prediccio_personalitzada()' al final del script") 