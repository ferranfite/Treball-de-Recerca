import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

print("🚀 MILLORES DEL MODEL DE PREDICCIÓ")
print("=" * 50)

# 1. Carregar dades amb la nova base fusionada
print("📖 Carregant dades fusionades...")

df_classificacio = pd.read_excel('BDD_EntrenamentModel_Estadístiques-La_Liga-FUSIONADA.xlsx', 
                                sheet_name='ClassificacióGeneral').drop('Temporada', axis=1, errors='ignore')

df_partits = pd.read_excel('BDD_EntrenamentModel_Estadístiques-La_Liga-FUSIONADA.xlsx', 
                           sheet_name='StatsPartit').drop('Temporada', axis=1, errors='ignore')

print(f"✅ Dades carregades: {df_classificacio.shape}, {df_partits.shape}")

# 2. Crear features més sofisticades
print("\n🔧 Creant features avançades...")

def crear_features_avancades(df_partits, df_classificacio):
    """Creem features més sofisticades per millorar la predicció"""
    df_nou = df_partits.copy()
    
    # Estadístiques dels equips
    stats_columns = [col for col in df_classificacio.columns if col != 'Team']
    
    # Features bàsiques dels equips
    for col in stats_columns:
        df_nou[f'Home_{col}'] = df_nou['Home Team'].map(
            df_classificacio.set_index('Team')[col]
        ).fillna(df_classificacio[col].mean())
        
        df_nou[f'Away_{col}'] = df_nou['Away Team'].map(
            df_classificacio.set_index('Team')[col]
        ).fillna(df_classificacio[col].mean())
    
    # NOVES FEATURES AVANÇADES
    
    # 1. Diferència de formació entre equips
    df_nou['Form_Difference'] = df_nou['Home_%_Victories'] - df_nou['Away_%_Victories']
    df_nou['Goals_Difference'] = df_nou['Home_GF'] - df_nou['Away_GF']
    df_nou['Defense_Difference'] = df_nou['Away_GA'] - df_nou['Home_GA']
    
    # 2. Ràtios de rendiment
    df_nou['Home_Attack_Ratio'] = df_nou['Home_GF'] / (df_nou['Home_GF'] + df_nou['Home_GA']).replace(0, 1)
    df_nou['Away_Attack_Ratio'] = df_nou['Away_GF'] / (df_nou['Away_GF'] + df_nou['Away_GA']).replace(0, 1)
    
    # 3. Momentum (últimes victòries)
    df_nou['Home_Momentum'] = df_nou['Home_%_Victories'] * df_nou['Home_GF']
    df_nou['Away_Momentum'] = df_nou['Away_%_Victories'] * df_nou['Away_GF']
    
    # 4. Eficiència defensiva (usant partits totals en lloc de Matches)
    total_partits_home = (df_nou['Home_Victory'] + df_nou['Home_Empats'] + df_nou['Home_Derrotes']).replace(0, 1)
    total_partits_away = (df_nou['Away_Victory'] + df_nou['Away_Empats'] + df_nou['Away_Derrotes']).replace(0, 1)
    df_nou['Home_Defense_Efficiency'] = df_nou['Home_Clean Sheets'] / total_partits_home
    df_nou['Away_Defense_Efficiency'] = df_nou['Away_Clean Sheets'] / total_partits_away
    
    # 5. Pressió ofensiva
    df_nou['Home_Attack_Pressure'] = df_nou['Home_GF'] * df_nou['Home_%_Victories']
    df_nou['Away_Attack_Pressure'] = df_nou['Away_GF'] * df_nou['Away_%_Victories']
    
    return df_nou

# Aplicar millores
df_complet_millorat = crear_features_avancades(df_partits, df_classificacio)

# Variables objectiu
df_complet_millorat['Victory'] = (df_complet_millorat['Home Goal'] > df_complet_millorat['Away Goal']).astype(int)
df_complet_millorat['Total_Goals'] = df_complet_millorat['Home Goal'] + df_complet_millorat['Away Goal']
df_complet_millorat['Goal_Difference'] = df_complet_millorat['Home Goal'] - df_complet_millorat['Away Goal']

def crear_resultat_categoric(row):
    if row['Home Goal'] > row['Away Goal']:
        return 'Home_Win'
    elif row['Home Goal'] < row['Away Goal']:
        return 'Away_Win'
    else:
        return 'Draw'

df_complet_millorat['Result'] = df_complet_millorat.apply(crear_resultat_categoric, axis=1)

# 3. Selecció de features millorada
print("\n🎯 Seleccionant features millorades...")

# Features del partit
features_partit_cols = [
    'foulsCommitted', 'yellowCards', 'redCards', 'offsides', 'wonCorners',
    'saves', 'possessionPct', 'totalShots', 'shotsOnTarget', 'shotPct',
    'penaltyKickGoals', 'penaltyKickShots', 'accuratePasses', 'totalPasses',
    'passPct', 'accurateCrosses', 'totalCrosses', 'crossPct', 'accurateLongBalls',
    'totalLongBalls', 'longballPct', 'blockedShots', 'effectiveTackles',
    'totalTackles', 'tacklePct', 'interceptions', 'effectiveClearance', 'totalClearance'
]

# Features dels equips
features_equips = [col for col in df_complet_millorat.columns if col.startswith('Home_') or col.startswith('Away_')]

# NOVES FEATURES AVANÇADES
features_avancades = [
    'Form_Difference', 'Goals_Difference', 'Defense_Difference',
    'Home_Attack_Ratio', 'Away_Attack_Ratio', 'Home_Momentum', 'Away_Momentum',
    'Home_Defense_Efficiency', 'Away_Defense_Efficiency', 'Home_Attack_Pressure', 'Away_Attack_Pressure'
]

X_features_millorades = features_partit_cols + features_equips + features_avancades

# Dataset final millorat
df_model_millorat = df_complet_millorat[X_features_millorades + ['Victory', 'Total_Goals', 'wonCorners', 'Result']].dropna()
df_model_millorat = df_model_millorat.loc[:, ~df_model_millorat.columns.duplicated()]

X_millorat = df_model_millorat[X_features_millorades]
y_victory_millorat = df_model_millorat['Victory']
y_goals_millorat = df_model_millorat['Total_Goals']
y_corners_millorat = df_model_millorat['wonCorners']
y_result_millorat = df_model_millorat['Result']

print(f"✅ Dataset millorat: {df_model_millorat.shape}")
print(f"📊 Features utilitzades: {len(X_features_millorades)}")

# 4. Normalització de dades
print("\n📏 Aplicant normalització...")

scaler = StandardScaler()
X_millorat_scaled = scaler.fit_transform(X_millorat)

# 5. Divisió de dades amb estratificació
X_train, X_test, y_victory_train, y_victory_test = train_test_split(
    X_millorat_scaled, y_victory_millorat, test_size=0.2, random_state=42, stratify=y_victory_millorat
)

_, _, y_goals_train, y_goals_test = train_test_split(
    X_millorat_scaled, y_goals_millorat, test_size=0.2, random_state=42
)

_, _, y_corners_train, y_corners_test = train_test_split(
    X_millorat_scaled, y_corners_millorat, test_size=0.2, random_state=42
)

_, _, y_result_train, y_result_test = train_test_split(
    X_millorat_scaled, y_result_millorat, test_size=0.2, random_state=42, stratify=y_result_millorat
)

# 6. Models millorats amb hiperparàmetres optimitzats
print("\n🤖 ENTRENANT MODELS MILLORATS")
print("=" * 50)

# Model 1: Victòria amb Gradient Boosting
print("\n1️⃣ Model de Predicció de Victòria (Gradient Boosting)")
model_victory_millorat = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
model_victory_millorat.fit(X_train, y_victory_train)
victory_pred_millorat = model_victory_millorat.predict(X_test)
victory_accuracy_millorat = accuracy_score(y_victory_test, victory_pred_millorat)
print(f"   ✅ Precisió: {victory_accuracy_millorat:.3f}")

# Model 2: Gols amb Random Forest optimitzat
print("\n2️⃣ Model de Predicció de Gols Totals (Random Forest Optimitzat)")
model_goals_millorat = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model_goals_millorat.fit(X_train, y_goals_train)
goals_pred_millorat = model_goals_millorat.predict(X_test)
goals_mse_millorat = mean_squared_error(y_goals_test, goals_pred_millorat)
print(f"   ✅ RMSE: {np.sqrt(goals_mse_millorat):.3f}")

# Model 3: Corners amb Gradient Boosting
print("\n3️⃣ Model de Predicció de Corners (Gradient Boosting)")
model_corners_millorat = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
model_corners_millorat.fit(X_train, y_corners_train)
corners_pred_millorat = model_corners_millorat.predict(X_test)
corners_mse_millorat = mean_squared_error(y_corners_test, corners_pred_millorat)
print(f"   ✅ RMSE: {np.sqrt(corners_mse_millorat):.3f}")

# Model 4: Resultat amb Random Forest optimitzat
print("\n4️⃣ Model de Predicció de Resultat (Random Forest Optimitzat)")
model_result_millorat = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=3,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42
)
model_result_millorat.fit(X_train, y_result_train)
result_pred_millorat = model_result_millorat.predict(X_test)
result_accuracy_millorat = accuracy_score(y_result_test, result_pred_millorat)
print(f"   ✅ Precisió: {result_accuracy_millorat:.3f}")

print("\n📊 Report de classificació millorat (Resultat):")
print(classification_report(y_result_test, result_pred_millorat))

# 7. Comparació de resultats
print("\n📈 COMPARACIÓ DE RESULTATS")
print("=" * 50)

print("MODEL ORIGINAL vs MODEL MILLORAT:")
print(f"Victòria:    73.7% → {victory_accuracy_millorat:.1%}")
print(f"Gols RMSE:   1.97 → {np.sqrt(goals_mse_millorat):.2f}")
print(f"Corners RMSE: 3.01 → {np.sqrt(corners_mse_millorat):.2f}")
print(f"Resultat:    38.2% → {result_accuracy_millorat:.1%}")

# 8. Anàlisi d'importància de features millorada
print("\n🔍 ANÀLISI D'IMPORTÀNCIA MILLORADA")
print("=" * 50)

feature_importance_victory_millorat = pd.DataFrame({
    'feature': X_features_millorades,
    'importance': model_victory_millorat.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🏆 Top 10 features millorades per a predicció de victòria:")
for i, row in feature_importance_victory_millorat.head(10).iterrows():
    print(f"  {i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")

print("\n✅ Millores aplicades amb èxit!")
print("🎯 Principals millores:")
print("  - Features avançades (diferències, ràtios, momentum)")
print("  - Normalització de dades")
print("  - Models més sofisticats (Gradient Boosting)")
print("  - Hiperparàmetres optimitzats")
print("  - Més dades (base fusionada)") 