import pandas as pd
import numpy as np
from datetime import datetime

print("🔄 FUSIONANT BASES DE DADES EXCEL")
print("=" * 50)

# 1. Llegir les dues bases de dades
print("📖 Llegint les bases de dades...")

# Base de dades 2023-2024
df_23_24_classificacio = pd.read_excel('BDD_EntrenamentModel_Estadístiques-La_Liga-2023-2024.xlsx', 
                                       sheet_name='ClassificacióGeneral')
df_23_24_partits = pd.read_excel('BDD_EntrenamentModel_Estadístiques-La_Liga-2023-2024.xlsx', 
                                  sheet_name='StatsPartit')

# Base de dades 2024-2025
df_24_25_classificacio = pd.read_excel('BDD_EntrenamentModel_Estadístiques-La_Liga-2024-2025.xlsx', 
                                       sheet_name='ClassificacióGeneral')
df_24_25_partits = pd.read_excel('BDD_EntrenamentModel_Estadístiques-La_Liga-2024-2025.xlsx', 
                                  sheet_name='StatsPartit')

print(f"✅ Base 2023-2024 - Classificació: {df_23_24_classificacio.shape}")
print(f"✅ Base 2023-2024 - Partits: {df_23_24_partits.shape}")
print(f"✅ Base 2024-2025 - Classificació: {df_24_25_classificacio.shape}")
print(f"✅ Base 2024-2025 - Partits: {df_24_25_partits.shape}")

# 2. Afegir columna de temporada
print("\n📅 Afegint columna de temporada...")

df_23_24_classificacio['Temporada'] = '2023-2024'
df_23_24_partits['Temporada'] = '2023-2024'

df_24_25_classificacio['Temporada'] = '2024-2025'
df_24_25_partits['Temporada'] = '2024-2025'

# 3. Fusionar les dades
print("\n🔗 Fusionant dades...")

# Fusionar partits (mantenir separats per temporada)
df_partits_fusionada = pd.concat([df_23_24_partits, df_24_25_partits], 
                                 ignore_index=True, sort=False)

# Fusionar classificacions (sumar estadístiques dels equips)
print("📊 Fusionant estadístiques dels equips...")

# Combinar les dues classificacions
df_classificacio_combinada = pd.concat([df_23_24_classificacio, df_24_25_classificacio], 
                                       ignore_index=True, sort=False)

# Eliminar columna Temporada per evitar problemes
df_classificacio_combinada = df_classificacio_combinada.drop('Temporada', axis=1, errors='ignore')

# Identificar columnes per sumar (valors enters) i per recalcular (percentatges)
print("📊 Identificant columnes per sumar i recalcular...")

# Columnes que són valors enters (sumar)
columnes_enters = [
    'MP', 'Victory', 'Empats', 'Derrotes', 'GF', 'GA', 'GD', 'Clean Sheets',
    'Home MP', 'Home Win', 'Home Draw', 'Home Loss', 'Home GF', 'Home GA', 'Home Clean Sheets'
]

# Columnes que són percentatges (recalcular després)
columnes_percentatges = [
    '%_Victories', '%_Empats', '%_Derrotes', '%_CS',
    '%_HomeWin', '%_HomeDraw', '%_HomeLoss', '%_HomeGoals'
]

# Columnes que són mitjanes (recalcular després)
columnes_mitjanes = [
    'M_GolsF', 'M_GolsC', 'M_HomeGF'
]

# Crear diccionari d'agregacions
agg_dict = {}

# Sumar només els valors enters
for col in columnes_enters:
    if col in df_classificacio_combinada.columns:
        agg_dict[col] = 'sum'

# Per percentatges i mitjanes, agafar el primer valor (es recalcularan després)
for col in columnes_percentatges + columnes_mitjanes:
    if col in df_classificacio_combinada.columns:
        agg_dict[col] = 'first'

print(f"Columnes enters a sumar: {len([col for col in columnes_enters if col in df_classificacio_combinada.columns])}")
print(f"Columnes percentatges a recalcular: {len([col for col in columnes_percentatges if col in df_classificacio_combinada.columns])}")

df_classificacio_fusionada = df_classificacio_combinada.groupby('Team').agg(agg_dict).reset_index()

# Recalcular percentatges i mitjanes
print("🔄 Recalculant percentatges i mitjanes...")

for idx, row in df_classificacio_fusionada.iterrows():
    # Percentatges generals
    if 'Victory' in row and 'MP' in row and row['MP'] > 0:
        df_classificacio_fusionada.loc[idx, '%_Victories'] = row['Victory'] / row['MP']
        df_classificacio_fusionada.loc[idx, '%_Empats'] = row['Empats'] / row['MP']
        df_classificacio_fusionada.loc[idx, '%_Derrotes'] = row['Derrotes'] / row['MP']
    
    # Percentatges de clean sheets
    if 'Clean Sheets' in row and 'MP' in row and row['MP'] > 0:
        df_classificacio_fusionada.loc[idx, '%_CS'] = row['Clean Sheets'] / row['MP']
    
    # Percentatges de casa
    if 'Home Win' in row and 'Home MP' in row and row['Home MP'] > 0:
        df_classificacio_fusionada.loc[idx, '%_HomeWin'] = row['Home Win'] / row['Home MP']
        df_classificacio_fusionada.loc[idx, '%_HomeDraw'] = row['Home Draw'] / row['Home MP']
        df_classificacio_fusionada.loc[idx, '%_HomeLoss'] = row['Home Loss'] / row['Home MP']
    
    # Percentatges de gols a casa
    if 'Home GF' in row and 'GF' in row and row['GF'] > 0:
        df_classificacio_fusionada.loc[idx, '%_HomeGoals'] = row['Home GF'] / row['GF']
    
    # Mitjanes
    if 'GF' in row and 'MP' in row and row['MP'] > 0:
        df_classificacio_fusionada.loc[idx, 'M_GolsF'] = row['GF'] / row['MP']
    if 'GA' in row and 'MP' in row and row['MP'] > 0:
        df_classificacio_fusionada.loc[idx, 'M_GolsC'] = row['GA'] / row['MP']
    if 'Home GF' in row and 'Home MP' in row and row['Home MP'] > 0:
        df_classificacio_fusionada.loc[idx, 'M_HomeGF'] = row['Home GF'] / row['Home MP']

print("✅ Percentatges i mitjanes recalculats correctament")

# Ordenar per victòries de més a menys
df_classificacio_fusionada = df_classificacio_fusionada.sort_values('Victory', ascending=False).reset_index(drop=True)

print(f"✅ Estadístiques fusionades per equip (ordenades per victòries)")

print(f"✅ Classificació fusionada: {df_classificacio_fusionada.shape}")
print(f"✅ Partits fusionats: {df_partits_fusionada.shape}")

# 4. Verificar la fusió
print("\n🔍 Verificant la fusió...")

print(f"\n📊 Distribució per temporada (Partits):")
print(df_partits_fusionada['Temporada'].value_counts())

print(f"\n🏆 Equips únics en classificació (estadístiques fusionades):")
equips_unic = df_classificacio_fusionada['Team'].unique()
print(f"Total equips: {len(equips_unic)}")
for equip in sorted(equips_unic):
    print(f"  - {equip}")

print(f"\n📈 Exemple d'estadístiques fusionades (Real Madrid):")
madrid_stats = df_classificacio_fusionada[df_classificacio_fusionada['Team'] == 'Real Madrid']
if not madrid_stats.empty:
    print(f"  - Victòries: {madrid_stats['Victory'].iloc[0]}")
    print(f"  - Gols a favor: {madrid_stats['GF'].iloc[0]}")
    print(f"  - Gols en contra: {madrid_stats['GA'].iloc[0]}")

print(f"\n🏆 Classificació ordenada per victòries:")
for i, (_, equip) in enumerate(df_classificacio_fusionada[['Victory', 'Team']].head(10).iterrows(), 1):
    print(f"  {i:2d}. {equip['Team']:<20} - {equip['Victory']} victòries")

# 5. Crear la nova base de dades Excel
print("\n💾 Creant nova base de dades Excel...")

with pd.ExcelWriter('BDD_EntrenamentModel_Estadístiques-La_Liga-FUSIONADA.xlsx', engine='openpyxl') as writer:
    df_classificacio_fusionada.to_excel(writer, sheet_name='ClassificacióGeneral', index=False)
    df_partits_fusionada.to_excel(writer, sheet_name='StatsPartit', index=False)

print("✅ Base de dades fusionada creada: 'BDD_EntrenamentModel_Estadístiques-La_Liga-FUSIONADA.xlsx'")

# 6. Estadístiques de la nova base de dades
print("\n📈 ESTADÍSTIQUES DE LA NOVA BASE DE DADES")
print("=" * 50)

print(f"📊 Classificació (estadístiques fusionades):")
print(f"  - Total equips únics: {len(df_classificacio_fusionada)}")
print(f"  - Variables: {len(df_classificacio_fusionada.columns)}")
print(f"  - Estadístiques sumades de les dues temporades")

print(f"\n⚽ Partits:")
print(f"  - Total partits: {len(df_partits_fusionada)}")
print(f"  - Partits per temporada: {len(df_partits_fusionada) // 2}")
print(f"  - Variables: {len(df_partits_fusionada.columns)}")

print(f"\n🎯 Avantatges de la fusió:")
print(f"  - Més dades per entrenar els models")
print(f"  - Millor generalització dels models")
print(f"  - Anàlisi de tendències entre temporades")
print(f"  - Més robustesa en les prediccions")

print("\n✅ Fusió completada amb èxit!")
print("💡 Ara pots usar 'BDD_EntrenamentModel_Estadístiques-La_Liga-FUSIONADA.xlsx' al teu model") 