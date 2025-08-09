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

# Identificar columnes numèriques (excloent 'Team')
columnes_numeriques = []
for col in df_classificacio_combinada.columns:
    if col != 'Team' and df_classificacio_combinada[col].dtype in ['int64', 'float64']:
        columnes_numeriques.append(col)

print(f"Columnes numèriques identificades: {len(columnes_numeriques)}")

# Crear diccionari d'agregacions
agg_dict = {}
for col in columnes_numeriques:
    agg_dict[col] = 'sum'

df_classificacio_fusionada = df_classificacio_combinada.groupby('Team').agg(agg_dict).reset_index()

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