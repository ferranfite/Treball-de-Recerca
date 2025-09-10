
import pandas as pd

print("🔄 FUSIONANT BASES DE DADES EXCEL")
print("=" * 50)

# Llegir bases de dades
df_23_24_classificacio = pd.read_excel('BDD_EntrenamentModel_Estadístiques-La_Liga-2023-2024.xlsx', sheet_name='ClassificacióGeneral')
df_23_24_partits = pd.read_excel('BDD_EntrenamentModel_Estadístiques-La_Liga-2023-2024.xlsx', sheet_name='StatsPartit')

df_24_25_classificacio = pd.read_excel('BDD_EntrenamentModel_Estadístiques-La_Liga-2024-2025.xlsx', sheet_name='ClassificacióGeneral')
df_24_25_partits = pd.read_excel('BDD_EntrenamentModel_Estadístiques-La_Liga-2024-2025.xlsx', sheet_name='StatsPartit')

df_25_26_classificacio = pd.read_excel('BDD_EntrenamentModel_Estadístiques-La_Liga-2025-2026.xlsx', sheet_name='ClassificacióGeneral')
df_25_26_classificacio = df_25_26_classificacio.drop('Points', axis=1, errors='ignore')
# Renombrar columnes específiques: win, draw, loss -> Victory, Empats, Derrotes
rename_map_25_26 = {}
for col in df_25_26_classificacio.columns:
    key = col.lower() if isinstance(col, str) else col
    if key == 'win':
        rename_map_25_26[col] = 'Victory'
    elif key == 'draw':
        rename_map_25_26[col] = 'Empats'
    elif key == 'loss':
        rename_map_25_26[col] = 'Derrotes'
if rename_map_25_26:
    df_25_26_classificacio = df_25_26_classificacio.rename(columns=rename_map_25_26)
df_25_26_partits = pd.read_excel('BDD_EntrenamentModel_Estadístiques-La_Liga-2025-2026.xlsx', sheet_name='StatsPartit')

df_equips_ascendits_classificacio = pd.read_excel('BDD_EntrenamentModel_Estadístiques-La_Liga-EquipsAscendits.xlsx', sheet_name='ClassificacióGeneral')
df_equips_ascendits_partits = pd.read_excel('BDD_EntrenamentModel_Estadístiques-La_Liga-EquipsAscendits.xlsx', sheet_name='StatsPartit')

# Afegir columna temporada
df_23_24_classificacio['Temporada'] = '2023-2024'
df_23_24_partits['Temporada'] = '2023-2024'
df_24_25_classificacio['Temporada'] = '2024-2025'
df_24_25_partits['Temporada'] = '2024-2025'
df_25_26_classificacio['Temporada'] = '2025-2026'
df_25_26_partits['Temporada'] = '2025-2026'
df_equips_ascendits_classificacio['Temporada'] = 'Ascendits'
df_equips_ascendits_partits['Temporada'] = 'Ascendits'

# Fusionar partits
df_partits_fusionada = pd.concat([df_23_24_partits, df_24_25_partits, df_25_26_partits, df_equips_ascendits_partits], ignore_index=True, sort=False)

# Fusionar classificacions
df_classificacio_combinada = pd.concat([df_23_24_classificacio, df_24_25_classificacio, df_25_26_classificacio, df_equips_ascendits_classificacio], ignore_index=True, sort=False)
df_classificacio_combinada = df_classificacio_combinada.drop('Temporada', axis=1, errors='ignore')

# Definir columnes enters, percentatges i mitjanes
columnes_enters = ['MP', 'Victory', 'Empats', 'Derrotes', 'GF', 'GA', 'GD', 'Clean Sheets',
                   'Home MP', 'Home Win', 'Home Draw', 'Home Loss', 'Home GF', 'Home GA', 'Home Clean Sheets']
columnes_percentatges = ['%_Victories', '%_Empats', '%_Derrotes', '%_CS', '%_HomeWin', '%_HomeDraw', '%_HomeLoss', '%_HomeGoals']
columnes_mitjanes = ['M_GolsF', 'M_GolsC', 'M_HomeGF']

agg_dict = {col:'sum' for col in columnes_enters if col in df_classificacio_combinada.columns}
agg_dict.update({col:'first' for col in columnes_percentatges+columnes_mitjanes if col in df_classificacio_combinada.columns})

df_classificacio_fusionada = df_classificacio_combinada.groupby('Team').agg(agg_dict).reset_index()

# Recalcular percentatges i mitjanes
for idx, row in df_classificacio_fusionada.iterrows():
    if row['MP'] > 0:
        df_classificacio_fusionada.loc[idx, '%_Victories'] = row['Victory']/row['MP']
        df_classificacio_fusionada.loc[idx, '%_Empats'] = row['Empats']/row['MP']
        df_classificacio_fusionada.loc[idx, '%_Derrotes'] = row['Derrotes']/row['MP']
        df_classificacio_fusionada.loc[idx, '%_CS'] = row['Clean Sheets']/row['MP']
    if row['Home MP'] > 0:
        df_classificacio_fusionada.loc[idx, '%_HomeWin'] = row['Home Win']/row['Home MP']
        df_classificacio_fusionada.loc[idx, '%_HomeDraw'] = row['Home Draw']/row['Home MP']
        df_classificacio_fusionada.loc[idx, '%_HomeLoss'] = row['Home Loss']/row['Home MP']
    if row['GF'] > 0:
        df_classificacio_fusionada.loc[idx, '%_HomeGoals'] = row['Home GF']/row['GF']
    if row['MP'] > 0:
        df_classificacio_fusionada.loc[idx, 'M_GolsF'] = row['GF']/row['MP']
        df_classificacio_fusionada.loc[idx, 'M_GolsC'] = row['GA']/row['MP']
    if row['Home MP'] > 0:
        df_classificacio_fusionada.loc[idx, 'M_HomeGF'] = row['Home GF']/row['Home MP']

df_classificacio_fusionada = df_classificacio_fusionada.sort_values('Victory', ascending=False).reset_index(drop=True)

# Crear nova base de dades Excel
with pd.ExcelWriter('BDD_EntrenamentModel_Estadístiques-La_Liga-FUSIONADA.xlsx', engine='openpyxl') as writer:
    df_classificacio_fusionada.to_excel(writer, sheet_name='ClassificacióGeneral', index=False)
    df_partits_fusionada.to_excel(writer, sheet_name='StatsPartit', index=False)

print("✅ Base de dades fusionada amb la temporada 2025-2026 creada correctament!")
