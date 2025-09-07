# Aquest codi serveix per a descarregar la BDD de la La Liga 2025-2026
# de moment no funciona perquè hi ha un captcha ja ho solucionarem

import requests

url = "https://www.excel4soccer.com/download/soccer-stats-in-excel-la-liga-2025-2026/?wpdmdl=2030&_wpdmkey=68bd69bd12e6d"
nom_fitxer = "BDD_EntrenamentModel_Estadístiques-La_Liga-2025-2026.xlsx"

resposta = requests.get(url)

with open(nom_fitxer, "wb") as f:
    f.write(resposta.content)

print("Descarregat correctament")
