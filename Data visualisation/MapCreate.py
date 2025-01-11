import pandas as pd
import xgboost as xgb
import folium
import geopandas as gpd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np


df = pd.read_csv("mutations_d75.csv", delimiter=';')

with open("/departements/Paris.p", 'rb') as file:
    gdf_paris = pickle.load(file)


print(type(gdf_paris))

# Liste des biens pour l'onglet 1 (logements)
liste_biens_1 = ['UN APPARTEMENT', 'BATI MIXTE - LOGEMENT/ACTIVITE', 'APPARTEMENT INDETERMINE', 
                 'DEUX APPARTEMENTS', 'UNE MAISON', 'BATI MIXTE - LOGEMENTS', 'DES MAISONS', 'MAISON - INDETERMINEE']
# Liste des biens pour l'onglet 2 (locaux commerciaux)
liste_biens_2 = ['ACTIVITE', 'BATI MIXTE - LOGEMENT/ACTIVITE']


df_logements = df[df['libtypbien'].isin(liste_biens_1)]
df_locaux = df[df['libtypbien'].isin(liste_biens_2)]

# Liste des colonnes de surface
surface_columns = ['sbati', 'sbatmai', 'sbatapt', 'sbatact', 'sapt1pp', 'sapt2pp', 'sapt3pp', 'sapt4pp', 'sapt5pp', 'smai1pp', 'smai2pp', 'smai3pp', 'smai4pp', 'smai5pp']

# Pour récupérer la surface du bien 
def get_min_non_zero(row):
    values = row[surface_columns]
    non_zero_values = values[values > 0]  
    if len(non_zero_values) > 0:
        return non_zero_values.min()  # Retourner la plus petite valeur non nulle
    return 0  # Retourner 0 si toutes les valeurs sont nulles

# Appliquer cette fonction pour calculer la surface dans les deux DataFrames
df_logements['surface'] = df_logements.apply(get_min_non_zero, axis=1)
df_locaux['surface'] = df_locaux.apply(get_min_non_zero, axis=1)

print("0 dans Locaux : " +str((df_logements['surface'] == 0).sum()))
print("NaN dans Locaux : " +str(df_logements['surface'].isna().sum()))

print("0 dans Locaux : " +str((df_locaux['surface'] == 0).sum()))
print("NaN dans Locaux : " +str(df_locaux['surface'].isna().sum()))


# Étape de traitement des NaN ou infinies

df_logements = df_logements.replace([np.inf, -np.inf], np.nan)
df_locaux = df_locaux.replace([np.inf, -np.inf], np.nan)
df_logements['surface'].fillna(0, inplace=True)
df_locaux['surface'].fillna(0, inplace=True)

# Prédiction des prix au mètre carré pour les logements
df_logements = df_logements[['valeurfonc', 'nbdispo', 'nbcomm', 'surface']]  # Sélectionner les features pertinentes
X_logements = df_logements.drop(columns='valeurfonc')
y_logements = df_logements['valeurfonc']

X_train, X_test, y_train, y_test = train_test_split(X_logements, y_logements, test_size=0.2, random_state=42)

#################################################### Debug 

def check_invalid_values(y):
    if y.isna().any():
        print("Il y a des NaN dans y_train")
    if np.isinf(y).any():
        print("Il y a des valeurs infinies dans y_train")
    if (np.abs(y) > 1e10).any():
        print("Il y a des valeurs trop grandes dans y_train")


check_invalid_values(y_train)

####################################################


#################################################### Modèles de prédictions


################################# LOGEMENST
model_logements = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', n_estimators=100)
model_logements.fit(X_train, y_train)

y_pred_logements = model_logements.predict(X_test)
mse_logements = mean_squared_error(y_test, y_pred_logements)
print(f'Mean Squared Error (Logements): {mse_logements}')

################################# LOCAUX COMMERCIAUX
df_locaux = df_locaux[['valeurfonc', 'nbdispo', 'nbcomm', 'surface']]  # Sélectionner les features pertinentes
X_locaux = df_locaux.drop(columns='valeurfonc')
y_locaux = df_locaux['valeurfonc']

X_train_locaux, X_test_locaux, y_train_locaux, y_test_locaux = train_test_split(X_locaux, y_locaux, test_size=0.2, random_state=42)

model_locaux = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', n_estimators=100)
model_locaux.fit(X_train_locaux, y_train_locaux)

y_pred_locaux = model_locaux.predict(X_test_locaux)
mse_locaux = mean_squared_error(y_test_locaux, y_pred_locaux)
print(f'Mean Squared Error (Locaux commerciaux): {mse_locaux}')

#################################################### CARTE 

m = folium.Map(location=[48.8566, 2.3522], zoom_start=10) 


for index, row in gdf_paris.iterrows():
    commune_code = row['code_insee']
    nom_commune = row['nom']
    lat, lon = row['geometry'].centroid.y, row['geometry'].centroid.x


    prix_2024 = df_logements[df_logements['l_codinsee'].apply(lambda x: commune_code in x)]['valeurfonc'].mean()
    if pd.isna(prix_2024):
        prix_2024 = 0  # Si aucune donnée, mettre 0

    folium.Marker(
        location=[lat, lon],
        popup=f"{nom_commune}: {prix_2024:.2f} €/m² (2024)",
        icon=folium.Icon(color="blue")
    ).add_to(m)


m.save("carte_ile_de_france.html")
print("Carte générée avec succès !")
