import pandas as pd
import xgboost as xgb
import folium
import geopandas as gpd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import numpy as np


df = pd.read_csv("mutations_d75.csv", delimiter=';')
df_confiance = pd.read_csv("confiance_menage.csv")
df_petrole = pd.read_csv("petrole.csv")
df_taux = pd.read_csv("taux.csv")

with open("./departements/Paris.p", 'rb') as file:
    gdf_paris = pickle.load(file)

print(type(gdf_paris))

# Liste des biens pour l'onglet 1 (logements)
liste_biens_1 = ['UN APPARTEMENT', 'BATI MIXTE - LOGEMENT/ACTIVITE', 'APPARTEMENT INDETERMINE', 
                 'DEUX APPARTEMENTS', 'UNE MAISON', 'BATI MIXTE - LOGEMENTS', 'DES MAISONS', 'MAISON - INDETERMINEE']
# Liste des biens pour l'onglet 2 (locaux commerciaux)
liste_biens_2 = ['ACTIVITE', 'BATI MIXTE - LOGEMENT/ACTIVITE']


df_logements = df[df['libtypbien'].isin(liste_biens_1)]
df_locaux = df[df['libtypbien'].isin(liste_biens_2)]

print("filtrage type de bien  : check")

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
df_logements = df_logements[df_logements['surface'] != 0]

df_locaux['surface'] = df_locaux.apply(get_min_non_zero, axis=1)
df_locaux = df_locaux[df_locaux['surface'] != 0]

print("0 dans Logements : " +str((df_logements['surface'] == 0).sum()))
print("NaN dans Logements : " +str(df_logements['surface'].isna().sum()))

print("0 dans Locaux : " +str((df_locaux['surface'] == 0).sum()))
print("NaN dans Locaux : " +str(df_locaux['surface'].isna().sum()))


# Étape de traitement des NaN ou infinies
df_logements = df_logements.replace([np.inf, -np.inf], np.nan)
df_locaux = df_locaux.replace([np.inf, -np.inf], np.nan)
df_logements['surface'].fillna(0, inplace=True)
df_locaux['surface'].fillna(0, inplace=True)

# Préparation des données externes
df_logements['datemut'] = pd.to_datetime(df_logements['datemut'])
df_logements['year_month'] = df_logements['datemut'].dt.to_period('M')

df_confiance['date'] = pd.to_datetime(df_confiance['date'])
df_confiance['year_month'] = df_confiance['date'].dt.to_period('M')
df_petrole['date'] = pd.to_datetime(df_petrole['date'])
df_petrole['year_month'] = df_petrole['date'].dt.to_period('M')
df_taux['date'] = pd.to_datetime(df_taux['date'])
df_taux['year_month'] = df_taux['date'].dt.to_period('M')

# Fusion des données
df_logements = df_logements.merge(df_confiance[['year_month', 'Indicateur synthétique']], 
                                 on='year_month', 
                                 how='left')
df_logements = df_logements.merge(df_petrole[['year_month', 'Europe Brent Spot Price FOB (Dollars per Barrel)']], 
                                 on='year_month', 
                                 how='left')
df_logements = df_logements.merge(df_taux[['year_month', 'taux_directeur']], 
                                 on='year_month', 
                                 how='left')

# Renommage des colonnes
df_logements = df_logements.rename(columns={
    'Indicateur synthétique': 'confiance_menages',
    'Europe Brent Spot Price FOB (Dollars per Barrel)': 'cours_petrole'
})

# Encodage des variables catégorielles
le = LabelEncoder()
df_logements['l_codinsee_encoded'] = le.fit_transform(df_logements['l_codinsee'])
df_logements['coddep_encoded'] = le.fit_transform(df_logements['coddep'])

# Prédiction des prix au mètre carré pour les logements
numeric_features = ['surface', 'confiance_menages', 'taux_directeur', 'cours_petrole']
categorical_features = ['coddep_encoded', 'l_codinsee_encoded']
features = numeric_features + categorical_features

X_logements = df_logements[features]
y_logements = df_logements['valeurfonc']

X_train, X_test, y_train, y_test = train_test_split(X_logements, y_logements, test_size=0.2, random_state=42)

print("split des données : check")
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


################################# LOGEMENTS
# Paramètres pour GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200],
    'min_child_weight': [1, 3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Création et optimisation du modèle
base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=2
)

grid_search.fit(X_train, y_train)
print("Meilleurs paramètres:", grid_search.best_params_)
model_logements = grid_search.best_estimator_

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