import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import pickle
import json
def convert_to_numeric(df, exclude_columns=None):
    exclude_columns = exclude_columns or []
    for column in df.columns:
        if column not in exclude_columns:
            if df[column].dtype == 'object':
                # Remplacer les tirets par NaN
                df[column] = df[column].replace(['-'], np.nan)
                # Convertir en numérique
                df[column] = pd.to_numeric(df[column].str.replace(',', '.'), errors='coerce')
    return df

def preprocess_data():
    print("Début du prétraitement des données...")
    
    print("Chargement des données macro-économiques...")
    confiance_df = pd.read_csv('confiance_menage.csv', sep=';', encoding='latin-1')
    confiance_df = confiance_df[['date', 'Indicateur synthï¿½tique']].rename(columns={'Indicateur synthï¿½tique': 'confiance_menages'})
    confiance_df = confiance_df.dropna(subset=['date'])
    confiance_df['date'] = pd.to_datetime(confiance_df['date'], format='%b-%y', errors='coerce')
    confiance_df['date'] = confiance_df['date'].apply(lambda x: x.replace(year=x.year + 100) if x.year < 2000 else x)
    confiance_df = convert_to_numeric(confiance_df, exclude_columns=['date'])
    
    
    taux_df = pd.read_csv('taux.csv', sep=';', encoding='latin-1')
    taux_df['ï»¿date'] = taux_df['ï»¿date'].str.replace('\ufeff', '')  # Suppression BOM
    taux_df['ï»¿date'] = pd.to_datetime(taux_df['ï»¿date'], format='%Y-%m-%d')
    taux_df.columns = ['date', 'taux_directeur']
    taux_df = convert_to_numeric(taux_df, exclude_columns=['date'])
    
    petrole_df = pd.read_csv('petrole.csv', sep=';', encoding='latin-1')
    petrole_df['date'] = pd.to_datetime(petrole_df['date'], format='%Y')
    petrole_df = petrole_df[['date', 'Europe Brent Spot Price FOB (Dollars per Barrel)']].rename(columns={'Europe Brent Spot Price FOB (Dollars per Barrel)': 'cours_petrole'})
    petrole_df = convert_to_numeric(petrole_df, exclude_columns=['date'])
    
    print("Préparation des données macro...")
    macro_df = pd.merge(confiance_df, taux_df, on='date', how='outer')
    macro_df = pd.merge(macro_df, petrole_df, on='date', how='outer')
    macro_df = macro_df.sort_values('date').interpolate(method='linear')
    macro_df['year_month'] = macro_df['date'].dt.to_period('M')
    macro_df.to_pickle('processed_data/macro_data.pkl')

    # Chargement des données immobilières en chunks
    print("Chargement des données immobilières...")
    chunk_size = 50000  # Ajuster selon la mémoire disponible
    dfs = []
    
    for dept in ['75', '77', '78', '91', '92', '93', '94', '95']:
        try:
            for chunk in pd.read_csv(f'mutations_d{dept}.csv', delimiter=';', chunksize=chunk_size):
                chunk['departement'] = dept
                # Prétraitement minimal pour le chunk
                chunk['date_mutation'] = pd.to_datetime(chunk['datemut'])
                chunk['year_month'] = chunk['date_mutation'].dt.to_period('M')
                chunk['l_codinsee'] = chunk['l_codinsee'].str.extract(r'\'(\d+)\'')
                chunk['type_bien'] = chunk['libtypbien'].fillna('Non spécifié')
                chunk['surface'] = chunk.apply(lambda x: x['sbati'] if x['sterr'] == 0 else x['sterr'], axis=1)
                chunk['prix_m2'] = np.where(chunk['surface'] > 0, chunk['valeurfonc'] / chunk['surface'], np.nan)
                
                # Sélection des colonnes nécessaires uniquement
                chunk = chunk[['year_month', 'l_codinsee', 'type_bien', 'surface', 'prix_m2', 'departement', 'valeurfonc']]
                dfs.append(chunk)
        except FileNotFoundError:
            print(f"Fichier mutations_d{dept}.csv non trouvé")
            continue
    
    if not dfs:
        raise FileNotFoundError("Aucun fichier de données immobilières trouvé")
    
    print("Fusion des chunks...")
    df = pd.concat(dfs, ignore_index=True)
    del dfs  
    
    df = pd.merge(df, macro_df, on='year_month', how='left')
    
    df = df[df['prix_m2'].notna()]
    Q1, Q3 = df['prix_m2'].quantile([0.01, 0.99])
    df = df[(df['prix_m2'] >= Q1) & (df['prix_m2'] <= Q3)]

    df.to_pickle('processed_data/clean_data.pkl')

    # Statistiques
    print("Calcul des statistiques...")
    stats_annuelles = df.groupby('year_month').agg({
        'prix_m2': ['mean', 'median', 'std', 'count'],
        'valeurfonc': ['mean', 'sum'],
        'surface': 'mean'
    }).round(2)
    stats_annuelles.to_pickle('processed_data/stats_annuelles.pkl')

    stats_communes = df.groupby('l_codinsee').agg({
        'prix_m2': ['mean', 'median', 'std', 'count'],
        'valeurfonc': ['mean', 'sum'],
        'surface': 'mean'
    }).round(2)
    stats_communes.to_pickle('processed_data/stats_communes.pkl')

    print("Préparation des modèles...")
    numeric_features = ['surface', 'confiance_menages', 'taux_directeur', 'cours_petrole']
    categorical_features = ['type_bien', 'departement', 'l_codinsee']
    

    encoders = {}
    X = df[numeric_features].copy()
    X = X.astype('float32')  
    
    for feature in categorical_features:
        le = LabelEncoder()
        X[feature] = le.fit_transform(df[feature].astype(str))
        encoders[feature] = le
    
    # Sauvegarder les encodeurs
    with open('processed_data/label_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    y = df['prix_m2'].astype('float32')

    # Imputation
    print("Imputation des valeurs manquantes...")
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Entraînement des modèles
    print("Entraînement des modèles...")
    models = {
        'linear': LinearRegression(),
        'rf': RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1),
        'xgb': xgb.XGBRegressor(max_depth=6, n_estimators=100, learning_rate=0.1, n_jobs=-1)
    }
    
    for name, model in models.items():
        print(f"Entraînement du modèle {name}...")
        model.fit(X, y)
        with open(f'processed_data/model_{name}.pkl', 'wb') as f:
            pickle.dump(model, f)

    print("Prétraitement terminé avec succès!")

if __name__ == "__main__":
    preprocess_data()