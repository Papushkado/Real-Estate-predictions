import pandas as pd
import xgboost as xgb
import geopandas as gpd
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from datetime import datetime
import os
import re

def load_department_data(department_files):
    dfs = []
    for file in department_files:
        df = pd.read_csv(file, delimiter=';', encoding='utf-8')
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Liste des biens
RESIDENTIAL_PROPERTIES = [
    'UN APPARTEMENT', 'BATI MIXTE - LOGEMENT/ACTIVITE', 
    'APPARTEMENT INDETERMINE', 'DEUX APPARTEMENTS', 'UNE MAISON', 
    'BATI MIXTE - LOGEMENTS', 'DES MAISONS', 'MAISON - INDETERMINEE'
]

COMMERCIAL_PROPERTIES = ['ACTIVITE', 'BATI MIXTE - LOGEMENT/ACTIVITE']

# Définition des features
NUMERIC_FEATURES = ['surface']
CATEGORICAL_FEATURES = ['libtypbien', 'coddep', 'l_codinsee']

# Colonnes de surface
SURFACE_COLUMNS = [
    'sbati', 'sbatmai', 'sbatapt', 'sbatact', 
    'sapt1pp', 'sapt2pp', 'sapt3pp', 'sapt4pp', 'sapt5pp', 
    'smai1pp', 'smai2pp', 'smai3pp', 'smai4pp', 'smai5pp'
]

def clean_feature_name(name):
    # Remplacer les caractères spéciaux par des underscores
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
    # S'assurer que le nom commence par une lettre
    if clean_name[0].isdigit():
        clean_name = 'f_' + clean_name
    return clean_name

def get_min_non_zero_surface(row):
    values = row[SURFACE_COLUMNS]
    non_zero_values = values[values > 0]
    return non_zero_values.min() if len(non_zero_values) > 0 else 0

class FeatureProcessor:
    def __init__(self):
        self.encoders = {}
        self.feature_names = {}
        
    def fit(self, df, categorical_features):
        for feature in categorical_features:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoder.fit(df[[feature]])
            self.encoders[feature] = encoder
            
            # Générer et stocker des noms de features nettoyés
            categories = encoder.categories_[0]
            self.feature_names[feature] = [
                f"{clean_feature_name(feature)}_{clean_feature_name(val)}"
                for val in categories
            ]
            
    def transform(self, df, categorical_features):
        transformed_dfs = []
        
        # Traitement des features numériques
        numeric_df = df[NUMERIC_FEATURES].copy()
        transformed_dfs.append(numeric_df)
        
        # Traitement des features catégorielles
        for feature in categorical_features:
            encoder = self.encoders[feature]
            feature_encoded = encoder.transform(df[[feature]])
            feature_df = pd.DataFrame(
                feature_encoded,
                columns=self.feature_names[feature],
                index=df.index
            )
            transformed_dfs.append(feature_df)
        
        # Combiner toutes les features
        final_df = pd.concat(transformed_dfs, axis=1)
        return final_df

def prepare_data(df, property_types):
    # Filtrer par type de bien
    df_filtered = df[df['libtypbien'].isin(property_types)].copy()
    
    # Calculer la surface
    df_filtered['surface'] = df_filtered.apply(get_min_non_zero_surface, axis=1)
    df_filtered = df_filtered[df_filtered['surface'] != 0]
    
    # Calculer le prix au m²
    df_filtered['prix_m2'] = df_filtered['valeurfonc'] / df_filtered['surface']
    
    # Nettoyer les prix au m²
    df_filtered = df_filtered[df_filtered['prix_m2'].notna()]
    df_filtered = df_filtered[~df_filtered['prix_m2'].isin([np.inf, -np.inf])]
    df_filtered = df_filtered[df_filtered['prix_m2'] < 50000]
    df_filtered = df_filtered[df_filtered['prix_m2'] > 100]
    
    # Convertir les dates
    df_filtered['datemut'] = pd.to_datetime(df_filtered['datemut'])
    df_filtered['year'] = df_filtered['datemut'].dt.year
    df_filtered['month'] = df_filtered['datemut'].dt.month
    
    print("\nStatistiques des features apres preparation:")
    print(df_filtered[['prix_m2', 'surface']].describe())
    
    return df_filtered

def train_model(df, feature_processor):
    # Préparation des features
    X = feature_processor.transform(df, CATEGORICAL_FEATURES)
    y = df['prix_m2']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(
        max_depth=6,
        n_estimators=200,
        learning_rate=0.1,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        n_jobs=-1,
        random_state=42
    )
    
    print("Debut de l'entrainement")
    model.fit(X_train, y_train)
    
    # Calculer et afficher les métriques de performance
    test_predictions = model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_predictions)
    print(f"Test MSE: {test_mse}")
    
    # Afficher l'importance des features
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    })
    print("\nTop 10 features les plus importantes:")
    print(feature_importance.nlargest(10, 'importance'))
    
    return model, X_test, y_test

def calculate_yearly_prices(df, year):
    year_data = df[df['year'] == year]
    
    # Débogage
    print(f"\nAnalyse des donnees pour l'annee {year}:")
    print(f"Nombre total de transactions: {len(year_data)}")
    print("\nDistribution des prix au m² par commune:")
    print(year_data.groupby('l_codinsee')['prix_m2'].describe())
    
    return year_data.groupby('l_codinsee').agg({
        'prix_m2': ['mean', 'count']
    }).reset_index()

def predict_future_prices(model, df, feature_processor):
    latest_data = df[df['year'] == df['year'].max()].copy()
    latest_data['year'] = 2025
    
    print(f"\nAnalyse des donnees pour les predictions:")
    print(f"Nombre d'observations pour la prediction: {len(latest_data)}")
    print("\nDistribution des codes INSEE:")
    print(latest_data['l_codinsee'].value_counts().head())
    
    X_pred = feature_processor.transform(latest_data, CATEGORICAL_FEATURES)
    predictions = model.predict(X_pred)
    
    result_df = pd.DataFrame({
        'l_codinsee': latest_data['l_codinsee'],
        'predicted_price': predictions
    })
    
    # Débogage des prédictions
    print("\nDistribution des prix predits:")
    print(result_df['predicted_price'].describe())
    
    return result_df.groupby('l_codinsee')['predicted_price'].mean().reset_index()

def format_price(price):
    return f"{int(price):,} € / m²".replace(',', ' ')

def calculate_variation(current, previous):
    if previous == 0 or current == 0:
        return 0  # Pas de variation calculable
    return ((current - previous) / previous) * 100

def generate_popup_content(commune, prices_last_year, prices_current, prices_future):
    last_year_price = prices_last_year.get(commune, 0)
    current_price = prices_current.get(commune, 0)
    future_price = prices_future.get(commune, 0)
    
    # Si on n'a pas de prix actuel, on ne peut pas calculer de variations
    if current_price == 0:
        return {
            'commune': commune,
            'il_y_a_1_an': 'N/A',
            'aujourd_hui': 'N/A',
            'dans_1_an': 'N/A'
        }
    
    yearly_var = calculate_variation(current_price, last_year_price)
    future_var = calculate_variation(future_price, current_price)
    
    return {
        'commune': commune,
        'il_y_a_1_an': format_price(last_year_price) if last_year_price > 0 else 'N/A',
        'aujourd_hui': f"{yearly_var:.1f}% ({format_price(current_price)})",
        'dans_1_an': f"{future_var:.1f}% ({format_price(future_price)})" if future_price > 0 else 'N/A'
    }

def main():
    print("\nChargement des donnees")
    department_files = [f for f in os.listdir() if f.startswith("mutations_d75") and f.endswith(".csv")]
    df = load_department_data(department_files)
    
    print("\nPreparation des donnees")
    df_residential = prepare_data(df, RESIDENTIAL_PROPERTIES)")
    df_commercial = prepare_data(df, COMMERCIAL_PROPERTIES)
    
    feature_processor_residential = FeatureProcessor()
    feature_processor_residential.fit(df_residential, CATEGORICAL_FEATURES)
    
    feature_processor_commercial = FeatureProcessor()
    feature_processor_commercial.fit(df_commercial, CATEGORICAL_FEATURES)
    
    print("\nEntrainement des modeles")
    model_residential, X_test_res, y_test_res = train_model(df_residential, feature_processor_residential)
    model_commercial, X_test_com, y_test_com = train_model(df_commercial, feature_processor_commercial)
    
    print("\nCalcul des predictions")
    results = {}
    for property_type, df_filtered, model, feature_processor in [
        ('residential', df_residential, model_residential, feature_processor_residential),
        ('commercial', df_commercial, model_commercial, feature_processor_commercial)
    ]:
        prices_2023 = calculate_yearly_prices(df_filtered, 2023)
        prices_2024 = calculate_yearly_prices(df_filtered, 2024)
        predictions_2025 = predict_future_prices(model, df_filtered, feature_processor)
        
        communes_data = {}
        for commune in prices_2024['l_codinsee'].unique():
            # On ne garde que les prédictions pour un unique code INSEE et pas plusieurs 
            if not isinstance(commune, str):
                commune_str = str(commune)
            else:
                commune_str = commune
                
            if ',' not in commune_str and len(eval(commune_str)) == 1:
                commune_data = generate_popup_content(
                    commune,
                    dict(zip(prices_2023['l_codinsee'], prices_2023['prix_m2']['mean'])),
                    dict(zip(prices_2024['l_codinsee'], prices_2024['prix_m2']['mean'])),
                    dict(zip(predictions_2025['l_codinsee'], predictions_2025['predicted_price']))
                )
                communes_data[commune] = commune_data
        
        results[property_type] = communes_data
    
    print("\nSauvegarde des resultats")
    with open('property_prices.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("All Inclusive")

if __name__ == "__main__":
    main()