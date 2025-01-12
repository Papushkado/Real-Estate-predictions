import React from 'react';

const ModelExplanation = () => {
  return (
    <div className="bg-white p-8 rounded-lg shadow-lg max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-6">Notre Modèle de Prédiction by Stephen Cohen, Kien Pham Trung, Antoine Rech--Tronville </h2>
      
      <div className="space-y-8">
        <section>
          <h3 className="text-xl font-semibold mb-3">Approche Méthodologique</h3>
          <p className="text-gray-700 mb-4">
            Notre modèle utilise l'algorithme XGBoost pour prédire l'évolution des prix immobiliers à Paris.
            Cette approche a été choisie pour sa capacité à capturer des relations complexes dans les données
            immobilières et sa robustesse face aux valeurs aberrantes.
          </p>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h4 className="font-medium mb-2">Points clés de la méthodologie :</h4>
            <ul className="list-disc pl-5 space-y-2 text-gray-600">
              <li>Entraînement séparé pour les biens résidentiels et commerciaux</li>
              <li>Utilisation exclusive des données intrinsèques aux biens</li>
              <li>Validation croisée pour éviter le surapprentissage</li>
              <li>Mise à jour régulière avec les nouvelles transactions</li>
            </ul>
          </div>
        </section>

        <section>
          <h3 className="text-xl font-semibold mb-3">Variables du Modèle</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-medium mb-2">Variables Numériques</h4>
              <ul className="list-disc pl-5 space-y-1 text-gray-600">
                <li>Surface du bien</li>
                <li>Prix historiques</li>
                <li>Nombre de transactions</li>
              </ul>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-medium mb-2">Variables Catégorielles</h4>
              <ul className="list-disc pl-5 space-y-1 text-gray-600">
                <li>Type de bien</li>
                <li>Arrondissement</li>
                <li>Code INSEE</li>
              </ul>
            </div>
          </div>
        </section>

        <section>
          <h3 className="text-xl font-semibold mb-3">Performance et Validation</h3>
          <p className="text-gray-700 mb-4">
            Le modèle est évalué sur plusieurs métriques pour assurer sa fiabilité :
          </p>
          <div className="bg-gray-50 p-4 rounded-lg">
            <ul className="list-disc pl-5 space-y-2 text-gray-600">
              <li>Erreur moyenne de prédiction inférieure à 5% sur les données de test</li>
              <li>Validation sur des données historiques pour vérifier la précision des tendances</li>
              <li>Tests de robustesse face aux variations saisonnières</li>
            </ul>
          </div>
        </section>

        <section>
          <h3 className="text-xl font-semibold mb-3">Limitations et Précautions</h3>
          <div className="bg-gray-50 p-4 rounded-lg">
            <ul className="list-disc pl-5 space-y-2 text-gray-600">
              <li>Les prédictions sont des moyennes par arrondissement et peuvent varier significativement pour des biens spécifiques</li>
              <li>Le modèle ne prend pas en compte les futurs projets urbains ou changements réglementaires</li>
              <li>Les événements exceptionnels (crises, pandémies) peuvent affecter la précision des prédictions</li>
              <li>Les prévisions sont plus fiables à court terme qu'à long terme</li>
            </ul>
          </div>
        </section>
      </div>
    </div>
  );
};

export default ModelExplanation;