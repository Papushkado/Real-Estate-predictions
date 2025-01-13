import React, { useState, useEffect } from 'react';
import PriceMap from './PriceMap';
import ModelExplanation from './ModelExplanation';

const RealEstateApp = () => {
  const [propertyData, setPropertyData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('residential');

  useEffect(() => {
    // Charger les données des prix
    const loadData = async () => {
      try {
        const response = await fetch('/property_prices.json');
        const data = await response.json();
        setPropertyData(data);
      } catch (error) {
        console.error('Error loading property data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  if (loading) {
    return <div className="flex justify-center items-center h-screen">Chargement des données...</div>;
  }

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-center mb-6">
          Prix Immobiliers à Paris
        </h1>
        
        <div className="mb-6">
          <div className="flex justify-center space-x-4">
            <button
              onClick={() => setActiveTab('residential')}
              className={`px-4 py-2 rounded-lg transition-colors ${
                activeTab === 'residential'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white hover:bg-gray-100'
              }`}
            >
              Résidentiel
            </button>
            <button
              onClick={() => setActiveTab('commercial')}
              className={`px-4 py-2 rounded-lg transition-colors ${
                activeTab === 'commercial'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white hover:bg-gray-100'
              }`}
            >
              Commercial
            </button>
            <button
              onClick={() => setActiveTab('model')}
              className={`px-4 py-2 rounded-lg transition-colors ${
                activeTab === 'model'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white hover:bg-gray-100'
              }`}
            >
              Notre Modèle
            </button>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6">
          {activeTab === 'residential' && (
            <div>
              <h2 className="text-xl font-semibold mb-4">Prix des Biens Résidentiels</h2>
              <p className="mb-4 text-gray-600">
                Visualisation des prix au m² des appartements et maisons à Paris par arrondissement.
                Survolez un arrondissement pour voir les détails des prix.
              </p>
              <PriceMap data={propertyData} propertyType="residential" />
            </div>
          )}

          {activeTab === 'commercial' && (
            <div>
              <h2 className="text-xl font-semibold mb-4">Prix des Locaux Commerciaux</h2>
              <p className="mb-4 text-gray-600">
                Visualisation des prix au m² des locaux commerciaux à Paris par arrondissement.
                Survolez un arrondissement pour voir les détails des prix.
              </p>
              <PriceMap data={propertyData} propertyType="commercial" />
            </div>
          )}

          {activeTab === 'model' && <ModelExplanation />}
        </div>
      </div>
    </div>
  );
};

export default RealEstateApp;