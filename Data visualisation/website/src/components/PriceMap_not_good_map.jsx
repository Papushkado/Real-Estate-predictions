import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

const PriceMap = ({ data, propertyType }) => {
  const mapRef = useRef(null);
  const mapInstance = useRef(null);
  const geoJsonLayerRef = useRef(null);

  // Fonction pour extraire le prix actuel des données
  const getCurrentPrice = (priceData) => {
    try {
      if (!priceData?.aujourd_hui) {
        console.log('Pas de données pour aujourd_hui');
        return 0;
      }
      console.log('Données brutes:', priceData.aujourd_hui);
      
      // Gérer les deux formats possibles : "11 460 € / m²" ou "-0,1% (11 460 € / m²)"
      const match = priceData.aujourd_hui.match(/(\d[\d\s]*)\s*€/);
      if (!match) {
        console.log('Pas de correspondance trouvée');
        return 0;
      }
      
      const price = Number(match[1].replace(/\s/g, ''));
      console.log('Prix extrait:', price);
      return price;
    } catch (error) {
      console.error('Erreur dans getCurrentPrice:', error);
      return 0;
    }
  };

  // Fonction pour déterminer la couleur en fonction du prix
  const getColor = (price) => {
    if (price > 15000) return '#67000d';
    if (price > 13000) return '#a50f15';
    if (price > 11000) return '#cb181d';
    if (price > 9000) return '#ef3b2c';
    if (price > 7000) return '#fb6a4a';
    return '#fc9272';
  };

  // Fonction pour définir le style des zones sur la carte
  const style = (feature) => {
    try {
      const communeCode = `['${feature.properties.c_ar}']`;
      console.log("Code commune:", communeCode);
      console.log("Données pour cette commune:", data[propertyType][communeCode]);
      
      const priceData = data[propertyType][communeCode];
      const price = getCurrentPrice(priceData);
      console.log("Prix calculé:", price); // Debug
  
      return {
        fillColor: getColor(price),
        weight: 2,
        opacity: 1,
        color: 'black', // Changé de white à black pour mieux voir les frontières
        fillOpacity: 0.7,
        dashArray: '3'
      };
    } catch (error) {
      console.error('Erreur dans style():', error);
      return {
        fillColor: '#red',
        weight: 2,
        opacity: 1,
        color: 'red',
        fillOpacity: 0.7
      };
    }
  };

  // Fonction pour créer le contenu des popups
  const createPopupContent = (feature) => {
    try {
      const communeCode = `['${feature.properties.c_ar}']`;
      const priceData = data[propertyType][communeCode];
      
      if (!priceData) {
        console.log('Pas de données pour:', communeCode);
        return 'Données non disponibles';
      }
  
      return `
        <div class="p-4 min-w-[200px]">
          <h3 class="text-lg font-bold mb-2">Paris ${feature.properties.l_ar}</h3>
          <div class="space-y-2">
            <p class="text-sm">
              <span class="font-medium">Il y a 1 an:</span>
              <span class="ml-2">${priceData.il_y_a_1_an}</span>
            </p>
            <p class="text-sm">
              <span class="font-medium">Aujourd'hui:</span>
              <span class="ml-2">${priceData.aujourd_hui}</span>
            </p>
            <p class="text-sm">
              <span class="font-medium">Dans 1 an:</span>
              <span class="ml-2">${priceData.dans_1_an}</span>
            </p>
          </div>
        </div>
      `;
    } catch (error) {
      console.error('Erreur dans createPopupContent:', error);
      return 'Erreur lors du chargement des données';
    }
  };

  useEffect(() => {
    console.log("PriceMap mounted, data:", data);
    
    if (!mapRef.current || !data) {
      console.log("Missing ref or data");
      return;
    }

    // Fix pour l'icône de marqueur Leaflet
    delete L.Icon.Default.prototype._getIconUrl;
    L.Icon.Default.mergeOptions({
      iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
      iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
      shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
    });

    // Initialiser la carte si elle n'existe pas
    if (!mapInstance.current) {
      console.log("Initializing map");
      mapInstance.current = L.map(mapRef.current).setView([48.856614, 2.3522219], 12);

      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
      }).addTo(mapInstance.current);
    }

    // Charger et afficher les données GeoJSON
    
const loadGeoJson = async () => {
    try {
      console.log("Début du chargement GeoJSON");
      const response = await fetch('/Real-ESTATE-PREDICTIONS/paris-districts.geojson');
      console.log("Statut de la réponse:", response.status);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
  
      let geojsonData;
      try {
        const text = await response.text();
        console.log("Texte brut reçu:", text.substring(0, 200)); // Affiche les 200 premiers caractères
        geojsonData = JSON.parse(text);
      } catch (parseError) {
        console.error("Erreur lors du parsing JSON:", parseError);
        throw parseError;
      }
  
      console.log("Données GeoJSON chargées:", geojsonData);
  
      if (!geojsonData || !geojsonData.features) {
        throw new Error('Format GeoJSON invalide');
      }
  
      // Nettoyer la couche précédente si elle existe
      if (geoJsonLayerRef.current) {
        mapInstance.current.removeLayer(geoJsonLayerRef.current);
      }
  
      // Créer la nouvelle couche GeoJSON
      geoJsonLayerRef.current = L.geoJSON(geojsonData, {
        style: style,
        onEachFeature: (feature, layer) => {
          layer.bindPopup(createPopupContent(feature));
          
          layer.on({
            mouseover: (e) => {
              const layer = e.target;
              layer.setStyle({
                weight: 4,
                color: '#666',
                dashArray: '',
                fillOpacity: 0.9
              });
              layer.openPopup();
            },
            mouseout: (e) => {
              geoJsonLayerRef.current.resetStyle(e.target);
              e.target.closePopup();
            }
          });
        }
      }).addTo(mapInstance.current);
  
      // Ajuster la vue de la carte pour qu'elle englobe les données GeoJSON
      mapInstance.current.fitBounds(geoJsonLayerRef.current.getBounds());
  
      // Ajouter la légende
      const legend = L.control({ position: 'bottomright' });
      legend.onAdd = () => {
        const div = L.DomUtil.create('div', 'info legend bg-white p-3 rounded shadow-lg');
        const grades = [0, 7000, 9000, 11000, 13000, 15000];
        
        div.innerHTML = '<h4 class="font-bold mb-2">Prix au m²</h4>';
        for (let i = 0; i < grades.length; i++) {
          div.innerHTML += `
            <div class="flex items-center mb-1">
              <i style="background:${getColor(grades[i] + 1)}; width:20px; height:20px; margin-right:5px;"></i>
              <span>${grades[i]}${grades[i + 1] ? '&ndash;' + grades[i + 1] : '+'} €</span>
            </div>
          `;
        }
        return div;
      };
      legend.addTo(mapInstance.current);
  
    } catch (error) {
      console.error('Error loading GeoJSON:', error);
    }
  };
    loadGeoJson();

    // Nettoyage lors du démontage du composant
    return () => {
      console.log("Cleaning up map");
      if (mapInstance.current) {
        mapInstance.current.remove();
        mapInstance.current = null;
      }
    };
  }, [data, propertyType]);

  return (
    <div 
      ref={mapRef} 
      className="h-[600px] w-full rounded-lg shadow-lg" 
      style={{ backgroundColor: '#f0f0f0' }}
    />
  );
};

export default PriceMap;