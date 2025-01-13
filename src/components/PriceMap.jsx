import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

const PriceMap = ({ data, propertyType }) => {
  const mapRef = useRef(null);
  const mapInstance = useRef(null);
  const geoJsonLayerRef = useRef(null);
  const legendRef = useRef(null);

  // Fonction pour convertir le code d'arrondissement en code JSON
  const getJsonCode = (arCode) => {
    // Convertit "1" en "['75101']", "2" en "['75102']", etc.
    const paddedCode = arCode.toString().padStart(2, '0');
    return `['751${paddedCode}']`;
  };

  // Fonction pour extraire le prix actuel des données
  const getCurrentPrice = (priceData) => {
    if (!priceData?.aujourd_hui) return 0;
    
    const match = priceData.aujourd_hui.match(/-?\d+\.?\d*%\s*\((\d+\s*\d*)\s*€/);
    if (!match) return 0;
    
    return Number(match[1].replace(/\s/g, ''));
  };

  // Fonction pour extraire et analyser un pourcentage
  const extractPercentage = (str) => {
    const match = str.match(/-?\d+\.?\d*%/);
    if (!match) return { value: 0, text: '0%' };
    const text = match[0];
    const value = parseFloat(text);
    return { value, text };
  };

  // Fonction pour obtenir la classe de couleur basée sur la valeur
  const getColorClass = (value) => {
    if (value > 0) return 'color: #22c55e;'; // vert
    if (value < 0) return 'color: #ef4444;'; // rouge
    return 'color: #666666;'; // gris neutre
  };

  // Configuration des paliers de prix et couleurs
  const priceRanges = [
    { min: 15000, color: '#67000d', label: '> 15 000 €/m²' },
    { min: 13000, color: '#a50f15', label: '13 000 - 15 000 €/m²' },
    { min: 11000, color: '#cb181d', label: '11 000 - 13 000 €/m²' },
    { min: 9000, color: '#ef3b2c', label: '9 000 - 11 000 €/m²' },
    { min: 7000, color: '#fb6a4a', label: '7 000 - 9 000 €/m²' },
    { min: 0, color: '#fc9272', label: '< 7 000 €/m²' }
  ];

  // Fonction pour déterminer la couleur en fonction du prix
  const getColor = (price) => {
    for (const range of priceRanges) {
      if (price >= range.min) return range.color;
    }
    return priceRanges[priceRanges.length - 1].color;
  };

  // Fonction pour définir le style des zones sur la carte
  const style = (feature) => {
    try {
      const jsonCode = getJsonCode(feature.properties.c_ar);
      const priceData = data[propertyType][jsonCode];
      const price = getCurrentPrice(priceData);
  
      return {
        fillColor: getColor(price),
        weight: 2,
        opacity: 1,
        color: 'white',
        fillOpacity: 0.7,
        dashArray: '3'
      };
    } catch (error) {
      console.error('Erreur dans style():', error);
      return {
        fillColor: '#fc9272',
        weight: 2,
        opacity: 1,
        color: 'white',
        fillOpacity: 0.7
      };
    }
  };

  // Fonction pour créer le contenu des popups
  const createPopupContent = (feature) => {
    try {
      const jsonCode = getJsonCode(feature.properties.c_ar);
      const priceData = data[propertyType][jsonCode];
      
      if (!priceData) {
        return 'Données non disponibles';
      }
      
      const currentPrice = getCurrentPrice(priceData);
      const currentVariation = extractPercentage(priceData.aujourd_hui);
      const futureVariation = extractPercentage(priceData.dans_1_an);
      
      return `
        <div class="p-4 min-w-[250px]">
          <h3 class="text-lg font-bold mb-2">Paris ${feature.properties.l_ar}</h3>
          <div class="space-y-2">
            <p class="text-sm">
              <span class="font-medium">Au 30 juin 2024</span>
              <span class="ml-2">${priceData.il_y_a_1_an}</span>
            </p>
            <p class="text-sm">
              <span class="font-medium">Aujourd'hui:</span>
              <span class="ml-2">${currentPrice.toLocaleString()} €/m²</span>
              <span class="ml-1" style="${getColorClass(currentVariation.value)}">(${currentVariation.text})</span>
            </p>
            <p class="text-sm">
              <span class="font-medium">Au 30 juin 2025:</span>
              <span class="ml-1" style="${getColorClass(futureVariation.value)}">${priceData.dans_1_an}</span>
            </p>
          </div>
        </div>
      `;
    } catch (error) {
      console.error('Erreur dans createPopupContent:', error);
      return 'Erreur lors du chargement des données';
    }
  };

  // Fonction pour créer la légende
  const createLegend = (map) => {
    const legend = L.control({ position: 'bottomright' });

    legend.onAdd = () => {
      const div = L.DomUtil.create('div', 'info legend');
      div.style.backgroundColor = 'white';
      div.style.padding = '10px';
      div.style.borderRadius = '5px';
      div.style.boxShadow = '0 1px 5px rgba(0,0,0,0.4)';

      div.innerHTML = '<h4 style="margin:0 0 10px 0;font-weight:bold">Prix au m²</h4>';

      // Ajout des paliers de prix dans la légende
      priceRanges.forEach(range => {
        div.innerHTML += `
          <div style="margin-bottom:5px">
            <i style="background:${range.color};width:18px;height:18px;float:left;margin-right:8px;opacity:0.7"></i>
            <span>${range.label}</span>
          </div>
        `;
      });

      return div;
    };

    return legend;
  };

  useEffect(() => {
    if (!mapRef.current || !data) return;

    // Fix pour l'icône de marqueur Leaflet
    delete L.Icon.Default.prototype._getIconUrl;
    L.Icon.Default.mergeOptions({
      iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
      iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
      shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
    });

    // Initialiser la carte si elle n'existe pas
    if (!mapInstance.current) {
      mapInstance.current = L.map(mapRef.current, {
        center: [48.856614, 2.3522219],
        zoom: 12,
        maxZoom: 16,
        minZoom: 11,
        maxBounds: [
          [48.815573, 2.224199], // Sud-Ouest
          [48.902145, 2.469920]  // Nord-Est
        ]
      });

      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
      }).addTo(mapInstance.current);

      // Ajouter la légende
      legendRef.current = createLegend(mapInstance.current);
      legendRef.current.addTo(mapInstance.current);
    }

    // Charger et afficher les données GeoJSON
    const loadGeoJson = async () => {
      try {
        const response = await fetch('/paris-districts.geojson');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
    
        const geojsonData = await response.json();
        
        if (geoJsonLayerRef.current) {
          mapInstance.current.removeLayer(geoJsonLayerRef.current);
        }
        
        geoJsonLayerRef.current = L.geoJSON(geojsonData, {
          style: style,
          onEachFeature: (feature, layer) => {
            const popup = L.popup({
              maxWidth: 300,
              className: 'custom-popup'
            }).setContent(createPopupContent(feature));
            
            layer.bindPopup(popup);
            
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
              },
              click: (e) => {
                mapInstance.current.fitBounds(e.target.getBounds());
              }
            });
          }
        }).addTo(mapInstance.current);
    
      } catch (error) {
        console.error('Error loading data:', error);
      }
    };
    
    loadGeoJson();

    // Nettoyage lors du démontage du composant
    return () => {
      if (mapInstance.current) {
        if (legendRef.current) {
          legendRef.current.remove();
        }
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