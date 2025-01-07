# Real-Estate-predictions
Machine Learning project for M2 actuarial ISUP
--- 
- [Real-Estate-predictions](#real-estate-predictions)
  - [Machine Learning project for M2 actuarial ISUP](#machine-learning-project-for-m2-actuarial-isup)
- [Machine Learning](#machine-learning)
  - [Données](#données)
    - [Transactions immobilières](#transactions-immobilières)
    - [Autres données](#autres-données)
  - [Rapport](#rapport)
- [Data Visualisation](#data-visualisation)


---

# Machine Learning

## Données 

### Transactions immobilières

Voici ce que l'on peut lire au sujet de l'aggregation des données fourni par **CEREMA** : 

```
Depuis mai 2019, les données "Demande de Valeurs Foncières" (DVF) sont disponibles en open-data sur le site data.gouv.fr.

La DGALN et le Cerema propose "DVF+ open-data", qui permet d'accéder librement à cette même donnée sous la forme d'une base de données géolocalisée aisément exploitable pour l'observation des marchés fonciers et immobiliers.

La structuration de la donnée DVF proposée s'appuie sur le modèle de données partagé dit "DVF+", issu des travaux menés à l'initiative du groupe national DVF et qui existe depuis 2013. Ce modèle, développé pour faciliter les analyses, fournit notamment une table des mutations dans laquelle chaque ligne correspond aux informations et à la localisation d'une transaction. 

La géolocalisation s'appuie sur les différents millésimes du Plan cadastral informatisé également disponibles en open-data sur data.gouv.fr.

Chacune des variables du modèle DVF+ est calculée uniquement à partir des données brutes de DVF. Les variables calculées s’appliquent sur l’ensemble du territoire et relèvent d’une méthodologie partagée. Il n’y a pas de données exogènes à ce stade hormis les données de géolocalisation issue du PCI Vecteur.

A noter que le modèle DVF+ constitue également le socle pour la constitution de la base de données DV3F.
```

Voici le [lien](https://datafoncier.cerema.fr/donnees/autres-donnees-foncieres/dvfplus-open-data) pour pouvoir accéder aux données.


Les données se trouvent dans data. Pour les fichiers de transactions sont de la forme mutations_d{departement}.csv et sont **à jour jusqu'à octobre 2024**.

### Autres données

Pour tenter de prédire les prix futurs, nous allons utiliser les données explicatives suivantes (les liens de téléchargements sont disponibles en cliquant sur le nom):

- [**Les taux en vigueur**](https://webstat.banque-france.fr/fr/catalogue/estr/ESTR.B.EU000A2X2A25.WT)
- [**Le cours du pétrol**](https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=RBRTE&f=A)
- [**La confiance des ménages**](https://www.insee.fr/fr/statistiques/7758403)

## Rapport

Le rapport est disponible sous format pdf ou ipynb selon les préférences et concerne les données de l'île-de-France. (75, 77, 78, 91, 92, 93, 94, 95)

--- 

# Data Visualisation

