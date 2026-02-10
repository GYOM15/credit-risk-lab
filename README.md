# Analyse et Modélisation du Risque de Crédit

## Présentation du projet

Ce projet explore différentes approches de Data Science appliquées à l’analyse du risque de crédit bancaire.  
L’objectif principal n’est pas uniquement de construire des modèles prédictifs performants, mais d’illustrer l’ensemble du pipeline Data Science et les décisions méthodologiques associées à un contexte métier réel.

Le projet se décompose en plusieurs axes :

- Prédiction de la solvabilité d’un client (classification)
- Prédiction du montant de prêt optimal pour un client solvable (régression)
- Analyse comparative de différentes stratégies de modélisation et d’optimisation
- Traduction des performances modèles en impact financier métier

---

## Données utilisées

Les données utilisées dans ce projet sont **100 % synthétiques**, générées afin de reproduire des profils clients bancaires réalistes.

Elles incluent notamment :

- Données socio-démographiques
- Revenus et stabilité professionnelle
- Comportement financier et épargne
- Historique d’endettement
- Variables dérivées construites via feature engineering

L’utilisation de données synthétiques permet :

- d’explorer librement différentes stratégies de modélisation
- de simuler des problématiques industrielles
- de travailler dans un cadre reproductible et pédagogique

## Aperçu des données

Exemple simplifié du jeu de données client :

| Age  | Salaire_Annuel | Ratio_Epargne_Salaire | Niveau_Etude | Defaut_Paiement|
|------|----------------|-----------------------|--------------|----------------|
| 35   | 42000          | 0.18                  | Licence      | 0              |
| 52   | 31000          | 0.05                  | Bac          | 1              |
| 28   | 55000          | 0.25                  | Master       | 0              |

Les variables combinent :
- informations socio-démographiques
- indicateurs financiers
- variables métier construites par feature engineering

---

## Objectifs Data Science

Ce projet met volontairement l’accent sur l’exploration des différentes étapes d’un pipeline Data Science complet :

- Analyse exploratoire (EDA)
- Feature Engineering métier
- Gestion des valeurs manquantes
- Encodage et normalisation
- Gestion du déséquilibre des classes
- Comparaison multi-modèles
- Validation croisée stratifiée
- Optimisation du seuil de décision
- Calibration probabiliste
- Évaluation financière métier

---

## Modèles étudiés

### Classification – Solvabilité client
- Régression Logistique pondérée
- Régression Logistique avec SMOTE
- XGBoost

### Régression – Montant de prêt (phase suivante du projet)
- Modèles de régression supervisée
- Optimisation du montant accordé sous contrainte de risque

---

## Méthodologie clé

Le projet met en avant plusieurs approches critiques en Data Science appliquée :

- Comparaison entre rééquilibrage des classes (SMOTE vs pondération)
- Analyse de robustesse via validation croisée
- Ajustement métier du seuil de décision
- Calibration des probabilités pour une interprétation fiable
- Simulation financière pour évaluer la rentabilité des modèles

---

## Résultats principaux

Les analyses montrent que :

- La régression logistique pondérée offre le meilleur compromis robustesse / interprétabilité
- L’optimisation métier du seuil de décision permet d’adapter la politique de risque bancaire
- La calibration probabiliste améliore la fiabilité des scores de risque
- L’évaluation financière permet de traduire les performances ML en décisions opérationnelles

---

## Technologies utilisées

- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Matplotlib / Seaborn

---

## Perspectives d’amélioration

- Industrialisation complète via pipelines sklearn
- Ajout d’outils d’interprétabilité avancée (SHAP)
- Simulation métier multi-scénarios
- Validation sur données réelles

---

## Auteur

Projet réalisé dans un objectif d’apprentissage avancé en Data Science appliquée au secteur bancaire et au risque crédit.

