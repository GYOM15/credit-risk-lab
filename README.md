# Système de Décision de Crédit Bancaire

## Vue d'ensemble

Pipeline complet de Machine Learning pour l'analyse du risque client et la décision automatisée de crédit bancaire. Ce projet intègre des techniques avancées de preprocessing, de modélisation prédictive et de règles métier pour produire un système de décision robuste et explicable.

---

## Sommaire

- [Technologies et Librairies](#technologies-et-librairies)
- [Application Web](#application-web)
- [Architecture du Projet](#architecture-du-projet)
- [Données Utilisées](#données-utilisées)
- [Méthodologie](#méthodologie)
  - [1. Analyse Exploratoire des Données](#1-analyse-exploratoire-des-données-eda)
  - [2. Preprocessing Pipeline](#2-preprocessing-pipeline)
  - [3. Modélisation - Classification](#3-modélisation---classification)
  - [4. Modélisation - Régression](#4-modélisation---régression)
  - [5. Explicabilité - SHAP](#5-explicabilité---shap)
  - [6. Système Expert End-to-End](#6-système-expert-end-to-end)
  - [7. Tests Unitaires](#7-tests-unitaires)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure des Données](#structure-des-données)
- [Métriques de Performance](#métriques-de-performance)
- [Auteur](#auteur)
- [Licence](#licence)
- [Contact](#contact)

---

## Technologies et Librairies

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-FF4785?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

**Librairies principales :**
- **scikit-learn** : Pipeline ML, transformers personnalisés, modèles
- **pandas / numpy** : Manipulation et analyse de données
- **matplotlib / seaborn** : Visualisations
- **XGBoost** : Modèle de gradient boosting
- **SHAP** : Explicabilité des modèles
- **Streamlit** : Interface web interactive
- **scipy** : Tests statistiques

---

## Application Web

L'application Streamlit offre une interface intuitive pour tester le système de décision :

![Application Streamlit](./docs/streamlit_app.png)

**Fonctionnalités :**
- Simulateur de demande de crédit individuelle
- Analyse en batch via upload CSV
- Visualisation des décisions et justifications
- Documentation du système

---

## Architecture du Projet

```
projet-credit/
├── src/
│   ├── preprocessing/
│   │   └── custom_transformers.py    # Transformers personnalisés
│   └── credit_engine.py               # Système de décision
├── notebooks/
│   └── projet_data_science.ipynb     # Notebook d'analyse complet
├── outputs/
│   └── models/                        # Modèles sauvegardés
├── app.py                             # Application Streamlit
└── README.md
```

---

## Données Utilisées

Les données utilisées dans ce projet sont **100% synthétiques**, générées afin de reproduire des profils clients bancaires réalistes dans un contexte d'étude de cas pédagogique.

### Contexte et Problématique

Le dataset a été conçu pour illustrer des **défis réels** rencontrés en Machine Learning appliqué au crédit bancaire :

- **Taille limitée** : 1000 lignes seulement
- **Déséquilibre de classe sévère** : 11% de défauts de paiement (89% non-défaut)
- **Données réalistes mais imparfaites** : Valeurs manquantes, outliers, colinéarités

**Objectif pédagogique :** Explorer et comparer différentes stratégies de modélisation face à ces contraintes (SMOTE, class weighting, optimisation de seuils, calibration).

### Contenu des Données

Les données incluent notamment :
- **Données socio-démographiques** : Age, Niveau d'études, Ville
- **Informations financières** : Salaire annuel, Épargne totale, Score de crédit externe
- **Variables cibles** :
  - `Defaut_Paiement` (classification) : Oui/Non
  - `Montant_Pret_Accorde` (régression) : Montant en euros
- **Variables dérivées** : Construites via feature engineering (ex: Ratio_Epargne_Salaire)

**Note importante :** Les variables cibles (`Defaut_Paiement`, `Montant_Pret_Accorde`) n'existent que dans les données d'entraînement. Pour l'utilisation en production du modèle, seules les features descriptives du client sont nécessaires.

### Aperçu des Données

#### Données d'Entraînement (avec targets)

Exemple du jeu de données historique utilisé pour entraîner les modèles :

| Age | Salaire_Annuel | Epargne_Totale | Score_Credit_Externe | Niveau_Etude | Ville     | Defaut_Paiement | Montant_Pret_Accorde |
|-----|----------------|----------------|----------------------|--------------|-----------|-----------------|----------------------|
| 35  | 42000          | 7500           | 70                   | Licence      | Paris     | Non             | 25000                |
| 52  | 31000          | 1500           | 55                   | Bac          | Lyon      | Oui             | 12000                |
| 28  | 55000          | 13750          | 80                   | Master       | Marseille | Non             | 32000                |

#### Données pour Prédiction (nouveau client)

Données nécessaires pour obtenir une décision de crédit (sans les targets) :

| Age | Salaire_Annuel | Epargne_Totale | Score_Credit_Externe | Niveau_Etude | Ville     |
|-----|----------------|----------------|----------------------|--------------|-----------|
| 40  | 48000          | 9600           | 75                   | Master       | Lyon      |
| 33  | 38000          | 5700           | 65                   | Licence      | Toulouse  |
| 45  | 62000          | 18600          | 85                   | Doctorat     | Paris     |

Les variables combinent :
- Informations socio-démographiques
- Indicateurs financiers
- Variables métier construites par feature engineering

L'utilisation de données synthétiques permet :
- D'explorer librement différentes stratégies de modélisation
- De simuler des problématiques industrielles réalistes
- De travailler dans un cadre reproductible et pédagogique
- De partager le projet sans contraintes de confidentialité

---

## Méthodologie

### 1. Analyse Exploratoire des Données (EDA)

**Objectif :** Comprendre la structure des données et identifier les patterns.

**Approches :**
- **Analyse univariée** : Distribution des variables numériques et catégorielles
- **Détection d'outliers** : Méthode IQR (plus robuste que Z-score pour distributions asymétriques)
- **Analyse bivariée** : Corrélations (Pearson et Spearman), relations features-target
- **Identification de colinéarité** : Matrice de corrélation pour détecter redondances

**Résultats clés :**
- Déséquilibre de classe : 11% de défauts de paiement
- Corrélation forte entre Salaire et Épargne (0.87) nécessitant feature engineering
- Distribution asymétrique des variables financières (Salaire, Montant)

---

### 2. Preprocessing Pipeline

**Architecture modulaire** avec transformers personnalisés respectant l'API scikit-learn.

#### Transformers Implémentés

**DataCleaningTransformer**
- Remplace les valeurs intruses (`-`, `?`, `null`, etc.) par `np.nan`
- Assure la cohérence des types de données

**FeatureEngineeringTransformer**
- Création du ratio `Epargne_Totale / Salaire_Annuel`
- Suppression de `Epargne_Totale` (colinéarité avec salaire)
- Gestion des valeurs infinies

**SmartImputerTransformer**
- **Variables numériques** : Médiane si |skew| > 0.5, sinon moyenne
- **Variables catégorielles** : Mode (ou "Inconnu" si >10% manquant)
- **Variables ordinales** : Médiane sur codes ordinaux
- Les valeurs d'imputation sont apprises uniquement sur le train

**CustomEncodingTransformer**
- **Niveau_Etude** : OrdinalEncoder (ordre : bac < licence < master < doctorat)
- **Ville** : Filtrage top villes (seuil 10 occurrences, couverture 85%) + OneHotEncoder
- Gestion des valeurs inconnues en test

**MultiStrategyScaler**
- **Log1p + StandardScaler** : Salaire_Annuel, Ratio_Epargne_Salaire
- **StandardScaler** : Age
- **RobustScaler** : Score_Credit_Externe (résistant aux outliers)

**Pipeline complet :**
```python
preprocessing_pipeline = Pipeline([
    ('cleaning', DataCleaningTransformer()),
    ('feature_eng', FeatureEngineeringTransformer()),
    ('imputation', SmartImputerTransformer()),
    ('encoding', CustomEncodingTransformer()),
    ('scaling', MultiStrategyScaler())
])
```

**Principe clé :** `fit()` uniquement sur le train, `transform()` sur validation et test pour éviter toute fuite de données.

---

### 3. Modélisation - Classification

**Objectif :** Prédire le risque de défaut de paiement.

#### Gestion du Déséquilibre

Trois approches comparées pour gérer le déséquilibre de classe (11% défauts) :

1. **SMOTE (Synthetic Minority Over-sampling Technique)**
   - Génération synthétique d'échantillons minoritaires
   - Pipeline : SMOTE + LogisticRegression

2. **Class Weight Balanced**
   - Ajustement automatique des poids par classe
   - Plus rapide, pas de génération de données synthétiques

3. **XGBoost avec scale_pos_weight**
   - Poids calculé : ratio majorité/minorité (89/11 ≈ 8)
   - Adapté aux données déséquilibrées

#### Optimisation des Hyperparamètres

**GridSearchCV retenu** (plutôt que RandomizedSearchCV) pour les raisons suivantes :
- **Petite quantité de données** : 1000 lignes (600 train, 200 val, 200 test)
- **Données synthétiques** : Patterns réguliers et prévisibles
- **Petit espace de recherche** : 10-50 combinaisons d'hyperparamètres
- **Résultat** : Meilleures performances observées avec GridSearchCV qui teste systématiquement toutes les combinaisons vs RandomizedSearchCV qui échantillonne aléatoirement

Configuration :
- **StratifiedKFold** : 5 folds pour préserver les proportions de classes
- **Métrique d'optimisation** : ROC-AUC
- **Grilles de recherche** :
  - Logistic Regression : C=[0.01, 0.1, 1, 10, 100], penalty=['l1', 'l2']
  - XGBoost : n_estimators=[200,300,500], max_depth=[2,3], learning_rate=[0.005,0.01,0.02]

#### Audit de Stabilité

**K-Fold Cross-Validation** pour évaluer la robustesse :
- Entraînement sur 5 folds différents
- Analyse de la variance des coefficients
- Métriques de stabilité : Coefficient de Variation (CV = std / mean)
- **Résultat** : 3 features identifiées comme instables (CV > 0.5)

#### Optimisation du Seuil

**Approche métier-driven** plutôt que seuil par défaut (0.5) :

1. **Seuil optimal F1-Score** : Équilibre précision/recall
2. **Seuil de sécurité** : Garantir recall minimum de 75%
   - Justification : Capturer au moins 75% des défauts pour limiter les pertes
   - Seuil final retenu : ~0.30

#### Calibration des Probabilités

**CalibratedClassifierCV** appliqué au modèle final :
- Méthode : Sigmoid
- CV : 5 folds
- **Objectif** : Les probabilités prédites reflètent le risque réel
- Améliore la fiabilité des décisions basées sur les seuils de probabilité

#### Modèle Final

**LogisticRegression avec class_weight='balanced' + Calibration**
- AUC : 0.85+
- Recall : 75%+ (garanti par optimisation du seuil)
- Précision : Variable selon le seuil choisi
- Stabilité : Écart-type < 0.05 sur K-Fold

---

### 4. Modélisation - Régression

**Objectif :** Estimer le montant du prêt pour les clients solvables.

#### Préparation Spécifique

- **Filtrage** : Uniquement clients avec défaut=0 (solvables)
- **Transformation log** : `log1p(Montant_Pret_Accorde)` pour gérer l'asymétrie
- **Smearing Factor** : Correction du biais de Jensen lors de la transformation inverse
  - Formule : `smearing_factor = mean(exp(residuals))`
  - Prédiction finale : `montant = exp(pred_log) * smearing_factor`

#### Modèles Comparés

Benchmark avec **RandomizedSearchCV** (ici, espace de recherche plus large) :
- **Ridge Regression** (avec log + smearing) : Retenu
- **Random Forest** : Testé
- **XGBoost** : Testé

#### Performances du Modèle Ridge

- **MAE** : ~2000€
- **RMSE** : ~2800€
- **MAPE** : 7.3% (excellent)
- **R²** : 0.85+

#### Analyse de Robustesse

**Analyse par segments** pour vérifier l'équité :
- Tranches d'âge (Jeunes, Actifs, Matures, Seniors)
- Niveaux de revenu (Bas, Moyen, Haut)
- Scores de crédit (Risque, Intermédiaire, Premium)

**Résultat** : Performance homogène, pas de biais significatif.

**Intervalles de confiance** :
- Calcul : ±(MAPE × z × montant) avec z=1.96 (95% confiance)
- Communication de l'incertitude au client

---

### 5. Explicabilité - SHAP

**SHAP (SHapley Additive exPlanations)** pour interpréter les décisions.

#### Classification
- **Summary plot** : Contribution globale des features
- **Feature importance** : Variables les plus influentes
- **Force plot** : Explication individuelle (client #10)
- Identification des drivers de risque

#### Régression
- **Summary plot** : Variables augmentant/diminuant le montant
- **Feature importance** : Facteurs déterminant le montant du prêt
- Transparence sur les prédictions

**Avantage** : Justification explicable pour chaque décision (conformité RGPD).

---

### 6. Système Expert End-to-End

Intégration ML + Règles Métier pour la décision finale.

#### Fonctions Métier

**get_dynamic_rate(proba_default)**
- Taux d'intérêt selon le risque :
  - < 5% risque : 3%
  - 5-15% : 5%
  - 15-25% : 8%
  - ≥25% : 12%

**compute_max_authorized_capital()**
- Capacité d'endettement maximale
- Seuil : 33% du revenu mensuel

**make_credit_decision()**
- Compare montant ML vs capacité réelle
- Ajuste si dépassement
- Calcule mensualité et ratio d'endettement

#### Fonction Principale : decide()

**Workflow de décision :**
1. **Extraction variables métier** (salaire)
2. **Vérification salaire minimum** (18,000€/an)
3. **Classification** : Client solvable ? (seuil 0.30)
4. **Régression** : Estimation du montant
5. **Règles métier** : Ajustement selon capacité
6. **Output** : Décision finale (Accepté/Refusé/Ajusté)

#### Classe CreditDecisionSystem

Wrapper pour simplifier l'utilisation du système :
```python
system = CreditDecisionSystem('./models/')
decision = system.predict(client_dataframe)
```

**Avantages :**
- Chargement unique des modèles (mise en cache)
- API simple et intuitive
- Preprocessing automatique
- Facilite l'intégration dans une application

---

### 7. Tests Unitaires

Tests complets des transformers pour garantir la qualité :

- **DataCleaningTransformer** : Remplacement intrus, préservation valeurs valides
- **FeatureEngineeringTransformer** : Création ratio, gestion infinis
- **SmartImputerTransformer** : Imputation correcte, préservation fit/transform
- **CustomEncodingTransformer** : Encodage ordinal/one-hot, gestion inconnus
- **MultiStrategyScaler** : Application correcte des stratégies
- **Pipeline complet** : Intégration, absence de NaN final
- **Edge cases** : DataFrame vide, ligne unique, colonne all-NaN

**Taux de réussite** : 100% des tests passent

---

## Installation

### Prérequis

```bash
Python 3.11+
```

### Installation des dépendances

```bash
pip install -r requirements.txt
```

**requirements.txt :**
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.11.0
xgboost>=2.0.0
shap>=0.42.0
streamlit>=1.28.0
jupyter>=1.0.0
```

---

## Utilisation

### 1. Pipeline de Preprocessing

```python
from src.preprocessing.custom_transformers import (
    DataCleaningTransformer,
    FeatureEngineeringTransformer,
    SmartImputerTransformer,
    CustomEncodingTransformer,
    MultiStrategyScaler
)
from sklearn.pipeline import Pipeline

# Créer le pipeline
preprocessing_pipeline = Pipeline([
    ('cleaning', DataCleaningTransformer()),
    ('feature_eng', FeatureEngineeringTransformer()),
    ('imputation', SmartImputerTransformer()),
    ('encoding', CustomEncodingTransformer()),
    ('scaling', MultiStrategyScaler())
])

# Entraîner et transformer
preprocessing_pipeline.fit(X_train)
X_train_transformed = preprocessing_pipeline.transform(X_train)
X_test_transformed = preprocessing_pipeline.transform(X_test)
```

### 2. Système de Décision

```python
from src.credit_engine import CreditDecisionSystem
import pandas as pd

# Charger le système
system = CreditDecisionSystem('./outputs/models/')

# Client à évaluer
client = pd.DataFrame({
    'Age': [35],
    'Salaire_Annuel': [45000],
    'Epargne_Totale': [10000],
    'Score_Credit_Externe': [75],
    'Niveau_Etude': ['Master'],
    'Ville': ['Lyon']
})

# Obtenir la décision
decision = system.predict(client, duration_months=60)
print(decision)
```

### 3. Application Web

```bash
streamlit run app.py
```

Ouvrir le navigateur à `http://localhost:8501`

---

## Structure des Données

### Input (Données Brutes pour Prédiction)

Données nécessaires pour faire une prédiction sur un nouveau client :

| Colonne | Type | Description |
|---------|------|-------------|
| Age | int | Âge du client |
| Salaire_Annuel | float | Salaire annuel en euros |
| Epargne_Totale | float | Épargne totale en euros |
| Score_Credit_Externe | int | Score de crédit (0-100) |
| Niveau_Etude | str | Niveau d'études (Bac/Licence/Master/Doctorat) |
| Ville | str | Ville de résidence |

**Note :** Les variables cibles (`Defaut_Paiement`, `Montant_Pret_Accorde`) sont présentes uniquement dans les données d'entraînement historiques. Elles ne sont pas nécessaires pour utiliser le modèle et obtenir une décision de crédit.

### Output (Décision)

```python
{
    "Decision": "Accepté" | "Refusé" | "Ajusté",
    "Montant_Modele": 25000.00,          # Montant prédit par ML
    "Montant_Max_Autorise": 28000.00,    # Capacité maximale
    "Montant_Final": 25000.00,           # Montant accordé
    "Mensualite": 467.53,                # Mensualité calculée
    "Ratio_Endettement": 0.25,           # 25% du revenu
    "Niveau_Risque": 0.15,               # 15% de risque de défaut
    "Taux_Propose": 0.05,                # 5% d'intérêt
    "Justification": "Montant conforme à la capacité d'endettement"
}
```

---

## Métriques de Performance

### Classification

| Métrique | Valeur |
|----------|--------|
| AUC-ROC | 0.85+ |
| Recall | 75%+ (garanti) |
| Précision | Variable selon seuil |
| Stabilité (std K-Fold) | < 0.05 |

### Régression

| Métrique | Valeur |
|----------|--------|
| MAE | ~2000€ |
| RMSE | ~2800€ |
| MAPE | 7.3% |
| R² | 0.85+ |

---

## Auteur

Projet développé dans le cadre d'un pipeline complet de Data Science appliqué au crédit bancaire.

**Compétences démontrées :**
- Pipeline ML avec transformers personnalisés scikit-learn
- Gestion du déséquilibre de classe (SMOTE, class weights, scale_pos_weight)
- Optimisation d'hyperparamètres (GridSearchCV vs RandomizedSearchCV)
- Calibration de modèles (CalibratedClassifierCV)
- Explicabilité (SHAP)
- Tests unitaires sur transformers personnalisés
- Déploiement d'interface web (Streamlit)
- Architecture modulaire et réutilisable

---

## Licence

Ce projet est à des fins éducatives et de démonstration. Les données utilisées sont entièrement synthétiques.

---

## Contact

Pour toute question ou suggestion, n'hésitez pas à me contacter.
