# 📈 NASDAQ Trading Analysis & Prediction - Mini Projet ENSIAS

## 🎯 Vue d'ensemble

Application Streamlit complète pour l'analyse des données du NASDAQ, développement de stratégies de trading et backtesting avec machine learning.

## ✅ Modules Implémentés

### Module 1: 🔍 Scraping des données

- Téléchargement des données via **yfinance**
- Sélection de tickers NASDAQ populaires (AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, NFLX)
- Période temporelle configurable (par défaut: 2020-01-01)
- Aperçu des données brutes avec statistiques
- Export CSV des données brutes

### Module 2: 🧹 Nettoyage des données

- **Analyse de qualité:**

  - Détection des valeurs manquantes
  - Statistiques descriptives (min, max, moyenne, volatilité)
  - Analyse par colonne

- **Options de nettoyage:**

  - Suppression des lignes avec valeurs manquantes
  - Forward Fill / Backward Fill
  - Remplissage par la moyenne
  - Détection d'anomalies (méthode Z-score)
  - Seuil configurable des anomalies

- **Résultats:**
  - Comparaison avant/après
  - Visualisation graphique
  - Export des données nettoyées

### Module 3: 📊 Visualisation exploratoire

- **Graphiques implémentés:**
  1. Évolution du prix de clôture (courbe + remplissage)
  2. Volume d'échange quotidien (histogramme)
  3. Distribution des prix (histogramme + moyenne/médiane)
  4. Distribution des rendements quotidiens
  5. Matrice de corrélation (heatmap)
  6. Graphique chandelier (candlestick) pour les 30 derniers jours

### Module 4: ⚙️ Feature Engineering

**Indicateurs techniques créés:**

- Moyennes mobiles (MA_7, MA_21, MA_50)
- RSI (Relative Strength Index) - détection suracheté/survendu
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands (BB_Upper, BB_Middle, BB_Lower)
- Volatilité (rolling 21 jours)
- ATR (Average True Range)
- Daily Return (rendements quotidiens)

**Features temporelles:**

- Day of Week
- Month
- Quarter

**Features de lag:**

- Close_Lag1, Close_Lag2, Close_Lag3

**Variable cible:**

- Target: prédire si le prix montrera demain (1) ou baissera (0)

### Module 5: 🎯 Sélection de Features

- **Matrice de corrélation:**

  - Heatmap complète des indicateurs techniques
  - Identification des features hautement corrélées

- **Feature Importance:**
  - Entraînement d'un Random Forest
  - Classement par importance
  - Visualisation en barre horizontale
  - Export des résultats

### Module 6: 🔍 Réduction de Dimensionnalité

- **PCA (Principal Component Analysis):**

  - Variance expliquée par composante
  - Variance cumulée
  - Projection 2D avec coloration par cible
  - Identification du nombre optimal de composantes (95%)

- **t-SNE:**
  - Clustering automatique des comportements de marché
  - Visualisation 2D des patterns cachés
  - Calcul optimisé avec perplexity=30

### Module 7: 🤖 Modélisation Machine Learning

**Modèles entraînés:**

1. **Decision Tree** (max_depth=10)
2. **Random Forest** (100 estimateurs, max_depth=10)
3. **XGBoost** (100 estimateurs, max_depth=5, learning_rate=0.1)

**Métriques évaluées:**

- Accuracy
- Precision
- Recall
- F1-Score
- Matrices de confusion

**Données:**

- Train/Test Split: 80/20
- Normalisation StandardScaler
- Feature scaling obligatoire

### Module 8: 📈 Backtesting

**Stratégies implémentées:**

1. **Buy & Hold**

   - Baseline simple
   - Achat au premier jour, vente au dernier

2. **Moving Average Crossover**

   - MA rapide: 7 jours
   - MA lente: 21 jours
   - Signal d'achat: MA7 > MA21
   - Signal de vente: MA7 ≤ MA21

3. **RSI Mean Reversion**
   - Achat quand RSI < 30 (survendu)
   - Vente quand RSI > 70 (suracheté)
   - Paramètre RSI: 14 jours

**Métriques de performance:**

- Rendement total (%)
- Courbe d'équité
- Maximum Drawdown
- Valeur finale du portefeuille

## 📦 Dépendances Installées

```
streamlit          # Framework web
pandas             # Manipulation de données
numpy              # Calculs numériques
yfinance           # Données financières
matplotlib         # Visualisation
seaborn            # Visualisation avancée
scikit-learn       # Machine Learning
xgboost            # Gradient Boosting
certifi            # SSL certificates
```

## 🚀 Comment Utiliser

### Installation

```bash
cd "C:\Users\Random\Documents\ENSIAS\S3\P2\data preprocessing\project\app"
pip install -r requirements.txt
```

### Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvrira à `http://localhost:8501`

## 📊 Flux Recommandé

1. **Scraping** → Télécharger les données NASDAQ
2. **Nettoyage** → Supprimer outliers et valeurs manquantes
3. **Visualisation** → Explorer les patterns visuels
4. **Feature Engineering** → Créer les indicateurs techniques
5. **Sélection de Features** → Analyser l'importance et corrélation
6. **Réduction de Dimensionnalité** → PCA et t-SNE
7. **Modélisation ML** → Entraîner et évaluer les modèles
8. **Backtesting** → Tester les stratégies de trading

## 🔧 Améliorations Possibles

### Court terme

- [ ] Ajouter cross-validation
- [ ] Implémenter LSTM pour prédictions temporelles
- [ ] Ajouter Sharpe Ratio et autres métriques
- [ ] Support multi-tickers simultanés

### Moyen terme

- [ ] Intégration avec API réelle (temps réel)
- [ ] Cache des données pour performance
- [ ] Optimisation hyperparamètres (GridSearchCV)
- [ ] Analyse de sentiment des news

### Long terme

- [ ] Ensemble learning (stacking)
- [ ] Reinforcement learning pour stratégies dynamiques
- [ ] Déploiement cloud (AWS/GCP)
- [ ] Dashboard temps réel

## 📝 Architecture Fichiers

```
app/
├── app.py                 # Application principale Streamlit
├── requirements.txt       # Dépendances Python
└── .venv/               # Virtual environment
```

## 🎓 Concepts Clés Utilisés

- **Analyse Technique:** RSI, MACD, Bollinger Bands, ATR
- **Statistiques:** Z-score pour anomalies, rolling statistics
- **ML Classification:** Arbres de décision, Forêts aléatoires, XGBoost
- **Réduction Dimension:** PCA, t-SNE
- **Backtesting:** Simulation trading historique
- **Pandas:** Manipulation de séries temporelles
- **Matplotlib/Seaborn:** Visualisation de données

## ⚠️ Limitations Actuelles

- Pas de commissions de trading (impacte légèrement résultats)
- Pas de slippage (décalage prix d'exécution)
- Données historiques uniquement (pas temps réel)
- Max ~5000 données pour t-SNE
- Pas de gestion de mémoire pour très gros datasets

## ✨ Points Forts

✅ Interface intuitive et bien structurée
✅ 8 modules complets et fonctionnels
✅ Gestion robuste des données MultiIndex
✅ Visualisations riches et informatives
✅ Modèles ML multiples avec comparaison
✅ Stratégies de trading programmables
✅ Export CSV pour tous les résultats

---

**Auteur:** Projet ENSIAS - S3 2025
**Date:** January 7, 2026
**Status:** ✅ Fonctionnel et Testé
