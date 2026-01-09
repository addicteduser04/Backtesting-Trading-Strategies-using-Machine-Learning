# 📋 Guide Détaillé - NASDAQ Trading Analysis

## 🎯 Objectifs du Projet

1. **Collecte & Nettoyage:** Scraper et nettoyer les données du NASDAQ
2. **Analyse Exploratoire:** Visualiser les patterns et tendances
3. **Feature Engineering:** Créer des indicateurs techniques sophistiqués
4. **Machine Learning:** Prédire les mouvements de prix (hausse/baisse)
5. **Backtesting:** Valider les stratégies sur données historiques

---

## 📊 Modules Détaillés

### 1️⃣ SCRAPING DES DONNÉES (Section 1)

**Fonctionnalités:**

- Sélection de 8 tickers NASDAQ majeurs
- Période configurable (par défaut: 5 ans)
- Téléchargement via API Yahoo Finance
- Affichage des 20 premières lignes
- Statistiques globales (nombre de lignes, colonnes, période)
- Export CSV du dataset brut

**Colonnes téléchargées:**

```
- Open: Prix d'ouverture du jour
- High: Prix maximal du jour
- Low: Prix minimal du jour
- Close: Prix de clôture (PRINCIPAL)
- Volume: Nombre d'actions échangées
- Adj Close: Prix ajusté pour les splits/dividendes
```

**Exemple de sortie:**

```
Date        Open  High   Low  Close  Volume  Adj Close
2020-01-02  75.09 75.15  74.37 74.36  135647600 73.41
2020-01-03  74.29 75.11  73.19 75.09  146009600 74.12
...
```

---

### 2️⃣ NETTOYAGE DES DONNÉES (Section 2)

**Étape 1 - Diagnostic:**

- Compte des valeurs manquantes
- Pourcentage manquant par colonne
- Statistiques descriptives

**Étape 2 - Traitement des valeurs manquantes:**

| Méthode       | Utilité                        | Impact                |
| ------------- | ------------------------------ | --------------------- |
| Suppression   | Données très incomplètes       | Réduit taille dataset |
| Forward Fill  | Données ponctuelles manquantes | Préserve continuité   |
| Backward Fill | Pour données futures           | Moins courant         |
| Moyenne       | Distribution uniforme          | Peut déformer données |

**Étape 3 - Détection d'anomalies (Z-score):**

```
Z-score = |X - μ| / σ

Si Z-score > seuil (défaut 3σ) → Anomalie
- 1σ écarte 32% des données
- 2σ écarte 5% des données
- 3σ écarte 0.3% des données
```

**Résultats:**

- Nombre de lignes supprimées
- Graphique avant/après
- Dataset nettoyé exportable

---

### 3️⃣ VISUALISATION EXPLORATOIRE (Section 3)

#### Graphique 1: Prix de Clôture

- Courbe temporelle avec zone remplie
- Montre la tendance générale

#### Graphique 2: Volume d'Échange

- Histogramme quotidien
- Identifie jours avec forte activité

#### Graphique 3: Distribution des Prix

- Histogramme avec moyenne et médiane
- Détecte asymétrie de la distribution

#### Graphique 4: Rendements Quotidiens

```
Rendement = (Prix_t - Prix_t-1) / Prix_t-1 × 100%
```

- Perte/Gain quotidien en pourcentage
- Distribu

tion centrée sur 0

#### Graphique 5: Matrice de Corrélation

- Corrélation entre Open, High, Low, Close, Volume
- Valeurs proches de 1: forte corrélation positive
- Valeurs proches de 0: pas de corrélation
- Valeurs proches de -1: corrélation négative

#### Graphique 6: Chandelier (Candlestick)

```
│ High   (Max du jour)
├─┤
│ Close  (Clôture) - Vert si hausse, Rouge si baisse
├─┤
│ Open   (Ouverture)
├─┤
│ Low    (Min du jour)
```

---

### 4️⃣ FEATURE ENGINEERING (Section 4)

#### A) Moyennes Mobiles

```
MA_n = SUM(Prix_n) / n

- MA_7: Tendance court terme
- MA_21: Tendance moyen terme
- MA_50: Tendance long terme
```

**Signaux:**

- Si Prix > MA: Tendance haussière
- Si Prix < MA: Tendance baissière

#### B) RSI (Relative Strength Index)

```
Gain moyen = SUM(prix hausse) / 14
Perte moyen = SUM(prix baisse) / 14
RS = Gain / Perte
RSI = 100 - (100 / (1 + RS))

Range: 0 à 100
```

**Interprétation:**

- RSI > 70: Suracheté (potentiellement baisse)
- RSI < 30: Survendu (potentiellement hausse)
- 30-70: Zone neutre

#### C) MACD (Moving Average Convergence Divergence)

```
MACD = EMA_12 - EMA_26
Signal = EMA_9(MACD)
```

**Signaux:**

- MACD > Signal: Signal haussier
- MACD < Signal: Signal baissier
- Croisements: Points d'inflexion potentiels

#### D) Bollinger Bands

```
Middle Band = MA_20
Upper Band = Middle + (2 × σ_20)
Lower Band = Middle - (2 × σ_20)
```

**Signaux:**

- Prix > Upper: Potentiellement suracheté
- Prix < Lower: Potentiellement survendu
- Bandes étroites: Faible volatilité → Éclatement attendu

#### E) Volatilité

```
Volatilité = STD(Daily_Return) sur 21 jours
```

- Faible volatilité: Marché calme
- Haute volatilité: Marché turbulent

#### F) ATR (Average True Range)

```
True Range = MAX(
  High - Low,
  ABS(High - Close_t-1),
  ABS(Low - Close_t-1)
)
ATR = MA_14(True Range)
```

Mesure l'amplitude moyenne des mouvements

---

### 5️⃣ SÉLECTION DE FEATURES (Section 5)

#### Analyse de Corrélation

**Matrice de corrélation:** Montre quels indicateurs capturent les mêmes informations

**Interprétation:**

- |r| > 0.8: Forte redondance → Éliminer une
- |r| 0.5-0.8: Corrélation modérée → Garder les deux
- |r| < 0.5: Complémentaires → Très utiles

#### Feature Importance (Random Forest)

```
Algorithme entraîné sur 100 arbres
Importance = Fréquence/Gain d'information apporté
```

**Top Features:** À utiliser en priorité
**Faibles Features:** Peut être supprimées

---

### 6️⃣ RÉDUCTION DE DIMENSIONNALITÉ (Section 6)

#### PCA (Principal Component Analysis)

**Concept:** Trouver les combinaisons linéaires de features qui capturent le maximum de variance

**Variance expliquée:**

- PC1: ~40% de la variance
- PC2: ~25% de la variance
- Ensemble: Cumulé jusqu'à 95%

**Utilité:**

- Visualisation 2D/3D
- Réduction du bruit
- Accélération des modèles

#### t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Concept:** Préserver la structure locale des données en 2D/3D

**Avantages:**

- Révèle les clusters naturels
- Meilleur que PCA pour la visualisation
- Coûteux en calcul

**Cas d'usage:** Détection de patterns cachés

---

### 7️⃣ MODÉLISATION ML (Section 7)

#### Préparation des Données

```
Target = 1 si Prix_t+1 > Prix_t
Target = 0 si Prix_t+1 ≤ Prix_t

Train: 80% (historique)
Test: 20% (non vus par le modèle)
```

#### Modèles Utilisés

**1. Decision Tree** (Arbre de Décision)

- Avantages: Simple, interprétable
- Inconvénients: Prone à l'overfitting
- max_depth=10: Limite la profondeur

**2. Random Forest** (Forêt Aléatoire)

- Ensemble de 100 arbres
- Chaque arbre voit un sous-ensemble aléatoire
- Réduit l'overfitting
- Calcul parallélisable

**3. XGBoost** (Gradient Boosting)

- Les arbres apprennent des erreurs des précédents
- Plus puissant mais demande plus de tuning
- Très utilisé en compétition

#### Métriques d'Évaluation

**Accuracy:** (TP + TN) / Total

- Pourcentage de prédictions correctes
- À utiliser si classes équilibrées

**Precision:** TP / (TP + FP)

- Quand on prédit HAUSSE, c'est juste X% du temps
- Important si faux positif coûteux

**Recall:** TP / (TP + FN)

- % de vraies HAUSSES qu'on capture
- Important ne pas manquer les opportunités

**F1-Score:** 2 × (Precision × Recall) / (Precision + Recall)

- Moyenne harmonique
- Bon compromis

**Matrice de Confusion:**

```
          Réalité
        Hausse Baisse
Pred Hausse [TP] [FP]
     Baisse [FN] [TN]
```

---

### 8️⃣ BACKTESTING (Section 8)

#### Stratégie 1: Buy & Hold

```
Achat: Jour 1
Vente: Dernier jour
```

**Baseline simple** pour comparer avec autres stratégies

#### Stratégie 2: Moving Average Crossover

```
Signal d'achat: MA7 croise MA21 vers le haut
Signal de vente: MA7 croise MA21 vers le bas

Logic:
- Si MA7 > MA21: On est en hausse → Acheter
- Si MA7 ≤ MA21: On est en baisse → Vendre
```

**Avantages:**

- Capture les tendances moyennes
- Filtre le bruit

**Inconvénients:**

- Lent à réagir
- Whipsaw en marché latéral

#### Stratégie 3: RSI Mean Reversion

```
RSI < 30: Survendu → Acheter
RSI > 70: Suracheté → Vendre

Logique: Les extrêmes tendent à se normaliser
```

**Avantages:**

- Réagit rapidement
- Bon en marché oscillant

**Inconvénients:**

- Peut combattre une tendance forte

#### Métriques de Performance

**Rendement Total:**

```
Return = (Valeur_finale - Capital_initial) / Capital_initial × 100%
```

**Drawdown (DD):**

```
DD = Valeur_actuelle / Valeur_max_historique - 1
Max DD = Pire DD observé
```

**Exemple:**

- Capital: 10,000$
- Max atteint: 12,000$
- Valeur actuelle: 11,000$
- DD: 11,000/12,000 - 1 = -8.3%

---

## 💡 Conseils d'Utilisation

### Ordre Recommandé

1. Choisir un ticker (ex: AAPL)
2. Scraper 2-3 ans de données
3. Nettoyer (vérifier qu'aucune donnée manquante)
4. Visualiser (comprendre le comportement)
5. Créer les features
6. Analyser la sélection
7. Réduire la dimensionnalité
8. Entraîner les modèles
9. Tester les stratégies

### Points d'Attention

- ⚠️ Ne pas faire de **lookahead bias** (utiliser future data)
- ⚠️ Les **frais de transaction** réduisent profits (non implémentés)
- ⚠️ **Slippage:** Prix d'exécution peut être différent (non modélisé)
- ⚠️ **Survivorship bias:** Données peuvent être biaisées

### Améliorations Pour Rapport

- Justifier le choix des paramètres (pourquoi MA_7, RSI_14, etc.?)
- Analyser les résultats (pourquoi cet algo marche mieux?)
- Comparer avec benchmarks (indice NASDAQ)
- Discuter des limitations réelles du backtesting

---

## 📈 Résumé des Formules Clés

| Indicateur      | Formule                     |
| --------------- | --------------------------- |
| **MA_n**        | SUM(Prix) / n               |
| **RSI**         | 100 - (100 / (1 + RS))      |
| **MACD**        | EMA_12 - EMA_26             |
| **BB Upper**    | MA_20 + 2×σ                 |
| **Volatilité**  | STD(Rendements)             |
| **Rendement %** | (P_t - P_t-1) / P_t-1 × 100 |

---

**Bonne chance! 🚀**
