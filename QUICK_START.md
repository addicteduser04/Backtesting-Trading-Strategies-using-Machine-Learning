# 🚀 Quick Start Guide

## Installation rapide (2 minutes)

### 1. Installer les dépendances

```bash
cd "C:\Users\Random\Documents\ENSIAS\S3\P2\data preprocessing\project\app"
pip install -r requirements.txt
```

### 2. Lancer l'application

```bash
streamlit run app.py
```

✅ L'app s'ouvre automatiquement à `http://localhost:8501`

---

## Flux Quick Demo (5 minutes)

### Étape 1: Scraper les données (1 min)

1. Sélectionner un ticker (ex: **AAPL**)
2. Garder les dates par défaut
3. Cliquer 🚀 **Lancer le scraping**
4. Attendre quelques secondes
5. Télécharger le CSV (optionnel)

### Étape 2: Nettoyer les données (1 min)

1. Cocher "Supprimer les lignes avec valeurs manquantes"
2. Cocher "Détecter et traiter les anomalies"
3. Garder seuil par défaut (3.0)
4. Cliquer 🧹 **Nettoyer les données**
5. Voir la comparaison avant/après

### Étape 3: Visualiser (1 min)

1. Passer à "📊 Visualisation"
2. Scroll les 6 graphiques
3. Observer les patterns et tendances

### Étape 4: Features (1 min)

1. Passer à "⚙️ Feature Engineering"
2. Cliquer ⚙️ **Générer les features**
3. Voir les indicateurs techniques

### Étape 5: Backtesting (1 min)

1. Passer à "📈 Backtesting"
2. Sélectionner "Buy & Hold"
3. Cliquer ▶️ **Lancer le Backtest**
4. Voir le graphique de performance

---

## Fichiers Importants

```
app.py
├── 900 lignes de code
├── 8 sections complètes
└── Prêt à l'emploi

requirements.txt
├── Streamlit, Pandas, NumPy
├── Matplotlib, Seaborn
├── scikit-learn, XGBoost
└── yfinance, certifi

README.md
└── Documentation complète

GUIDE_DETAILLE.md
└── Explication détaillée de chaque module
```

---

## Résolution de Problèmes

### ❌ "ModuleNotFoundError: No module named..."

**Solution:**

```bash
pip install [nom_module]
```

### ❌ "Connection error" (yfinance)

**Solution:** Vérifier la connexion internet, Yahoo Finance peut être temporairement indisponible

### ❌ "Streamlit not found"

**Solution:**

```bash
pip install streamlit
```

### ❌ L'app est lente

**Solution:** Réduire la période de dates ou utiliser moins de données

---

## Conseils Importants

### ✅ Faire

- Tester avec un seul ticker d'abord (AAPL)
- Utiliser 2-3 ans de données pour démarrer
- Bien observer les graphiques avant de passer à ML
- Comparer les stratégies de backtesting

### ❌ Ne pas faire

- Trop de tickers en même temps
- 20+ ans de données (très lent)
- Changer tous les paramètres à la fois
- Oublier que le passé ne garantit pas l'avenir

---

## Performance Attendue

| Étape            | Temps     |
| ---------------- | --------- |
| Scraping (2 ans) | 5-10 sec  |
| Nettoyage        | 1-2 sec   |
| Visualisation    | 3-5 sec   |
| Features         | 2-3 sec   |
| ML Training      | 10-15 sec |
| t-SNE            | 20-30 sec |
| Backtesting      | 1-2 sec   |

**Total:** ~1-2 minutes pour complet

---

## Customisation Facile

### Changer les paramètres MA

Dans **Feature Engineering**, modifier:

```python
df['MA_7'] = close_prices.rolling(window=7).mean()
# Changer 7 en 5 ou 10
```

### Ajouter une stratégie

Dans **Backtesting**, ajouter:

```python
elif test_strategy == "Nouvelle":
    # Votre logique
```

### Changer les modèles

Dans **Modélisation ML**, modifier `n_estimators`, `max_depth`, etc.

---

## Documentation Complète

- 📖 **README.md** → Vue d'ensemble et architecture
- 📚 **GUIDE_DETAILLE.md** → Explications approfondies
- 💻 **app.py** → Code commenté (900 lignes)

---

## Contact & Support

**Questions fréquentes:**

- Q: Pourquoi certains modèles marchent mieux?

  - A: Dépend de la distribution des données et du ticker

- Q: Peut-on utiliser en trading réel?

  - A: Pas sans améliorations (commissions, slippage, temps réel)

- Q: Comment améliorer la précision?
  - A: Plus de features, meilleur tuning, ensemble learning

---

**Bon trading! 📈**
