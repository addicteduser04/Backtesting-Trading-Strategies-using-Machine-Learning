import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle

# Helper function to get column by prefix
def get_column_by_prefix(df, prefix):
    """Find column that contains prefix, handling flattened MultiIndex"""
    # First try exact match
    if prefix in df.columns:
        return prefix
    # Then try prefix match
    cols = [col for col in df.columns if str(col).startswith(prefix)]
    if cols:
        return cols[0]
    # Finally try contains match (e.g., 'Close_AAPL' for 'Close')
    cols = [col for col in df.columns if prefix in str(col)]
    return cols[0] if cols else None

# Configuration de la page
st.set_page_config(page_title="NASDAQ Trading Analysis", layout="wide", page_icon="📈")

# Titre principal
st.title("📈 Analyse et Prédiction NASDAQ - Mini Projet")
st.markdown("---")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisir une section:", 
                        ["🔍 Scraping des données", 
                         "🧹 Nettoyage", 
                         "📊 Visualisation",
                         "⚙️ Feature Engineering",
                         "🎯 Sélection de Features",
                         "🔍 Réduction de Dimensionnalité",
                         "🤖 Modélisation ML",
                         "📈 Backtesting"])

# ==================== SECTION 1: SCRAPING ====================
if page == "🔍 Scraping des données":
    st.header("1. Scraping des données NASDAQ")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuration")
        
        # Liste des tickers NASDAQ populaires
        tickers_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
        ticker = st.selectbox("Sélectionner un ticker NASDAQ:", tickers_list)
        
        start_date = st.date_input("Date de début:", datetime(2020, 1, 1))
        end_date = st.date_input("Date de fin:", datetime.now())
        
        if st.button("🚀 Lancer le scraping", type="primary"):
            with st.spinner(f"Scraping des données {ticker}..."):
                try:
                    # Téléchargement des données via yfinance
                    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    
                    # Flatten MultiIndex columns - handle both MultiIndex and tuple columns
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = ['_'.join(str(c) for c in col).strip() for col in df.columns.values]
                    elif df.columns.dtype == 'object' and len(df.columns) > 0 and isinstance(df.columns[0], tuple):
                        df.columns = ['_'.join(str(c) for c in col).strip() for col in df.columns]
                    else:
                        df.columns = [str(col) for col in df.columns]
                    
                    # Sauvegarde dans session state
                    st.session_state['raw_data'] = df
                    st.session_state['ticker'] = ticker
                    
                    st.success(f"✅ Données récupérées avec succès! {len(df)} lignes")
                    
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
    
    with col2:
        st.subheader("Informations")
        st.info("""
        **Source des données:** Yahoo Finance
        
        **Données récupérées:**
        - Open (Prix d'ouverture)
        - High (Prix max)
        - Low (Prix min)
        - Close (Prix de clôture)
        - Volume (Volume d'échange)
        - Adj Close (Prix ajusté)
        """)
    
    # Affichage des données
    if 'raw_data' in st.session_state:
        st.subheader("Aperçu des données brutes")
        df = st.session_state['raw_data']
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Lignes", len(df))
        col2.metric("Colonnes", len(df.columns))
        col3.metric("Période", f"{(df.index[-1] - df.index[0]).days} jours")
        col4.metric("Valeurs manquantes", df.isnull().sum().sum())
        
        st.dataframe(df.head(20), use_container_width=True)
        
        # Bouton de téléchargement
        csv = df.to_csv()
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name=f"{ticker}_raw_data.csv",
            mime="text/csv"
        )

# ==================== SECTION 2: NETTOYAGE ====================
elif page == "🧹 Nettoyage":
    st.header("2. Nettoyage des données")
    
    if 'raw_data' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord scraper des données dans la section précédente.")
    else:
        df = st.session_state['raw_data'].copy()
        
        # Flatten MultiIndex columns - handle both MultiIndex and tuple columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(str(c) for c in col).strip() for col in df.columns.values]
        elif df.columns.dtype == 'object' and len(df.columns) > 0 and isinstance(df.columns[0], tuple):
            # Handle case where columns are tuples but not MultiIndex
            df.columns = ['_'.join(str(c) for c in col).strip() for col in df.columns]
        else:
            # Ensure all columns are strings
            df.columns = [str(col) for col in df.columns]
        
        st.subheader("📊 Analyse de la qualité des données")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total des lignes", len(df))
            st.metric("Valeurs manquantes", df.isnull().sum().sum())
        
        with col2:
            missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
            st.write("**% Manquantes par colonne:**")
            for col, pct in missing_pct.items():
                st.write(f"- {col}: {pct}%")
        
        with col3:
            st.write("**Statistiques descriptives:**")
            close_col = get_column_by_prefix(df, 'Close')
            if close_col:
                st.write(f"Prix moyen: ${float(df[close_col].mean()):.2f}")
                st.write(f"Prix min: ${float(df[close_col].min()):.2f}")
                st.write(f"Prix max: ${float(df[close_col].max()):.2f}")
                st.write(f"Volatilité (std): {float(df[close_col].std()):.2f}")
        
        st.markdown("---")
        st.subheader("🛠️ Options de nettoyage")
        
        col1, col2 = st.columns(2)
        
        with col1:
            remove_missing = st.checkbox("Supprimer les lignes avec valeurs manquantes")
            fill_method = st.selectbox("Méthode de remplissage:", 
                                      ["Aucun", "Forward Fill", "Backward Fill", "Moyenne"])
        
        with col2:
            remove_outliers = st.checkbox("Détecter et traiter les anomalies")
            if remove_outliers:
                std_threshold = st.slider("Seuil (écarts-types):", 1.0, 5.0, 3.0)
        
        if st.button("🧹 Nettoyer les données", type="primary"):
            df_cleaned = df.copy()
            
            # 1. Traitement des valeurs manquantes
            if remove_missing:
                df_cleaned = df_cleaned.dropna()
                st.info(f"✓ Lignes supprimées: {len(df) - len(df_cleaned)}")
            
            elif fill_method == "Forward Fill":
                df_cleaned = df_cleaned.fillna(method='ffill')
                st.info("✓ Forward fill appliqué")
            
            elif fill_method == "Backward Fill":
                df_cleaned = df_cleaned.fillna(method='bfill')
                st.info("✓ Backward fill appliqué")
            
            elif fill_method == "Moyenne":
                df_cleaned = df_cleaned.fillna(df_cleaned.mean())
                st.info("✓ Remplissage par moyenne appliqué")
            
            # 2. Détection des anomalies (méthode Z-score)
            if remove_outliers:
                for col_prefix in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    col = get_column_by_prefix(df_cleaned, col_prefix)
                    if col:
                        mean = df_cleaned[col].mean()
                        std = df_cleaned[col].std()
                        z_scores = np.abs((df_cleaned[col] - mean) / std)
                        df_cleaned = df_cleaned[z_scores < std_threshold]
                
                st.info(f"✓ Anomalies détectées et supprimées: {len(df) - len(df_cleaned)} lignes")
            
            # Sauvegarde
            st.session_state['cleaned_data'] = df_cleaned
            st.success(f"✅ Nettoyage terminé! Dataset final: {len(df_cleaned)} lignes")
            
            # Comparaison avant/après
            st.subheader("📈 Comparaison avant/après")
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            close_col = get_column_by_prefix(df, 'Close')
            if close_col:
                axes[0].plot(df.index, df[close_col], label='Avant nettoyage', alpha=0.7)
                axes[0].set_title('Données brutes')
                axes[0].set_xlabel('Date')
                axes[0].set_ylabel('Prix de clôture ($)')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                axes[1].plot(df_cleaned.index, df_cleaned[close_col], label='Après nettoyage', color='green', alpha=0.7)
                axes[1].set_title('Données nettoyées')
                axes[1].set_xlabel('Date')
                axes[1].set_ylabel('Prix de clôture ($)')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.dataframe(df_cleaned.head(20), use_container_width=True)

# ==================== SECTION 3: VISUALISATION ====================
elif page == "📊 Visualisation":
    st.header("3. Visualisation exploratoire")
    
    if 'cleaned_data' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord nettoyer les données.")
    else:
        df = st.session_state['cleaned_data'].copy()
        
        # Graphique 1: Prix de clôture
        st.subheader("📈 Évolution du prix de clôture")
        fig, ax = plt.subplots(figsize=(14, 6))
        close_col = get_column_by_prefix(df, 'Close')
        if close_col:
            close_values = df[close_col].values.flatten()
            ax.plot(df.index, close_values, linewidth=2, color='#1f77b4')
            ax.fill_between(df.index, close_values, alpha=0.3)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Prix de clôture ($)', fontsize=12)
            ax.set_title(f"Prix de clôture - {st.session_state.get('ticker', 'NASDAQ')}", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Graphique 2: Volume
        st.subheader("📊 Volume d'échange")
        fig, ax = plt.subplots(figsize=(14, 5))
        volume_col = get_column_by_prefix(df, 'Volume')
        if volume_col:
            volume_values = df[volume_col].values.flatten()
            ax.bar(df.index, volume_values, color='orange', alpha=0.7)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Volume', fontsize=12)
            ax.set_title('Volume d\'échange quotidien', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique 3: Distribution des prix
            st.subheader("📉 Distribution des prix")
            fig, ax = plt.subplots(figsize=(7, 5))
            close_col = get_column_by_prefix(df, 'Close')
            if close_col:
                close_values = df[close_col].values.flatten()
                ax.hist(close_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
                close_mean = float(df[close_col].mean())
                close_median = float(df[close_col].median())
                ax.axvline(close_mean, color='red', linestyle='--', label=f"Moyenne: ${close_mean:.2f}")
                ax.axvline(close_median, color='green', linestyle='--', label=f"Médiane: ${close_median:.2f}")
                ax.set_xlabel('Prix de clôture ($)', fontsize=12)
                ax.set_ylabel('Fréquence', fontsize=12)
                ax.set_title('Distribution des prix', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                st.pyplot(fig)
        
        with col2:
            # Graphique 4: Rendements quotidiens
            st.subheader("📊 Rendements quotidiens")
            close_col = get_column_by_prefix(df, 'Close')
            if close_col:
                df_viz = df.copy()
                df_viz['Returns'] = df_viz[close_col].pct_change() * 100
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.hist(df_viz['Returns'].dropna(), bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
                ax.axvline(0, color='black', linestyle='--', linewidth=2)
                ax.set_xlabel('Rendement (%)', fontsize=12)
                ax.set_ylabel('Fréquence', fontsize=12)
                ax.set_title('Distribution des rendements quotidiens', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                st.pyplot(fig)
        
        # Graphique 5: Heatmap de corrélation
        st.subheader("🔥 Matrice de corrélation")
        # Find columns dynamically
        cols_to_corr = []
        for col_name in ['Open', 'High', 'Low', 'Close', 'Volume']:
            col = get_column_by_prefix(df, col_name)
            if col:
                cols_to_corr.append(col)
        
        if cols_to_corr:
            corr_data = df[cols_to_corr].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                        square=True, linewidths=1, ax=ax, fmt='.2f')
            ax.set_title('Corrélation entre les variables', fontsize=14, fontweight='bold')
            st.pyplot(fig)
        
        # Graphique 6: Chandelier (Candlestick) simplifié
        st.subheader("🕯️ Graphique Chandelier (dernier mois)")
        
        # Get column names
        close_col = get_column_by_prefix(df, 'Close')
        open_col = get_column_by_prefix(df, 'Open')
        low_col = get_column_by_prefix(df, 'Low')
        high_col = get_column_by_prefix(df, 'High')
        
        if close_col and open_col and low_col and high_col:
            df_recent = df.tail(30)
            
            fig, ax = plt.subplots(figsize=(14, 6))
            for i, (idx, row) in enumerate(df_recent.iterrows()):
                close = row[close_col].item() if isinstance(row[close_col], pd.Series) else float(row[close_col])
                open_ = row[open_col].item() if isinstance(row[open_col], pd.Series) else float(row[open_col])
                low = row[low_col].item() if isinstance(row[low_col], pd.Series) else float(row[low_col])
                high = row[high_col].item() if isinstance(row[high_col], pd.Series) else float(row[high_col])
                color = 'green' if close > open_ else 'red'
                ax.plot([i, i], [low, high], color='black', linewidth=1)
                ax.plot([i, i], [open_, close], color=color, linewidth=5)
            
            ax.set_xlabel('Jours', fontsize=12)
            ax.set_ylabel('Prix ($)', fontsize=12)
            ax.set_title('Chandelier - 30 derniers jours', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

# ==================== SECTION 4: FEATURE ENGINEERING ====================
elif page == "⚙️ Feature Engineering":
    st.header("4. Feature Engineering")
    
    if 'cleaned_data' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord nettoyer les données.")
    else:
        df = st.session_state['cleaned_data'].copy()
        
        st.subheader("🔧 Création des features techniques")
        
        st.info("""
        **Features qui seront créées:**
        - Moyennes mobiles (MA_7, MA_21, MA_50)
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - Bollinger Bands
        - Rendements et Volatilité
        - Features temporelles
        """)
        
        if st.button("⚙️ Générer les features", type="primary"):
            with st.spinner("Calcul des indicateurs techniques..."):
                
                # Get column names using helper function
                close_col = get_column_by_prefix(df, 'Close')
                high_col = get_column_by_prefix(df, 'High')
                low_col = get_column_by_prefix(df, 'Low')
                
                if not close_col:
                    st.error("Column 'Close' not found!")
                else:
                    # Gérer la structure potentiellement MultiIndex
                    close_prices = df[close_col].squeeze() if isinstance(df[close_col], pd.DataFrame) else df[close_col]
                    
                    # 1. Moyennes mobiles
                    df['MA_7'] = close_prices.rolling(window=7).mean()
                    df['MA_21'] = close_prices.rolling(window=21).mean()
                    df['MA_50'] = close_prices.rolling(window=50).mean()
                    
                    # 2. RSI (Relative Strength Index)
                    delta = close_prices.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['RSI'] = 100 - (100 / (1 + rs))
                    
                    # 3. MACD
                    ema_12 = close_prices.ewm(span=12, adjust=False).mean()
                    ema_26 = close_prices.ewm(span=26, adjust=False).mean()
                    df['MACD'] = ema_12 - ema_26
                    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                    
                    # 4. Bollinger Bands
                    df['BB_Middle'] = close_prices.rolling(window=20).mean()
                    bb_std = close_prices.rolling(window=20).std()
                    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
                    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
                    
                    # 5. Rendements et volatilité
                    df['Daily_Return'] = close_prices.pct_change()
                    df['Volatility'] = df['Daily_Return'].rolling(window=21).std()
                    
                    # 6. ATR (Average True Range)
                    high = df[high_col].squeeze() if high_col and isinstance(df[high_col], pd.DataFrame) else df[high_col] if high_col else close_prices
                    low = df[low_col].squeeze() if low_col and isinstance(df[low_col], pd.DataFrame) else df[low_col] if low_col else close_prices
                    high_low = high - low
                    high_close = np.abs(high - close_prices.shift())
                    low_close = np.abs(low - close_prices.shift())
                    ranges = pd.concat([high_low, high_close, low_close], axis=1)
                    true_range = ranges.max(axis=1)
                    df['ATR'] = true_range.rolling(14).mean()
                    
                    # 7. Features temporelles
                    df['Day_of_Week'] = df.index.dayofweek
                    df['Month'] = df.index.month
                    df['Quarter'] = df.index.quarter
                    
                    # 8. Lag features
                    df['Close_Lag1'] = close_prices.shift(1)
                    df['Close_Lag2'] = close_prices.shift(2)
                    df['Close_Lag3'] = close_prices.shift(3)
                    
                    # 9. Target (pour ML)
                    df['Target'] = (close_prices.shift(-1) > close_prices).astype(int)
                    
                    # Supprimer les NaN générés
                    df_features = df.dropna()
                
                st.session_state['features_data'] = df_features
                st.success(f"✅ {len(df_features.columns)} features créées!")
                
                # Affichage des features
                st.subheader("📋 Aperçu des features")
                st.dataframe(df_features.tail(20), use_container_width=True)
                
                # Statistiques
                col1, col2, col3 = st.columns(3)
                col1.metric("Total features", len(df_features.columns))
                col2.metric("Lignes finales", len(df_features))
                col3.metric("Target: Hausse", f"{df_features['Target'].sum()} ({df_features['Target'].mean()*100:.1f}%)")
                
                # Visualisation des indicateurs
                st.subheader("📊 Visualisation des indicateurs")
                
                # Moyennes mobiles
                fig, ax = plt.subplots(figsize=(14, 6))
                close_col = get_column_by_prefix(df_features, 'Close')
                if close_col:
                    ax.plot(df_features.index, df_features[close_col], label='Close', linewidth=2)
                    ax.plot(df_features.index, df_features['MA_7'], label='MA 7', alpha=0.7)
                    ax.plot(df_features.index, df_features['MA_21'], label='MA 21', alpha=0.7)
                    ax.plot(df_features.index, df_features['MA_50'], label='MA 50', alpha=0.7)
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Prix ($)')
                    ax.set_title('Prix et Moyennes Mobiles', fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # RSI
                    rsi_col = get_column_by_prefix(df_features, 'RSI')
                    fig, ax = plt.subplots(figsize=(7, 5))
                    if rsi_col:
                        ax.plot(df_features.index, df_features[rsi_col], color='purple', linewidth=2)
                        ax.axhline(70, color='red', linestyle='--', label='Suracheté')
                        ax.axhline(30, color='green', linestyle='--', label='Survendu')
                        ax.fill_between(df_features.index, 30, 70, alpha=0.1)
                        ax.set_xlabel('Date')
                        ax.set_ylabel('RSI')
                        ax.set_title('RSI (Relative Strength Index)', fontsize=14, fontweight='bold')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                
                with col2:
                    # MACD
                    macd_col = get_column_by_prefix(df_features, 'MACD')
                    signal_col = get_column_by_prefix(df_features, 'MACD_Signal')
                    fig, ax = plt.subplots(figsize=(7, 5))
                    if macd_col and signal_col:
                        ax.plot(df_features.index, df_features[macd_col], label='MACD', linewidth=2)
                        ax.plot(df_features.index, df_features[signal_col], label='Signal', linewidth=2)
                        ax.axhline(0, color='black', linestyle='--', linewidth=1)
                        ax.set_xlabel('Date')
                        ax.set_ylabel('MACD')
                        ax.set_title('MACD', fontsize=14, fontweight='bold')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                
                # Bollinger Bands
                fig, ax = plt.subplots(figsize=(14, 6))
                if close_col:
                    ax.plot(df_features.index, df_features[close_col], label='Close', linewidth=2, color='blue')
                    ax.plot(df_features.index, df_features['BB_Upper'], label='BB Upper', linestyle='--', color='red', alpha=0.7)
                    ax.plot(df_features.index, df_features['BB_Middle'], label='BB Middle', linestyle='--', color='gray', alpha=0.7)
                    ax.plot(df_features.index, df_features['BB_Lower'], label='BB Lower', linestyle='--', color='green', alpha=0.7)
                    ax.fill_between(df_features.index, df_features['BB_Lower'], df_features['BB_Upper'], alpha=0.1)
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Prix ($)')
                    ax.set_title('Bollinger Bands', fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Téléchargement
                csv = df_features.to_csv()
                st.download_button(
                    label="📥 Télécharger les données avec features",
                    data=csv,
                    file_name=f"{st.session_state.get('ticker', 'NASDAQ')}_features.csv",
                    mime="text/csv"
                )

# ==================== SECTION 5: SELECTION DE FEATURES ====================
elif page == "🎯 Sélection de Features":
    st.header("5. Sélection de Caractéristiques")
    
    if 'cleaned_data' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord nettoyer les données.")
    else:
        # Utiliser les données avec features déjà créées ou en créer de nouvelles
        df = st.session_state['cleaned_data'].copy()
        
        st.subheader("🔧 Création des features pour sélection")
        
        # Créer les features si pas déjà existantes
        features_to_create = ['MA_7', 'MA_21', 'MA_50', 'RSI', 'MACD', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'Volatility', 'Daily_Return']
        
        if not all(col in df.columns for col in features_to_create):
            st.info("🔄 Création des features techniques...")
            
            close_col = get_column_by_prefix(df, 'Close')
            if close_col:
                # Ma mobiles
                df['MA_7'] = df[close_col].squeeze().rolling(window=7).mean()
                df['MA_21'] = df[close_col].squeeze().rolling(window=21).mean()
                df['MA_50'] = df[close_col].squeeze().rolling(window=50).mean()
                
                # RSI
                close_prices = df[close_col].squeeze() if isinstance(df[close_col], pd.DataFrame) else df[close_col]
                delta = close_prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # MACD
                ema_12 = close_prices.ewm(span=12, adjust=False).mean()
                ema_26 = close_prices.ewm(span=26, adjust=False).mean()
                df['MACD'] = ema_12 - ema_26
                
                # Bollinger Bands
                df['BB_Middle'] = close_prices.rolling(window=20).mean()
                bb_std = close_prices.rolling(window=20).std()
                df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
                df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
                
                # Volatilité et rendements
                df['Daily_Return'] = close_prices.pct_change()
                df['Volatility'] = df['Daily_Return'].rolling(window=21).std()
                
                # Supprimer les NaN
                df = df.dropna()
                st.session_state['featured_data'] = df
        
        st.subheader("📊 Analyse de Corrélation")
        
        # Sélectionner les colonnes numériques pour la corrélation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            correlation_matrix = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'Corrélation'})
            ax.set_title('Matrice de Corrélation - Indicateurs Techniques', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            
            st.write("**Observations:**")
            st.info("- Les indicateurs fortement corrélés offrent moins d'information")
            st.info("- Les indicateurs peu corrélés capturent différents aspects du marché")
        
        st.subheader("🎯 Feature Importance (Random Forest)")
        
        # Préparer les données pour le ML
        df_ml = df.copy()
        
        # Créer la variable cible (prédire si le prix monte ou descend demain)
        close_col = get_column_by_prefix(df_ml, 'Close')
        if close_col:
            close_prices = df_ml[close_col].squeeze() if isinstance(df_ml[close_col], pd.DataFrame) else df_ml[close_col]
            df_ml['Target'] = (close_prices.shift(-1) > close_prices).astype(int)
            df_ml = df_ml.dropna()
            
            # Features à utiliser
            feature_cols = [str(col) for col in numeric_cols if col not in ['Close', 'Open', 'High', 'Low', 'Volume'] and col in df_ml.columns]
            
            if len(feature_cols) > 0 and len(df_ml) > 50:
                X = df_ml[feature_cols].fillna(0)
                y = df_ml['Target']
                
                # Scaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
                
                # Entraîner un Random Forest pour obtenir l'importance
                rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
                rf.fit(X_scaled, y)
                
                # Afficher l'importance
                feature_importance = pd.DataFrame({
                    'Feature': [str(f) for f in feature_cols],
                    'Importance': rf.feature_importances_
                }).sort_values('Importance', ascending=False).reset_index(drop=True)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(feature_importance['Feature'].values, feature_importance['Importance'].values, color='steelblue')
                ax.set_xlabel('Importance', fontsize=12)
                ax.set_title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
                ax.invert_yaxis()
                st.pyplot(fig)
                
                st.dataframe(feature_importance, use_container_width=True)
                
                # Sauvegarder
                st.session_state['feature_importance'] = feature_importance
                st.session_state['X_scaled'] = X_scaled
                st.session_state['y_target'] = y
                st.session_state['feature_cols'] = feature_cols

# ==================== SECTION 6: REDUCTION DE DIMENSIONNALITE ====================
elif page == "🔍 Réduction de Dimensionnalité":
    st.header("6. Réduction de Dimensionnalité")
    
    if 'X_scaled' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord effectuer la sélection de features.")
    else:
        X_scaled = st.session_state['X_scaled']
        y_target = st.session_state['y_target']
        
        st.subheader("📉 PCA (Principal Component Analysis)")
        
        # PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Variance expliquée
        ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Composante Principale', fontsize=12)
        ax1.set_ylabel('Variance Expliquée', fontsize=12)
        ax1.set_title('Variance Expliquée par chaque PC', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Variance cumulée
        ax2.plot(range(1, len(cumsum) + 1), cumsum, 'o-', color='steelblue', linewidth=2)
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
        ax2.set_xlabel('Nombre de Composantes', fontsize=12)
        ax2.set_ylabel('Variance Cumulée', fontsize=12)
        ax2.set_title('Variance Cumulée', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # PCA 2D
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_target, cmap='viridis', alpha=0.6)
        ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
        ax.set_title('PCA - 2D Projection', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Target (0=Baisse, 1=Hausse)')
        st.pyplot(fig)
        
        st.session_state['X_pca_2d'] = X_pca_2d
        st.session_state['pca_2d'] = pca_2d
        
        st.subheader("🔮 t-SNE (t-Distributed Stochastic Neighbor Embedding)")
        
        if st.button("Calculer t-SNE (peut prendre quelques secondes)..."):
            with st.spinner("Calcul en cours..."):
                tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                X_tsne = tsne.fit_transform(X_scaled)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_target, cmap='viridis', alpha=0.6)
                ax.set_xlabel('t-SNE 1', fontsize=12)
                ax.set_ylabel('t-SNE 2', fontsize=12)
                ax.set_title('t-SNE - Market Behavior Clustering', fontsize=14, fontweight='bold')
                plt.colorbar(scatter, ax=ax, label='Target (0=Baisse, 1=Hausse)')
                st.pyplot(fig)
                
                st.session_state['X_tsne'] = X_tsne
                st.session_state['tsne'] = tsne

# ==================== SECTION 7: MODELISATION ML ====================
elif page == "🤖 Modélisation ML":
    st.header("7. Modélisation Machine Learning")
    
    if 'X_scaled' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord effectuer la sélection de features.")
    else:
        X_scaled = st.session_state['X_scaled']
        y_target = st.session_state['y_target']
        
        st.subheader("🚀 Entraînement des Modèles")
        
        # Diviser les données
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_target, test_size=0.2, random_state=42)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Données d'entraînement:** {len(X_train)} samples")
            st.write(f"**Données de test:** {len(X_test)} samples")
        
        with col2:
            st.write(f"**Classe 0 (Baisse):** {(y_train == 0).sum()} samples")
            st.write(f"**Classe 1 (Hausse):** {(y_train == 1).sum()} samples")
        
        # Entraîner les modèles
        models = {}
        
        st.info("⏳ Entraînement des modèles en cours...")
        
        # 1. Decision Tree
        dt = DecisionTreeClassifier(max_depth=10, random_state=42)
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)
        models['Decision Tree'] = {'model': dt, 'predictions': y_pred_dt}
        
        # 2. Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        models['Random Forest'] = {'model': rf, 'predictions': y_pred_rf}
        
        # 3. XGBoost
        xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=0)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        models['XGBoost'] = {'model': xgb, 'predictions': y_pred_xgb}
        
        # Résultats
        st.success("✅ Entraînement terminé!")
        
        st.subheader("📊 Résultats des Modèles")
        
        results = []
        for model_name, model_data in models.items():
            y_pred = model_data['predictions']
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            results.append({
                'Modèle': model_name,
                'Accuracy': f"{acc:.4f}",
                'Precision': f"{prec:.4f}",
                'Recall': f"{recall:.4f}",
                'F1-Score': f"{f1:.4f}"
            })
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        # Sauvegarder les modèles
        st.session_state['models'] = models
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        
        # Afficher les matrices de confusion
        st.subheader("🎯 Matrices de Confusion")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, (model_name, model_data) in enumerate(models.items()):
            y_pred = model_data['predictions']
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
            axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Prédiction')
            axes[idx].set_ylabel('Réalité')
        
        st.pyplot(fig)

# ==================== SECTION 8: BACKTESTING ====================
elif page == "📈 Backtesting":
    st.header("8. Backtesting des Stratégies")
    
    if 'featured_data' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord créer les features.")
    else:
        df = st.session_state['featured_data'].copy()
        close_col = get_column_by_prefix(df, 'Close')
        
        if close_col:
            close_prices = df[close_col].squeeze() if isinstance(df[close_col], pd.DataFrame) else df[close_col]
            
            st.subheader("🧪 Implémentation des Stratégies de Trading")
            
            col1, col2 = st.columns(2)
            
            with col1:
                test_strategy = st.selectbox("Sélectionner une stratégie:", 
                                             ["Buy & Hold", 
                                              "Moving Average Crossover",
                                              "RSI Mean Reversion",
                                              "ML-Based (si disponible)"])
            
            with col2:
                initial_capital = st.number_input("Capital initial ($)", value=10000, min_value=1000)
            
            # Implémenter les stratégies
            def backtest_buy_hold(prices, initial_capital):
                shares = initial_capital / prices.iloc[0]
                final_value = shares * prices.iloc[-1]
                returns = (final_value - initial_capital) / initial_capital * 100
                equity_curve = shares * prices
                return equity_curve, returns
            
            def backtest_ma_crossover(prices, initial_capital, fast=7, slow=21):
                ma_fast = prices.rolling(window=fast).mean()
                ma_slow = prices.rolling(window=slow).mean()
                
                signals = np.zeros(len(prices))
                signals[ma_fast > ma_slow] = 1  # Signal d'achat
                signals[ma_fast <= ma_slow] = 0  # Signal de vente
                
                position = 0
                shares = 0
                equity = [initial_capital]
                
                for i in range(1, len(prices)):
                    if signals[i] == 1 and position == 0:
                        shares = equity[-1] / prices.iloc[i]
                        position = 1
                        equity.append(shares * prices.iloc[i])
                    elif signals[i] == 0 and position == 1:
                        equity.append(shares * prices.iloc[i])
                        position = 0
                        shares = 0
                    else:
                        if position == 1:
                            equity.append(shares * prices.iloc[i])
                        else:
                            equity.append(equity[-1])
                
                # Ensure we have the correct number of values
                while len(equity) < len(prices):
                    if position == 1:
                        equity.append(shares * prices.iloc[-1])
                    else:
                        equity.append(equity[-1])
                
                equity = pd.Series(equity[:len(prices)], index=prices.index)
                returns = (equity.iloc[-1] - initial_capital) / initial_capital * 100
                return equity, returns
            
            def backtest_rsi_mean_reversion(prices, initial_capital):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                signals = np.zeros(len(prices))
                signals[rsi < 30] = 1  # Oversold - achat
                signals[rsi > 70] = 0  # Overbought - vente
                
                position = 0
                shares = 0
                equity = [initial_capital]
                
                for i in range(1, len(prices)):
                    if signals[i] == 1 and position == 0:
                        shares = equity[-1] / prices.iloc[i]
                        position = 1
                        equity.append(shares * prices.iloc[i])
                    elif signals[i] == 0 and position == 1:
                        equity.append(shares * prices.iloc[i])
                        position = 0
                        shares = 0
                    else:
                        if position == 1:
                            equity.append(shares * prices.iloc[i])
                        else:
                            equity.append(equity[-1])
                
                # Ensure we have the correct number of values
                while len(equity) < len(prices):
                    if position == 1:
                        equity.append(shares * prices.iloc[-1])
                    else:
                        equity.append(equity[-1])
                
                equity = pd.Series(equity[:len(prices)], index=prices.index)
                returns = (equity.iloc[-1] - initial_capital) / initial_capital * 100
                return equity, returns
            
            if st.button("▶️ Lancer le Backtest", type="primary"):
                st.info("Exécution du backtest...")
                
                equity_curves = {}
                returns_dict = {}
                
                # Exécuter les stratégies
                if test_strategy == "Buy & Hold":
                    equity, ret = backtest_buy_hold(close_prices, initial_capital)
                    equity_curves['Buy & Hold'] = equity
                    returns_dict['Buy & Hold'] = ret
                
                elif test_strategy == "Moving Average Crossover":
                    equity, ret = backtest_ma_crossover(close_prices, initial_capital)
                    equity_curves['MA Crossover'] = equity
                    returns_dict['MA Crossover'] = ret
                
                elif test_strategy == "RSI Mean Reversion":
                    equity, ret = backtest_rsi_mean_reversion(close_prices, initial_capital)
                    equity_curves['RSI Mean Reversion'] = equity
                    returns_dict['RSI Mean Reversion'] = ret
                
                # Afficher les résultats
                st.subheader("📈 Résultats du Backtest")
                
                fig, ax = plt.subplots(figsize=(14, 6))
                for strategy_name, equity in equity_curves.items():
                    ax.plot(equity.index, equity.values, label=f'{strategy_name} - Return: {returns_dict[strategy_name]:.2f}%', linewidth=2)
                
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Valeur du Portefeuille ($)', fontsize=12)
                ax.set_title('Backtesting - Evolution du Portefeuille', fontsize=14, fontweight='bold')
                ax.legend(loc='best', fontsize=10)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Afficher les métriques
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Retours:**")
                    for strategy_name, ret in returns_dict.items():
                        color = "green" if ret > 0 else "red"
                        st.markdown(f"- **{strategy_name}**: <span style='color:{color}'>{ret:.2f}%</span>", unsafe_allow_html=True)
                
                with col2:
                    st.write("**Métriques Additionnelles:**")
                    for strategy_name, equity in equity_curves.items():
                        max_value = equity.max()
                        max_drawdown = (equity / equity.expanding().max() - 1).min() * 100
                        st.write(f"- {strategy_name}: Max DD = {max_drawdown:.2f}%")
                
                st.session_state['backtest_results'] = {
                    'equity_curves': equity_curves,
                    'returns': returns_dict
                }

# ==================== FOOTER ====================
st.sidebar.markdown("---")
st.sidebar.info("""
**Mini-Projet ENSIAS**
Ingénierie des données
Semestre 3 - 2025
""")