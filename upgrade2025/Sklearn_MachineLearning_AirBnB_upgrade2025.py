#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning mit Scikit-Learn - AirBnB Preisvorhersage
Upgrade 2025 - Modernisierte Version

Autor: Andreas Traut
Datum: Dezember 2025
Version: 2025.1

Dieses Skript demonstriert einen vollständigen Machine-Learning-Workflow:
1. Daten laden
2. Explorative Datenanalyse
3. Datenvorverarbeitung mit Pipelines
4. Modelltraining und -evaluation
5. Hyperparameter-Optimierung
6. Modellpersistenz

Anforderungen:
    - Python >= 3.10
    - scikit-learn >= 1.2
    - pandas >= 2.0
    - numpy >= 1.24
    
    Siehe requirements.txt für alle Abhängigkeiten.

Daten:
    AirBnB listings.csv von http://insideairbnb.com/get-the-data.html
    Speichern unter: datasets/AirBnB/listings.csv
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Tuple, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from scipy import stats
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Warnings filtern
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# Reproduzierbarkeit
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Feature-Namen für Extra-Features
REVIEWS_FEATURE = "number_of_reviews"
REVIEWS_PER_MONTH_FEATURE = "reviews_per_month"

# Pfade
PROJECT_ROOT = Path(".")
DATASET_NAME = "AirBnB"
IMAGES_PATH = PROJECT_ROOT / "media"
DATASET_PATH = PROJECT_ROOT / "datasets" / "AirBnB"

# Sicherstellen, dass Verzeichnisse existieren
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def save_figure(
    fig_id: str,
    prefix: str = DATASET_NAME,
    tight_layout: bool = True,
    fig_extension: str = "png",
    resolution: int = 300
) -> None:
    """
    Speichert die aktuelle Matplotlib-Figur.
    
    Args:
        fig_id: Identifier für die Figur
        prefix: Präfix für Dateinamen
        tight_layout: Ob tight_layout verwendet werden soll
        fig_extension: Dateiendung (png, jpg, pdf, etc.)
        resolution: DPI für die gespeicherte Figur
    """
    path = IMAGES_PATH / f"{prefix}_{fig_id}.{fig_extension}"
    logger.info(f"Speichere Figur: {path}")
    
    try:
        if tight_layout:
            plt.tight_layout()
        
        plt.savefig(path, format=fig_extension, dpi=resolution)
        logger.info(f"Figur erfolgreich gespeichert: {path}")
    except (IOError, OSError) as e:
        logger.error(f"Fehler beim Speichern der Figur {path}: {e}")
    except Exception as e:
        logger.error(f"Unerwarteter Fehler beim Speichern der Figur: {e}")


def load_data(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    """
    Lädt AirBnB-Datensatz aus CSV-Datei.
    
    Args:
        dataset_path: Pfad zum Dataset-Verzeichnis
        
    Returns:
        DataFrame mit den geladenen Daten
        
    Raises:
        FileNotFoundError: Wenn die Datei nicht gefunden wird
    """
    csv_path = dataset_path / "listings.csv"
    
    if not csv_path.exists():
        error_msg = (
            f"Dataset nicht gefunden: {csv_path}\n\n"
            f"Bitte laden Sie die Daten von http://insideairbnb.com/get-the-data.html herunter\n"
            f"und speichern Sie die listings.csv unter: {dataset_path}"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Lade Daten von: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
    
    return df


def create_price_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erstellt Preiskategorien für stratifiziertes Sampling.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame mit hinzugefügter 'price_cat' Spalte
    """
    df = df.copy()
    df["price_cat"] = pd.cut(
        df["price"],
        bins=[-1, 50, 100, 200, 400, np.inf],
        labels=[50, 100, 200, 400, 500]
    )
    return df


def split_stratified(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Erstellt stratifizierten Train-Test-Split basierend auf Preiskategorien.
    
    Args:
        df: Input DataFrame mit 'price_cat' Spalte
        test_size: Anteil der Test-Daten
        random_state: Seed für Reproduzierbarkeit
        
    Returns:
        Tuple aus (train_set, test_set)
    """
    logger.info(f"Erstelle stratifizierten Split (test_size={test_size})")
    
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )
    
    for train_idx, test_idx in splitter.split(df, df["price_cat"]):
        train_set = df.loc[train_idx].copy()
        test_set = df.loc[test_idx].copy()
    
    # Entferne temporäre Kategoriespalte
    for dataset in (train_set, test_set):
        dataset.drop("price_cat", axis=1, inplace=True)
    
    logger.info(f"Train-Set: {len(train_set)} Zeilen, Test-Set: {len(test_set)} Zeilen")
    
    return train_set, test_set


def explore_data(df: pd.DataFrame) -> None:
    """
    Führt explorative Datenanalyse durch und erstellt Visualisierungen.
    
    Args:
        df: Input DataFrame
    """
    logger.info("Starte explorative Datenanalyse")
    
    # Scatter-Plot: Longitude vs Latitude mit Preis als Farbe
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        df["longitude"],
        df["latitude"],
        alpha=0.4,
        s=df["price"] / 100,
        c=df["price"],
        cmap="jet",
        label="price"
    )
    plt.colorbar(scatter, label="Price")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("AirBnB Listings - Geografische Verteilung und Preise")
    plt.legend()
    save_figure("prices_scatterplot")
    plt.close()
    
    # Korrelationsmatrix
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    price_corr = corr_matrix["price"].sort_values(ascending=False)
    logger.info(f"Korrelationen mit Preis:\n{price_corr}")
    
    # Scatter-Matrix für wichtige Features
    attributes = [
        "price", "number_of_reviews", "availability_365",
        "reviews_per_month", "minimum_nights"
    ]
    # Nur Features verwenden, die existieren
    available_attrs = [attr for attr in attributes if attr in df.columns]
    
    fig, axes = plt.subplots(
        len(available_attrs), len(available_attrs),
        figsize=(12, 12)
    )
    scatter_matrix(df[available_attrs], ax=axes, figsize=(12, 12))
    plt.suptitle("Scatter Matrix - Wichtige Features")
    save_figure("scatter_matrix_plot")
    plt.close()


def add_extra_features(X: np.ndarray, feature_names: list) -> np.ndarray:
    """
    Fügt berechnete Features hinzu.
    
    Args:
        X: Feature-Matrix
        feature_names: Liste der Feature-Namen
        
    Returns:
        Erweiterte Feature-Matrix
    """
    # Finde Indizes der benötigten Features
    try:
        reviews_idx = feature_names.index(REVIEWS_FEATURE)
        reviews_per_month_idx = feature_names.index(REVIEWS_PER_MONTH_FEATURE)
        
        # Berechne neues Feature
        reviews_product = X[:, reviews_idx] * X[:, reviews_per_month_idx]
        
        # Füge neues Feature hinzu
        return np.c_[X, reviews_product]
    except (ValueError, IndexError):
        logger.warning("Konnte Extra-Features nicht erstellen")
        return X


def build_preprocessing_pipeline(
    numeric_features: list,
    categorical_features: list
) -> ColumnTransformer:
    """
    Erstellt Preprocessing-Pipeline mit ColumnTransformer.
    
    Args:
        numeric_features: Liste numerischer Feature-Namen
        categorical_features: Liste kategorischer Feature-Namen
        
    Returns:
        Konfigurierte ColumnTransformer Pipeline
    """
    logger.info("Erstelle Preprocessing-Pipeline")
    
    # Numerische Pipeline
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler())
    ])
    
    # Kategorische Pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Kombiniere beide Pipelines
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])
    
    return preprocessor


def train_and_evaluate_models(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series
) -> dict:
    """
    Trainiert und evaluiert verschiedene Modelle.
    
    Args:
        X_train: Training Features
        y_train: Training Labels
        X_test: Test Features
        y_test: Test Labels
        
    Returns:
        Dictionary mit trainierten Modellen und Scores
    """
    results = {}
    
    # 1. Linear Regression
    logger.info("Trainiere Linear Regression...")
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    
    lin_pred_train = lin_reg.predict(X_train)
    lin_rmse_train = np.sqrt(mean_squared_error(y_train, lin_pred_train))
    
    lin_pred_test = lin_reg.predict(X_test)
    lin_rmse_test = np.sqrt(mean_squared_error(y_test, lin_pred_test))
    
    logger.info(f"Linear Regression - Train RMSE: {lin_rmse_train:.2f}, Test RMSE: {lin_rmse_test:.2f}")
    results['linear_regression'] = {
        'model': lin_reg,
        'train_rmse': lin_rmse_train,
        'test_rmse': lin_rmse_test
    }
    
    # 2. Decision Tree
    logger.info("Trainiere Decision Tree...")
    tree_reg = DecisionTreeRegressor(random_state=RANDOM_STATE)
    tree_reg.fit(X_train, y_train)
    
    tree_pred_train = tree_reg.predict(X_train)
    tree_rmse_train = np.sqrt(mean_squared_error(y_train, tree_pred_train))
    
    tree_pred_test = tree_reg.predict(X_test)
    tree_rmse_test = np.sqrt(mean_squared_error(y_test, tree_pred_test))
    
    logger.info(f"Decision Tree - Train RMSE: {tree_rmse_train:.2f}, Test RMSE: {tree_rmse_test:.2f}")
    results['decision_tree'] = {
        'model': tree_reg,
        'train_rmse': tree_rmse_train,
        'test_rmse': tree_rmse_test
    }
    
    # 3. Random Forest
    logger.info("Trainiere Random Forest...")
    forest_reg = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    forest_reg.fit(X_train, y_train)
    
    forest_pred_train = forest_reg.predict(X_train)
    forest_rmse_train = np.sqrt(mean_squared_error(y_train, forest_pred_train))
    
    forest_pred_test = forest_reg.predict(X_test)
    forest_rmse_test = np.sqrt(mean_squared_error(y_test, forest_pred_test))
    
    logger.info(f"Random Forest - Train RMSE: {forest_rmse_train:.2f}, Test RMSE: {forest_rmse_test:.2f}")
    results['random_forest'] = {
        'model': forest_reg,
        'train_rmse': forest_rmse_train,
        'test_rmse': forest_rmse_test
    }
    
    # Cross-Validation für Random Forest
    logger.info("Führe Cross-Validation für Random Forest durch...")
    cv_scores = cross_val_score(
        forest_reg, X_train, y_train,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    cv_rmse_scores = np.sqrt(-cv_scores)
    logger.info(f"Random Forest CV RMSE: {cv_rmse_scores.mean():.2f} (+/- {cv_rmse_scores.std():.2f})")
    results['random_forest']['cv_rmse_mean'] = cv_rmse_scores.mean()
    results['random_forest']['cv_rmse_std'] = cv_rmse_scores.std()
    
    return results


def optimize_model(
    X_train: np.ndarray,
    y_train: pd.Series,
    n_iter: int = 10
) -> RandomForestRegressor:
    """
    Optimiert Random Forest Hyperparameter mit RandomizedSearchCV.
    
    Args:
        X_train: Training Features
        y_train: Training Labels
        n_iter: Anzahl der Iterationen für RandomizedSearch
        
    Returns:
        Optimiertes Modell
    """
    logger.info("Starte Hyperparameter-Optimierung...")
    
    param_distributions = {
        'n_estimators': randint(50, 200),
        'max_features': randint(2, 10),
        'max_depth': randint(10, 50),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10)
    }
    
    forest_reg = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    logger.info(f"Beste Parameter: {random_search.best_params_}")
    logger.info(f"Bester CV Score (RMSE): {np.sqrt(-random_search.best_score_):.2f}")
    
    return random_search.best_estimator_


def save_model(model, filename: str = "model.pkl") -> None:
    """
    Speichert trainiertes Modell mit joblib.
    
    Args:
        model: Zu speicherndes Modell
        filename: Dateiname für gespeichertes Modell
    """
    model_path = PROJECT_ROOT / filename
    logger.info(f"Speichere Modell: {model_path}")
    joblib.dump(model, model_path)
    logger.info(f"Modell erfolgreich gespeichert")


def main() -> None:
    """Hauptfunktion - führt kompletten ML-Workflow aus."""
    logger.info("=" * 80)
    logger.info("Machine Learning mit Scikit-Learn - AirBnB Preisvorhersage")
    logger.info("Upgrade 2025")
    logger.info("=" * 80)
    
    try:
        # 1. Daten laden
        df = load_data()
        
        # Optional: Entferne ID-Spalte falls vorhanden
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        
        # 2. Erstelle Preiskategorien für stratifizierten Split
        df_with_cat = create_price_categories(df)
        
        # 3. Train-Test-Split
        train_set, test_set = split_stratified(df_with_cat)
        
        # 4. Explorative Datenanalyse (nur auf Training-Daten)
        explore_data(train_set)
        
        # 5. Bereite Features und Labels vor
        X_train = train_set.drop("price", axis=1)
        y_train = train_set["price"].copy()
        X_test = test_set.drop("price", axis=1)
        y_test = test_set["price"].copy()
        
        # 6. Definiere Feature-Typen
        numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = ['room_type'] if 'room_type' in X_train.columns else []
        
        logger.info(f"Numerische Features ({len(numeric_features)}): {numeric_features}")
        logger.info(f"Kategorische Features ({len(categorical_features)}): {categorical_features}")
        
        # 7. Erstelle und wende Preprocessing-Pipeline an
        preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)
        
        X_train_prepared = preprocessor.fit_transform(X_train)
        X_test_prepared = preprocessor.transform(X_test)
        
        logger.info(f"Daten vorbereitet - Shape: {X_train_prepared.shape}")
        
        # 8. Trainiere und evaluiere Modelle
        results = train_and_evaluate_models(
            X_train_prepared, y_train,
            X_test_prepared, y_test
        )
        
        # 9. Optimiere bestes Modell (Random Forest)
        best_model = optimize_model(X_train_prepared, y_train, n_iter=10)
        
        # 10. Finale Evaluation
        final_predictions = best_model.predict(X_test_prepared)
        final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
        logger.info(f"Finales optimiertes Modell - Test RMSE: {final_rmse:.2f}")
        
        # 11. Speichere Modell
        save_model(best_model, "airbnb_price_model.pkl")
        
        # 12. Feature Importance
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            # Erstelle Feature-Namen (vereinfacht)
            feature_names_output = numeric_features + categorical_features
            
            # Zeige Top 10 wichtigste Features
            indices = np.argsort(importances)[::-1][:10]
            logger.info("Top 10 wichtigste Features:")
            for i, idx in enumerate(indices):
                if idx < len(feature_names_output):
                    logger.info(f"  {i+1}. {feature_names_output[idx]}: {importances[idx]:.4f}")
        
        logger.info("=" * 80)
        logger.info("Workflow erfolgreich abgeschlossen!")
        logger.info("=" * 80)
        
    except FileNotFoundError as e:
        logger.error(f"Fehler: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unerwarteter Fehler: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
