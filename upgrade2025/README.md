# Machine Learning with Python - Upgrade 2025

**Autor:** Andreas Traut  
**Datum:** Dezember 2025  
**Version:** 2025.1

## 📋 Inhaltsverzeichnis

- [Über dieses Upgrade](#über-dieses-upgrade)
- [Ziele des Repositories](#ziele-des-repositories)
- [Quickstart](#quickstart)
- [Unterstützte Versionen](#unterstützte-versionen)
- [Was wurde aktualisiert?](#was-wurde-aktualisiert)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Dateien in diesem Ordner](#dateien-in-diesem-ordner)
- [Reproduzierbarkeit](#reproduzierbarkeit)
- [Lizenz](#lizenz)

## 🎯 Über dieses Upgrade

Dieser Ordner (`/upgrade2025/`) enthält modernisierte Versionen der Machine-Learning-Beispiele aus diesem Repository, aktualisiert nach den Best Practices von 2025. Die **ursprünglichen Dateien bleiben unverändert** – alle Aktualisierungen befinden sich ausschließlich in diesem Ordner.

## 🔍 Ziele des Repositories

Dieses Repository zeigt die Unterschiede und Gemeinsamkeiten zwischen **"Small Data"** (Scikit-Learn/Pandas) und **"Big Data"** (Spark) Ansätzen im Machine Learning. Der Fokus liegt auf:

- Praktischen, wiederverwendbaren Code-Beispielen
- Vergleich von Scikit-Learn und Apache Spark ML
- Verständnis der Unterschiede zwischen kleinen und großen Datensätzen
- Verwendung von IDEs zusätzlich zu Jupyter-Notebooks

### Small Data vs. Big Data

**Small Data (Scikit-Learn):**
- Datensätze, die in den Arbeitsspeicher passen
- Einfache, schnelle Entwicklung
- Umfangreiche Bibliotheken (pandas, scikit-learn, matplotlib)
- Ideal für Prototyping und kleinere Projekte

**Big Data (Apache Spark):**
- Verteilte Verarbeitung großer Datensätze
- Skalierbare Algorithmen
- Komplexere Infrastruktur
- Für produktive, große Anwendungen

## 🚀 Quickstart

### Lokale Installation mit venv

```bash
# Python Virtual Environment erstellen
python3 -m venv venv

# Environment aktivieren
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Dependencies installieren
pip install -r upgrade2025/requirements.txt

# Python-Skript ausführen
python upgrade2025/Sklearn_MachineLearning_AirBnB_upgrade2025.py

# Jupyter Notebook starten
jupyter lab
# Dann die Notebooks im Ordner upgrade2025/ öffnen
```

### Mit Conda

```bash
# Conda Environment erstellen
conda create -n ml-python python=3.10
conda activate ml-python

# Dependencies installieren
pip install -r upgrade2025/requirements.txt

# Oder mit conda:
conda install pandas numpy scikit-learn matplotlib seaborn jupyterlab
```

### Mit Docker

**Hinweis:** Die Beispiele in diesem `/upgrade2025/` Ordner sind für lokale Ausführung optimiert und benötigen kein Docker. 

Ein Docker-Setup für die Spark-Beispiele ("Big Data") finden Sie im Hauptverzeichnis des Repositories. Für die "Small Data" Beispiele in diesem Ordner genügt eine lokale Installation mit Python und den in `requirements.txt` aufgeführten Paketen.

## 📦 Unterstützte Versionen

- **Python:** >= 3.10
- **pandas:** >= 2.0
- **numpy:** >= 1.24
- **scikit-learn:** >= 1.2
- **matplotlib:** >= 3.5
- **seaborn:** >= 0.12
- **jupyterlab:** >= 4.0
- **joblib:** >= 1.2

Siehe `requirements.txt` für genaue Versionsangaben.

## ✨ Was wurde aktualisiert?

### API-Änderungen und Modernisierung

1. **OneHotEncoder:** `handle_unknown='ignore'` Parameter hinzugefügt für robustere Verarbeitung unbekannter Kategorien
2. **SimpleImputer:** Moderne API statt veralteter `Imputer`
3. **ColumnTransformer:** Konsistente Verwendung für verschiedene Feature-Typen
4. **FunctionTransformer:** Für benutzerdefinierte Transformationen
5. **LinearRegression:** Veralteter `normalize` Parameter entfernt (jetzt `StandardScaler` in Pipeline)

### Code-Qualität

1. **F-Strings:** Statt `%s` oder `.format()` für bessere Lesbarkeit
2. **Type Hints:** Optional hinzugefügt für bessere Code-Dokumentation
3. **Logging:** `logging` Modul statt `print()` Statements
4. **Modulare Struktur:** Funktionen statt langer Skripte
5. **Docstrings:** Klare Dokumentation aller Funktionen

### Reproduzierbarkeit

1. **random_state:** Konsequent in allen stochastischen Operationen gesetzt
2. **Seeds:** Dokumentiert und konsistent verwendet
3. **Versionierung:** Klare Angaben zu Package-Versionen

### Error Handling

1. Bessere Fehlerbehandlung beim Laden von Dateien
2. Klare Fehlermeldungen mit Hinweisen zur Lösung
3. Validierung von Eingabedaten

## 🔄 Machine Learning Workflow

### 1. Daten einlesen

```python
import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """Lädt CSV-Datei und gibt DataFrame zurück."""
    return pd.read_csv(filepath)
```

### 2. Explorative Datenanalyse (EDA)

- Datenstruktur verstehen (`info()`, `describe()`)
- Visualisierungen erstellen (Histogramme, Scatter-Plots)
- Korrelationen analysieren
- Fehlende Werte identifizieren

### 3. Datenvorverarbeitung

**Fehlende Werte behandeln:**
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
# oder strategy="mean", "most_frequent", "constant"
```

**Kategorische Features encodieren:**
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
```

**Numerische Features skalieren:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

### 4. Pipeline aufbauen

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Numerische Pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

# Kategorische Pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Kombinierte Pipeline
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numeric_features),
    ('cat', cat_pipeline, categorical_features)
])
```

### 5. Modellwahl und Training

**Regression:**
```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

**Klassifikation:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### 6. Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X_train, y_train,
    cv=5,  # 5-fold cross-validation
    scoring='neg_mean_squared_error'
)
rmse_scores = np.sqrt(-scores)
```

### 7. Hyperparameter-Optimierung

**Grid Search:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    model, param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

**Randomized Search:**
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distributions = {
    'n_estimators': randint(10, 200),
    'max_depth': randint(5, 30)
}

random_search = RandomizedSearchCV(
    model, param_distributions,
    n_iter=20, cv=5,
    random_state=42
)
```

### 8. Evaluation

```python
from sklearn.metrics import mean_squared_error, r2_score

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")
```

### 9. Modell Persistenz

```python
import joblib

# Modell speichern
joblib.dump(best_model, 'model.pkl')

# Modell laden
loaded_model = joblib.load('model.pkl')
```

## 📁 Dateien in diesem Ordner

### Python-Skripte

**`Sklearn_MachineLearning_AirBnB_upgrade2025.py`**
- Modernisierte Version des AirBnB-Beispiels
- Preisvorhersage für AirBnB-Listings
- Demonstriert kompletten ML-Workflow
- Modular strukturiert mit Funktionen
- Verwendet aktuelle scikit-learn APIs

### Jupyter Notebooks

**`Movies_Machine_Learning_Predict_NaNs_upgrade2025.ipynb`**
- Vorhersage fehlender Revenue-Werte in Movies-Dataset
- Zeigt Umgang mit Missing Data
- DecisionTree und RandomForest Regressoren

**`Movies_Machine_Learning_StratifiedSample_upgrade2025.ipynb`**
- Stratifiziertes Sampling für ausgewogene Train/Test-Splits
- Vergleich verschiedener Sampling-Strategien
- Pipeline-Erstellung und Cross-Validation

### Konfiguration

**`requirements.txt`**
- Minimale empfohlene Versionen aller Dependencies
- Für reproduzierbare Umgebungen

**`CHANGELOG.md`**
- Detaillierte Liste aller Änderungen
- Begründung für Updates

## 🔁 Reproduzierbarkeit

Für reproduzierbare Ergebnisse:

1. **random_state setzen:**
   ```python
   # In train_test_split
   train_test_split(X, y, test_size=0.2, random_state=42)
   
   # In Modellen
   RandomForestRegressor(n_estimators=100, random_state=42)
   
   # In Cross-Validation
   cross_val_score(model, X, y, cv=5, random_state=42)
   
   # In Grid/Randomized Search
   GridSearchCV(model, param_grid, cv=5, random_state=42)
   ```

2. **Numpy seed setzen:**
   ```python
   import numpy as np
   np.random.seed(42)
   ```

3. **Exakte Versionen verwenden:**
   ```bash
   pip freeze > requirements-exact.txt
   ```

## 📊 Datenquellen

### AirBnB Dataset
- **Quelle:** [Inside Airbnb](http://insideairbnb.com/get-the-data.html)
- **Lizenz:** Creative Commons CC0 1.0 Universal "Public Domain Dedication"
- **Pfad:** `datasets/AirBnB/listings.csv`

### Movies Dataset
- **Quelle:** [Kaggle - IMDB Movies](https://www.kaggle.com/datasets)
- **Pfad:** `datasets/movies/`

**Hinweis:** Die Datasets müssen separat heruntergeladen werden. Die Skripte geben klare Anweisungen, falls Daten fehlen.

## 📚 Weitere Ressourcen

### Dokumentation
- [Scikit-Learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)

### Tutorials
- [Scikit-Learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)

## 📝 Lizenz

Dieses Werk ist lizenziert unter der **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License**.

Um eine Kopie dieser Lizenz zu sehen, besuchen Sie:
http://creativecommons.org/licenses/by-nc-sa/4.0/

## 🤝 Beiträge

Dieses Upgrade wurde erstellt, um die Code-Beispiele auf aktuelle Best Practices zu bringen. Für Fragen oder Verbesserungsvorschläge öffnen Sie bitte ein Issue im GitHub-Repository.

---

**Ursprüngliches Repository:** [AndreasTraut/Machine-Learning-with-Python](https://github.com/AndreasTraut/Machine-Learning-with-Python)
