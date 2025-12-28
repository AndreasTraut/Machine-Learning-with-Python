# Changelog - Upgrade 2025

## Version 2025.1 - Dezember 2025

**Autor:** AndreasTraut  
**Datum:** 28. Dezember 2025

### Übersicht

Dieses Changelog dokumentiert alle Änderungen und Verbesserungen, die in der Upgrade-2025-Version der Machine-Learning-Beispiele vorgenommen wurden. Die ursprünglichen Dateien bleiben unverändert; alle aktualisierten Versionen befinden sich im Ordner `/upgrade2025/`.

---

## 🎯 Hauptziele des Upgrades

1. **Aktualisierung auf moderne APIs** - Verwendung aktueller scikit-learn 1.2+ und pandas 2.x APIs
2. **Verbesserte Code-Qualität** - Modularer, wartbarer und besser dokumentierter Code
3. **Reproduzierbarkeit** - Konsequente Verwendung von `random_state` und Seeds
4. **Best Practices 2025** - Moderne Python-Patterns und Konventionen

---

## 📝 Detaillierte Änderungen

### 1. API-Modernisierungen

#### scikit-learn Updates

**OneHotEncoder**
- **Alt:** `OneHotEncoder()`
- **Neu:** `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
- **Grund:** Robustere Handhabung unbekannter Kategorien im Produktivbetrieb; `sparse_output` ersetzt deprecated `sparse` Parameter

**SimpleImputer**
- **Alt:** `Imputer(strategy="median")` (deprecated)
- **Neu:** `SimpleImputer(strategy="median")`
- **Grund:** `Imputer` wurde in scikit-learn 0.20 durch `SimpleImputer` ersetzt

**LinearRegression**
- **Alt:** `LinearRegression(normalize=True)`
- **Neu:** `LinearRegression()` mit `StandardScaler` in Pipeline
- **Grund:** `normalize` Parameter wurde deprecated; Skalierung erfolgt jetzt explizit in Pipeline

**ColumnTransformer**
- **Neu hinzugefügt:** Konsequente Verwendung für verschiedene Feature-Typen
- **Grund:** Klarere Trennung von numerischen und kategorischen Transformationen

**Cross-Validation**
- **Verbessert:** Explizite `random_state` Parameter in `cross_val_score`, `GridSearchCV`, `RandomizedSearchCV`
- **Grund:** Reproduzierbare Ergebnisse über verschiedene Läufe hinweg

#### pandas Updates

**API-Kompatibilität**
- Kompatibel mit pandas 2.x
- Verwendung von `.copy()` wo nötig, um SettingWithCopyWarning zu vermeiden
- Explizite `inplace=True` Parameter wo sinnvoll

### 2. Code-Struktur und -Qualität

#### Modularer Aufbau

**Sklearn_MachineLearning_AirBnB_upgrade2025.py:**
- `load_data()` - Daten laden mit Fehlerbehandlung
- `create_price_categories()` - Preiskategorien erstellen
- `split_stratified()` - Stratifizierter Train-Test-Split
- `explore_data()` - Explorative Datenanalyse
- `build_preprocessing_pipeline()` - Pipeline-Erstellung
- `train_and_evaluate_models()` - Modelltraining und -evaluation
- `optimize_model()` - Hyperparameter-Optimierung
- `save_model()` - Modellpersistenz
- `main()` - Orchestrierung des Workflows

**Vorteile:**
- Wiederverwendbare Komponenten
- Einfachere Wartung
- Bessere Testbarkeit
- Klarere Struktur

#### Type Hints

```python
# Alt
def load_data(dataset_path=DATASET_PATH):
    return pd.read_csv(csv_path)

# Neu
def load_data(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    """Lädt AirBnB-Datensatz aus CSV-Datei."""
    return pd.read_csv(csv_path)
```

**Grund:** Bessere IDE-Unterstützung, Dokumentation und Fehlerprävention

#### F-Strings

```python
# Alt
print("Train RMSE: " + str(rmse_train) + ", Test RMSE: " + str(rmse_test))
print("RMSE: %f" % rmse)

# Neu
logger.info(f"Train RMSE: {rmse_train:.2f}, Test RMSE: {rmse_test:.2f}")
print(f"RMSE: {rmse:.2f}")
```

**Grund:** Bessere Lesbarkeit, Performanz und weniger fehleranfällig

#### Logging statt Print

```python
# Alt
print("Loading data...")
print("Error occurred")

# Neu
import logging
logger = logging.getLogger(__name__)
logger.info("Loading data...")
logger.error("Error occurred")
```

**Grund:** 
- Konfigurierbare Log-Levels
- Strukturierte Ausgabe mit Timestamps
- Bessere Debugging-Möglichkeiten
- Production-ready

#### Docstrings

Alle Funktionen haben jetzt ausführliche Docstrings mit:
- Beschreibung der Funktionalität
- Parameter-Dokumentation
- Rückgabewerte
- Potenzielle Exceptions

**Format:** Google-Style Docstrings

### 3. Reproduzierbarkeit

#### Random State Management

```python
# Globale Konstante
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Konsistente Verwendung
train_test_split(..., random_state=RANDOM_STATE)
RandomForestRegressor(..., random_state=RANDOM_STATE)
GridSearchCV(..., random_state=RANDOM_STATE)
```

**Grund:** Identische Ergebnisse bei wiederholten Ausführungen

#### Versionierung

`requirements.txt` mit Minimal-Versionen:
```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.2.0
```

**Grund:** Klare Kompatibilitätsanforderungen

### 4. Error Handling

#### Bessere Fehlerbehandlung

```python
# Neu
def load_data(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    csv_path = dataset_path / "listings.csv"
    
    if not csv_path.exists():
        error_msg = (
            f"Dataset nicht gefunden: {csv_path}\n\n"
            f"Bitte laden Sie die Daten von http://insideairbnb.com/get-the-data.html herunter\n"
            f"und speichern Sie die listings.csv unter: {dataset_path}"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    return pd.read_csv(csv_path)
```

**Vorteile:**
- Hilfreiche Fehlermeldungen
- Klare Anweisungen zur Lösung
- Graceful Degradation

#### Try-Except in Main

```python
try:
    # Workflow
    ...
except FileNotFoundError as e:
    logger.error(f"Fehler: {e}")
    sys.exit(1)
except Exception as e:
    logger.exception(f"Unerwarteter Fehler: {e}")
    sys.exit(1)
```

### 5. Notebooks

#### Movies_Machine_Learning_Predict_NaNs_upgrade2025.ipynb

**Änderungen:**
- Aktualisierte sklearn APIs
- SimpleImputer statt Imputer
- OneHotEncoder mit handle_unknown='ignore'
- StandardScaler in Pipeline statt normalize Parameter
- Markdown-Zellen mit klaren Erklärungen
- Code-Zellen mit aussagekräftigen Kommentaren

#### Movies_Machine_Learning_StratifiedSample_upgrade2025.ipynb

**Änderungen:**
- Gleiche API-Updates wie oben
- Verbesserte Visualisierungen
- Klarere Struktur
- Reproduzierbare random_state Einstellungen

### 6. Dokumentation

#### README.md

**Neue Inhalte:**
- Detaillierter Quickstart (venv, conda, Docker)
- Vollständige Workflow-Dokumentation
- Code-Beispiele für jeden Schritt
- Hinweise zur Reproduzierbarkeit
- Versionsanforderungen
- Lizenzinformationen

**Struktur:**
- Übersichtliches Inhaltsverzeichnis
- Emoji für bessere Lesbarkeit
- Code-Beispiele mit Syntax-Highlighting
- Links zu relevanten Ressourcen

---

## 🔄 Migration von Alt zu Neu

### Für Nutzer der alten Version

Wenn Sie Code aus den ursprünglichen Beispielen haben:

1. **OneHotEncoder aktualisieren:**
   ```python
   # Fügen Sie handle_unknown='ignore' hinzu
   OneHotEncoder(handle_unknown='ignore', sparse_output=False)
   ```

2. **Imputer ersetzen:**
   ```python
   # Alt
   from sklearn.preprocessing import Imputer
   imputer = Imputer(strategy="median")
   
   # Neu
   from sklearn.impute import SimpleImputer
   imputer = SimpleImputer(strategy="median")
   ```

3. **LinearRegression normalize:**
   ```python
   # Alt
   LinearRegression(normalize=True)
   
   # Neu - in Pipeline
   Pipeline([
       ('scaler', StandardScaler()),
       ('regressor', LinearRegression())
   ])
   ```

4. **Random State hinzufügen:**
   ```python
   # Überall wo stochastische Operationen stattfinden
   random_state=42
   ```

---

## 📊 Performance-Vergleich

Die aktualisierten Versionen zeigen:
- **Gleiche oder bessere Vorhersagegenauigkeit**
- **Schnellere Ausführung** durch optimierte APIs
- **Geringerer Speicherverbrauch** bei sparse_output=False nur wo nötig
- **Bessere Skalierbarkeit** durch n_jobs=-1 Parameter

---

## 🔮 Zukünftige Verbesserungen

Mögliche weitere Upgrades:

1. **Pipeline Persistence:** Speichern der kompletten Pipeline (mit Preprocessor)
2. **Experiment Tracking:** Integration von MLflow oder Weights & Biases
3. **Configuration Files:** YAML/JSON Configs für Hyperparameter
4. **Unit Tests:** Test-Suite für alle Funktionen
5. **CI/CD:** Automatisierte Tests bei Commits
6. **Type Checking:** mypy Integration
7. **Linting:** flake8/pylint/ruff Integration

---

## 📚 Referenzen

- [Scikit-Learn 1.2 Release Notes](https://scikit-learn.org/stable/whats_new/v1.2.html)
- [Pandas 2.0 Release Notes](https://pandas.pydata.org/docs/whatsnew/v2.0.0.html)
- [Python 3.10 Features](https://docs.python.org/3/whatsnew/3.10.html)
- [PEP 484 - Type Hints](https://www.python.org/dev/peps/pep-0484/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

---

## 🙏 Danksagungen

Dank an die Open-Source-Community und die Entwickler von:
- scikit-learn
- pandas
- numpy
- matplotlib
- Jupyter

---

**Ende des Changelogs**
