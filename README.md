# 🌱 BioVision Analytics Hub

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **An interactive machine learning dashboard for above-ground biomass prediction using multi-source satellite data (GEDI, Sentinel-1, Sentinel-2)**

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Data Pipeline](#-data-pipeline)
- [Models](#-models)
- [Dashboard Features](#-dashboard-features)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

---

## 🌟 Overview

**BioVision Analytics Hub** is an end-to-end machine learning platform for predicting above-ground biomass (AGB) in forest ecosystems using satellite remote sensing data. This project integrates data from NASA's GEDI mission, ESA's Sentinel-1 (SAR), and Sentinel-2 (optical) satellites to create robust predictive models with comprehensive spatial analysis capabilities.

### 🎯 Key Objectives

- **Data Integration**: Merge multi-source satellite data (GEDI L4A, Sentinel-1, Sentinel-2, DEM, Land Cover)
- **Feature Engineering**: Extract and engineer spectral indices, terrain metrics, and statistical features
- **Model Development**: Train and optimize ensemble ML models (Random Forest, LightGBM, XGBoost, SVR)
- **Interactive Dashboard**: Provide real-time model insights, diagnostics, and spatial analysis
- **Spatial Analysis**: Identify biomass hotspots, clustering patterns, and spatial autocorrelation

---

## ✨ Features

### 🔬 **Advanced ML Pipeline**
- **4 Ensemble Models**: Random Forest, LightGBM, XGBoost, Support Vector Regression
- **Automated Hyperparameter Tuning**: RandomizedSearchCV with cross-validation
- **Feature Engineering**: Vegetation indices (NDVI, NDMI, NDWI), spectral ratios, polynomial features
- **Feature Selection**: Variance Threshold, F-test, Mutual Information, RFE, Lasso

### 📊 **Model Diagnostics**
- **Learning Curves**: Training/validation performance tracking
- **Residual Analysis**: Normality tests, homoscedasticity assessment
- **Bias-Variance Tradeoff**: Cross-validation stability analysis
- **Feature Importance**: Traditional and permutation-based importance

### 🗺️ **Spatial Analysis**
- **Geographic Clustering**: K-Means spatial pattern detection
- **Spatial Autocorrelation**: Moran's I and Geary's C statistics
- **Hotspot Analysis**: Local Outlier Factor (LOF) for anomaly detection
- **Spatial Interpolation**: IDW and Nearest Neighbor interpolation

### 🎨 **Interactive Dashboard**
- **Real-time Model Training**: Train models with custom hyperparameters
- **Interactive Visualizations**: Plotly-based charts with zoom, pan, hover
- **Spatial Maps**: Interactive mapbox visualizations with density layers
- **Export Functionality**: Save models, results, and figures

---

## 🖼️ Demo

### Dashboard Preview

![Dashboard Overview](assets/dashboard_overview.png)

### Model Performance Comparison

![Model Comparison](assets/model_comparison.png)

### Spatial Analysis

![Spatial Analysis](assets/spatial_analysis.png)

> **Note**: Add screenshots to `assets/` folder for visual appeal

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/MichaelTheAnalyst/BioVision-Analytics-Hub.git
cd BioVision-Analytics-Hub
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import streamlit; import pandas; import sklearn; print('✅ All dependencies installed!')"
```

---

## ⚡ Quick Start

### Option 1: Launch Dashboard (Recommended)

```bash
# Windows
scripts\launch_dashboard.bat

# macOS/Linux
streamlit run src/dashboard/app.py
```

The dashboard will automatically open in your default web browser at `http://localhost:8501`

### Option 2: Run Individual Scripts

```python
# Train models
python src/models/train_random_forest.py
python src/models/train_lightgbm.py

# Generate visualizations
python src/visualization/plot_feature_importance.py
```

### Option 3: Use Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

---

## 📁 Project Structure

```
BioVision-Analytics-Hub/
│
├── 📂 src/                              # Source code
│   ├── data_preprocessing/              # Data extraction & preprocessing
│   │   ├── gedi_preprocessing.py        # GEDI data processing
│   │   ├── sentinel1_extraction.py      # SAR data extraction
│   │   ├── sentinel2_extraction.py      # Optical data extraction
│   │   └── data_merger.py               # Multi-source data integration
│   │
│   ├── models/                          # Model training & evaluation
│   │   ├── train_random_forest.py       # RF model training
│   │   ├── train_lightgbm.py            # LightGBM training
│   │   ├── train_xgboost.py             # XGBoost training
│   │   ├── train_svr.py                 # SVR training
│   │   └── model_evaluation.py          # Model metrics & comparison
│   │
│   ├── visualization/                   # Visualization scripts
│   │   ├── plot_feature_importance.py   # Feature importance plots
│   │   ├── plot_spatial_analysis.py     # Spatial visualization
│   │   └── plot_model_diagnostics.py    # Learning curves, residuals
│   │
│   ├── dashboard/                       # Streamlit dashboard
│   │   ├── app.py                       # Main dashboard application
│   │   ├── core.py                      # Core model training
│   │   ├── feature_analysis.py          # Feature importance module
│   │   ├── feature_engineering.py       # Feature engineering module
│   │   └── model_diagnostics.py         # Diagnostics module
│   │
│   └── utils/                           # Utility functions
│       ├── data_loader.py               # Data loading utilities
│       ├── feature_engineering.py       # Feature creation
│       └── spatial_utils.py             # Spatial analysis utilities
│
├── 📂 data/                             # Data directory (not in Git)
│   ├── raw/                             # Raw satellite data
│   ├── interim/                         # Intermediate processed data
│   └── processed/                       # Final training data
│
├── 📂 models/                           # Trained models (not in Git)
│   ├── saved_models/                    # Serialized model files
│   └── scalers/                         # Feature scalers
│
├── 📂 outputs/                          # Generated outputs (not in Git)
│   ├── figures/                         # Plots and visualizations
│   ├── results/                         # Model results & metrics
│   └── reports/                         # Analysis reports
│
├── 📂 notebooks/                        # Jupyter notebooks
│   ├── 01_data_exploration.ipynb        # EDA
│   ├── 02_feature_engineering.ipynb     # Feature creation
│   └── 03_model_experiments.ipynb       # Model experimentation
│
├── 📂 docs/                             # Documentation
│   ├── INSTALLATION.md                  # Detailed setup guide
│   ├── USER_GUIDE.md                    # User documentation
│   ├── ARCHITECTURE.md                  # System architecture
│   └── API.md                           # API reference
│
├── 📂 config/                           # Configuration files
│   └── config.yaml                      # Hyperparameters & settings
│
├── 📂 scripts/                          # Utility scripts
│   ├── launch_dashboard.bat             # Windows launcher
│   └── setup.sh                         # Setup automation
│
├── 📂 tests/                            # Unit tests
│   ├── test_data_preprocessing.py
│   ├── test_models.py
│   └── test_utils.py
│
├── 📂 assets/                           # Media assets for README
│   ├── dashboard_overview.png
│   └── logo.png
│
├── 📄 README.md                         # This file
├── 📄 requirements.txt                  # Python dependencies
├── 📄 .gitignore                        # Git ignore rules
├── 📄 LICENSE                           # MIT License
├── 📄 CONTRIBUTING.md                   # Contribution guidelines
└── 📄 CHANGELOG.md                      # Version history
```

---

## 🔄 Data Pipeline

### 1. **Data Acquisition**
- **GEDI L4A**: Above-ground biomass density (NASA)
- **Sentinel-1**: C-band SAR imagery (ESA)
- **Sentinel-2**: Multispectral optical imagery (ESA)
- **DEM**: Digital Elevation Model (terrain analysis)
- **Land Cover**: ESA WorldCover classification

### 2. **Preprocessing**
```python
# Example: Load and preprocess data
from src.data_preprocessing import GEDIPreprocessor, SentinelExtractor

gedi = GEDIPreprocessor('data/raw/gedi_l4a.csv')
gedi_data = gedi.filter_quality().extract_roi()

sentinel = SentinelExtractor('ROI_south.kml')
s1_data = sentinel.extract_sentinel1()
s2_data = sentinel.extract_sentinel2()
```

### 3. **Feature Engineering**
- **Spectral Indices**: NDVI, NDMI, NDWI, NDCI, ChlRe, REPO, MCARI
- **Band Ratios**: NIR/Red, SWIR1/SWIR2, etc.
- **Terrain Features**: Slope, aspect, elevation
- **Statistical Features**: Mean, std, max, min per band

### 4. **Model Training**
```python
# Example: Train Random Forest model
from src.models import train_random_forest

model, metrics = train_random_forest(
    data='data/processed/training_data.csv',
    hyperparameter_tuning=True,
    n_iter=20
)
```

---

## 🤖 Models

### Model Performance Summary

| Model | RMSE | R² Score | MAE | Training Time |
|-------|------|----------|-----|---------------|
| **Random Forest** | 25.34 | 0.87 | 18.21 | 45s |
| **LightGBM** | 23.12 | 0.89 | 16.84 | 12s |
| **XGBoost** | 24.56 | 0.88 | 17.92 | 38s |
| **SVR** | 28.91 | 0.84 | 21.45 | 120s |

> **Note**: Performance metrics are example values. Actual results depend on your dataset.

### Hyperparameter Optimization

All models use **RandomizedSearchCV** with:
- **Cross-validation**: 5-fold stratified
- **Iterations**: 20 random combinations
- **Scoring**: Negative RMSE
- **Parallel processing**: n_jobs=-1

---

## 🎨 Dashboard Features

### 1. **📊 Model Performance**
- Train all 4 models with one click
- Compare RMSE, R², MAE, training time
- Radar charts and bar plots
- Prediction scatter plots

### 2. **🎯 Feature Importance**
- Traditional feature importance (tree-based)
- Permutation importance
- Feature correlation heatmaps
- Top features analysis

### 3. **🔧 Feature Engineering**
- Automated feature creation
- Multiple selection techniques
- PCA dimensionality reduction
- Feature set evaluation

### 4. **📈 Model Diagnostics**
- Learning curves (train/validation)
- Residual analysis (normality, homoscedasticity)
- Bias-variance tradeoff
- Cross-validation stability

### 5. **🗺️ Spatial Analysis**
- Geographic clustering (K-Means)
- Spatial autocorrelation (Moran's I, Geary's C)
- Hotspot detection (LOF)
- Interactive maps with Mapbox

---

## 📚 Documentation

- **[Installation Guide](docs/INSTALLATION.md)**: Detailed setup instructions
- **[User Guide](docs/USER_GUIDE.md)**: How to use the dashboard
- **[Architecture](docs/ARCHITECTURE.md)**: System design and workflow
- **[API Reference](docs/API.md)**: Function and class documentation

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/MichaelTheAnalyst/BioVision-Analytics-Hub.git
cd BioVision-Analytics-Hub
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **NASA GEDI Mission**: Global Ecosystem Dynamics Investigation
- **ESA Copernicus**: Sentinel-1 and Sentinel-2 data
- **University of Southampton**: Academic support and resources
- **Open-source community**: Scikit-learn, Streamlit, Plotly, Pandas

---

## 📧 Contact

**Masood Nazari**  
AI Engineer | Data Engineer | Data Science Enthusiast  
📍 Southampton, UK  
🔗 [LinkedIn](https://www.linkedin.com/in/masood-nazari)  
📧 [Email](mailto:michaelnazary@gmail.com)  
🐙 [GitHub](https://github.com/MichaelTheAnalyst)

---

## 📊 Project Statistics

![GitHub stars](https://img.shields.io/github/stars/MichaelTheAnalyst/BioVision-Analytics-Hub?style=social)
![GitHub forks](https://img.shields.io/github/forks/MichaelTheAnalyst/BioVision-Analytics-Hub?style=social)
![GitHub issues](https://img.shields.io/github/issues/MichaelTheAnalyst/BioVision-Analytics-Hub)
![GitHub pull requests](https://img.shields.io/github/issues-pr/MichaelTheAnalyst/BioVision-Analytics-Hub)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

Made with ❤️ by Masood Nazari

</div>
