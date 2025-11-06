import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, shapiro
import time
from datetime import datetime

# Spatial Analysis imports
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

# Set page config
st.set_page_config(
    page_title="BioVision Analytics Hub",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main Header Styling */
    .main-header {
        font-size: 3.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(135deg, #2E8B57, #3CB371);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Subtitle Styling */
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
        opacity: 0.8;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #2E8B57;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s ease-in-out;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #2E8B57, #3CB371);
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        background: linear-gradient(135deg, #3CB371, #2E8B57);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #f0f8f0, #e8f5e8);
        border-radius: 10px;
        border: 2px solid #2E8B57;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 2rem;
        color: #2E8B57;
        margin-bottom: 1.5rem;
        font-weight: 600;
        border-bottom: 3px solid #2E8B57;
        padding-bottom: 0.5rem;
    }
    
    /* Success Messages */
    .success-box {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Warning Messages */
    .warning-box {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Info Messages */
    .info-box {
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
        border: 2px solid #17a2b8;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Chart Containers */
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Footer Styling */
    .footer {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Signature Styling */
    .signature {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        border: 2px solid #2E8B57;
    }
    
    .signature h3 {
        color: #2E8B57;
        margin-bottom: 0.5rem;
        font-weight: 700;
        font-size: 1.5rem;
    }
    
    .signature p {
        color: #666;
        margin: 0.5rem 0;
        font-size: 1rem;
    }
    
    .signature a {
        color: #0077B5;
        text-decoration: none;
        font-weight: 600;
        transition: color 0.3s ease;
    }
    
    .signature a:hover {
        color: #0056b3;
        text-decoration: underline;
    }
    
    /* Progress Bar Styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2E8B57, #3CB371);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 10px;
        font-weight: 600;
        color: #2E8B57;
    }
    
    /* Dataframe Styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class StreamlitAGBDashboard:
    def __init__(self):
        self.data = None
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    @st.cache_data
    def load_data(_self):
        """Load and cache data"""
        try:
            data = pd.read_csv('FEI data/opt_means_cleaned.csv')
            return data
        except FileNotFoundError:
            st.error("Data file not found. Please ensure 'FEI data/opt_means_cleaned.csv' exists.")
            return None
    
    @st.cache_data
    def preprocess_data(_self, data):
        """Preprocess data for modeling"""
        X = data.drop(columns=['AGB_2017'])
        y = data['AGB_2017']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler
    
    def train_models(self, X_train_scaled, y_train, X_test_scaled, y_test):
        """Train all models with hyperparameter tuning"""
        # Hyperparameter distributions for RandomizedSearchCV
        rf_params = {
            'n_estimators': [100, 200, 500, 1000],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        
        lgbm_params = {
            'n_estimators': [100, 200, 500, 1000],
            'num_leaves': [31, 50, 100, 200],
            'max_depth': [5, 7, 10, 15, -1],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        xgb_params = {
            'n_estimators': [100, 200, 500, 1000],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
        
        models_config = {
            'Random Forest': (RandomForestRegressor(random_state=42), rf_params),
            'LightGBM': (LGBMRegressor(random_state=42, verbose=-1), lgbm_params),
            'XGBoost': (XGBRegressor(random_state=42, verbosity=0), xgb_params),
            'SVR': (SVR(), {})  # SVR without tuning for now
        }
        
        results = {}
        feature_importance = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (model_name, (base_model, params)) in enumerate(models_config.items()):
            status_text.text(f'Training {model_name}...')
            start_time = time.time()
            
            if params:  # Use RandomizedSearchCV for models with parameters
                model = RandomizedSearchCV(
                    base_model, params, n_iter=20, cv=3, 
                    random_state=42, n_jobs=-1, verbose=0
                )
                model.fit(X_train_scaled, y_train)
                best_model = model.best_estimator_
            else:  # Use base model for SVR
                best_model = base_model
                best_model.fit(X_train_scaled, y_train)
            
            training_time = time.time() - start_time
            
            # Make predictions
            y_pred = best_model.predict(X_test_scaled)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results[model_name] = {
                'RMSE': rmse,
                'R²': r2,
                'MAE': mae,
                'Training Time (s)': training_time,
                'predictions': y_pred,
                'model': best_model
            }
            
            # Get feature importance if available
            if hasattr(best_model, 'feature_importances_'):
                feature_importance[model_name] = best_model.feature_importances_
            
            progress_bar.progress((i + 1) / len(models_config))
        
        status_text.text('Training completed!')
        progress_bar.empty()
        status_text.empty()
        
        return results, feature_importance
    
    def create_performance_comparison(self, results):
        """Create interactive performance comparison"""
        # Prepare data
        models = list(results.keys())
        rmse_values = [results[model]['RMSE'] for model in models]
        r2_values = [results[model]['R²'] for model in models]
        mae_values = [results[model]['MAE'] for model in models]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('RMSE Comparison', 'R² Comparison', 'MAE Comparison', 'Performance Radar'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatterpolar"}]]
        )
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # RMSE bar chart
        fig.add_trace(
            go.Bar(x=models, y=rmse_values, name='RMSE', marker_color=colors),
            row=1, col=1
        )
        
        # R² bar chart
        fig.add_trace(
            go.Bar(x=models, y=r2_values, name='R²', marker_color=colors),
            row=1, col=2
        )
        
        # MAE bar chart
        fig.add_trace(
            go.Bar(x=models, y=mae_values, name='MAE', marker_color=colors),
            row=2, col=1
        )
        
        # Radar chart for best model
        best_model_idx = np.argmin(rmse_values)
        best_model = models[best_model_idx]
        
        # Normalize metrics for radar chart
        rmse_norm = 1 - (rmse_values[best_model_idx] - min(rmse_values)) / (max(rmse_values) - min(rmse_values)) if max(rmse_values) != min(rmse_values) else 1
        r2_norm = r2_values[best_model_idx]
        mae_norm = 1 - (mae_values[best_model_idx] - min(mae_values)) / (max(mae_values) - min(mae_values)) if max(mae_values) != min(mae_values) else 1
        
        fig.add_trace(
            go.Scatterpolar(
                r=[rmse_norm, r2_norm, mae_norm],
                theta=['RMSE (normalized)', 'R²', 'MAE (normalized)'],
                fill='toself',
                name=f'{best_model} Performance',
                line_color=colors[best_model_idx]
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=700,
            showlegend=False,
            title_text="Model Performance Comparison Dashboard"
        )
        
        return fig
    
    def create_feature_importance_plot(self, feature_importance, feature_names):
        """Create interactive feature importance plot"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(feature_importance.keys()),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, (model_name, importance) in enumerate(feature_importance.items()):
            # Get top 10 features
            top_indices = np.argsort(importance)[-10:][::-1]
            top_features = [feature_names[idx] for idx in top_indices]
            top_importance = importance[top_indices]
            
            row, col = positions[i]
            fig.add_trace(
                go.Bar(
                    x=top_importance,
                    y=top_features,
                    orientation='h',
                    name=model_name,
                    marker_color=colors[i]
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            height=700,
            showlegend=False,
            title_text="Feature Importance Across Models (Top 10 Features)"
        )
        
        return fig
    
    def create_prediction_scatter(self, results, y_test):
        """Create prediction vs actual scatter plots"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(results.keys())
        )
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, (model_name, model_results) in enumerate(results.items()):
            y_pred = model_results['predictions']
            row, col = positions[i]
            
            # Scatter plot
            fig.add_trace(
                go.Scatter(
                    x=y_test,
                    y=y_pred,
                    mode='markers',
                    name=model_name,
                    marker=dict(color=colors[i], size=6, opacity=0.7),
                    hovertemplate=f'<b>{model_name}</b><br>Actual: %{{x:.2f}}<br>Predicted: %{{y:.2f}}<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Perfect prediction line
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Perfect Prediction',
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            height=700,
            showlegend=False,
            title_text="Actual vs Predicted Values"
        )
        
        fig.update_xaxes(title_text="Actual AGB")
        fig.update_yaxes(title_text="Predicted AGB")
        
        return fig
    
    def create_correlation_heatmap(self, data):
        """Create correlation heatmap"""
        # Calculate correlation matrix
        feature_cols = [col for col in data.columns if col != 'AGB_2017']
        corr_matrix = data[feature_cols + ['AGB_2017']].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 8},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            xaxis_title="Features",
            yaxis_title="Features",
            height=600
        )
        
        return fig
    
    def create_engineered_features(self, data):
        """Create engineered features"""
        X = data.drop(columns=['AGB_2017'])
        y = data['AGB_2017']
        
        # Vegetation indices combinations
        if 'NDVI' in X.columns and 'NDMI' in X.columns:
            X['NDVI_NDMI_ratio'] = X['NDVI'] / (X['NDMI'] + 1e-8)
            X['NDVI_NDMI_sum'] = X['NDVI'] + X['NDMI']
        
        if 'NDVI' in X.columns and 'NDWI' in X.columns:
            X['NDVI_NDWI_ratio'] = X['NDVI'] / (X['NDWI'] + 1e-8)
        
        # Spectral band ratios (if B01-B12 exist)
        band_cols = [col for col in X.columns if col.startswith('B') and len(col) <= 3]
        if len(band_cols) >= 2:
            for i in range(len(band_cols)-1):
                for j in range(i+1, len(band_cols)):
                    X[f'{band_cols[i]}_{band_cols[j]}_ratio'] = X[band_cols[i]] / (X[band_cols[j]] + 1e-8)
        
        # Statistical features
        X['mean_spectral'] = X[band_cols].mean(axis=1) if band_cols else 0
        X['std_spectral'] = X[band_cols].std(axis=1) if band_cols else 0
        X['max_spectral'] = X[band_cols].max(axis=1) if band_cols else 0
        X['min_spectral'] = X[band_cols].min(axis=1) if band_cols else 0
        
        # Polynomial features (degree 2 for top features)
        top_features = ['NDVI', 'ChlRe', 'REPO', 'NDMI', 'NDWI']
        available_top_features = [f for f in top_features if f in X.columns]
        
        if len(available_top_features) >= 2:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly_features = poly.fit_transform(X[available_top_features])
            poly_feature_names = [f'poly_{i}' for i in range(poly_features.shape[1])]
            
            for i, name in enumerate(poly_feature_names):
                X[name] = poly_features[:, i]
        
        return X, y
    
    def apply_feature_selection(self, X, y):
        """Apply various feature selection techniques"""
        selection_results = {}
        
        # Variance Threshold
        variance_selector = VarianceThreshold(threshold=0.01)
        X_var_selected = variance_selector.fit_transform(X)
        var_selected_features = X.columns[variance_selector.get_support()].tolist()
        selection_results['Variance Threshold'] = {
            'n_features': len(var_selected_features),
            'features': var_selected_features
        }
        
        # F-test
        f_selector = SelectKBest(score_func=f_regression, k=min(20, X.shape[1]))
        f_selector.fit(X, y)
        f_selected_features = X.columns[f_selector.get_support()].tolist()
        selection_results['F-test'] = {
            'n_features': len(f_selected_features),
            'features': f_selected_features
        }
        
        # Mutual Information
        mi_selector = SelectKBest(score_func=mutual_info_regression, k=min(20, X.shape[1]))
        mi_selector.fit(X, y)
        mi_selected_features = X.columns[mi_selector.get_support()].tolist()
        selection_results['Mutual Information'] = {
            'n_features': len(mi_selected_features),
            'features': mi_selected_features
        }
        
        # RFE with Random Forest
        rf_for_rfe = RandomForestRegressor(n_estimators=100, random_state=42)
        rfe = RFE(estimator=rf_for_rfe, n_features_to_select=min(20, X.shape[1]))
        rfe.fit(X, y)
        rfe_selected_features = X.columns[rfe.get_support()].tolist()
        selection_results['RFE'] = {
            'n_features': len(rfe_selected_features),
            'features': rfe_selected_features
        }
        
        # L1-based selection (Lasso)
        lasso = Lasso(alpha=0.01, random_state=42)
        lasso.fit(X, y)
        lasso_selected_features = X.columns[lasso.coef_ != 0].tolist()
        selection_results['Lasso'] = {
            'n_features': len(lasso_selected_features),
            'features': lasso_selected_features
        }
        
        return selection_results
    
    def apply_pca(self, X, n_components=0.95):
        """Apply PCA for dimensionality reduction"""
        if isinstance(n_components, float):
            pca = PCA(n_components=n_components)
        else:
            pca = PCA(n_components=min(n_components, X.shape[1]))
        
        X_pca = pca.fit_transform(X)
        
        return X_pca, pca
    
    def evaluate_feature_sets(self, X_original, X_engineered, y, X_train_scaled, y_train, X_test_scaled, y_test):
        """Evaluate different feature sets"""
        feature_sets = {
            'Original': X_original,
            'Engineered': X_engineered
        }
        
        # Apply feature selection to engineered features
        selection_results = self.apply_feature_selection(X_engineered, y)
        
        # Create selected feature sets
        for method, result in selection_results.items():
            if result['features']:
                selected_features = result['features']
                # Ensure all features exist in X_engineered
                available_features = [f for f in selected_features if f in X_engineered.columns]
                if available_features:
                    feature_sets[f'{method} Selected'] = X_engineered[available_features]
        
        # Evaluate each feature set
        evaluation_results = {}
        
        for set_name, feature_set in feature_sets.items():
            try:
                # Scale features
                scaler = StandardScaler()
                X_train_scaled_set = scaler.fit_transform(feature_set.iloc[:len(X_train_scaled)])
                X_test_scaled_set = scaler.transform(feature_set.iloc[len(X_train_scaled):])
                
                # Train Random Forest on this feature set
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train_scaled_set, y_train)
                
                # Evaluate
                y_pred = rf_model.predict(X_test_scaled_set)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                evaluation_results[set_name] = {
                    'n_features': feature_set.shape[1],
                    'RMSE': rmse,
                    'R²': r2
                }
            except Exception as e:
                st.warning(f"Error evaluating {set_name}: {str(e)}")
                continue
        
        return evaluation_results
    
    def analyze_learning_curves(self, models, X_train_scaled, y_train, X_test_scaled, y_test):
        """Analyze learning curves for all models"""
        learning_curve_data = {}
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        for model_name, model_results in models.items():
            model = model_results['model']
            
            train_sizes_abs, train_scores, test_scores = learning_curve(
                model, X_train_scaled, y_train, 
                train_sizes=train_sizes, cv=3, 
                scoring='neg_mean_squared_error', random_state=42
            )
            
            # Convert to RMSE
            train_rmse = np.sqrt(-train_scores)
            test_rmse = np.sqrt(-test_scores)
            
            learning_curve_data[model_name] = {
                'train_sizes': train_sizes_abs,
                'train_rmse': train_rmse.mean(axis=1),
                'test_rmse': test_rmse.mean(axis=1),
                'train_std': train_rmse.std(axis=1),
                'test_std': test_rmse.std(axis=1)
            }
        
        return learning_curve_data
    
    def analyze_residuals(self, models, y_test):
        """Analyze residuals for all models"""
        residual_data = {}
        
        for model_name, model_results in models.items():
            y_pred = model_results['predictions']
            residuals = y_test - y_pred
            
            # Normality test
            try:
                shapiro_stat, shapiro_p = shapiro(residuals)
                is_normal = shapiro_p > 0.05
            except:
                shapiro_stat, shapiro_p = 0, 0
                is_normal = False
            
            # Homoscedasticity (simple check)
            residual_variance = np.var(residuals)
            
            residual_data[model_name] = {
                'residuals': residuals,
                'predictions': y_pred,
                'shapiro_stat': shapiro_stat,
                'shapiro_p': shapiro_p,
                'is_normal': is_normal,
                'variance': residual_variance
            }
        
        return residual_data
    
    def analyze_bias_variance_tradeoff(self, models, X_train_scaled, y_train, X_test_scaled, y_test):
        """Analyze bias-variance tradeoff using cross-validation"""
        bias_variance_data = {}
        
        for model_name, model_results in models.items():
            model = model_results['model']
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, 
                cv=5, scoring='neg_mean_squared_error'
            )
            
            # Convert to RMSE
            cv_rmse = np.sqrt(-cv_scores)
            
            # Calculate bias and variance estimates
            bias_estimate = cv_rmse.mean()
            variance_estimate = cv_rmse.std() ** 2
            
            bias_variance_data[model_name] = {
                'cv_mean': cv_rmse.mean(),
                'cv_std': cv_rmse.std(),
                'cv_scores': cv_rmse,
                'bias_estimate': bias_estimate,
                'variance_estimate': variance_estimate,
                'total_error': bias_estimate + variance_estimate
            }
        
        return bias_variance_data
    
    def create_bias_variance_plot(self, bias_variance_data):
        """Create comprehensive bias-variance tradeoff visualization"""
        models = list(bias_variance_data.keys())
        bias_values = [data['bias_estimate'] for data in bias_variance_data.values()]
        variance_values = [data['variance_estimate'] for data in bias_variance_data.values()]
        total_error_values = [data['total_error'] for data in bias_variance_data.values()]
        cv_std_values = [data['cv_std'] for data in bias_variance_data.values()]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Bias-Variance Tradeoff', 'Cross-Validation Stability', 'Error Components', 'Model Comparison'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # 1. Bias-Variance Scatter Plot
        fig.add_trace(
            go.Scatter(
                x=bias_values,
                y=variance_values,
                mode='markers+text',
                text=models,
                textposition="top center",
                marker=dict(size=15, color=colors[:len(models)]),
                name='Models',
                hovertemplate='<b>%{text}</b><br>Bias: %{x:.4f}<br>Variance: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Cross-Validation Stability (CV Standard Deviation)
        fig.add_trace(
            go.Bar(
                x=models,
                y=cv_std_values,
                marker_color=colors[:len(models)],
                name='CV Std Dev',
                hovertemplate='<b>%{x}</b><br>CV Std Dev: %{y:.4f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Error Components Stacked Bar
        fig.add_trace(
            go.Bar(
                x=models,
                y=bias_values,
                name='Bias²',
                marker_color='#FF6B6B',
                hovertemplate='<b>%{x}</b><br>Bias²: %{y:.4f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=variance_values,
                name='Variance',
                marker_color='#4ECDC4',
                hovertemplate='<b>%{x}</b><br>Variance: %{y:.4f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Total Error vs CV Mean
        fig.add_trace(
            go.Scatter(
                x=[data['cv_mean'] for data in bias_variance_data.values()],
                y=total_error_values,
                mode='markers+text',
                text=models,
                textposition="top center",
                marker=dict(size=15, color=colors[:len(models)]),
                name='Total Error vs CV Mean',
                hovertemplate='<b>%{text}</b><br>CV Mean: %{x:.4f}<br>Total Error: %{y:.4f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Bias-Variance Tradeoff Analysis",
            showlegend=True,
            barmode='stack'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Bias Estimate", row=1, col=1)
        fig.update_yaxes(title_text="Variance Estimate", row=1, col=1)
        
        fig.update_xaxes(title_text="Models", row=1, col=2)
        fig.update_yaxes(title_text="CV Standard Deviation", row=1, col=2)
        
        fig.update_xaxes(title_text="Models", row=2, col=1)
        fig.update_yaxes(title_text="Error Components", row=2, col=1)
        
        fig.update_xaxes(title_text="CV Mean RMSE", row=2, col=2)
        fig.update_yaxes(title_text="Total Error (Bias² + Variance)", row=2, col=2)
        
        return fig
    
    def create_learning_curves_plot(self, learning_curve_data):
        """Create learning curves visualization with clear train/test distinction"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(learning_curve_data.keys()),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, (model_name, data) in enumerate(learning_curve_data.items()):
            if i < len(positions):
                row, col = positions[i]
                
                # Training curve - SOLID LINE
                fig.add_trace(
                    go.Scatter(
                        x=data['train_sizes'],
                        y=data['train_rmse'],
                        mode='lines+markers',
                        name=f'🚀 {model_name} (Train)',
                        line=dict(color=colors[i], width=3),
                        marker=dict(size=6, symbol='circle'),
                        showlegend=True,
                        legendgroup=f"group{i}",
                        legendgrouptitle_text=f"📊 {model_name}",
                        hovertemplate=f'<b>{model_name} Train</b><br>Training Size: %{{x}}<br>RMSE: %{{y:.4f}}<extra></extra>'
                    ),
                    row=row, col=col
                )
                
                # Test curve - DASHED LINE
                fig.add_trace(
                    go.Scatter(
                        x=data['train_sizes'],
                        y=data['test_rmse'],
                        mode='lines+markers',
                        name=f'📈 {model_name} (Test)',
                        line=dict(color=colors[i], width=3, dash='dash'),
                        marker=dict(size=6, symbol='diamond'),
                        showlegend=True,
                        legendgroup=f"group{i}",
                        hovertemplate=f'<b>{model_name} Test</b><br>Training Size: %{{x}}<br>RMSE: %{{y:.4f}}<extra></extra>'
                    ),
                    row=row, col=col
                )
                
                # Add confidence intervals for test curve
                fig.add_trace(
                    go.Scatter(
                        x=data['train_sizes'],
                        y=data['test_rmse'] + data['test_std'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row, col=col
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data['train_sizes'],
                        y=data['test_rmse'] - data['test_std'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor=colors[i],
                        opacity=0.15,
                        showlegend=False,
                        hoverinfo='skip',
                        name=f'{model_name} Confidence Interval'
                    ),
                    row=row, col=col
                )
        
        # Update layout without legend
        fig.update_layout(
            height=800,
            title_text="📈 Learning Curves Analysis - Train vs Test Performance",
            showlegend=False
        )
        
        # Update axes with better labels
        fig.update_xaxes(title_text="Training Set Size", row=1, col=1)
        fig.update_xaxes(title_text="Training Set Size", row=1, col=2)
        fig.update_xaxes(title_text="Training Set Size", row=2, col=1)
        fig.update_xaxes(title_text="Training Set Size", row=2, col=2)
        
        fig.update_yaxes(title_text="RMSE (Lower is Better)", row=1, col=1)
        fig.update_yaxes(title_text="RMSE (Lower is Better)", row=1, col=2)
        fig.update_yaxes(title_text="RMSE (Lower is Better)", row=2, col=1)
        fig.update_yaxes(title_text="RMSE (Lower is Better)", row=2, col=2)
        
        return fig
    
    def create_residuals_analysis_plot(self, residual_data):
        """Create residuals analysis visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(residual_data.keys()),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, (model_name, data) in enumerate(residual_data.items()):
            if i < len(positions):
                row, col = positions[i]
                
                # Residuals vs predicted
                fig.add_trace(
                    go.Scatter(
                        x=data['predictions'],
                        y=data['residuals'],
                        mode='markers',
                        name=model_name,
                        marker=dict(color=colors[i], size=6, opacity=0.7),
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Add horizontal line at y=0
                fig.add_trace(
                    go.Scatter(
                        x=[data['predictions'].min(), data['predictions'].max()],
                        y=[0, 0],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=700,
            title_text="Residuals Analysis",
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Predicted Values")
        fig.update_yaxes(title_text="Residuals")
        
        return fig
    
    def load_spatial_data(self, data_path='merged_gedi_sentinel2_data_with_indices.csv'):
        """Load spatial data with coordinates"""
        try:
            spatial_data = pd.read_csv(data_path)
            # Extract coordinates and AGB values
            spatial_data = spatial_data[['Longitude_gedi', 'Latitude_gedi', 'AGB_L4A']].dropna()
            return spatial_data
        except Exception as e:
            st.error(f"Error loading spatial data: {str(e)}")
            return None
    
    def perform_geographic_clustering(self, spatial_data, n_clusters=5):
        """Perform geographic clustering to identify spatial patterns"""
        try:
            # Prepare coordinates for clustering
            coords = spatial_data[['Longitude_gedi', 'Latitude_gedi']].values
            
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            spatial_data['Cluster'] = kmeans.fit_predict(coords)
            
            # Calculate cluster statistics
            cluster_stats = spatial_data.groupby('Cluster').agg({
                'AGB_L4A': ['mean', 'std', 'count'],
                'Longitude_gedi': 'mean',
                'Latitude_gedi': 'mean'
            }).round(2)
            
            return spatial_data, cluster_stats
            
        except Exception as e:
            st.error(f"Error in geographic clustering: {str(e)}")
            return None, None
    
    def calculate_spatial_autocorrelation(self, spatial_data):
        """Calculate Moran's I and Geary's C statistics"""
        try:
            coords = spatial_data[['Longitude_gedi', 'Latitude_gedi']].values
            agb_values = spatial_data['AGB_L4A'].values
            
            # Calculate distance matrix
            distances = pdist(coords)
            distance_matrix = squareform(distances)
            
            # Moran's I calculation (simplified version)
            n = len(agb_values)
            mean_agb = np.mean(agb_values)
            
            # Calculate spatial weights (inverse distance)
            weights = 1 / (1 + distance_matrix)
            np.fill_diagonal(weights, 0)
            
            # Moran's I
            numerator = 0
            denominator = 0
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        numerator += weights[i, j] * (agb_values[i] - mean_agb) * (agb_values[j] - mean_agb)
                        denominator += weights[i, j]
            
            morans_i = (n / (2 * denominator)) * (numerator / np.sum((agb_values - mean_agb) ** 2))
            
            # Geary's C (simplified version)
            numerator_c = 0
            denominator_c = 0
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        numerator_c += weights[i, j] * (agb_values[i] - agb_values[j]) ** 2
                        denominator_c += weights[i, j]
            
            gearys_c = ((n - 1) / (2 * denominator_c)) * (numerator_c / np.sum((agb_values - mean_agb) ** 2))
            
            return {
                'Morans_I': morans_i,
                'Gearys_C': gearys_c,
                'Interpretation_Morans_I': 'Positive spatial autocorrelation' if morans_i > 0 else 'Negative spatial autocorrelation',
                'Interpretation_Gearys_C': 'Positive spatial autocorrelation' if gearys_c < 1 else 'Negative spatial autocorrelation'
            }
            
        except Exception as e:
            st.error(f"Error calculating spatial autocorrelation: {str(e)}")
            return None
    
    def perform_hotspot_analysis(self, spatial_data, radius=0.01):
        """Identify high/low biomass concentration areas"""
        try:
            coords = spatial_data[['Longitude_gedi', 'Latitude_gedi']].values
            agb_values = spatial_data['AGB_L4A'].values
            
            # Use Local Outlier Factor for hotspot detection
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            lof_scores = lof.fit_predict(coords)
            
            # Create hotspot classification
            spatial_data['LOF_Score'] = lof.negative_outlier_factor_
            spatial_data['Hotspot_Type'] = 'Normal'
            spatial_data.loc[spatial_data['LOF_Score'] < np.percentile(spatial_data['LOF_Score'], 10), 'Hotspot_Type'] = 'Low_AGB'
            spatial_data.loc[spatial_data['LOF_Score'] > np.percentile(spatial_data['LOF_Score'], 90), 'Hotspot_Type'] = 'High_AGB'
            
            # Calculate hotspot statistics
            hotspot_stats = spatial_data.groupby('Hotspot_Type').agg({
                'AGB_L4A': ['mean', 'std', 'count'],
                'LOF_Score': 'mean'
            }).round(2)
            
            return spatial_data, hotspot_stats
            
        except Exception as e:
            st.error(f"Error in hotspot analysis: {str(e)}")
            return None, None
    
    def perform_spatial_interpolation(self, spatial_data, method='idw', grid_resolution=100):
        """Perform spatial interpolation (IDW or Kriging)"""
        try:
            coords = spatial_data[['Longitude_gedi', 'Latitude_gedi']].values
            agb_values = spatial_data['AGB_L4A'].values
            
            # Create interpolation grid
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            
            x_grid = np.linspace(x_min, x_max, grid_resolution)
            y_grid = np.linspace(y_min, y_max, grid_resolution)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
            
            if method == 'idw':
                # Inverse Distance Weighting
                interpolated_values = np.zeros_like(X_grid)
                
                for i in range(grid_resolution):
                    for j in range(grid_resolution):
                        grid_point = np.array([X_grid[i, j], Y_grid[i, j]])
                        distances = np.sqrt(np.sum((coords - grid_point) ** 2, axis=1))
                        
                        # Avoid division by zero
                        distances = np.where(distances == 0, 1e-10, distances)
                        
                        # IDW weights
                        weights = 1 / (distances ** 2)
                        interpolated_values[i, j] = np.sum(weights * agb_values) / np.sum(weights)
            
            elif method == 'nearest':
                # Nearest neighbor interpolation
                tree = cKDTree(coords)
                grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
                _, indices = tree.query(grid_points)
                interpolated_values = agb_values[indices].reshape(X_grid.shape)
            
            return {
                'X_grid': X_grid,
                'Y_grid': Y_grid,
                'interpolated_values': interpolated_values,
                'method': method
            }
            
        except Exception as e:
            st.error(f"Error in spatial interpolation: {str(e)}")
            return None
    
    def create_spatial_visualizations(self, spatial_data, cluster_data=None, hotspot_data=None, interpolation_data=None):
        """Create comprehensive spatial visualizations with interactive maps"""
        try:
            # 1. Basic spatial scatter plot with map
            fig_scatter = px.scatter_mapbox(
                spatial_data, 
                lat='Latitude_gedi', 
                lon='Longitude_gedi',
                color='AGB_L4A',
                size='AGB_L4A',
                title='Spatial Distribution of AGB Values',
                color_continuous_scale='viridis',
                hover_data=['AGB_L4A'],
                zoom=8,
                mapbox_style="open-street-map"
            )
            fig_scatter.update_layout(height=600, margin={"r":0,"t":30,"l":0,"b":0})
            
            # 2. Clustering visualization with map
            if cluster_data is not None:
                fig_clusters = px.scatter_mapbox(
                    cluster_data,
                    lat='Latitude_gedi',
                    lon='Longitude_gedi',
                    color='Cluster',
                    title='Geographic Clustering Results',
                    hover_data=['AGB_L4A', 'Cluster'],
                    zoom=8,
                    mapbox_style="open-street-map"
                )
                fig_clusters.update_layout(height=600, margin={"r":0,"t":30,"l":0,"b":0})
            else:
                fig_clusters = None
            
            # 3. Hotspot visualization with map
            if hotspot_data is not None:
                fig_hotspots = px.scatter_mapbox(
                    hotspot_data,
                    lat='Latitude_gedi',
                    lon='Longitude_gedi',
                    color='Hotspot_Type',
                    title='Hotspot Analysis Results',
                    hover_data=['AGB_L4A', 'LOF_Score'],
                    zoom=8,
                    mapbox_style="open-street-map"
                )
                fig_hotspots.update_layout(height=600, margin={"r":0,"t":30,"l":0,"b":0})
            else:
                fig_hotspots = None
            
            # 4. Interpolation visualization with map overlay
            if interpolation_data is not None:
                # Create a heatmap overlay on the map
                fig_interpolation = go.Figure()
                
                # Add the heatmap layer
                fig_interpolation.add_trace(go.Densitymapbox(
                    lat=spatial_data['Latitude_gedi'],
                    lon=spatial_data['Longitude_gedi'],
                    z=spatial_data['AGB_L4A'],
                    radius=20,
                    colorscale='viridis',
                    name=f'{interpolation_data["method"].upper()} Interpolation'
                ))
                
                fig_interpolation.update_layout(
                    title=f'Spatial Interpolation Heatmap ({interpolation_data["method"].upper()})',
                    mapbox=dict(
                        style="open-street-map",
                        center=dict(
                            lat=spatial_data['Latitude_gedi'].mean(),
                            lon=spatial_data['Longitude_gedi'].mean()
                        ),
                        zoom=8
                    ),
                    height=600,
                    margin={"r":0,"t":30,"l":0,"b":0}
                )
            else:
                fig_interpolation = None
            
            return fig_scatter, fig_clusters, fig_hotspots, fig_interpolation
            
        except Exception as e:
            st.error(f"Error creating spatial visualizations: {str(e)}")
            return None, None, None, None

def main():
    # Initialize dashboard
    dashboard = StreamlitAGBDashboard()
    
    # Title with enhanced styling
    st.markdown('<h1 class="main-header">🌲 BioVision Analytics Hub</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced Above Ground Biomass Intelligence Platform</p>', unsafe_allow_html=True)
    
    # Add a decorative separator
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <div style="display: inline-block; width: 100px; height: 4px; background: linear-gradient(90deg, #2E8B57, #3CB371); border-radius: 2px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with enhanced styling
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #2E8B57, #3CB371); border-radius: 15px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0; font-size: 1.5rem;">🌲 BioVision Analytics Hub</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.9rem;">Dashboard Controls</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    data = dashboard.load_data()
    if data is None:
        st.stop()
    
    # Sidebar info
    st.sidebar.success(f"✅ Data loaded: {data.shape[0]} samples, {data.shape[1]} features")
    
    # Quick stats
    st.sidebar.markdown("### 📊 Quick Stats")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Samples", data.shape[0])
    with col2:
        st.metric("Features", data.shape[1] - 1)
    
    # Data info
    st.sidebar.markdown("### 📋 Data Info")
    st.sidebar.info(f"**Target:** AGB_2017")
    st.sidebar.info(f"**Missing Values:** {data.isnull().sum().sum()}")
    st.sidebar.info(f"**Memory Usage:** {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Model status
    if 'results' in st.session_state:
        st.sidebar.markdown("### 🤖 Model Status")
        st.sidebar.success("✅ Models Trained")
        best_model = min(st.session_state.results.keys(), key=lambda x: st.session_state.results[x]['RMSE'])
        st.sidebar.info(f"**Best Model:** {best_model}")
        st.sidebar.info(f"**Best RMSE:** {st.session_state.results[best_model]['RMSE']:.4f}")
    else:
        st.sidebar.markdown("### 🤖 Model Status")
        st.sidebar.warning("⚠️ Models Not Trained")
        st.sidebar.info("Go to Model Performance to train models")
    
    # Dashboard sections
    st.sidebar.markdown("### 🎯 Navigation")
    st.sidebar.markdown("Select a section to explore:")
    
    # Create buttons for each section
    if st.sidebar.button("📊 Model Performance", use_container_width=True):
        st.session_state.selected_section = "📊 Model Performance"
    
    if st.sidebar.button("🎯 Feature Importance", use_container_width=True):
        st.session_state.selected_section = "🎯 Feature Importance"
    
    if st.sidebar.button("📈 Prediction Analysis", use_container_width=True):
        st.session_state.selected_section = "📈 Prediction Analysis"
    
    if st.sidebar.button("🔗 Correlation Analysis", use_container_width=True):
        st.session_state.selected_section = "🔗 Correlation Analysis"
    
    if st.sidebar.button("🔧 Feature Engineering", use_container_width=True):
        st.session_state.selected_section = "🔧 Feature Engineering"
    
    if st.sidebar.button("📊 Model Diagnostics", use_container_width=True):
        st.session_state.selected_section = "📊 Model Diagnostics"
    
    if st.sidebar.button("🗺️ Spatial Analysis", use_container_width=True):
        st.session_state.selected_section = "🗺️ Spatial Analysis"
    
    if st.sidebar.button("📋 Data Overview", use_container_width=True):
        st.session_state.selected_section = "📋 Data Overview"
    
    # Initialize default section if not set
    if 'selected_section' not in st.session_state:
        st.session_state.selected_section = "📊 Model Performance"
    
    selected_section = st.session_state.selected_section
    
    # Show current section
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Current Section:** {selected_section}")
    
    # Helpful tips
    st.sidebar.markdown("### 💡 Quick Tips")
    if selected_section == "📊 Model Performance":
        st.sidebar.info("💡 Start by training models to unlock all features")
    elif selected_section == "🎯 Feature Importance":
        st.sidebar.info("💡 Train models first to see feature importance")
    elif selected_section == "📈 Prediction Analysis":
        st.sidebar.info("💡 Analyze prediction accuracy and residuals")
    elif selected_section == "🔗 Correlation Analysis":
        st.sidebar.info("💡 Explore feature relationships with target")
    elif selected_section == "🔧 Feature Engineering":
        st.sidebar.info("💡 Create new features and evaluate performance")
    elif selected_section == "📊 Model Diagnostics":
        st.sidebar.info("💡 Check for overfitting and model stability")
    elif selected_section == "🗺️ Spatial Analysis":
        st.sidebar.info("💡 Explore geographic patterns in biomass data")
    elif selected_section == "📋 Data Overview":
        st.sidebar.info("💡 Understand your dataset structure and distributions")
    
    # Keyboard shortcuts info
    st.sidebar.markdown("### ⌨️ Navigation")
    st.sidebar.markdown("Use the buttons above to navigate between sections")
    st.sidebar.markdown("Each section provides different insights into your data")
    
    # Preprocess data
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = dashboard.preprocess_data(data)
    feature_names = X_train.columns.tolist()
    
    if selected_section == "📊 Model Performance":
        st.markdown('<h1 class="section-header">📊 Model Performance Comparison</h1>', unsafe_allow_html=True)
        
        if st.button("🚀 Train Models", type="primary"):
            # Add a beautiful loading message
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 15px; margin: 1rem 0;">
                <div style="font-size: 2rem; margin-bottom: 1rem;">🌲</div>
                <h3 style="color: #2E8B57; margin-bottom: 0.5rem;">Training Models...</h3>
                <p style="color: #666; margin: 0;">Please wait while we train the machine learning models</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("Training models..."):
                results, feature_importance = dashboard.train_models(
                    X_train_scaled, y_train, X_test_scaled, y_test
                )
                
                # Store in session state
                st.session_state.results = results
                st.session_state.feature_importance = feature_importance
        
        if 'results' in st.session_state:
            results = st.session_state.results
            
            # Performance metrics
            st.subheader("📈 Performance Metrics")
            
            cols = st.columns(4)
            for i, (model_name, model_results) in enumerate(results.items()):
                with cols[i]:
                    st.metric(
                        label=f"{model_name}",
                        value=f"RMSE: {model_results['RMSE']:.4f}",
                        delta=f"R²: {model_results['R²']:.4f}"
                    )
            
            # Performance comparison chart with enhanced styling
            st.markdown('<h3 style="color: #2E8B57; margin: 1.5rem 0 1rem 0;">📊 Interactive Performance Comparison</h3>', unsafe_allow_html=True)
            fig = dashboard.create_performance_comparison(results)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Best model highlight with enhanced styling
            best_model = min(results.keys(), key=lambda x: results[x]['RMSE'])
            st.markdown(f"""
            <div class="success-box">
                <h4 style="color: #28a745; margin: 0 0 0.5rem 0;">🏆 Best Performing Model</h4>
                <p style="margin: 0; font-size: 1.1rem;"><strong>{best_model}</strong> with RMSE: <strong>{results[best_model]['RMSE']:.4f}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Save models functionality
            st.subheader("💾 Save Models & Results")
            
            if st.button("💾 Save All Models and Results", type="primary"):
                try:
                    # Create timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Save models
                    for model_name, model_results in results.items():
                        model_filename = f"AGB_{model_name.replace(' ', '_')}_{timestamp}.pkl"
                        joblib.dump(model_results['model'], model_filename)
                        
                        scaler_filename = f"AGB_Scaler_{timestamp}.pkl"
                        joblib.dump(scaler, scaler_filename)
                    
                    # Save results summary
                    summary_data = []
                    for model_name, model_results in results.items():
                        summary_data.append({
                            'Model': model_name,
                            'RMSE': model_results['RMSE'],
                            'R²': model_results['R²'],
                            'MAE': model_results['MAE'],
                            'Training Time (s)': model_results['Training Time (s)']
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_filename = f"AGB_Model_Results_{timestamp}.csv"
                    summary_df.to_csv(summary_filename, index=False)
                    
                    st.markdown(f"""
                    <div class="success-box">
                        <h4 style="color: #28a745; margin: 0 0 0.5rem 0;">✅ Models and Results Saved Successfully!</h4>
                        <p style="margin: 0;">📁 Files saved with timestamp: <strong>{timestamp}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display saved files
                    with st.expander("📁 Saved Files"):
                        st.write("**Models:**")
                        for model_name in results.keys():
                            st.write(f"- AGB_{model_name.replace(' ', '_')}_{timestamp}.pkl")
                        st.write(f"**Scaler:** AGB_Scaler_{timestamp}.pkl")
                        st.write(f"**Results:** AGB_Model_Results_{timestamp}.csv")
                        
                except Exception as e:
                    st.error(f"❌ Error saving models: {str(e)}")
            
            # Comprehensive Insights Section
            st.markdown("---")
            st.header("🧠 Comprehensive Insights & Recommendations")
            
            # Model Performance Insights
            st.subheader("📊 Model Performance Insights")
            
            best_model = min(results.keys(), key=lambda x: results[x]['RMSE'])
            worst_model = max(results.keys(), key=lambda x: results[x]['RMSE'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "🏆 Best Model", 
                    best_model,
                    f"RMSE: {results[best_model]['RMSE']:.4f}"
                )
            
            with col2:
                st.metric(
                    "📉 Worst Model", 
                    worst_model,
                    f"RMSE: {results[worst_model]['RMSE']:.4f}"
                )
            
            with col3:
                performance_gap = results[worst_model]['RMSE'] - results[best_model]['RMSE']
                st.metric(
                    "📊 Performance Gap", 
                    f"{performance_gap:.4f}",
                    "RMSE difference"
                )
            
            # Training Efficiency Insights
            st.subheader("⚡ Training Efficiency Insights")
            
            fastest_model = min(results.keys(), key=lambda x: results[x]['Training Time (s)'])
            slowest_model = max(results.keys(), key=lambda x: results[x]['Training Time (s)'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"🚀 **Fastest Training**: {fastest_model} ({results[fastest_model]['Training Time (s)']:.2f}s)")
                st.info(f"🐌 **Slowest Training**: {slowest_model} ({results[slowest_model]['Training Time (s)']:.2f}s)")
            
            with col2:
                speed_ratio = results[slowest_model]['Training Time (s)'] / results[fastest_model]['Training Time (s)']
                st.info(f"⚡ **Speed Ratio**: {slowest_model} is {speed_ratio:.1f}x slower than {fastest_model}")
            
            # Feature Engineering Recommendations
            if 'evaluation_results' in st.session_state:
                st.subheader("🔧 Feature Engineering Recommendations")
                
                evaluation_results = st.session_state.evaluation_results
                best_feature_set = max(evaluation_results.keys(), key=lambda x: evaluation_results[x]['R²'])
                
                st.success(f"🎯 **Recommended Feature Set**: {best_feature_set}")
                st.info(f"📈 **Performance**: R² = {evaluation_results[best_feature_set]['R²']:.4f}")
                st.info(f"🔢 **Feature Count**: {evaluation_results[best_feature_set]['n_features']} features")
            
            # Actionable Next Steps
            st.subheader("🎯 Actionable Next Steps")
            
            next_steps = []
            
            # General recommendations based on model performance
            best_r2 = max([data['R²'] for data in results.values()])
            if best_r2 < 0.6:
                next_steps.append("🔧 **Improve Model Performance**: Focus on feature engineering and hyperparameter tuning")
                next_steps.append("📊 **Collect More Data**: Consider expanding the dataset for better generalization")
                next_steps.append("🔄 **Try Different Algorithms**: Experiment with ensemble methods or deep learning")
            else:
                next_steps.append("🚀 **Deploy Best Model**: Start with the highest-performing model")
                next_steps.append("📊 **Monitor Performance**: Set up continuous monitoring and retraining")
            
            for step in next_steps:
                st.write(f"• {step}")
            
            # Summary Statistics
            st.subheader("📊 Summary Statistics")
            
            summary_stats = {
                "Total Models Trained": len(results),
                "Best R² Score": max([data['R²'] for data in results.values()]),
                "Worst R² Score": min([data['R²'] for data in results.values()]),
                "Average Training Time": np.mean([data['Training Time (s)'] for data in results.values()]),
                "Total Features": len(feature_names),
                "Dataset Size": f"{len(data)} samples"
            }
            
            col1, col2, col3 = st.columns(3)
            
            for i, (metric, value) in enumerate(summary_stats.items()):
                col = col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3
                with col:
                    if isinstance(value, float):
                        st.metric(metric, f"{value:.4f}")
                    else:
                        st.metric(metric, value)
    
    elif selected_section == "🎯 Feature Importance":
        st.markdown('<h1 class="section-header">🎯 Feature Importance Analysis</h1>', unsafe_allow_html=True)
        
        if 'feature_importance' in st.session_state:
            feature_importance = st.session_state.feature_importance
            
            if feature_importance:
                # Feature importance plot
                fig = dashboard.create_feature_importance_plot(feature_importance, feature_names)
                st.plotly_chart(fig, use_container_width=True)
                
                # Permutation importance analysis
                if st.button("🔍 Calculate Permutation Importance", type="primary"):
                    with st.spinner("Calculating permutation importance..."):
                        results = st.session_state.results
                        permutation_importance_data = {}
                        
                        for model_name, model_results in results.items():
                            model = model_results['model']
                            
                            # Calculate permutation importance
                            perm_importance = permutation_importance(
                                model, X_test_scaled, y_test, 
                                n_repeats=10, random_state=42, n_jobs=-1
                            )
                            
                            permutation_importance_data[model_name] = {
                                'importance': perm_importance.importances_mean,
                                'std': perm_importance.importances_std
                            }
                        
                        # Store in session state
                        st.session_state.permutation_importance = permutation_importance_data
                
                if 'permutation_importance' in st.session_state:
                    permutation_importance_data = st.session_state.permutation_importance
                    
                    st.subheader("🔍 Permutation Importance Analysis")
                    
                    # Create permutation importance plot
                    fig_perm = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=list(permutation_importance_data.keys()),
                        specs=[[{"type": "bar"}, {"type": "bar"}],
                               [{"type": "bar"}, {"type": "bar"}]]
                    )
                    
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
                    
                    for i, (model_name, data) in enumerate(permutation_importance_data.items()):
                        if i < len(positions):
                            row, col = positions[i]
                            
                            # Get top 10 features
                            top_indices = np.argsort(data['importance'])[-10:][::-1]
                            top_features = [feature_names[idx] for idx in top_indices]
                            top_importance = data['importance'][top_indices]
                            top_std = data['std'][top_indices]
                            
                            fig_perm.add_trace(
                                go.Bar(
                                    x=top_importance,
                                    y=top_features,
                                    orientation='h',
                                    name=model_name,
                                    marker_color=colors[i],
                                    error_x=dict(type='data', array=top_std, visible=True)
                                ),
                                row=row, col=col
                            )
                    
                    fig_perm.update_layout(
                        height=700,
                        showlegend=False,
                        title_text="Permutation Importance Analysis (Top 10 Features)"
                    )
                    
                    st.plotly_chart(fig_perm, use_container_width=True)
                    
                    # Feature importance comparison
                    st.subheader("📊 Feature Importance Comparison")
                    
                    # Create comparison DataFrame
                    importance_comparison = []
                    for model_name, data in feature_importance.items():
                        top_indices = np.argsort(data)[-10:][::-1]
                        for i, idx in enumerate(top_indices):
                            importance_comparison.append({
                                'Feature': feature_names[idx],
                                'Model': model_name,
                                'Importance': data[idx],
                                'Rank': i + 1
                            })
                    
                    importance_df = pd.DataFrame(importance_comparison)
                    st.dataframe(importance_df, use_container_width=True)
                    
                    # Top features summary
                    st.subheader("🎯 Top Features Summary")
                    
                    for model_name, importance in feature_importance.items():
                        with st.expander(f"{model_name} - Top 10 Features"):
                            top_indices = np.argsort(importance)[-10:][::-1]
                            top_features_df = pd.DataFrame({
                                'Feature': [feature_names[idx] for idx in top_indices],
                                'Importance': importance[top_indices],
                                'Rank': range(1, 11)
                            })
                            st.dataframe(top_features_df, use_container_width=True)
            else:
                st.warning("No feature importance data available. Please train models first.")
        else:
            st.warning("Please train models first in the Model Performance section.")
    
    elif selected_section == "📈 Prediction Analysis":
        st.markdown('<h1 class="section-header">📈 Prediction Analysis</h1>', unsafe_allow_html=True)
        
        if 'results' in st.session_state:
            results = st.session_state.results
            
            # Prediction scatter plots
            fig = dashboard.create_prediction_scatter(results, y_test)
            st.plotly_chart(fig, use_container_width=True)
            
            # Residual analysis
            st.subheader("📊 Residual Analysis")
            
            selected_model = st.selectbox("Select Model for Residual Analysis", list(results.keys()))
            
            if selected_model:
                y_pred = results[selected_model]['predictions']
                residuals = y_test - y_pred
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Residuals histogram
                    fig_hist = px.histogram(
                        x=residuals, 
                        title=f"{selected_model} - Residuals Distribution",
                        labels={'x': 'Residuals', 'y': 'Frequency'}
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Residuals vs predicted
                    fig_scatter = px.scatter(
                        x=y_pred, 
                        y=residuals,
                        title=f"{selected_model} - Residuals vs Predicted",
                        labels={'x': 'Predicted Values', 'y': 'Residuals'}
                    )
                    fig_scatter.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Please train models first in the Model Performance section.")
    
    elif selected_section == "🔗 Correlation Analysis":
        st.markdown('<h1 class="section-header">🔗 Correlation Analysis</h1>', unsafe_allow_html=True)
        
        # Correlation heatmap
        fig = dashboard.create_correlation_heatmap(data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Target correlation
        st.subheader("🎯 Correlation with AGB Target")
        
        feature_cols = [col for col in data.columns if col != 'AGB_2017']
        correlations = []
        
        for feature in feature_cols:
            corr = data[feature].corr(data['AGB_2017'])
            correlations.append({'Feature': feature, 'Correlation': corr, 'Abs_Correlation': abs(corr)})
        
        corr_df = pd.DataFrame(correlations).sort_values('Abs_Correlation', ascending=False)
        
        # Top positive and negative correlations
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Positive Correlations**")
            positive_corr = corr_df[corr_df['Correlation'] > 0].head(10)
            fig_pos = px.bar(
                positive_corr, 
                x='Correlation', 
                y='Feature',
                orientation='h',
                title="Top Positive Correlations with AGB"
            )
            st.plotly_chart(fig_pos, use_container_width=True)
        
        with col2:
            st.write("**Top Negative Correlations**")
            negative_corr = corr_df[corr_df['Correlation'] < 0].head(10)
            fig_neg = px.bar(
                negative_corr, 
                x='Correlation', 
                y='Feature',
                orientation='h',
                title="Top Negative Correlations with AGB"
            )
            st.plotly_chart(fig_neg, use_container_width=True)
    
    elif selected_section == "🔧 Feature Engineering":
        st.markdown('<h1 class="section-header">🔧 Feature Engineering</h1>', unsafe_allow_html=True)
        
        # Create engineered features
        X_engineered, y_engineered = dashboard.create_engineered_features(data)
        
        # Display engineered features
        st.subheader("📊 Engineered Features")
        st.dataframe(X_engineered.head(5), use_container_width=True)
        
        # Feature selection results
        st.subheader("📋 Feature Selection Results")
        selection_results = dashboard.apply_feature_selection(X_engineered, y_engineered)
        for method, result in selection_results.items():
            st.write(f"**{method}**")
            st.write(f"Number of features selected: {result['n_features']}")
            st.write(f"Selected features: {', '.join(result['features'])}")
        
        # PCA analysis
        st.subheader("📊 PCA Analysis")
        X_pca, pca = dashboard.apply_pca(X_engineered)
        st.write(f"Number of components selected by PCA: {pca.n_components_}")
        
        # Scatter plot of PCA components
        if X_pca.shape[1] >= 2:
            fig_pca_scatter = px.scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                title="PCA Scatter Plot of Engineered Features",
                labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', 'y': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'}
            )
            st.plotly_chart(fig_pca_scatter, use_container_width=True)
        else:
            st.warning("Not enough components for PCA scatter plot.")
        
        # Feature set evaluation
        if st.button("🚀 Evaluate Feature Sets", type="primary"):
            with st.spinner("Evaluating different feature sets..."):
                X_original = data.drop(columns=['AGB_2017'])
                evaluation_results = dashboard.evaluate_feature_sets(
                    X_original, X_engineered, y_engineered, 
                    X_train_scaled, y_train, X_test_scaled, y_test
                )
                
                # Store in session state
                st.session_state.evaluation_results = evaluation_results
        
        if 'evaluation_results' in st.session_state:
            evaluation_results = st.session_state.evaluation_results
            
            st.subheader("📊 Feature Set Performance Comparison")
            
            # Create comparison DataFrame
            comparison_data = []
            for set_name, results in evaluation_results.items():
                comparison_data.append({
                    'Feature Set': set_name,
                    'Number of Features': results['n_features'],
                    'RMSE': results['RMSE'],
                    'R²': results['R²']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Performance comparison chart
            fig_comparison = go.Figure()
            
            fig_comparison.add_trace(go.Bar(
                x=comparison_df['Feature Set'],
                y=comparison_df['RMSE'],
                name='RMSE',
                marker_color='#FF6B6B'
            ))
            
            fig_comparison.add_trace(go.Bar(
                x=comparison_df['Feature Set'],
                y=comparison_df['R²'],
                name='R²',
                marker_color='#4ECDC4',
                yaxis='y2'
            ))
            
            fig_comparison.update_layout(
                title="Feature Set Performance Comparison",
                xaxis_title="Feature Sets",
                yaxis=dict(title="RMSE", side="left"),
                yaxis2=dict(title="R²", side="right", overlaying="y"),
                height=500,
                barmode='group'
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Best feature set recommendation
            best_set = comparison_df.loc[comparison_df['R²'].idxmax()]
            st.success(f"🏆 **Best Feature Set**: {best_set['Feature Set']} with R²: {best_set['R²']:.4f} and {best_set['Number of Features']} features")
    
    elif selected_section == "📊 Model Diagnostics":
        st.markdown('<h1 class="section-header">📊 Model Diagnostics</h1>', unsafe_allow_html=True)
        
        if 'results' in st.session_state:
            results = st.session_state.results
            
            # Learning curves
            st.subheader("📈 Learning Curves")
            
            # Add explanation box
            with st.expander("ℹ️ How to Read Learning Curves", expanded=True):
                st.markdown("""
                **Learning Curves Guide:**
                
                🚀 **Solid Lines (Training Performance)**: Shows how well the model performs on the data it was trained on
                - Usually decreases as more data is added
                - Lower values indicate better training performance
                
                📈 **Dashed Lines (Test Performance)**: Shows how well the model generalizes to unseen data
                - More important for real-world performance
                - Should be close to training performance (no overfitting)
                
                **What to Look For:**
                - ✅ **Good**: Training and test curves are close together and both decrease
                - ⚠️ **Overfitting**: Training curve is much lower than test curve
                - ⚠️ **Underfitting**: Both curves are high and don't decrease much
                - ✅ **Ideal**: Test curve plateaus while training curve continues to improve slightly
                
                **Shaded Areas**: Confidence intervals showing the variability in test performance
                """)
            
            learning_curve_data = dashboard.analyze_learning_curves(results, X_train_scaled, y_train, X_test_scaled, y_test)
            fig_learning_curves = dashboard.create_learning_curves_plot(learning_curve_data)
            st.plotly_chart(fig_learning_curves, use_container_width=True)
            
            # Residuals analysis
            st.subheader("📊 Residuals Analysis")
            residual_data = dashboard.analyze_residuals(results, y_test)
            fig_residuals = dashboard.create_residuals_analysis_plot(residual_data)
            st.plotly_chart(fig_residuals, use_container_width=True)
            
            # Bias-Variance Tradeoff
            st.subheader("📊 Bias-Variance Tradeoff")
            bias_variance_data = dashboard.analyze_bias_variance_tradeoff(results, X_train_scaled, y_train, X_test_scaled, y_test)
            
            # Create comprehensive bias-variance visualization
            fig_bias_variance = dashboard.create_bias_variance_plot(bias_variance_data)
            st.plotly_chart(fig_bias_variance, use_container_width=True)
            
            # Detailed bias-variance analysis
            st.subheader("📈 Detailed Bias-Variance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Bias-Variance Breakdown:**")
                for model_name, data in bias_variance_data.items():
                    with st.expander(f"{model_name} Analysis"):
                        st.metric("Bias Estimate", f"{data['bias_estimate']:.4f}")
                        st.metric("Variance Estimate", f"{data['variance_estimate']:.4f}")
                        st.metric("Total Error", f"{data['total_error']:.4f}")
                        st.metric("CV Stability (Std Dev)", f"{data['cv_std']:.4f}")
                        
                        # Interpretation
                        if data['cv_std'] < 0.1:
                            stability = "High Stability"
                            stability_color = "green"
                        elif data['cv_std'] < 0.2:
                            stability = "Moderate Stability"
                            stability_color = "orange"
                        else:
                            stability = "Low Stability"
                            stability_color = "red"
                        
                        st.info(f"**Stability Assessment:** {stability}")
                        
                        # Bias-Variance tradeoff assessment
                        bias_variance_ratio = data['bias_estimate'] / (data['variance_estimate'] + 1e-8)
                        if bias_variance_ratio > 2:
                            assessment = "High Bias (Underfitting)"
                            assessment_color = "red"
                        elif bias_variance_ratio < 0.5:
                            assessment = "High Variance (Overfitting)"
                            assessment_color = "orange"
                        else:
                            assessment = "Balanced"
                            assessment_color = "green"
                        
                        st.info(f"**Tradeoff Assessment:** {assessment}")
            
            with col2:
                st.write("**Cross-Validation Scores:**")
                for model_name, data in bias_variance_data.items():
                    with st.expander(f"{model_name} CV Scores"):
                        cv_df = pd.DataFrame({
                            'Fold': range(1, len(data['cv_scores']) + 1),
                            'RMSE': data['cv_scores']
                        })
                        st.dataframe(cv_df, use_container_width=True)
                        
                        # CV score distribution
                        fig_cv = px.histogram(
                            cv_df, 
                            x='RMSE',
                            title=f"{model_name} - CV Score Distribution",
                            nbins=5
                        )
                        st.plotly_chart(fig_cv, use_container_width=True)
            
            # Summary insights
            st.subheader("🧠 Bias-Variance Insights")
            
            best_stability = min(bias_variance_data.keys(), key=lambda x: bias_variance_data[x]['cv_std'])
            best_balance = min(bias_variance_data.keys(), key=lambda x: bias_variance_data[x]['total_error'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Most Stable Model",
                    best_stability,
                    f"CV Std: {bias_variance_data[best_stability]['cv_std']:.4f}"
                )
            
            with col2:
                st.metric(
                    "Best Balanced Model",
                    best_balance,
                    f"Total Error: {bias_variance_data[best_balance]['total_error']:.4f}"
                )
            
            with col3:
                avg_stability = np.mean([data['cv_std'] for data in bias_variance_data.values()])
                st.metric(
                    "Average Stability",
                    f"{avg_stability:.4f}",
                    "CV Standard Deviation"
                )
        else:
            st.warning("Please train models first in the Model Performance section.")
    
    elif selected_section == "🗺️ Spatial Analysis":
        st.markdown('<h1 class="section-header">🗺️ Spatial Analysis</h1>', unsafe_allow_html=True)
        
        # Load spatial data
        spatial_data = dashboard.load_spatial_data()
        
        if spatial_data is not None:
            st.markdown(f"""
            <div class="success-box">
                <h4 style="color: #28a745; margin: 0 0 0.5rem 0;">✅ Spatial Data Loaded Successfully</h4>
                <p style="margin: 0;">📊 {spatial_data.shape[0]} points with coordinates</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display basic spatial statistics
            st.subheader("📊 Spatial Data Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Points", len(spatial_data))
                st.metric("Mean AGB", f"{spatial_data['AGB_L4A'].mean():.2f}")
            
            with col2:
                st.metric("Min AGB", f"{spatial_data['AGB_L4A'].min():.2f}")
                st.metric("Max AGB", f"{spatial_data['AGB_L4A'].max():.2f}")
            
            with col3:
                st.metric("Std AGB", f"{spatial_data['AGB_L4A'].std():.2f}")
                st.metric("Coverage Area", f"{(spatial_data['Longitude_gedi'].max() - spatial_data['Longitude_gedi'].min()) * (spatial_data['Latitude_gedi'].max() - spatial_data['Latitude_gedi'].min()):.4f}°²")
            
            # Spatial Analysis Controls
            st.subheader("🔧 Spatial Analysis Controls")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_clusters = st.slider("Number of Clusters", 2, 10, 5)
                interpolation_method = st.selectbox("Interpolation Method", ["idw", "nearest"])
            
            with col2:
                grid_resolution = st.slider("Grid Resolution", 50, 200, 100)
                analysis_type = st.multiselect(
                    "Select Analysis Types",
                    ["Geographic Clustering", "Spatial Autocorrelation", "Hotspot Analysis", "Spatial Interpolation"],
                    default=["Geographic Clustering", "Spatial Autocorrelation"]
                )
            
            # Run selected analyses
            if st.button("🚀 Run Spatial Analysis", type="primary"):
                with st.spinner("Performing spatial analysis..."):
                    
                    cluster_data = None
                    cluster_stats = None
                    autocorr_results = None
                    hotspot_data = None
                    hotspot_stats = None
                    interpolation_data = None
                    
                    # Geographic Clustering
                    if "Geographic Clustering" in analysis_type:
                        cluster_data, cluster_stats = dashboard.perform_geographic_clustering(spatial_data, n_clusters)
                    
                    # Spatial Autocorrelation
                    if "Spatial Autocorrelation" in analysis_type:
                        autocorr_results = dashboard.calculate_spatial_autocorrelation(spatial_data)
                    
                    # Hotspot Analysis
                    if "Hotspot Analysis" in analysis_type:
                        hotspot_data, hotspot_stats = dashboard.perform_hotspot_analysis(spatial_data)
                    
                    # Spatial Interpolation
                    if "Spatial Interpolation" in analysis_type:
                        interpolation_data = dashboard.perform_spatial_interpolation(spatial_data, interpolation_method, grid_resolution)
                    
                    # Store results in session state
                    st.session_state.spatial_results = {
                        'cluster_data': cluster_data,
                        'cluster_stats': cluster_stats,
                        'autocorr_results': autocorr_results,
                        'hotspot_data': hotspot_data,
                        'hotspot_stats': hotspot_stats,
                        'interpolation_data': interpolation_data
                    }
            
            # Display results
            if 'spatial_results' in st.session_state:
                results = st.session_state.spatial_results
                
                # Create visualizations
                fig_scatter, fig_clusters, fig_hotspots, fig_interpolation = dashboard.create_spatial_visualizations(
                    spatial_data, 
                    results['cluster_data'], 
                    results['hotspot_data'], 
                    results['interpolation_data']
                )
                
                # Basic spatial distribution
                st.subheader("🗺️ Spatial Distribution")
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Geographic Clustering Results
                if results['cluster_data'] is not None:
                    st.subheader("🎯 Geographic Clustering Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.plotly_chart(fig_clusters, use_container_width=True)
                    
                    with col2:
                        st.write("**Cluster Statistics:**")
                        if results['cluster_stats'] is not None:
                            st.dataframe(results['cluster_stats'], use_container_width=True)
                
                # Spatial Autocorrelation Results
                if results['autocorr_results'] is not None:
                    st.subheader("📊 Spatial Autocorrelation Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Moran's I", 
                            f"{results['autocorr_results']['Morans_I']:.4f}",
                            help=results['autocorr_results']['Interpretation_Morans_I']
                        )
                    
                    with col2:
                        st.metric(
                            "Geary's C", 
                            f"{results['autocorr_results']['Gearys_C']:.4f}",
                            help=results['autocorr_results']['Interpretation_Gearys_C']
                        )
                    
                    st.info(f"**Interpretation:** {results['autocorr_results']['Interpretation_Morans_I']}")
                
                # Hotspot Analysis Results
                if results['hotspot_data'] is not None:
                    st.subheader("🔥 Hotspot Analysis Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.plotly_chart(fig_hotspots, use_container_width=True)
                    
                    with col2:
                        st.write("**Hotspot Statistics:**")
                        if results['hotspot_stats'] is not None:
                            st.dataframe(results['hotspot_stats'], use_container_width=True)
                
                # Spatial Interpolation Results
                if results['interpolation_data'] is not None:
                    st.subheader("🗺️ Spatial Interpolation Results")
                    st.plotly_chart(fig_interpolation, use_container_width=True)
                    
                    st.info(f"**Method:** {results['interpolation_data']['method'].upper()}")
                    st.info(f"**Grid Resolution:** {grid_resolution}x{grid_resolution}")
                
                # Spatial Insights
                st.subheader("🧠 Spatial Analysis Insights")
                
                insights = []
                
                if results['cluster_data'] is not None:
                    insights.append("🎯 **Geographic Clustering:** Identified distinct spatial regions with similar AGB characteristics")
                
                if results['autocorr_results'] is not None:
                    if results['autocorr_results']['Morans_I'] > 0.3:
                        insights.append("📊 **Strong Spatial Autocorrelation:** AGB values show significant spatial clustering")
                    elif results['autocorr_results']['Morans_I'] > 0:
                        insights.append("📊 **Moderate Spatial Autocorrelation:** Some spatial clustering in AGB values")
                    else:
                        insights.append("📊 **Weak Spatial Autocorrelation:** AGB values are randomly distributed")
                
                if results['hotspot_data'] is not None:
                    insights.append("🔥 **Hotspot Detection:** Identified areas with unusually high or low AGB concentrations")
                
                if results['interpolation_data'] is not None:
                    insights.append("🗺️ **Spatial Interpolation:** Created continuous surface maps for AGB estimation")
                
                for insight in insights:
                    st.write(insight)
        
        else:
            st.error("❌ Could not load spatial data. Please ensure the spatial data file is available.")
    
    elif selected_section == "📋 Data Overview":
        st.markdown('<h1 class="section-header">📋 Data Overview</h1>', unsafe_allow_html=True)
        
        # Dataset info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", data.shape[0])
        with col2:
            st.metric("Total Features", data.shape[1] - 1)
        with col3:
            st.metric("Target Variable", "AGB_2017")
        with col4:
            st.metric("Missing Values", data.isnull().sum().sum())
        
        # Data preview
        st.subheader("📊 Data Preview")
        st.dataframe(data.head(20), use_container_width=True)
        
        # Statistical summary
        st.subheader("📈 Statistical Summary")
        st.dataframe(data.describe(), use_container_width=True)
        
        # Target distribution
        st.subheader("🎯 AGB Target Distribution")
        fig_dist = px.histogram(
            data, 
            x='AGB_2017',
            title="Distribution of AGB Values",
            labels={'AGB_2017': 'AGB Value', 'count': 'Frequency'}
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Feature distributions
        st.subheader("📊 Feature Distributions")
        selected_features = st.multiselect(
            "Select features to visualize",
            options=[col for col in data.columns if col != 'AGB_2017'],
            default=[col for col in data.columns if col != 'AGB_2017'][:5]
        )
        
        if selected_features:
            fig_features = px.box(
                data[selected_features],
                title="Feature Distributions (Box Plots)"
            )
            st.plotly_chart(fig_features, use_container_width=True)
    
    # Footer with enhanced styling
    st.markdown("""
    <div class="footer">
        <div style="margin-bottom: 1rem;">
            <span style="font-size: 1.5rem;">🌲</span>
            <span style="font-weight: 600; color: #2E8B57; font-size: 1.2rem;">BioVision Analytics Hub</span>
        </div>
        <p style="color: #666; margin: 0.5rem 0;">
            Powered by <strong>Streamlit</strong> & <strong>Plotly</strong>
        </p>
        <p style="color: #888; margin: 0.5rem 0; font-style: italic;">
            Transforming Biomass Data into Actionable Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Signature with enhanced styling
    st.markdown("""
    <div class="signature">
        <h3>Masood Nazari</h3>
        <p>AI Engineer | Data Engineer | Data Science Enthusiast</p>
        <p>📍 Southampton, UK | 🔗 <a href='https://www.linkedin.com/in/masood-nazari' target='_blank'>www.linkedin.com/in/masood-nazari</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
