"""
🚀 TimeGPT Advanced Forecasting Dashboard
Professional Time Series Forecasting with Nixtla Ecosystem

Author: Pablo Poletti
GitHub: https://github.com/PabloPoletti
Contact: lic.poletti@gmail.com
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple, Optional
import logging

# Nixtla ecosystem imports
try:
    from nixtla import NixtlaClient
    from statsforecast import StatsForecast
    from statsforecast.models import (
        AutoARIMA, AutoETS, AutoCES, AutoTheta,
        SeasonalNaive, Naive, RandomWalkWithDrift,
        CrostonClassic, IMAPA, TSB, ADIDA
    )
    from neuralforecast import NeuralForecast
    from neuralforecast.models import (
        NBEATS, NHITS, TFT, PatchTST, TimesNet,
        MLP, RNN, LSTM, GRU, TCN, DilatedRNN,
        DeepAR, InformerTST
    )
    from utilsforecast.losses import mse, mae, mape, smape
    from utilsforecast.evaluation import evaluate
except ImportError as e:
    st.error(f"Error importing Nixtla libraries: {e}")
    st.stop()

# ML and optimization
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# Page config
st.set_page_config(
    page_title="TimeGPT Advanced Forecasting",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .forecast-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants
COLORS = {
    'primary': '#FF6B6B',
    'secondary': '#4ECDC4', 
    'accent': '#45B7D1',
    'success': '#96CEB4',
    'warning': '#FECA57',
    'error': '#FF9FF3'
}

@st.cache_data
def load_sample_data() -> Dict[str, pd.DataFrame]:
    """Load sample time series datasets for demonstration"""
    
    # Generate multiple realistic time series
    datasets = {}
    
    # 1. Economic indicators (monthly)
    dates_monthly = pd.date_range('2015-01-01', '2024-12-01', freq='MS')
    np.random.seed(42)
    
    # GDP-like series with trend and seasonality
    trend = np.linspace(100, 150, len(dates_monthly))
    seasonal = 5 * np.sin(2 * np.pi * np.arange(len(dates_monthly)) / 12)
    noise = np.random.normal(0, 2, len(dates_monthly))
    gdp = trend + seasonal + noise
    
    datasets['Economic_GDP'] = pd.DataFrame({
        'ds': dates_monthly,
        'y': gdp,
        'unique_id': 'GDP'
    })
    
    # 2. Sales data (daily)
    dates_daily = pd.date_range('2020-01-01', '2024-12-01', freq='D')
    np.random.seed(123)
    
    # Sales with weekly and yearly patterns
    trend_daily = np.linspace(1000, 2000, len(dates_daily))
    weekly_pattern = 200 * np.sin(2 * np.pi * np.arange(len(dates_daily)) / 7)
    yearly_pattern = 300 * np.sin(2 * np.pi * np.arange(len(dates_daily)) / 365.25)
    noise_daily = np.random.normal(0, 50, len(dates_daily))
    sales = trend_daily + weekly_pattern + yearly_pattern + noise_daily
    sales = np.maximum(sales, 100)  # Ensure positive values
    
    datasets['Sales_Daily'] = pd.DataFrame({
        'ds': dates_daily,
        'y': sales,
        'unique_id': 'Sales'
    })
    
    # 3. Energy consumption (hourly sample)
    dates_hourly = pd.date_range('2023-01-01', '2024-06-01', freq='H')
    np.random.seed(456)
    
    # Energy with daily and seasonal patterns
    hour_of_day = dates_hourly.hour
    day_of_year = dates_hourly.dayofyear
    
    base_consumption = 50
    daily_pattern = 20 * np.sin(2 * np.pi * hour_of_day / 24)
    seasonal_pattern = 15 * np.sin(2 * np.pi * day_of_year / 365.25)
    weekend_effect = -5 * (dates_hourly.weekday >= 5).astype(int)
    noise_hourly = np.random.normal(0, 3, len(dates_hourly))
    
    energy = base_consumption + daily_pattern + seasonal_pattern + weekend_effect + noise_hourly
    energy = np.maximum(energy, 10)
    
    # Take sample for performance
    sample_size = min(2000, len(energy))
    idx = np.linspace(0, len(energy)-1, sample_size).astype(int)
    
    datasets['Energy_Hourly'] = pd.DataFrame({
        'ds': dates_hourly[idx],
        'y': energy[idx],
        'unique_id': 'Energy'
    })
    
    return datasets

def create_ensemble_models() -> Dict[str, List]:
    """Create ensemble of statistical and neural forecasting models"""
    
    # Statistical models (StatsForecast)
    statistical_models = [
        AutoARIMA(season_length=12),
        AutoETS(season_length=12),
        AutoTheta(season_length=12),
        AutoCES(season_length=12),
        SeasonalNaive(season_length=12),
        RandomWalkWithDrift(),
    ]
    
    # Neural models (NeuralForecast)
    neural_models = [
        NBEATS(input_size=24, h=12, max_steps=100),
        NHITS(input_size=24, h=12, max_steps=100),
        MLP(input_size=24, h=12, max_steps=100),
        RNN(input_size=24, h=12, max_steps=100),
        LSTM(input_size=24, h=12, max_steps=100),
        TCN(input_size=24, h=12, max_steps=100),
    ]
    
    return {
        'statistical': statistical_models,
        'neural': neural_models
    }

def optimize_hyperparameters(data: pd.DataFrame, model_type: str) -> Dict:
    """Optimize hyperparameters using Optuna"""
    
    def objective(trial):
        if model_type == 'AutoARIMA':
            # AutoARIMA hyperparameters
            max_p = trial.suggest_int('max_p', 1, 5)
            max_q = trial.suggest_int('max_q', 1, 5)
            max_d = trial.suggest_int('max_d', 0, 2)
            
            model = AutoARIMA(
                season_length=12,
                max_p=max_p,
                max_q=max_q,
                max_d=max_d
            )
        
        elif model_type == 'NBEATS':
            # NBEATS hyperparameters
            input_size = trial.suggest_categorical('input_size', [12, 24, 36])
            hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512])
            
            model = NBEATS(
                input_size=input_size,
                h=12,
                hidden_size=hidden_size,
                max_steps=50  # Reduced for optimization
            )
        
        # Perform time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        errors = []
        
        for train_idx, val_idx in tscv.split(data):
            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]
            
            try:
                if model_type == 'AutoARIMA':
                    sf = StatsForecast(models=[model], freq='MS')
                    forecasts = sf.forecast(df=train_data, h=len(val_data))
                    predictions = forecasts['AutoARIMA'].values
                else:
                    nf = NeuralForecast(models=[model], freq='MS')
                    forecasts = nf.fit(train_data).predict()
                    predictions = forecasts['NBEATS'].values
                
                error = mean_absolute_error(val_data['y'].values, predictions)
                errors.append(error)
                
            except Exception:
                return float('inf')
        
        return np.mean(errors)
    
    # Run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20, show_progress_bar=False)
    
    return study.best_params

def perform_cross_validation(data: pd.DataFrame, models: List, n_splits: int = 5) -> pd.DataFrame:
    """Perform time series cross-validation"""
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(data)):
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]
        
        for model in models:
            try:
                model_name = model.__class__.__name__
                
                # Fit and predict based on model type
                if hasattr(model, 'forecast'):  # Statistical model
                    sf = StatsForecast(models=[model], freq='MS')
                    forecasts = sf.forecast(df=train_data, h=len(val_data))
                    predictions = forecasts[model_name].values
                else:  # Neural model
                    nf = NeuralForecast(models=[model], freq='MS')
                    forecasts = nf.fit(train_data).predict()
                    predictions = forecasts[model_name].values
                
                # Calculate metrics
                mae_score = mean_absolute_error(val_data['y'].values, predictions)
                mse_score = mean_squared_error(val_data['y'].values, predictions)
                mape_score = np.mean(np.abs((val_data['y'].values - predictions) / val_data['y'].values)) * 100
                
                results.append({
                    'Fold': fold + 1,
                    'Model': model_name,
                    'MAE': mae_score,
                    'MSE': mse_score,
                    'MAPE': mape_score
                })
                
            except Exception as e:
                st.warning(f"Error with {model.__class__.__name__}: {str(e)}")
                continue
    
    return pd.DataFrame(results)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 TimeGPT Advanced Forecasting</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="forecast-info">
    🎯 <strong>Professional Time Series Forecasting with Nixtla Ecosystem</strong><br>
    Featuring TimeGPT, StatsForecast, NeuralForecast with advanced optimization techniques
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Configuration")
        
        # Dataset selection
        datasets = load_sample_data()
        dataset_name = st.selectbox("📊 Select Dataset:", list(datasets.keys()))
        data = datasets[dataset_name]
        
        # Forecasting parameters
        st.markdown("### 📈 Forecasting Setup")
        forecast_horizon = st.slider("🔮 Forecast Horizon:", 6, 48, 12)
        confidence_level = st.slider("📊 Confidence Level:", 80, 99, 95)
        
        # Model selection
        st.markdown("### 🤖 Model Selection")
        use_timegpt = st.checkbox("🚀 Use TimeGPT (requires API key)", value=False)
        use_statistical = st.checkbox("📊 Statistical Models", value=True)
        use_neural = st.checkbox("🧠 Neural Models", value=True)
        use_ensemble = st.checkbox("🎭 Ensemble Methods", value=True)
        
        # Advanced options
        st.markdown("### ⚙️ Advanced Options")
        optimize_hyperparams = st.checkbox("🔧 Hyperparameter Optimization", value=False)
        cross_validation = st.checkbox("✅ Cross Validation", value=True)
        
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Data overview
        st.subheader("📊 Data Overview")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['ds'], 
            y=data['y'],
            mode='lines',
            name='Historical Data',
            line=dict(color=COLORS['primary'], width=2)
        ))
        
        fig.update_layout(
            title=f"Time Series: {dataset_name}",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data statistics
        st.markdown("### 📈 Data Statistics")
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.metric("📊 Data Points", len(data))
        with stats_col2:
            st.metric("📅 Date Range", f"{data['ds'].dt.date.min()} to {data['ds'].dt.date.max()}")
        with stats_col3:
            st.metric("📈 Mean Value", f"{data['y'].mean():.2f}")
        with stats_col4:
            st.metric("📊 Std Deviation", f"{data['y'].std():.2f}")
    
    with col2:
        # Model performance summary
        st.subheader("🏆 Model Performance")
        
        if cross_validation and (use_statistical or use_neural):
            with st.spinner("Running cross-validation..."):
                models = []
                
                if use_statistical:
                    stat_models = create_ensemble_models()['statistical'][:3]  # Limit for performance
                    models.extend(stat_models)
                
                if use_neural:
                    neural_models = create_ensemble_models()['neural'][:2]  # Limit for performance
                    models.extend(neural_models)
                
                if models:
                    cv_results = perform_cross_validation(data, models, n_splits=3)
                    
                    # Show average performance
                    avg_performance = cv_results.groupby('Model').agg({
                        'MAE': 'mean',
                        'MSE': 'mean', 
                        'MAPE': 'mean'
                    }).round(3)
                    
                    st.dataframe(avg_performance, height=200)
        
        # Quick insights
        st.markdown("### 💡 Quick Insights")
        
        # Trend analysis
        recent_trend = data['y'].tail(12).mean() - data['y'].head(12).mean()
        if recent_trend > 0:
            st.success(f"📈 Upward trend: +{recent_trend:.2f}")
        else:
            st.error(f"📉 Downward trend: {recent_trend:.2f}")
        
        # Seasonality detection
        if len(data) > 24:
            seasonal_strength = np.std(data['y'].rolling(12).mean().dropna())
            st.info(f"🔄 Seasonality strength: {seasonal_strength:.2f}")
        
        # Volatility
        volatility = data['y'].std()
        st.warning(f"📊 Volatility: {volatility:.2f}")
    
    # Forecasting section
    if st.button("🚀 Generate Forecasts", type="primary"):
        st.markdown("---")
        st.subheader("🔮 Forecasting Results")
        
        with st.spinner("Generating forecasts..."):
            forecasts_dict = {}
            
            # Statistical forecasting
            if use_statistical:
                try:
                    stat_models = [AutoARIMA(season_length=12), AutoETS(season_length=12)]
                    sf = StatsForecast(models=stat_models, freq='MS')
                    stat_forecasts = sf.forecast(df=data, h=forecast_horizon, level=[confidence_level])
                    
                    for model in stat_models:
                        model_name = model.__class__.__name__
                        forecasts_dict[f"📊 {model_name}"] = stat_forecasts[model_name].values
                        
                except Exception as e:
                    st.error(f"Statistical forecasting error: {e}")
            
            # Neural forecasting  
            if use_neural:
                try:
                    neural_models = [NBEATS(input_size=12, h=forecast_horizon, max_steps=50)]
                    nf = NeuralForecast(models=neural_models, freq='MS')
                    neural_forecasts = nf.fit(data).predict()
                    
                    for model in neural_models:
                        model_name = model.__class__.__name__
                        forecasts_dict[f"🧠 {model_name}"] = neural_forecasts[model_name].values
                        
                except Exception as e:
                    st.error(f"Neural forecasting error: {e}")
            
            # Ensemble forecasting
            if use_ensemble and len(forecasts_dict) > 1:
                ensemble_forecast = np.mean(list(forecasts_dict.values()), axis=0)
                forecasts_dict["🎭 Ensemble"] = ensemble_forecast
            
            # Plot forecasts
            if forecasts_dict:
                future_dates = pd.date_range(
                    start=data['ds'].max() + pd.DateOffset(months=1),
                    periods=forecast_horizon,
                    freq='MS'
                )
                
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=data['ds'],
                    y=data['y'],
                    mode='lines',
                    name='Historical',
                    line=dict(color=COLORS['primary'], width=2)
                ))
                
                # Forecasts
                colors = [COLORS['secondary'], COLORS['accent'], COLORS['success'], COLORS['warning']]
                for i, (name, forecast) in enumerate(forecasts_dict.items()):
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=forecast,
                        mode='lines+markers',
                        name=name,
                        line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                    ))
                
                fig.update_layout(
                    title="📈 Time Series Forecasting Results",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    template="plotly_white",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast table
                st.subheader("📋 Forecast Values")
                forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    **{name: values for name, values in forecasts_dict.items()}
                })
                st.dataframe(forecast_df, hide_index=True)
                
                # Download option
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecasts",
                    data=csv,
                    file_name=f"timegpt_forecasts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    🚀 <strong>TimeGPT Advanced Forecasting</strong> | 
    Built with Nixtla ecosystem | 
    <a href="https://github.com/PabloPoletti" target="_blank">GitHub</a> | 
    <a href="mailto:lic.poletti@gmail.com">Contact</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
