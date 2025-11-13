"""
forecasting.py
Demand forecasting using SARIMA and other time series models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from utils import (
    print_section_header,
    save_plot,
    save_dataframe,
    log_analysis_step
)

class DemandForecaster:
    """
    Demand forecasting using multiple time series methods
    """
    
    def __init__(self, sales_df):
        """Initialize with sales dataframe"""
        self.sales_df = sales_df
        self.forecast_results = {}
        
    def prepare_time_series(self, date_col='date', value_col='quantity', 
                           product_filter=None, freq='D'):
        """
        Prepare time series data for forecasting
        """
        df = self.sales_df.copy()
        
        # Filter by product if specified
        if product_filter:
            df = df[df['product_id'] == product_filter]
        
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Group by date and sum quantities
        ts_data = df.groupby(date_col)[value_col].sum().reset_index()
        ts_data = ts_data.set_index(date_col)
        
        # Resample to ensure consistent frequency
        ts_data = ts_data.resample(freq).sum()
        
        # Fill missing values
        ts_data = ts_data.fillna(0)
        
        return ts_data
    
    def decompose_time_series(self, ts_data, period=7, model='additive'):
        """
        Decompose time series into trend, seasonal, and residual components
        """
        print_section_header("📉 TIME SERIES DECOMPOSITION")
        
        if len(ts_data) < 2 * period:
            print(f"⚠️  Insufficient data for decomposition (need at least {2*period} periods)")
            return None
        
        try:
            decomposition = seasonal_decompose(
                ts_data.iloc[:, 0] if isinstance(ts_data, pd.DataFrame) else ts_data,
                model=model,
                period=period
            )
            
            # Plot decomposition
            fig, axes = plt.subplots(4, 1, figsize=(14, 10))
            
            decomposition.observed.plot(ax=axes[0], color='#3b82f6')
            axes[0].set_ylabel('Observed')
            axes[0].set_title('Time Series Decomposition')
            axes[0].grid(True, alpha=0.3)
            
            decomposition.trend.plot(ax=axes[1], color='#10b981')
            axes[1].set_ylabel('Trend')
            axes[1].grid(True, alpha=0.3)
            
            decomposition.seasonal.plot(ax=axes[2], color='#f59e0b')
            axes[2].set_ylabel('Seasonal')
            axes[2].grid(True, alpha=0.3)
            
            decomposition.resid.plot(ax=axes[3], color='#ef4444')
            axes[3].set_ylabel('Residual')
            axes[3].set_xlabel('Date')
            axes[3].grid(True, alpha=0.3)
            
            save_plot('time_series_decomposition.png')
            
            print("✅ Time series decomposition complete")
            
            return decomposition
            
        except Exception as e:
            print(f"⚠️  Decomposition failed: {e}")
            return None
    
    def forecast_sarima(self, ts_data, order=(1,1,1), seasonal_order=(1,1,1,7), 
                       forecast_periods=180, test_size=0.2):
        """
        Forecast using SARIMA model
        """
        print_section_header("📈 SARIMA DEMAND FORECASTING")
        
        # Prepare data
        if isinstance(ts_data, pd.DataFrame):
            ts_data = ts_data.iloc[:, 0]
        
        # Remove any infinite or NaN values
        ts_data = ts_data.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Split into train and test
        split_point = int(len(ts_data) * (1 - test_size))
        train = ts_data[:split_point]
        test = ts_data[split_point:]
        
        print(f"\n📊 Dataset Split:")
        print(f"   Training: {len(train)} periods ({train.index[0]} to {train.index[-1]})")
        print(f"   Testing: {len(test)} periods ({test.index[0]} to {test.index[-1]})")
        print(f"   Forecast horizon: {forecast_periods} periods")
        
        try:
            # Fit SARIMA model
            log_analysis_step("Fitting SARIMA model", f"Order: {order}, Seasonal: {seasonal_order}")
            
            model = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            results = model.fit(disp=False, maxiter=200)
            
            # Forecast on test set
            test_forecast = results.forecast(steps=len(test))
            
            # Calculate metrics
            mae = mean_absolute_error(test, test_forecast)
            rmse = np.sqrt(mean_squared_error(test, test_forecast))
            mape = mean_absolute_percentage_error(test, test_forecast) * 100
            
            print(f"\n📊 Model Performance on Test Set:")
            print(f"   MAE (Mean Absolute Error): {mae:.2f}")
            print(f"   RMSE (Root Mean Squared Error): {rmse:.2f}")
            print(f"   MAPE (Mean Absolute % Error): {mape:.2f}%")
            
            # Future forecast
            future_forecast = results.forecast(steps=forecast_periods)
            forecast_index = pd.date_range(
                start=ts_data.index[-1] + timedelta(days=1),
                periods=forecast_periods,
                freq='D'
            )
            
            # Get confidence intervals
            pred = results.get_prediction(
                start=len(train),
                end=len(train) + forecast_periods - 1
            )
            pred_ci = pred.conf_int()
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'date': forecast_index,
                'forecast': future_forecast.values,
                'lower_bound': pred_ci.iloc[-forecast_periods:, 0].values,
                'upper_bound': pred_ci.iloc[-forecast_periods:, 1].values
            })
            
            # Ensure non-negative forecasts
            forecast_df['forecast'] = forecast_df['forecast'].clip(lower=0)
            forecast_df['lower_bound'] = forecast_df['lower_bound'].clip(lower=0)
            forecast_df['upper_bound'] = forecast_df['upper_bound'].clip(lower=0)
            
            # Plot results
            self._plot_forecast_results(
                train, test, test_forecast, 
                forecast_index, future_forecast, pred_ci,
                forecast_periods
            )
            
            # Save results
            save_dataframe(forecast_df, 'demand_forecast_sarima.csv')
            
            # Store results
            self.forecast_results['sarima'] = {
                'model': results,
                'forecast': forecast_df,
                'metrics': {'mae': mae, 'rmse': rmse, 'mape': mape}
            }
            
            print(f"\n✅ SARIMA forecasting complete")
            
            return forecast_df
            
        except Exception as e:
            print(f"\n❌ SARIMA model failed: {e}")
            print("Falling back to simple moving average...")
            return self._fallback_forecast(ts_data, forecast_periods)
    
    def forecast_exponential_smoothing(self, ts_data, forecast_periods=180, 
                                      seasonal_periods=7):
        """
        Forecast using Exponential Smoothing (Holt-Winters)
        """
        print_section_header("📈 EXPONENTIAL SMOOTHING FORECASTING")
        
        if isinstance(ts_data, pd.DataFrame):
            ts_data = ts_data.iloc[:, 0]
        
        # Remove any infinite or NaN values
        ts_data = ts_data.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        try:
            # Fit model
            model = ExponentialSmoothing(
                ts_data,
                seasonal_periods=seasonal_periods,
                trend='add',
                seasonal='add',
                use_boxcox=False
            )
            
            results = model.fit()
            
            # Forecast
            forecast = results.forecast(steps=forecast_periods)
            forecast_index = pd.date_range(
                start=ts_data.index[-1] + timedelta(days=1),
                periods=forecast_periods,
                freq='D'
            )
            
            forecast_df = pd.DataFrame({
                'date': forecast_index,
                'forecast': forecast.values
            })
            
            # Ensure non-negative
            forecast_df['forecast'] = forecast_df['forecast'].clip(lower=0)
            
            save_dataframe(forecast_df, 'demand_forecast_exp_smoothing.csv')
            
            self.forecast_results['exp_smoothing'] = {
                'model': results,
                'forecast': forecast_df
            }
            
            print(f"✅ Exponential smoothing forecasting complete")
            
            return forecast_df
            
        except Exception as e:
            print(f"⚠️  Exponential smoothing failed: {e}")
            return None
    
    def _fallback_forecast(self, ts_data, forecast_periods):
        """
        Simple moving average forecast as fallback
        """
        if isinstance(ts_data, pd.DataFrame):
            ts_data = ts_data.iloc[:, 0]
        
        # Calculate moving average
        ma_window = min(30, len(ts_data) // 4)
        forecast_value = ts_data.rolling(window=ma_window).mean().iloc[-1]
        
        # Create forecast
        forecast_index = pd.date_range(
            start=ts_data.index[-1] + timedelta(days=1),
            periods=forecast_periods,
            freq='D'
        )
        
        forecast_df = pd.DataFrame({
            'date': forecast_index,
            'forecast': [forecast_value] * forecast_periods
        })
        
        save_dataframe(forecast_df, 'demand_forecast_moving_average.csv')
        
        print(f"✅ Moving average forecast complete (MA-{ma_window})")
        
        return forecast_df
    
    def _plot_forecast_results(self, train, test, test_forecast, 
                               forecast_index, future_forecast, pred_ci,
                               forecast_periods):
        """Plot forecasting results"""
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Full view
        ax1 = axes[0]
        ax1.plot(train.index, train.values, label='Training Data', 
                color='#3b82f6', linewidth=2, alpha=0.7)
        ax1.plot(test.index, test.values, label='Actual Test Data', 
                color='#10b981', linewidth=2, alpha=0.7)
        ax1.plot(test.index, test_forecast, label='Test Forecast', 
                color='#ef4444', linewidth=2, linestyle='--', alpha=0.7)
        ax1.plot(forecast_index, future_forecast, label='Future Forecast', 
                color='#f59e0b', linewidth=2, linestyle='--', alpha=0.7)
        
        # Confidence interval
        ax1.fill_between(
            forecast_index,
            pred_ci.iloc[-forecast_periods:, 0],
            pred_ci.iloc[-forecast_periods:, 1],
            alpha=0.2,
            color='#f59e0b',
            label='95% Confidence Interval'
        )
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Quantity')
        ax1.set_title('Demand Forecasting - Full Timeline')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Zoomed view (last 90 days + forecast)
        ax2 = axes[1]
        recent_train = train[-90:] if len(train) > 90 else train
        
        ax2.plot(recent_train.index, recent_train.values, 
                label='Recent History', color='#3b82f6', linewidth=2)
        ax2.plot(test.index, test.values, label='Actual', 
                color='#10b981', linewidth=2)
        ax2.plot(test.index, test_forecast, label='Test Forecast', 
                color='#ef4444', linewidth=2, linestyle='--')
        ax2.plot(forecast_index[:90], future_forecast[:90], 
                label='Future Forecast (90 days)', 
                color='#f59e0b', linewidth=2, linestyle='--')
        
        ax2.fill_between(
            forecast_index[:90],
            pred_ci.iloc[-forecast_periods:][:90].iloc[:, 0],
            pred_ci.iloc[-forecast_periods:][:90].iloc[:, 1],
            alpha=0.2,
            color='#f59e0b'
        )
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Quantity')
        ax2.set_title('Demand Forecasting - Recent & Near-term (90 Days)')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        save_plot('demand_forecast.png')
    
    def analyze_forecast_accuracy(self):
        """Analyze forecast accuracy across models"""
        print_section_header("📊 FORECAST ACCURACY COMPARISON")
        
        if not self.forecast_results:
            print("⚠️  No forecast results available")
            return
        
        comparison_data = []
        for model_name, results in self.forecast_results.items():
            if 'metrics' in results:
                metrics = results['metrics']
                comparison_data.append({
                    'Model': model_name.upper(),
                    'MAE': metrics['mae'],
                    'RMSE': metrics['rmse'],
                    'MAPE (%)': metrics['mape']
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            print("\n", comparison_df.to_string(index=False))
            save_dataframe(comparison_df, 'forecast_accuracy_comparison.csv')
            
            # Determine best model
            best_model = comparison_df.loc[comparison_df['MAPE (%)'].idxmin(), 'Model']
            print(f"\n🏆 Best Model (lowest MAPE): {best_model}")


def main(sales_df):
    """Run demand forecasting"""
    
    forecaster = DemandForecaster(sales_df)
    
    # Prepare time series
    log_analysis_step("Preparing time series data")
    ts_data = forecaster.prepare_time_series()
    
    print(f"   Time series length: {len(ts_data)} days")
    print(f"   Date range: {ts_data.index[0]} to {ts_data.index[-1]}")
    print(f"   Total quantity: {ts_data.values.sum():,.0f}")
    
    # Decompose time series
    forecaster.decompose_time_series(ts_data, period=7)
    
    # SARIMA forecast
    sarima_forecast = forecaster.forecast_sarima(
        ts_data,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        forecast_periods=180
    )
    
    # Exponential smoothing forecast
    exp_forecast = forecaster.forecast_exponential_smoothing(
        ts_data,
        forecast_periods=180,
        seasonal_periods=7
    )
    
    # Compare accuracy
    forecaster.analyze_forecast_accuracy()
    
    return forecaster


if __name__ == "__main__":
    print("Forecasting module loaded")
    print("Run from main.py to execute forecasting")