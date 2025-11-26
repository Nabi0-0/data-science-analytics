"""
Demand Forecasting Module
Time series forecasting using exponential smoothing and statistical models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class DemandForecaster:
    """Handles demand forecasting for inventory products"""
    
    def __init__(self, sales_df):
        self.sales_df = sales_df.copy()
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare sales data for forecasting"""
        # Ensure datetime format
        if 'SalesDate' in self.sales_df.columns:
            self.sales_df['SalesDate'] = pd.to_datetime(self.sales_df['SalesDate'], errors='coerce')
            self.sales_df = self.sales_df.dropna(subset=['SalesDate'])
    
    def aggregate_daily_sales(self, product_name=None, store=None):
        """Aggregate sales by day for a product or store"""
        df = self.sales_df.copy()
        
        if product_name:
            df = df[df['Brand'] == product_name]
        if store:
            df = df[df['Store'] == store]
        
        # Group by date
        daily_sales = df.groupby('SalesDate').agg({
            'SalesQuantity': 'sum',
            'SalesDollars': 'sum'
        }).reset_index()
        
        # Sort by date
        daily_sales = daily_sales.sort_values('SalesDate')
        
        return daily_sales
    
    def simple_moving_average(self, data, window=7):
        """Calculate simple moving average"""
        return data.rolling(window=window, min_periods=1).mean()
    
    def exponential_smoothing(self, data, alpha=0.3):
        """Simple exponential smoothing"""
        result = [data.iloc[0]]
        for i in range(1, len(data)):
            result.append(alpha * data.iloc[i] + (1 - alpha) * result[-1])
        return pd.Series(result, index=data.index)
    
    def double_exponential_smoothing(self, data, alpha=0.3, beta=0.1):
        """Double exponential smoothing (Holt's method) for trend"""
        level = [data.iloc[0]]
        trend = [0]
        
        for i in range(1, len(data)):
            level_new = alpha * data.iloc[i] + (1 - alpha) * (level[-1] + trend[-1])
            trend_new = beta * (level_new - level[-1]) + (1 - beta) * trend[-1]
            level.append(level_new)
            trend.append(trend_new)
        
        return pd.Series(level, index=data.index), pd.Series(trend, index=data.index)
    
    def forecast_simple(self, historical_data, periods=30):
        """Simple forecast using exponential smoothing"""
        # Use double exponential smoothing
        level, trend = self.double_exponential_smoothing(historical_data['SalesQuantity'])
        
        # Forecast
        last_level = level.iloc[-1]
        last_trend = trend.iloc[-1]
        
        forecasts = []
        for i in range(1, periods + 1):
            forecast = last_level + i * last_trend
            forecasts.append(max(0, forecast))  # Ensure non-negative
        
        return forecasts
    
    def calculate_forecast_accuracy(self, actual, predicted):
        """Calculate forecast accuracy metrics"""
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Remove any NaN or infinite values
        mask = ~(np.isnan(actual) | np.isnan(predicted) | np.isinf(actual) | np.isinf(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0:
            return {'mae': None, 'rmse': None, 'mape': None}
        
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        # MAPE (avoiding division by zero)
        mape = np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, 1))) * 100
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        }
    
    def forecast_product(self, product_name, periods=30):
        """Forecast demand for a specific product"""
        # Get historical data
        daily_sales = self.aggregate_daily_sales(product_name=product_name)
        
        if len(daily_sales) < 7:  # Need at least a week of data
            return None
        
        # Generate forecast
        forecasts = self.forecast_simple(daily_sales, periods=periods)
        
        # Calculate accuracy on last 30% of historical data
        test_size = max(1, int(len(daily_sales) * 0.3))
        train_data = daily_sales.iloc[:-test_size]
        test_data = daily_sales.iloc[-test_size:]
        
        if len(train_data) > 0:
            test_forecast = self.forecast_simple(train_data, periods=len(test_data))
            accuracy = self.calculate_forecast_accuracy(
                test_data['SalesQuantity'].values,
                test_forecast
            )
        else:
            accuracy = {'mae': None, 'rmse': None, 'mape': None}
        
        # Create forecast dates
        last_date = daily_sales['SalesDate'].max()
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]
        
        # Calculate statistics
        avg_daily_sales = daily_sales['SalesQuantity'].mean()
        std_daily_sales = daily_sales['SalesQuantity'].std()
        
        return {
            'product': product_name,
            'historical_days': len(daily_sales),
            'avg_daily_sales': float(avg_daily_sales),
            'std_daily_sales': float(std_daily_sales),
            'forecast_periods': periods,
            'forecast_values': forecasts,
            'forecast_dates': [d.strftime('%Y-%m-%d') for d in forecast_dates],
            'accuracy_metrics': accuracy,
            'total_forecast_demand': float(sum(forecasts)),
            'confidence_interval_95': {
                'lower': [max(0, f - 1.96 * std_daily_sales) for f in forecasts],
                'upper': [f + 1.96 * std_daily_sales for f in forecasts]
            }
        }
    
    def forecast_category(self, category, periods=30):
        """Forecast demand for a product category"""
        df = self.sales_df[self.sales_df['Classification'] == category].copy()
        
        if len(df) == 0:
            return None
        
        # Aggregate by date
        daily_sales = df.groupby('SalesDate').agg({
            'SalesQuantity': 'sum',
            'SalesDollars': 'sum'
        }).reset_index()
        
        daily_sales = daily_sales.sort_values('SalesDate')
        
        if len(daily_sales) < 7:
            return None
        
        # Generate forecast
        forecasts = self.forecast_simple(daily_sales, periods=periods)
        
        last_date = daily_sales['SalesDate'].max()
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]
        
        return {
            'category': category,
            'forecast_periods': periods,
            'forecast_values': forecasts,
            'forecast_dates': [d.strftime('%Y-%m-%d') for d in forecast_dates],
            'total_forecast_demand': float(sum(forecasts))
        }
    
    def get_seasonal_patterns(self, product_name):
        """Detect seasonal patterns in product sales"""
        daily_sales = self.aggregate_daily_sales(product_name=product_name)
        
        if len(daily_sales) < 28:  # Need at least 4 weeks
            return None
        
        # Add day of week
        daily_sales['DayOfWeek'] = daily_sales['SalesDate'].dt.dayofweek
        daily_sales['WeekOfYear'] = daily_sales['SalesDate'].dt.isocalendar().week
        
        # Average sales by day of week
        dow_pattern = daily_sales.groupby('DayOfWeek')['SalesQuantity'].mean().to_dict()
        
        # Convert day numbers to names
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_pattern_named = {day_names[k]: float(v) for k, v in dow_pattern.items() if k < 7}
        
        return {
            'product': product_name,
            'day_of_week_pattern': dow_pattern_named,
            'peak_day': day_names[max(dow_pattern, key=dow_pattern.get)],
            'lowest_day': day_names[min(dow_pattern, key=dow_pattern.get)]
        }


if __name__ == '__main__':
    # Example usage
    print("Demand Forecasting Module")
    print("This module provides time series forecasting capabilities")
    print("Import and use with: from forecasting import DemandForecaster")