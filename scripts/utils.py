"""
utils.py
Shared utility functions for Slooze Inventory Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def setup_output_directories():
    """Create output directories for results"""
    import os
    
    directories = ['output', 'output/csv', 'output/png', 'output/reports']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Output directories created")

def save_dataframe(df, filename, output_dir='output/csv'):
    """Save dataframe to CSV with proper formatting"""
    import os
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"✅ Saved: {filepath}")

def save_plot(filename, output_dir='output/png', dpi=300):
    """Save matplotlib plot with consistent formatting"""
    import os
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {filepath}")

def print_section_header(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def calculate_date_metrics(df, date_col='date'):
    """Add temporal features to dataframe"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    return df

def clean_currency_column(series):
    """Clean currency columns (remove $, commas)"""
    if series.dtype == 'object':
        return series.str.replace('$', '', regex=False)\
                     .str.replace(',', '', regex=False)\
                     .astype(float)
    return series

def clean_numeric_column(series):
    """Clean and convert to numeric"""
    return pd.to_numeric(series, errors='coerce')

def get_summary_statistics(df, numeric_columns):
    """Get summary statistics for numeric columns"""
    stats = df[numeric_columns].describe().T
    stats['missing'] = df[numeric_columns].isnull().sum()
    stats['missing_pct'] = (stats['missing'] / len(df)) * 100
    return stats

def identify_outliers(series, method='iqr', threshold=1.5):
    """Identify outliers using IQR or Z-score method"""
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (series < lower_bound) | (series > upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers = z_scores > threshold
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return outliers

def format_currency(value):
    """Format number as currency"""
    return f"${value:,.2f}"

def format_percentage(value):
    """Format number as percentage"""
    return f"{value:.2f}%"

def calculate_growth_rate(current, previous):
    """Calculate growth rate percentage"""
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100

def create_comparison_table(current, optimized, metric_name):
    """Create comparison table for current vs optimized metrics"""
    comparison = pd.DataFrame({
        'Metric': [metric_name],
        'Current': [current],
        'Optimized': [optimized],
        'Difference': [optimized - current],
        'Change %': [calculate_growth_rate(optimized, current)]
    })
    return comparison

def merge_with_validation(left_df, right_df, on, how='inner', validate=None):
    """Merge dataframes with validation and reporting"""
    original_left = len(left_df)
    original_right = len(right_df)
    
    merged = left_df.merge(right_df, on=on, how=how, validate=validate)
    
    print(f"   Merge on: {on}")
    print(f"   Left records: {original_left:,}")
    print(f"   Right records: {original_right:,}")
    print(f"   Merged records: {len(merged):,}")
    
    if how == 'left':
        print(f"   Unmatched left: {merged[on].isnull().sum():,}")
    elif how == 'right':
        print(f"   Unmatched right: {merged[on].isnull().sum():,}")
    
    return merged

def aggregate_time_series(df, date_col, value_col, freq='D'):
    """Aggregate data to time series"""
    ts = df.groupby(pd.Grouper(key=date_col, freq=freq))[value_col].sum()
    ts = ts.fillna(0)
    return ts

def calculate_cagr(start_value, end_value, periods):
    """Calculate Compound Annual Growth Rate"""
    if start_value <= 0:
        return 0
    return (((end_value / start_value) ** (1 / periods)) - 1) * 100

def create_pivot_summary(df, index, columns, values, aggfunc='sum'):
    """Create pivot table with totals"""
    pivot = pd.pivot_table(df, index=index, columns=columns, 
                          values=values, aggfunc=aggfunc, 
                          fill_value=0, margins=True)
    return pivot

def filter_date_range(df, date_col, start_date=None, end_date=None):
    """Filter dataframe by date range"""
    df = df.copy()
    if start_date:
        df = df[df[date_col] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[date_col] <= pd.to_datetime(end_date)]
    return df

def calculate_moving_average(series, window):
    """Calculate moving average"""
    return series.rolling(window=window, min_periods=1).mean()

def calculate_exponential_moving_average(series, span):
    """Calculate exponential moving average"""
    return series.ewm(span=span, adjust=False).mean()

def standardize_column_names(df):
    """Standardize column names (lowercase, underscores)"""
    df.columns = df.columns.str.lower()\
                           .str.replace(' ', '_')\
                           .str.replace('-', '_')\
                           .str.replace('/', '_')
    return df

def remove_duplicates_report(df, subset=None):
    """Remove duplicates and report"""
    original_len = len(df)
    df_clean = df.drop_duplicates(subset=subset)
    duplicates = original_len - len(df_clean)
    
    if duplicates > 0:
        print(f"   ⚠️  Removed {duplicates:,} duplicate records")
    
    return df_clean

def fill_missing_values(df, strategy='forward', columns=None):
    """Fill missing values with specified strategy"""
    df = df.copy()
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            if strategy == 'forward':
                df[col] = df[col].fillna(method='ffill')
            elif strategy == 'backward':
                df[col] = df[col].fillna(method='bfill')
            elif strategy == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif strategy == 'zero':
                df[col] = df[col].fillna(0)
            
            print(f"   Filled {missing_count:,} missing values in '{col}' using {strategy}")
    
    return df

def validate_data_quality(df, name="Dataset"):
    """Validate data quality and report issues"""
    print(f"\n📊 Data Quality Report: {name}")
    print(f"   Total Records: {len(df):,}")
    print(f"   Total Columns: {len(df.columns)}")
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    if missing.sum() > 0:
        print(f"\n   ⚠️  Missing Values:")
        for col in missing[missing > 0].index:
            print(f"      {col}: {missing[col]:,} ({missing_pct[col]:.2f}%)")
    else:
        print(f"   ✅ No missing values")
    
    # Duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"   ⚠️  Duplicate Records: {duplicates:,}")
    else:
        print(f"   ✅ No duplicate records")
    
    # Data types
    print(f"\n   Data Types:")
    for dtype in df.dtypes.value_counts().items():
        print(f"      {dtype[0]}: {dtype[1]} columns")

def generate_timestamp():
    """Generate formatted timestamp"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_analysis_step(step_name, details=None):
    """Log analysis step with timestamp"""
    timestamp = generate_timestamp()
    print(f"\n[{timestamp}] {step_name}")
    if details:
        print(f"   {details}")

class ProgressTracker:
    """Simple progress tracker for long-running operations"""
    
    def __init__(self, total_steps, description="Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        print(f"\n🔄 {description}: 0/{total_steps}")
    
    def update(self, step_name=""):
        self.current_step += 1
        pct = (self.current_step / self.total_steps) * 100
        print(f"   [{self.current_step}/{self.total_steps}] {step_name} ({pct:.0f}%)")
    
    def complete(self):
        print(f"✅ {self.description} complete!")

def create_color_palette(n_colors):
    """Create consistent color palette"""
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', 
              '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1']
    
    if n_colors <= len(colors):
        return colors[:n_colors]
    else:
        # Generate additional colors if needed
        return sns.color_palette("husl", n_colors).as_hex()

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for data"""
    from scipy import stats
    
    mean = np.mean(data)
    std_err = stats.sem(data)
    margin = std_err * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    
    return mean - margin, mean + margin

def format_large_number(num):
    """Format large numbers with K, M, B suffixes"""
    if abs(num) >= 1e9:
        return f"{num/1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"{num/1e6:.2f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"

def create_business_summary(metrics_dict):
    """Create formatted business summary"""
    print("\n" + "="*80)
    print(" BUSINESS SUMMARY")
    print("="*80)
    
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            if key.lower().find('revenue') >= 0 or key.lower().find('cost') >= 0:
                print(f"   {key}: {format_currency(value)}")
            elif key.lower().find('pct') >= 0 or key.lower().find('percent') >= 0:
                print(f"   {key}: {format_percentage(value)}")
            else:
                print(f"   {key}: {value:,.2f}")
        elif isinstance(value, int):
            print(f"   {key}: {value:,}")
        else:
            print(f"   {key}: {value}")

if __name__ == "__main__":
    print("Utils module loaded successfully")
    print("Available functions:")
    print("  - setup_output_directories()")
    print("  - save_dataframe()")
    print("  - save_plot()")
    print("  - calculate_date_metrics()")
    print("  - clean_currency_column()")
    print("  - get_summary_statistics()")
    print("  - identify_outliers()")
    print("  - validate_data_quality()")
    print("  - And many more...")