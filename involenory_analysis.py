"""
Slooze Inventory Analysis - Data Science & Analytics
Wine & Spirits Retail Optimization
Author: Data Science Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML libraries
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.optimize import minimize

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class InventoryAnalyzer:
    """
    Comprehensive inventory analysis system for retail wine & spirits
    """
    
    def __init__(self, sales_file, inventory_file, purchase_file):
        """Initialize with data file paths"""
        self.sales_df = None
        self.inventory_df = None
        self.purchase_df = None
        self.load_data(sales_file, inventory_file, purchase_file)
        
    def load_data(self, sales_file, inventory_file, purchase_file):
        """Load and preprocess all data files"""
        print("📊 Loading data files...")
        
        try:
            # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Current stock vs reorder point
        top_15 = reorder_data.nlargest(15, 'avg_daily_demand')
        x = range(len(top_15))
        width = 0.35
        
        axes[0, 0].bar([i - width/2 for i in x], top_15['current_stock'], 
                      width, label='Current Stock', color='#3b82f6', alpha=0.7)
        axes[0, 0].bar([i + width/2 for i in x], top_15['reorder_point'], 
                      width, label='Reorder Point', color='#ef4444', alpha=0.7)
        axes[0, 0].set_xlabel('Product')
        axes[0, 0].set_ylabel('Quantity')
        axes[0, 0].set_title('Current Stock vs Reorder Point (Top 15 Products)')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(top_15['product_name'], rotation=45, ha='right')
        axes[0, 0].legend()
        
        # Stockout risk distribution
        risk_bins = [0, 25, 50, 75, 100]
        risk_labels = ['Low', 'Medium', 'High', 'Critical']
        reorder_data['risk_category'] = pd.cut(reorder_data['stockout_risk'], 
                                               bins=risk_bins, labels=risk_labels)
        risk_counts = reorder_data['risk_category'].value_counts()
        
        colors_risk = ['#10b981', '#f59e0b', '#ef4444', '#991b1b']
        axes[0, 1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                      colors=colors_risk, startangle=90)
        axes[0, 1].set_title('Stockout Risk Distribution')
        
        # Days until stockout
        axes[1, 0].hist(reorder_data['days_until_stockout'].clip(0, 60), 
                       bins=20, color='#8b5cf6', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(reorder_data['avg_lead_time'].mean(), color='red', 
                          linestyle='--', label='Avg Lead Time')
        axes[1, 0].set_xlabel('Days Until Stockout')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Days Until Stockout Distribution')
        axes[1, 0].legend()
        
        # Safety stock levels
        top_safety = reorder_data.nlargest(10, 'safety_stock')
        axes[1, 1].barh(range(len(top_safety)), top_safety['safety_stock'], 
                       color='#10b981', alpha=0.7)
        axes[1, 1].set_yticks(range(len(top_safety)))
        axes[1, 1].set_yticklabels(top_safety['product_name'])
        axes[1, 1].set_xlabel('Safety Stock (units)')
        axes[1, 1].set_title('Top 10 Products by Safety Stock Requirements')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('reorder_point_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Reorder point analysis plot saved: reorder_point_analysis.png")
        
        # Save results
        reorder_output = reorder_data[[
            'product_id', 'product_name', 'current_stock', 'avg_daily_demand',
            'avg_lead_time', 'safety_stock', 'reorder_point', 
            'days_until_stockout', 'stockout_risk', 'status'
        ]].sort_values('stockout_risk', ascending=False)
        reorder_output.to_csv('reorder_points.csv', index=False)
        print("✅ Reorder points saved: reorder_points.csv")
        
        return reorder_data

    # ============================================================================
    # 5. LEAD TIME ANALYSIS
    # ============================================================================
    
    def lead_time_analysis(self):
        """
        Analyze supplier lead times and identify optimization opportunities
        """
        print("\n" + "="*80)
        print("⏱️  LEAD TIME ANALYSIS")
        print("="*80)
        
        # Calculate lead time
        self.purchase_df['lead_time'] = (
            self.purchase_df['delivery_date'] - self.purchase_df['order_date']
        ).dt.days
        
        # Overall statistics
        overall_stats = {
            'Mean Lead Time': self.purchase_df['lead_time'].mean(),
            'Median Lead Time': self.purchase_df['lead_time'].median(),
            'Std Dev': self.purchase_df['lead_time'].std(),
            'Min Lead Time': self.purchase_df['lead_time'].min(),
            'Max Lead Time': self.purchase_df['lead_time'].max()
        }
        
        print(f"\n📊 Overall Lead Time Statistics:")
        for key, value in overall_stats.items():
            print(f"   {key}: {value:.2f} days")
        
        # By supplier
        supplier_stats = self.purchase_df.groupby('supplier').agg({
            'lead_time': ['mean', 'std', 'min', 'max', 'count']
        }).reset_index()
        supplier_stats.columns = ['Supplier', 'Mean_LT', 'Std_LT', 'Min_LT', 'Max_LT', 'Orders']
        supplier_stats = supplier_stats.sort_values('Mean_LT')
        
        print(f"\n📦 Lead Time by Supplier:")
        print(supplier_stats.to_string(index=False))
        
        # By product
        product_stats = self.purchase_df.groupby('product_name').agg({
            'lead_time': ['mean', 'std', 'count']
        }).reset_index()
        product_stats.columns = ['Product', 'Mean_LT', 'Std_LT', 'Orders']
        product_stats = product_stats.sort_values('Mean_LT', ascending=False)
        
        # Calculate on-time delivery rate (assuming target is mean + 1 std)
        target_lead_time = overall_stats['Mean Lead Time'] + overall_stats['Std Dev']
        self.purchase_df['on_time'] = self.purchase_df['lead_time'] <= target_lead_time
        
        supplier_performance = self.purchase_df.groupby('supplier').agg({
            'on_time': lambda x: (x.sum() / len(x)) * 100,
            'lead_time': 'mean'
        }).reset_index()
        supplier_performance.columns = ['Supplier', 'On_Time_Rate', 'Avg_Lead_Time']
        supplier_performance = supplier_performance.sort_values('On_Time_Rate', ascending=False)
        
        print(f"\n🎯 Supplier Performance (On-Time Delivery):")
        print(supplier_performance.to_string(index=False))
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Lead time distribution
        axes[0, 0].hist(self.purchase_df['lead_time'], bins=30, 
                       color='#3b82f6', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(overall_stats['Mean Lead Time'], color='red', 
                          linestyle='--', linewidth=2, label=f"Mean: {overall_stats['Mean Lead Time']:.1f}")
        axes[0, 0].set_xlabel('Lead Time (days)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Lead Time Distribution')
        axes[0, 0].legend()
        
        # Lead time by supplier
        axes[0, 1].barh(range(len(supplier_stats)), supplier_stats['Mean_LT'], 
                       color='#10b981', alpha=0.7)
        axes[0, 1].set_yticks(range(len(supplier_stats)))
        axes[0, 1].set_yticklabels(supplier_stats['Supplier'])
        axes[0, 1].set_xlabel('Average Lead Time (days)')
        axes[0, 1].set_title('Average Lead Time by Supplier')
        axes[0, 1].invert_yaxis()
        
        # On-time delivery rate
        axes[1, 0].bar(range(len(supplier_performance)), 
                      supplier_performance['On_Time_Rate'], 
                      color='#f59e0b', alpha=0.7)
        axes[1, 0].set_xticks(range(len(supplier_performance)))
        axes[1, 0].set_xticklabels(supplier_performance['Supplier'], rotation=45, ha='right')
        axes[1, 0].set_ylabel('On-Time Delivery Rate (%)')
        axes[1, 0].set_title('Supplier On-Time Delivery Performance')
        axes[1, 0].axhline(y=90, color='red', linestyle='--', label='Target: 90%')
        axes[1, 0].legend()
        
        # Lead time trend over time
        monthly_lead_time = self.purchase_df.groupby(
            self.purchase_df['order_date'].dt.to_period('M')
        )['lead_time'].mean().reset_index()
        monthly_lead_time['order_date'] = monthly_lead_time['order_date'].dt.to_timestamp()
        
        axes[1, 1].plot(monthly_lead_time['order_date'], monthly_lead_time['lead_time'], 
                       marker='o', linewidth=2, color='#8b5cf6')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Average Lead Time (days)')
        axes[1, 1].set_title('Lead Time Trend Over Time')
        axes[1, 1].grid(True, alpha=0.3)
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('lead_time_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Lead time analysis plot saved: lead_time_analysis.png")
        
        # Save results
        supplier_stats.to_csv('supplier_lead_times.csv', index=False)
        print("✅ Supplier lead times saved: supplier_lead_times.csv")
        
        return supplier_stats

    # ============================================================================
    # 6. ADDITIONAL INSIGHTS
    # ============================================================================
    
    def seasonal_analysis(self):
        """
        Analyze seasonal patterns in sales
        """
        print("\n" + "="*80)
        print("🌤️  SEASONAL ANALYSIS")
        print("="*80)
        
        # Add temporal features
        sales_temp = self.sales_df.copy()
        sales_temp['month'] = sales_temp['date'].dt.month
        sales_temp['quarter'] = sales_temp['date'].dt.quarter
        sales_temp['day_of_week'] = sales_temp['date'].dt.dayofweek
        sales_temp['week_of_year'] = sales_temp['date'].dt.isocalendar().week
        
        # Monthly sales by category
        monthly_category = sales_temp.groupby(['month', 'category'])['quantity'].sum().reset_index()
        monthly_pivot = monthly_category.pivot(index='month', columns='category', values='quantity')
        
        # Quarterly analysis
        quarterly_sales = sales_temp.groupby('quarter').agg({
            'quantity': 'sum',
            'total_revenue': 'sum'
        }).reset_index()
        
        print(f"\n📊 Quarterly Sales Summary:")
        print(quarterly_sales.to_string(index=False))
        
        # Day of week analysis
        dow_sales = sales_temp.groupby('day_of_week').agg({
            'quantity': 'sum',
            'total_revenue': 'sum'
        }).reset_index()
        dow_sales['day_name'] = dow_sales['day_of_week'].map({
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
            4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        })
        
        print(f"\n📅 Sales by Day of Week:")
        print(dow_sales[['day_name', 'quantity', 'total_revenue']].to_string(index=False))
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Monthly sales by category
        monthly_pivot.plot(kind='line', marker='o', ax=axes[0, 0])
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Sales Quantity')
        axes[0, 0].set_title('Monthly Sales by Category')
        axes[0, 0].legend(title='Category')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Quarterly revenue
        axes[0, 1].bar(quarterly_sales['quarter'], quarterly_sales['total_revenue'], 
                      color='#3b82f6', alpha=0.7)
        axes[0, 1].set_xlabel('Quarter')
        axes[0, 1].set_ylabel('Total Revenue ($)')
        axes[0, 1].set_title('Quarterly Revenue')
        axes[0, 1].set_xticks(quarterly_sales['quarter'])
        
        # Day of week pattern
        axes[1, 0].bar(range(len(dow_sales)), dow_sales['quantity'], 
                      color='#10b981', alpha=0.7)
        axes[1, 0].set_xticks(range(len(dow_sales)))
        axes[1, 0].set_xticklabels(dow_sales['day_name'], rotation=45, ha='right')
        axes[1, 0].set_ylabel('Total Quantity')
        axes[1, 0].set_title('Sales by Day of Week')
        
        # Heatmap of weekly patterns
        weekly_pattern = sales_temp.groupby(['week_of_year', 'day_of_week'])['quantity'].sum().reset_index()
        weekly_pivot = weekly_pattern.pivot(index='week_of_year', columns='day_of_week', values='quantity')
        
        im = axes[1, 1].imshow(weekly_pivot.values, cmap='YlOrRd', aspect='auto')
        axes[1, 1].set_xlabel('Day of Week')
        axes[1, 1].set_ylabel('Week of Year')
        axes[1, 1].set_title('Sales Heatmap: Week vs Day')
        axes[1, 1].set_xticks(range(7))
        axes[1, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('seasonal_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Seasonal analysis plot saved: seasonal_analysis.png")
        
        return monthly_pivot

    def inventory_turnover_analysis(self):
        """
        Calculate inventory turnover ratios
        """
        print("\n" + "="*80)
        print("🔄 INVENTORY TURNOVER ANALYSIS")
        print("="*80)
        
        # Calculate COGS (Cost of Goods Sold)
        cogs = self.sales_df.merge(
            self.inventory_df[['product_id', 'unit_cost']], 
            on='product_id', 
            how='left'
        )
        cogs['cost'] = cogs['quantity'] * cogs['unit_cost']
        
        total_cogs = cogs.groupby('product_id').agg({
            'cost': 'sum',
            'product_name': 'first'
        }).reset_index()
        
        # Calculate average inventory
        avg_inventory = self.inventory_df.copy()
        avg_inventory['inventory_value'] = (
            avg_inventory['current_stock'] * avg_inventory['unit_cost']
        )
        
        # Merge and calculate turnover
        turnover_data = total_cogs.merge(
            avg_inventory[['product_id', 'inventory_value', 'current_stock']], 
            on='product_id', 
            how='left'
        )
        
        # Inventory turnover ratio = COGS / Average Inventory Value
        turnover_data['turnover_ratio'] = (
            turnover_data['cost'] / turnover_data['inventory_value']
        ).round(2)
        
        # Days inventory outstanding = 365 / Turnover Ratio
        turnover_data['days_inventory'] = (
            365 / turnover_data['turnover_ratio']
        ).round(0).astype(int)
        
        # Classify inventory health
        def classify_turnover(ratio):
            if ratio >= 8:
                return 'Excellent'
            elif ratio >= 6:
                return 'Good'
            elif ratio >= 4:
                return 'Average'
            else:
                return 'Poor'
        
        turnover_data['inventory_health'] = turnover_data['turnover_ratio'].apply(classify_turnover)
        
        print(f"\n📊 Inventory Turnover Summary:")
        print(f"   Average Turnover Ratio: {turnover_data['turnover_ratio'].mean():.2f}")
        print(f"   Median Turnover Ratio: {turnover_data['turnover_ratio'].median():.2f}")
        print(f"   Average Days Inventory: {turnover_data['days_inventory'].mean():.0f} days")
        
        # Health distribution
        health_dist = turnover_data['inventory_health'].value_counts()
        print(f"\n🏥 Inventory Health Distribution:")
        for health, count in health_dist.items():
            print(f"   {health}: {count} products")
        
        # Identify slow-moving inventory
        slow_moving = turnover_data[turnover_data['turnover_ratio'] < 4]\
            .sort_values('turnover_ratio')
        
        if len(slow_moving) > 0:
            print(f"\n⚠️  Slow-Moving Inventory (Turnover < 4):")
            print(slow_moving[['product_name', 'turnover_ratio', 'days_inventory']]\
                .head(10).to_string(index=False))
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Turnover ratio distribution
        axes[0, 0].hist(turnover_data['turnover_ratio'], bins=20, 
                       color='#3b82f6', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(turnover_data['turnover_ratio'].mean(), 
                          color='red', linestyle='--', linewidth=2, 
                          label=f"Mean: {turnover_data['turnover_ratio'].mean():.2f}")
        axes[0, 0].set_xlabel('Turnover Ratio')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Inventory Turnover Distribution')
        axes[0, 0].legend()
        
        # Top and bottom performers
        top_10 = turnover_data.nlargest(10, 'turnover_ratio')
        axes[0, 1].barh(range(len(top_10)), top_10['turnover_ratio'], 
                       color='#10b981', alpha=0.7)
        axes[0, 1].set_yticks(range(len(top_10)))
        axes[0, 1].set_yticklabels(top_10['product_name'])
        axes[0, 1].set_xlabel('Turnover Ratio')
        axes[0, 1].set_title('Top 10 Products by Turnover')
        axes[0, 1].invert_yaxis()
        
        # Inventory health pie chart
        health_colors = {'Excellent': '#10b981', 'Good': '#3b82f6', 
                        'Average': '#f59e0b', 'Poor': '#ef4444'}
        colors = [health_colors[h] for h in health_dist.index]
        axes[1, 0].pie(health_dist.values, labels=health_dist.index, 
                      autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1, 0].set_title('Inventory Health Distribution')
        
        # Days inventory vs turnover
        axes[1, 1].scatter(turnover_data['turnover_ratio'], 
                          turnover_data['days_inventory'], 
                          alpha=0.6, color='#8b5cf6')
        axes[1, 1].set_xlabel('Turnover Ratio')
        axes[1, 1].set_ylabel('Days Inventory Outstanding')
        axes[1, 1].set_title('Turnover Ratio vs Days Inventory')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('inventory_turnover.png', dpi=300, bbox_inches='tight')
        print("✅ Inventory turnover plot saved: inventory_turnover.png")
        
        # Save results
        turnover_output = turnover_data[[
            'product_id', 'product_name', 'turnover_ratio', 
            'days_inventory', 'inventory_health'
        ]].sort_values('turnover_ratio', ascending=False)
        turnover_output.to_csv('inventory_turnover.csv', index=False)
        print("✅ Inventory turnover saved: inventory_turnover.csv")
        
        return turnover_data

    def generate_executive_summary(self):
        """
        Generate comprehensive executive summary report
        """
        print("\n" + "="*80)
        print("📋 GENERATING EXECUTIVE SUMMARY")
        print("="*80)
        
        summary = []
        summary.append("="*80)
        summary.append("SLOOZE INVENTORY ANALYSIS - EXECUTIVE SUMMARY")
        summary.append("="*80)
        summary.append(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"\nAnalysis Period: {self.sales_df['date'].min()} to {self.sales_df['date'].max()}")
        
        # Key metrics
        total_revenue = self.sales_df['total_revenue'].sum()
        total_units = self.sales_df['quantity'].sum()
        num_products = self.sales_df['product_id'].nunique()
        
        summary.append(f"\n{'='*80}")
        summary.append("KEY PERFORMANCE INDICATORS")
        summary.append(f"{'='*80}")
        summary.append(f"Total Revenue: ${total_revenue:,.2f}")
        summary.append(f"Total Units Sold: {total_units:,}")
        summary.append(f"Active SKUs: {num_products}")
        summary.append(f"Average Transaction Value: ${total_revenue/len(self.sales_df):,.2f}")
        
        summary_text = "\n".join(summary)
        print(summary_text)
        
        # Save to file
        with open('executive_summary.txt', 'w') as f:
            f.write(summary_text)
        
        print("\n✅ Executive summary saved: executive_summary.txt")
        
        return summary_text


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    print("\n" + "="*80)
    print("🍷 SLOOZE INVENTORY ANALYSIS SYSTEM")
    print("="*80)
    print("\nWine & Spirits Retail Optimization")
    print("Comprehensive Data Science & Analytics Solution\n")
    
    # Initialize analyzer
    # Replace these with actual file paths
    analyzer = InventoryAnalyzer(
        sales_file='sales_data.csv',
        inventory_file='inventory_data.csv',
        purchase_file='purchase_data.csv'
    )
    
    # Run all analyses
    print("\n🚀 Starting comprehensive analysis...")
    
    # 1. Demand Forecasting
    forecast_df = analyzer.demand_forecasting(forecast_periods=180)
    
    # 2. ABC Analysis
    abc_results = analyzer.abc_analysis()
    
    # 3. EOQ Analysis
    eoq_results = analyzer.eoq_analysis(ordering_cost=50, holding_cost_rate=0.20)
    
    # 4. Reorder Point Analysis
    reorder_results = analyzer.reorder_point_analysis(service_level=0.95)
    
    # 5. Lead Time Analysis
    lead_time_results = analyzer.lead_time_analysis()
    
    # 6. Additional Insights
    seasonal_results = analyzer.seasonal_analysis()
    turnover_results = analyzer.inventory_turnover_analysis()
    
    # 7. Executive Summary
    summary = analyzer.generate_executive_summary()
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\n📁 Generated Files:")
    print("   - demand_forecast.csv")
    print("   - demand_forecast.png")
    print("   - abc_classification.csv")
    print("   - abc_analysis.png")
    print("   - eoq_results.csv")
    print("   - eoq_analysis.png")
    print("   - reorder_points.csv")
    print("   - reorder_point_analysis.png")
    print("   - supplier_lead_times.csv")
    print("   - lead_time_analysis.png")
    print("   - seasonal_analysis.png")
    print("   - inventory_turnover.csv")
    print("   - inventory_turnover.png")
    print("   - executive_summary.txt")
    
    print("\n🎯 Next Steps:")
    print("   1. Review critical stockout items and place orders")
    print("   2. Implement EOQ recommendations for cost savings")
    print("   3. Negotiate with underperforming suppliers")
    print("   4. Adjust safety stock levels based on forecast")
    print("   5. Consider liquidation of slow-moving inventory")
    
    print("\n✨ Thank you for using Slooze Inventory Analysis System!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main() Load datasets
            self.sales_df = pd.read_csv(sales_file)
            self.inventory_df = pd.read_csv(inventory_file)
            self.purchase_df = pd.read_csv(purchase_file)
            
            # Convert date columns
            date_cols = ['date', 'transaction_date', 'order_date', 'delivery_date']
            for df in [self.sales_df, self.inventory_df, self.purchase_df]:
                for col in date_cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
            
            print(f"✅ Loaded Sales: {len(self.sales_df)} records")
            print(f"✅ Loaded Inventory: {len(self.inventory_df)} records")
            print(f"✅ Loaded Purchases: {len(self.purchase_df)} records")
            
        except FileNotFoundError as e:
            print(f"⚠️  File not found: {e}")
            print("📝 Creating sample data for demonstration...")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data for demonstration"""
        np.random.seed(42)
        
        # Sample sales data
        dates = pd.date_range(start='2023-01-01', end='2024-10-31', freq='D')
        products = ['Red Wine', 'White Wine', 'Whiskey', 'Vodka', 'Champagne', 
                   'Beer', 'Gin', 'Rum', 'Tequila', 'Brandy']
        categories = ['Wine', 'Wine', 'Spirits', 'Spirits', 'Wine', 
                     'Beer', 'Spirits', 'Spirits', 'Spirits', 'Spirits']
        
        sales_data = []
        for date in dates:
            for i, product in enumerate(products):
                # Add seasonality and trend
                base_sales = np.random.poisson(50)
                seasonal = 20 * np.sin(2 * np.pi * date.dayofyear / 365)
                weekend = 15 if date.weekday() >= 5 else 0
                december = 30 if date.month == 12 else 0
                
                quantity = max(0, int(base_sales + seasonal + weekend + december))
                price = np.random.uniform(20, 100)
                
                sales_data.append({
                    'date': date,
                    'product_id': f'P{i+1:03d}',
                    'product_name': product,
                    'category': categories[i],
                    'quantity': quantity,
                    'unit_price': price,
                    'total_revenue': quantity * price,
                    'location': np.random.choice(['Store_A', 'Store_B', 'Store_C'])
                })
        
        self.sales_df = pd.DataFrame(sales_data)
        
        # Sample inventory data
        inventory_data = []
        for i, product in enumerate(products):
            inventory_data.append({
                'product_id': f'P{i+1:03d}',
                'product_name': product,
                'category': categories[i],
                'current_stock': np.random.randint(100, 500),
                'unit_cost': np.random.uniform(10, 50),
                'supplier': f'Supplier_{np.random.randint(1, 6)}'
            })
        
        self.inventory_df = pd.DataFrame(inventory_data)
        
        # Sample purchase data
        purchase_dates = pd.date_range(start='2023-01-01', end='2024-10-31', freq='W')
        purchase_data = []
        for date in purchase_dates:
            for i, product in enumerate(products):
                if np.random.random() > 0.3:  # 70% chance of ordering
                    purchase_data.append({
                        'order_date': date,
                        'delivery_date': date + timedelta(days=np.random.randint(3, 15)),
                        'product_id': f'P{i+1:03d}',
                        'product_name': product,
                        'quantity': np.random.randint(100, 500),
                        'unit_cost': np.random.uniform(10, 50),
                        'supplier': f'Supplier_{np.random.randint(1, 6)}'
                    })
        
        self.purchase_df = pd.DataFrame(purchase_data)
        
        print("✅ Sample data created successfully")

    # ============================================================================
    # 1. DEMAND FORECASTING
    # ============================================================================
    
    def demand_forecasting(self, product_id=None, forecast_periods=180):
        """
        Perform demand forecasting using SARIMA model
        """
        print("\n" + "="*80)
        print("📈 DEMAND FORECASTING ANALYSIS")
        print("="*80)
        
        if product_id is None:
            # Aggregate all products
            daily_sales = self.sales_df.groupby('date')['quantity'].sum().reset_index()
        else:
            daily_sales = self.sales_df[self.sales_df['product_id'] == product_id]\
                .groupby('date')['quantity'].sum().reset_index()
        
        # Create time series
        ts = daily_sales.set_index('date')['quantity']
        ts = ts.asfreq('D', fill_value=0)
        
        # Fill missing values
        ts = ts.fillna(method='ffill').fillna(0)
        
        print(f"\n📊 Time Series Summary:")
        print(f"   Period: {ts.index.min()} to {ts.index.max()}")
        print(f"   Total Days: {len(ts)}")
        print(f"   Mean Daily Sales: {ts.mean():.2f}")
        print(f"   Std Dev: {ts.std():.2f}")
        
        # Decompose time series
        if len(ts) >= 730:  # At least 2 years of data
            decomposition = seasonal_decompose(ts, model='additive', period=365)
            
            fig, axes = plt.subplots(4, 1, figsize=(14, 10))
            decomposition.observed.plot(ax=axes[0], title='Observed')
            decomposition.trend.plot(ax=axes[1], title='Trend')
            decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
            decomposition.resid.plot(ax=axes[3], title='Residual')
            plt.tight_layout()
            plt.savefig('demand_decomposition.png', dpi=300, bbox_inches='tight')
            print("\n✅ Seasonal decomposition saved: demand_decomposition.png")
        
        # Train SARIMA model
        print("\n🔄 Training SARIMA model...")
        
        # Split data
        train_size = int(len(ts) * 0.8)
        train, test = ts[:train_size], ts[train_size:]
        
        # Fit SARIMA model (1,1,1)(1,1,1,7) - weekly seasonality
        try:
            model = SARIMAX(train, 
                          order=(1, 1, 1), 
                          seasonal_order=(1, 1, 1, 7),
                          enforce_stationarity=False,
                          enforce_invertibility=False)
            
            results = model.fit(disp=False)
            
            # Forecast
            forecast = results.forecast(steps=len(test))
            
            # Calculate metrics
            mae = mean_absolute_error(test, forecast)
            rmse = np.sqrt(mean_squared_error(test, forecast))
            mape = mean_absolute_percentage_error(test, forecast) * 100
            
            print(f"\n📊 Model Performance:")
            print(f"   MAE:  {mae:.2f}")
            print(f"   RMSE: {rmse:.2f}")
            print(f"   MAPE: {mape:.2f}%")
            
            # Future forecast
            future_forecast = results.forecast(steps=forecast_periods)
            forecast_index = pd.date_range(start=ts.index[-1] + timedelta(days=1), 
                                          periods=forecast_periods, freq='D')
            
            # Get confidence intervals
            pred = results.get_prediction(start=len(train), 
                                         end=len(train) + forecast_periods - 1)
            pred_ci = pred.conf_int()
            
            # Plot results
            plt.figure(figsize=(14, 6))
            plt.plot(train.index, train, label='Training Data', color='blue', alpha=0.6)
            plt.plot(test.index, test, label='Actual Test Data', color='green', alpha=0.6)
            plt.plot(test.index, forecast, label='Forecast (Test)', color='red', linestyle='--')
            plt.plot(forecast_index, future_forecast, label='Future Forecast', color='orange', linestyle='--')
            
            plt.fill_between(forecast_index, 
                           pred_ci.iloc[-forecast_periods:, 0], 
                           pred_ci.iloc[-forecast_periods:, 1], 
                           alpha=0.2, color='orange', label='95% Confidence Interval')
            
            plt.xlabel('Date')
            plt.ylabel('Daily Sales')
            plt.title('Demand Forecasting with SARIMA Model')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('demand_forecast.png', dpi=300, bbox_inches='tight')
            print("✅ Forecast plot saved: demand_forecast.png")
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'date': forecast_index,
                'forecast': future_forecast.values,
                'lower_bound': pred_ci.iloc[-forecast_periods:, 0].values,
                'upper_bound': pred_ci.iloc[-forecast_periods:, 1].values
            })
            
            forecast_df.to_csv('demand_forecast.csv', index=False)
            print("✅ Forecast data saved: demand_forecast.csv")
            
            return forecast_df
            
        except Exception as e:
            print(f"⚠️  SARIMA model failed: {e}")
            print("📊 Using simple moving average forecast instead")
            
            # Simple moving average as fallback
            ma_window = 30
            forecast = train.rolling(window=ma_window).mean().iloc[-1]
            forecast_values = np.full(forecast_periods, forecast)
            
            forecast_df = pd.DataFrame({
                'date': pd.date_range(start=ts.index[-1] + timedelta(days=1), 
                                    periods=forecast_periods, freq='D'),
                'forecast': forecast_values
            })
            
            return forecast_df

    # ============================================================================
    # 2. ABC ANALYSIS
    # ============================================================================
    
    def abc_analysis(self):
        """
        Perform ABC analysis to classify inventory items
        """
        print("\n" + "="*80)
        print("📦 ABC ANALYSIS")
        print("="*80)
        
        # Calculate total revenue by product
        product_revenue = self.sales_df.groupby(['product_id', 'product_name']).agg({
            'total_revenue': 'sum',
            'quantity': 'sum'
        }).reset_index()
        
        # Sort by revenue
        product_revenue = product_revenue.sort_values('total_revenue', ascending=False)
        
        # Calculate cumulative percentage
        product_revenue['cumulative_revenue'] = product_revenue['total_revenue'].cumsum()
        total_revenue = product_revenue['total_revenue'].sum()
        product_revenue['cumulative_percentage'] = (product_revenue['cumulative_revenue'] / total_revenue) * 100
        
        # Classify into ABC categories
        def classify_abc(cum_pct):
            if cum_pct <= 70:
                return 'A'
            elif cum_pct <= 90:
                return 'B'
            else:
                return 'C'
        
        product_revenue['abc_category'] = product_revenue['cumulative_percentage'].apply(classify_abc)
        
        # Summary statistics
        abc_summary = product_revenue.groupby('abc_category').agg({
            'product_id': 'count',
            'total_revenue': 'sum'
        }).reset_index()
        abc_summary.columns = ['Category', 'Number_of_Products', 'Total_Revenue']
        abc_summary['Revenue_Percentage'] = (abc_summary['Total_Revenue'] / total_revenue) * 100
        
        print("\n📊 ABC Classification Summary:")
        print(abc_summary.to_string(index=False))
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pareto chart
        ax1 = axes[0]
        x = range(len(product_revenue))
        ax1.bar(x, product_revenue['total_revenue'], color='steelblue', alpha=0.7)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(x, product_revenue['cumulative_percentage'], color='red', marker='o', linewidth=2)
        ax1_twin.axhline(y=70, color='green', linestyle='--', label='70% (Category A)')
        ax1_twin.axhline(y=90, color='orange', linestyle='--', label='90% (Category B)')
        ax1.set_xlabel('Products (Ranked)')
        ax1.set_ylabel('Revenue', color='steelblue')
        ax1_twin.set_ylabel('Cumulative %', color='red')
        ax1.set_title('ABC Analysis - Pareto Chart')
        ax1_twin.legend()
        
        # Pie chart
        ax2 = axes[1]
        colors = ['#10b981', '#f59e0b', '#ef4444']
        ax2.pie(abc_summary['Revenue_Percentage'], labels=abc_summary['Category'], 
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Revenue Distribution by ABC Category')
        
        plt.tight_layout()
        plt.savefig('abc_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ ABC analysis plot saved: abc_analysis.png")
        
        # Save results
        product_revenue.to_csv('abc_classification.csv', index=False)
        print("✅ ABC classification saved: abc_classification.csv")
        
        return product_revenue

    # ============================================================================
    # 3. ECONOMIC ORDER QUANTITY (EOQ) ANALYSIS
    # ============================================================================
    
    def eoq_analysis(self, ordering_cost=50, holding_cost_rate=0.20):
        """
        Calculate Economic Order Quantity for each product
        
        Parameters:
        - ordering_cost: Cost per order (default: $50)
        - holding_cost_rate: Annual holding cost as % of unit cost (default: 20%)
        """
        print("\n" + "="*80)
        print("💰 ECONOMIC ORDER QUANTITY (EOQ) ANALYSIS")
        print("="*80)
        
        print(f"\n📋 Parameters:")
        print(f"   Ordering Cost: ${ordering_cost}")
        print(f"   Holding Cost Rate: {holding_cost_rate*100}%")
        
        # Calculate annual demand
        annual_demand = self.sales_df.groupby('product_id').agg({
            'quantity': 'sum',
            'product_name': 'first'
        }).reset_index()
        annual_demand.columns = ['product_id', 'annual_demand', 'product_name']
        
        # Merge with inventory for unit cost
        eoq_data = annual_demand.merge(
            self.inventory_df[['product_id', 'unit_cost']], 
            on='product_id', 
            how='left'
        )
        
        # Calculate EOQ
        # EOQ = sqrt((2 * D * S) / H)
        # where D = annual demand, S = ordering cost, H = holding cost per unit
        eoq_data['holding_cost_per_unit'] = eoq_data['unit_cost'] * holding_cost_rate
        eoq_data['eoq'] = np.sqrt(
            (2 * eoq_data['annual_demand'] * ordering_cost) / 
            eoq_data['holding_cost_per_unit']
        )
        eoq_data['eoq'] = eoq_data['eoq'].round(0).astype(int)
        
        # Calculate number of orders per year
        eoq_data['orders_per_year'] = (eoq_data['annual_demand'] / eoq_data['eoq']).round(2)
        
        # Calculate total annual costs
        eoq_data['annual_ordering_cost'] = eoq_data['orders_per_year'] * ordering_cost
        eoq_data['annual_holding_cost'] = (eoq_data['eoq'] / 2) * eoq_data['holding_cost_per_unit']
        eoq_data['total_annual_cost'] = (
            eoq_data['annual_ordering_cost'] + 
            eoq_data['annual_holding_cost']
        )
        
        # Simulate current ordering pattern (assume monthly orders)
        eoq_data['current_order_qty'] = (eoq_data['annual_demand'] / 12).round(0).astype(int)
        eoq_data['current_orders_per_year'] = 12
        eoq_data['current_annual_ordering_cost'] = eoq_data['current_orders_per_year'] * ordering_cost
        eoq_data['current_annual_holding_cost'] = (
            (eoq_data['current_order_qty'] / 2) * eoq_data['holding_cost_per_unit']
        )
        eoq_data['current_total_cost'] = (
            eoq_data['current_annual_ordering_cost'] + 
            eoq_data['current_annual_holding_cost']
        )
        
        # Calculate savings
        eoq_data['annual_savings'] = eoq_data['current_total_cost'] - eoq_data['total_annual_cost']
        eoq_data['savings_percentage'] = (
            (eoq_data['annual_savings'] / eoq_data['current_total_cost']) * 100
        ).round(2)
        
        print(f"\n📊 EOQ Summary:")
        print(f"   Total Products: {len(eoq_data)}")
        print(f"   Total Annual Savings Potential: ${eoq_data['annual_savings'].sum():,.2f}")
        print(f"   Average Savings per Product: ${eoq_data['annual_savings'].mean():,.2f}")
        
        # Top 5 savings opportunities
        print(f"\n🎯 Top 5 Savings Opportunities:")
        top_savings = eoq_data.nlargest(5, 'annual_savings')[
            ['product_name', 'current_order_qty', 'eoq', 'annual_savings']
        ]
        print(top_savings.to_string(index=False))
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # EOQ vs Current Order Quantity
        top_10 = eoq_data.nlargest(10, 'annual_demand')
        x = range(len(top_10))
        width = 0.35
        
        axes[0, 0].bar([i - width/2 for i in x], top_10['current_order_qty'], 
                      width, label='Current Order Qty', color='#ef4444', alpha=0.7)
        axes[0, 0].bar([i + width/2 for i in x], top_10['eoq'], 
                      width, label='Optimal EOQ', color='#10b981', alpha=0.7)
        axes[0, 0].set_xlabel('Product')
        axes[0, 0].set_ylabel('Order Quantity')
        axes[0, 0].set_title('Current vs Optimal Order Quantity (Top 10 Products)')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(top_10['product_name'], rotation=45, ha='right')
        axes[0, 0].legend()
        
        # Annual savings
        top_savings_10 = eoq_data.nlargest(10, 'annual_savings')
        axes[0, 1].barh(range(len(top_savings_10)), top_savings_10['annual_savings'], 
                       color='#10b981', alpha=0.7)
        axes[0, 1].set_yticks(range(len(top_savings_10)))
        axes[0, 1].set_yticklabels(top_savings_10['product_name'])
        axes[0, 1].set_xlabel('Annual Savings ($)')
        axes[0, 1].set_title('Top 10 Products by Potential Savings')
        axes[0, 1].invert_yaxis()
        
        # Cost comparison
        cost_comparison = pd.DataFrame({
            'Cost Type': ['Current System', 'EOQ System'],
            'Ordering Cost': [eoq_data['current_annual_ordering_cost'].sum(), 
                            eoq_data['annual_ordering_cost'].sum()],
            'Holding Cost': [eoq_data['current_annual_holding_cost'].sum(), 
                           eoq_data['annual_holding_cost'].sum()]
        })
        
        x_pos = range(len(cost_comparison))
        width = 0.35
        axes[1, 0].bar([i - width/2 for i in x_pos], cost_comparison['Ordering Cost'], 
                      width, label='Ordering Cost', color='#3b82f6', alpha=0.7)
        axes[1, 0].bar([i + width/2 for i in x_pos], cost_comparison['Holding Cost'], 
                      width, label='Holding Cost', color='#f59e0b', alpha=0.7)
        axes[1, 0].set_ylabel('Annual Cost ($)')
        axes[1, 0].set_title('Total Annual Cost Comparison')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(cost_comparison['Cost Type'])
        axes[1, 0].legend()
        
        # EOQ distribution
        axes[1, 1].hist(eoq_data['eoq'], bins=20, color='#8b5cf6', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('EOQ')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Optimal Order Quantities')
        axes[1, 1].axvline(eoq_data['eoq'].mean(), color='red', 
                          linestyle='--', label=f"Mean: {eoq_data['eoq'].mean():.0f}")
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('eoq_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ EOQ analysis plot saved: eoq_analysis.png")
        
        # Save results
        eoq_output = eoq_data[[
            'product_id', 'product_name', 'annual_demand', 'unit_cost',
            'current_order_qty', 'eoq', 'orders_per_year',
            'current_total_cost', 'total_annual_cost', 'annual_savings', 'savings_percentage'
        ]]
        eoq_output.to_csv('eoq_results.csv', index=False)
        print("✅ EOQ results saved: eoq_results.csv")
        
        return eoq_data

    # ============================================================================
    # 4. REORDER POINT ANALYSIS
    # ============================================================================
    
    def reorder_point_analysis(self, service_level=0.95):
        """
        Calculate reorder points for each product
        
        Parameters:
        - service_level: Desired service level (default: 95%)
        """
        print("\n" + "="*80)
        print("🔔 REORDER POINT ANALYSIS")
        print("="*80)
        
        print(f"\n📋 Parameters:")
        print(f"   Service Level: {service_level*100}%")
        
        # Calculate daily demand statistics
        daily_demand = self.sales_df.groupby(['product_id', 'date'])['quantity'].sum().reset_index()
        
        demand_stats = daily_demand.groupby('product_id').agg({
            'quantity': ['mean', 'std', 'count']
        }).reset_index()
        demand_stats.columns = ['product_id', 'avg_daily_demand', 'std_daily_demand', 'days_sold']
        
        # Calculate lead time from purchase data
        self.purchase_df['lead_time'] = (
            self.purchase_df['delivery_date'] - self.purchase_df['order_date']
        ).dt.days
        
        lead_time_stats = self.purchase_df.groupby('product_id').agg({
            'lead_time': ['mean', 'std']
        }).reset_index()
        lead_time_stats.columns = ['product_id', 'avg_lead_time', 'std_lead_time']
        
        # Merge data
        reorder_data = demand_stats.merge(lead_time_stats, on='product_id', how='left')
        reorder_data = reorder_data.merge(
            self.inventory_df[['product_id', 'product_name', 'current_stock']], 
            on='product_id', 
            how='left'
        )
        
        # Fill missing values
        reorder_data['avg_lead_time'] = reorder_data['avg_lead_time'].fillna(7)
        reorder_data['std_lead_time'] = reorder_data['std_lead_time'].fillna(2)
        reorder_data['std_daily_demand'] = reorder_data['std_daily_demand'].fillna(
            reorder_data['avg_daily_demand'] * 0.3
        )
        
        # Calculate safety stock
        # Z-score for service level
        z_score = stats.norm.ppf(service_level)
        
        # Safety stock = Z * sqrt(LT * σ_demand² + demand² * σ_LT²)
        reorder_data['safety_stock'] = (
            z_score * np.sqrt(
                reorder_data['avg_lead_time'] * reorder_data['std_daily_demand']**2 +
                reorder_data['avg_daily_demand']**2 * reorder_data['std_lead_time']**2
            )
        ).round(0).astype(int)
        
        # Calculate reorder point
        # ROP = (Average Daily Demand × Lead Time) + Safety Stock
        reorder_data['reorder_point'] = (
            (reorder_data['avg_daily_demand'] * reorder_data['avg_lead_time']) + 
            reorder_data['safety_stock']
        ).round(0).astype(int)
        
        # Calculate days until stockout
        reorder_data['days_until_stockout'] = (
            reorder_data['current_stock'] / reorder_data['avg_daily_demand']
        ).round(1)
        
        # Determine status
        def determine_status(row):
            if row['current_stock'] < row['reorder_point']:
                return 'CRITICAL - Order Now'
            elif row['current_stock'] < row['reorder_point'] * 1.2:
                return 'WARNING - Monitor Closely'
            else:
                return 'HEALTHY'
        
        reorder_data['status'] = reorder_data.apply(determine_status, axis=1)
        
        # Calculate stockout risk
        reorder_data['stockout_risk'] = (
            100 * (1 - (reorder_data['current_stock'] / reorder_data['reorder_point']))
        ).clip(0, 100).round(1)
        
        print(f"\n📊 Reorder Point Summary:")
        print(f"   Total Products: {len(reorder_data)}")
        print(f"   Critical (Need Immediate Order): {(reorder_data['status'] == 'CRITICAL - Order Now').sum()}")
        print(f"   Warning (Monitor): {(reorder_data['status'] == 'WARNING - Monitor Closely').sum()}")
        print(f"   Healthy: {(reorder_data['status'] == 'HEALTHY').sum()}")
        
        # Critical items
        critical_items = reorder_data[reorder_data['status'] == 'CRITICAL - Order Now']
        if len(critical_items) > 0:
            print(f"\n⚠️  CRITICAL ITEMS REQUIRING IMMEDIATE ACTION:")
            critical_display = critical_items[[
                'product_name', 'current_stock', 'reorder_point', 
                'days_until_stockout', 'stockout_risk'
            ]].sort_values('stockout_risk', ascending=False)
            print(critical_display.to_string(index=False))
        
        #