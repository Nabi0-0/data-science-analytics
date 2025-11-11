"""
eoq_reorder.py
Economic Order Quantity and Reorder Point Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from utils import (
    print_section_header,
    save_plot,
    save_dataframe,
    format_currency,
    log_analysis_step
)

class EOQReorderAnalyzer:
    """
    Analyze Economic Order Quantity and Reorder Points
    """
    
    def __init__(self, sales_df, inventory_df, purchases_df=None):
        """Initialize with sales, inventory, and purchase data"""
        self.sales_df = sales_df
        self.inventory_df = inventory_df
        self.purchases_df = purchases_df
        self.eoq_results = None
        self.reorder_results = None
    
    def calculate_eoq(self, ordering_cost=50, holding_cost_rate=0.20):
        """
        Calculate Economic Order Quantity for each product
        
        EOQ = √(2DS/H)
        D = Annual demand
        S = Ordering cost per order
        H = Holding cost per unit per year
        """
        print_section_header("💰 ECONOMIC ORDER QUANTITY (EOQ) ANALYSIS")
        
        print(f"\n📋 Parameters:")
        print(f"   Ordering Cost: {format_currency(ordering_cost)} per order")
        print(f"   Holding Cost Rate: {holding_cost_rate*100}% of unit cost per year")
        
        # Calculate annual demand from sales
        annual_demand = self.sales_df.groupby('brand').agg({
            'quantity': 'sum'
        }).reset_index()
        annual_demand.columns = ['brand', 'annual_demand']
        
        # Merge with inventory for unit cost
        if 'unit_cost' not in self.inventory_df.columns and 'price' in self.inventory_df.columns:
            self.inventory_df['unit_cost'] = self.inventory_df['price'] * 0.6  # Estimate
        
        eoq_data = annual_demand.merge(
            self.inventory_df[['brand', 'unit_cost', 'on_hand'] if 'on_hand' in self.inventory_df.columns 
                            else ['brand', 'unit_cost']],
            on='brand',
            how='left'
        )
        
        # Fill missing costs with median
        if eoq_data['unit_cost'].isnull().any():
            median_cost = eoq_data['unit_cost'].median()
            eoq_data['unit_cost'].fillna(median_cost, inplace=True)
        
        # Calculate EOQ
        eoq_data['holding_cost_per_unit'] = eoq_data['unit_cost'] * holding_cost_rate
        eoq_data['eoq'] = np.sqrt(
            (2 * eoq_data['annual_demand'] * ordering_cost) / 
            eoq_data['holding_cost_per_unit']
        )
        eoq_data['eoq'] = eoq_data['eoq'].round(0).astype(int)
        
        # Calculate number of orders per year
        eoq_data['orders_per_year'] = (eoq_data['annual_demand'] / eoq_data['eoq']).round(2)
        
        # Calculate total costs with EOQ
        eoq_data['annual_ordering_cost'] = eoq_data['orders_per_year'] * ordering_cost
        eoq_data['annual_holding_cost'] = (eoq_data['eoq'] / 2) * eoq_data['holding_cost_per_unit']
        eoq_data['total_cost_eoq'] = (
            eoq_data['annual_ordering_cost'] + 
            eoq_data['annual_holding_cost']
        )
        
        # Simulate current ordering pattern (assume monthly orders)
        eoq_data['current_order_qty'] = (eoq_data['annual_demand'] / 12).round(0).astype(int)
        eoq_data['current_orders_per_year'] = 12
        eoq_data['current_ordering_cost'] = eoq_data['current_orders_per_year'] * ordering_cost
        eoq_data['current_holding_cost'] = (
            (eoq_data['current_order_qty'] / 2) * eoq_data['holding_cost_per_unit']
        )
        eoq_data['total_cost_current'] = (
            eoq_data['current_ordering_cost'] + 
            eoq_data['current_holding_cost']
        )
        
        # Calculate savings
        eoq_data['annual_savings'] = eoq_data['total_cost_current'] - eoq_data['total_cost_eoq']
        eoq_data['savings_percentage'] = (
            (eoq_data['annual_savings'] / eoq_data['total_cost_current']) * 100
        ).round(2)
        
        # Summary
        total_savings = eoq_data['annual_savings'].sum()
        avg_savings = eoq_data['annual_savings'].mean()
        
        print(f"\n📊 EOQ Summary:")
        print(f"   Products Analyzed: {len(eoq_data):,}")
        print(f"   Total Annual Savings Potential: {format_currency(total_savings)}")
        print(f"   Average Savings per Product: {format_currency(avg_savings)}")
        
        # Top savings opportunities
        print(f"\n🎯 Top 10 Savings Opportunities:")
        top_10 = eoq_data.nlargest(10, 'annual_savings')[
            ['brand', 'current_order_qty', 'eoq', 'annual_savings', 'savings_percentage']
        ]
        print(top_10.to_string(index=False))
        
        # Store results
        self.eoq_results = eoq_data
        
        # Visualize
        self._plot_eoq_analysis(eoq_data)
        
        # Save
        save_dataframe(eoq_data, 'eoq_analysis.csv')
        
        print(f"\n✅ EOQ analysis complete")
        
        return eoq_data
    
    def calculate_reorder_points(self, service_level=0.95):
        """
        Calculate reorder points for each product
        
        ROP = (Average Daily Demand × Lead Time) + Safety Stock
        Safety Stock = Z × √(LT × σ²_demand + μ²_demand × σ²_LT)
        """
        print_section_header("🔔 REORDER POINT ANALYSIS")
        
        print(f"\n📋 Parameters:")
        print(f"   Service Level: {service_level*100}%")
        z_score = stats.norm.ppf(service_level)
        print(f"   Z-Score: {z_score:.2f}")
        
        # Calculate daily demand statistics
        daily_demand = self.sales_df.copy()
        
        # Ensure we have a date column
        date_col = [col for col in daily_demand.columns if 'date' in col.lower()][0]
        daily_demand[date_col] = pd.to_datetime(daily_demand[date_col])
        
        demand_stats = daily_demand.groupby(['brand', date_col])['quantity'].sum().reset_index()
        demand_stats = demand_stats.groupby('brand').agg({
            'quantity': ['mean', 'std', 'count']
        }).reset_index()
        demand_stats.columns = ['brand', 'avg_daily_demand', 'std_daily_demand', 'days_sold']
        
        # Calculate or estimate lead time
        if self.purchases_df is not None and len(self.purchases_df) > 0:
            # Calculate from actual purchase data
            date_cols = [col for col in self.purchases_df.columns if 'date' in col.lower()]
            if len(date_cols) >= 2:
                self.purchases_df['lead_time'] = (
                    pd.to_datetime(self.purchases_df[date_cols[1]]) - 
                    pd.to_datetime(self.purchases_df[date_cols[0]])
                ).dt.days
                
                lead_time_stats = self.purchases_df.groupby('brand').agg({
                    'lead_time': ['mean', 'std']
                }).reset_index()
                lead_time_stats.columns = ['brand', 'avg_lead_time', 'std_lead_time']
            else:
                lead_time_stats = None
        else:
            lead_time_stats = None
        
        # Merge data
        if lead_time_stats is not None:
            reorder_data = demand_stats.merge(lead_time_stats, on='brand', how='left')
        else:
            reorder_data = demand_stats.copy()
            reorder_data['avg_lead_time'] = 7  # Default 7 days
            reorder_data['std_lead_time'] = 2  # Default std dev
        
        # Merge with current inventory
        inventory_cols = ['brand', 'on_hand'] if 'on_hand' in self.inventory_df.columns else ['brand']
        if 'on_hand' not in inventory_cols and 'quantity' in self.inventory_df.columns:
            inventory_cols = ['brand', 'quantity']
            self.inventory_df = self.inventory_df.rename(columns={'quantity': 'on_hand'})
        
        reorder_data = reorder_data.merge(
            self.inventory_df[inventory_cols],
            on='brand',
            how='left'
        )
        
        # Fill missing values
        reorder_data['avg_lead_time'] = reorder_data['avg_lead_time'].fillna(7)
        reorder_data['std_lead_time'] = reorder_data['std_lead_time'].fillna(2)
        reorder_data['std_daily_demand'] = reorder_data['std_daily_demand'].fillna(
            reorder_data['avg_daily_demand'] * 0.3
        )
        if 'on_hand' in reorder_data.columns:
            reorder_data['on_hand'] = reorder_data['on_hand'].fillna(0)
        
        # Calculate safety stock
        reorder_data['safety_stock'] = (
            z_score * np.sqrt(
                reorder_data['avg_lead_time'] * reorder_data['std_daily_demand']**2 +
                reorder_data['avg_daily_demand']**2 * reorder_data['std_lead_time']**2
            )
        ).round(0).astype(int)
        
        # Calculate reorder point
        reorder_data['reorder_point'] = (
            (reorder_data['avg_daily_demand'] * reorder_data['avg_lead_time']) + 
            reorder_data['safety_stock']
        ).round(0).astype(int)
        
        # Calculate metrics if we have current inventory
        if 'on_hand' in reorder_data.columns:
            reorder_data['days_until_stockout'] = (
                reorder_data['on_hand'] / reorder_data['avg_daily_demand']
            ).round(1)
            
            # Determine status
            def determine_status(row):
                if pd.isna(row['on_hand']) or row['on_hand'] <= 0:
                    return 'OUT_OF_STOCK'
                elif row['on_hand'] < row['reorder_point']:
                    return 'CRITICAL'
                elif row['on_hand'] < row['reorder_point'] * 1.2:
                    return 'WARNING'
                else:
                    return 'HEALTHY'
            
            reorder_data['status'] = reorder_data.apply(determine_status, axis=1)
            
            # Calculate stockout risk
            reorder_data['stockout_risk'] = (
                100 * (1 - (reorder_data['on_hand'] / reorder_data['reorder_point']))
            ).clip(0, 100).round(1)
        else:
            reorder_data['status'] = 'UNKNOWN'
            reorder_data['stockout_risk'] = 0
        
        # Summary
        print(f"\n📊 Reorder Point Summary:")
        print(f"   Products Analyzed: {len(reorder_data):,}")
        
        if 'status' in reorder_data.columns:
            status_counts = reorder_data['status'].value_counts()
            print(f"\n   Status Distribution:")
            for status, count in status_counts.items():
                print(f"      {status}: {count:,}")
            
            # Critical items
            critical_items = reorder_data[reorder_data['status'].isin(['CRITICAL', 'OUT_OF_STOCK'])]
            if len(critical_items) > 0:
                print(f"\n⚠️  CRITICAL ITEMS REQUIRING IMMEDIATE ACTION:")
                critical_display = critical_items[['brand', 'on_hand', 'reorder_point', 
                                                   'days_until_stockout', 'stockout_risk']]\
                    .sort_values('stockout_risk', ascending=False).head(10)
                print(critical_display.to_string(index=False))
        
        # Store results
        self.reorder_results = reorder_data
        
        # Visualize
        self._plot_reorder_analysis(reorder_data)
        
        # Save
        save_dataframe(reorder_data, 'reorder_points.csv')
        
        print(f"\n✅ Reorder point analysis complete")
        
        return reorder_data
    
    def _plot_eoq_analysis(self, eoq_data):
        """Plot EOQ analysis results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Top 10 products - Current vs EOQ
        top_10 = eoq_data.nlargest(10, 'annual_demand')
        x = range(len(top_10))
        width = 0.35
        
        axes[0, 0].bar([i - width/2 for i in x], top_10['current_order_qty'], 
                      width, label='Current Order Qty', color='#ef4444', alpha=0.7)
        axes[0, 0].bar([i + width/2 for i in x], top_10['eoq'], 
                      width, label='Optimal EOQ', color='#10b981', alpha=0.7)
        axes[0, 0].set_xlabel('Product')
        axes[0, 0].set_ylabel('Order Quantity')
        axes[0, 0].set_title('Current vs Optimal Order Quantity (Top 10 by Demand)')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(top_10['brand'], rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Annual savings
        top_savings = eoq_data.nlargest(10, 'annual_savings')
        axes[0, 1].barh(range(len(top_savings)), top_savings['annual_savings'], 
                       color='#10b981', alpha=0.7)
        axes[0, 1].set_yticks(range(len(top_savings)))
        axes[0, 1].set_yticklabels(top_savings['brand'])
        axes[0, 1].set_xlabel('Annual Savings ($)')
        axes[0, 1].set_title('Top 10 Products by Savings Potential')
        axes[0, 1].invert_yaxis()
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # Cost comparison
        total_current = eoq_data['total_cost_current'].sum()
        total_eoq = eoq_data['total_cost_eoq'].sum()
        
        axes[1, 0].bar(['Current\nSystem', 'EOQ\nSystem'], 
                      [total_current, total_eoq],
                      color=['#ef4444', '#10b981'], alpha=0.7, width=0.6)
        axes[1, 0].set_ylabel('Total Annual Cost ($)')
        axes[1, 0].set_title('Total Annual Cost Comparison')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add savings annotation
        savings = total_current - total_eoq
        axes[1, 0].text(0.5, max(total_current, total_eoq) * 0.5,
                       f'Savings:\n{format_currency(savings)}',
                       ha='center', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # EOQ distribution
        axes[1, 1].hist(eoq_data['eoq'], bins=20, color='#8b5cf6', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(eoq_data['eoq'].mean(), color='red', 
                          linestyle='--', linewidth=2, label=f"Mean: {eoq_data['eoq'].mean():.0f}")
        axes[1, 1].axvline(eoq_data['eoq'].median(), color='orange', 
                          linestyle='--', linewidth=2, label=f"Median: {eoq_data['eoq'].median():.0f}")
        axes[1, 1].set_xlabel('EOQ (units)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Optimal Order Quantities')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        save_plot('eoq_analysis.png')
    
    def _plot_reorder_analysis(self, reorder_data):
        """Plot reorder point analysis results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Current stock vs reorder point (Top 15 by demand)
        if 'on_hand' in reorder_data.columns:
            top_15 = reorder_data.nlargest(15, 'avg_daily_demand')
            x = range(len(top_15))
            width = 0.35
            
            axes[0, 0].bar([i - width/2 for i in x], top_15['on_hand'], 
                          width, label='Current Stock', color='#3b82f6', alpha=0.7)
            axes[0, 0].bar([i + width/2 for i in x], top_15['reorder_point'], 
                          width, label='Reorder Point', color='#ef4444', alpha=0.7)
            axes[0, 0].set_xlabel('Product')
            axes[0, 0].set_ylabel('Quantity')
            axes[0, 0].set_title('Current Stock vs Reorder Point (Top 15 by Demand)')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(top_15['brand'], rotation=45, ha='right')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'Current inventory data not available',
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Current Stock vs Reorder Point')
        
        # Stockout risk distribution
        if 'status' in reorder_data.columns and reorder_data['status'].nunique() > 1:
            status_counts = reorder_data['status'].value_counts()
            colors_map = {'HEALTHY': '#10b981', 'WARNING': '#f59e0b', 
                         'CRITICAL': '#ef4444', 'OUT_OF_STOCK': '#991b1b'}
            colors = [colors_map.get(status, '#gray') for status in status_counts.index]
            
            axes[0, 1].pie(status_counts.values, labels=status_counts.index, 
                          autopct='%1.1f%%', colors=colors, startangle=90)
            axes[0, 1].set_title('Inventory Status Distribution')
        else:
            axes[0, 1].text(0.5, 0.5, 'Status data not available',
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Inventory Status Distribution')
        
        # Safety stock levels
        top_safety = reorder_data.nlargest(10, 'safety_stock')
        axes[1, 0].barh(range(len(top_safety)), top_safety['safety_stock'], 
                       color='#10b981', alpha=0.7)
        axes[1, 0].set_yticks(range(len(top_safety)))
        axes[1, 0].set_yticklabels(top_safety['brand'])
        axes[1, 0].set_xlabel('Safety Stock (units)')
        axes[1, 0].set_title('Top 10 Products by Safety Stock Requirements')
        axes[1, 0].invert_yaxis()
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Lead time distribution
        axes[1, 1].hist(reorder_data['avg_lead_time'], bins=15, 
                       color='#8b5cf6', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(reorder_data['avg_lead_time'].mean(), color='red', 
                          linestyle='--', linewidth=2, 
                          label=f"Mean: {reorder_data['avg_lead_time'].mean():.1f} days")
        axes[1, 1].set_xlabel('Average Lead Time (days)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Lead Time Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        save_plot('reorder_point_analysis.png')


def main(sales_df, inventory_df, purchases_df=None):
    """Run EOQ and reorder point analysis"""
    
    analyzer = EOQReorderAnalyzer(sales_df, inventory_df, purchases_df)
    
    # EOQ Analysis
    eoq_results = analyzer.calculate_eoq(
        ordering_cost=50,
        holding_cost_rate=0.20
    )
    
    # Reorder Point Analysis
    reorder_results = analyzer.calculate_reorder_points(
        service_level=0.95
    )
    
    return analyzer


if __name__ == "__main__":
    print("EOQ and Reorder Point Analysis module loaded")
    print("Run from main.py to execute analysis")
            