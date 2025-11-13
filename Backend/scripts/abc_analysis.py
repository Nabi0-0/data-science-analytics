"""
abc_analysis.py
ABC Analysis for inventory classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    print_section_header,
    save_plot,
    save_dataframe,
    format_currency,
    log_analysis_step
)

class ABCAnalyzer:
    """
    Perform ABC analysis to classify inventory items
    """
    
    def __init__(self, sales_df, inventory_df=None):
        """Initialize with sales and optional inventory dataframe"""
        self.sales_df = sales_df
        self.inventory_df = inventory_df
        self.abc_results = None
        
    def perform_abc_analysis(self, group_by='brand', value_col='total_revenue'):
        """
        Classify inventory into ABC categories based on revenue
        """
        print_section_header("📦 ABC ANALYSIS")
        
        # Aggregate sales by product
        if value_col not in self.sales_df.columns:
            # Calculate revenue if not present
            if 'quantity' in self.sales_df.columns and 'price' in self.sales_df.columns:
                self.sales_df['total_revenue'] = self.sales_df['quantity'] * self.sales_df['price']
                value_col = 'total_revenue'
            else:
                print("❌ Cannot find revenue or calculate it")
                return None
        
        # Group by product
        product_revenue = self.sales_df.groupby(group_by).agg({
            value_col: 'sum'
        }).reset_index()
        
        # Also get quantity if available
        if 'quantity' in self.sales_df.columns:
            product_qty = self.sales_df.groupby(group_by)['quantity'].sum().reset_index()
            product_revenue = product_revenue.merge(product_qty, on=group_by, how='left')
        
        # Rename columns
        product_revenue.columns = [group_by, 'total_revenue'] + \
            (['total_quantity'] if 'quantity' in product_revenue.columns else [])
        
        # Sort by revenue descending
        product_revenue = product_revenue.sort_values('total_revenue', ascending=False)
        
        # Calculate cumulative metrics
        total_revenue = product_revenue['total_revenue'].sum()
        product_revenue['cumulative_revenue'] = product_revenue['total_revenue'].cumsum()
        product_revenue['cumulative_percentage'] = (
            product_revenue['cumulative_revenue'] / total_revenue * 100
        )
        product_revenue['revenue_percentage'] = (
            product_revenue['total_revenue'] / total_revenue * 100
        )
        
        # Classify into ABC
        def classify_abc(cum_pct):
            if cum_pct <= 70:
                return 'A'
            elif cum_pct <= 90:
                return 'B'
            else:
                return 'C'
        
        product_revenue['abc_category'] = product_revenue['cumulative_percentage'].apply(classify_abc)
        
        # Calculate summary statistics
        abc_summary = product_revenue.groupby('abc_category').agg({
            group_by: 'count',
            'total_revenue': 'sum',
            'revenue_percentage': 'sum'
        }).reset_index()
        
        abc_summary.columns = ['Category', 'Number_of_Items', 'Total_Revenue', 'Revenue_Percentage']
        abc_summary['Item_Percentage'] = (
            abc_summary['Number_of_Items'] / len(product_revenue) * 100
        )
        
        print(f"\n📊 ABC Classification Summary:")
        print(f"   Total Items: {len(product_revenue):,}")
        print(f"   Total Revenue: {format_currency(total_revenue)}")
        print()
        
        for _, row in abc_summary.iterrows():
            print(f"   Category {row['Category']}:")
            print(f"      Items: {row['Number_of_Items']:,} ({row['Item_Percentage']:.1f}%)")
            print(f"      Revenue: {format_currency(row['Total_Revenue'])} ({row['Revenue_Percentage']:.1f}%)")
            print()
        
        # Store results
        self.abc_results = product_revenue
        
        # Create visualizations
        self._plot_abc_analysis(product_revenue, abc_summary, total_revenue)
        
        # Save results
        save_dataframe(product_revenue, 'abc_classification.csv')
        save_dataframe(abc_summary, 'abc_summary.csv')
        
        print("✅ ABC Analysis complete")
        
        return product_revenue, abc_summary
    
    def _plot_abc_analysis(self, product_revenue, abc_summary, total_revenue):
        """Create ABC analysis visualizations"""
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Pareto Chart
        ax1 = fig.add_subplot(gs[0, :])
        x = range(len(product_revenue))
        
        bars = ax1.bar(x, product_revenue['total_revenue'], 
                      color='#3b82f6', alpha=0.6, label='Revenue')
        
        # Color bars by ABC category
        colors = {'A': '#10b981', 'B': '#f59e0b', 'C': '#ef4444'}
        for i, (idx, row) in enumerate(product_revenue.iterrows()):
            bars[i].set_color(colors[row['abc_category']])
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(x, product_revenue['cumulative_percentage'], 
                     color='red', marker='o', linewidth=2, markersize=3,
                     label='Cumulative %')
        ax1_twin.axhline(y=70, color='#10b981', linestyle='--', linewidth=2, 
                        label='70% (Category A)')
        ax1_twin.axhline(y=90, color='#f59e0b', linestyle='--', linewidth=2,
                        label='90% (Category B)')
        
        ax1.set_xlabel('Products (Ranked by Revenue)', fontsize=11)
        ax1.set_ylabel('Revenue ($)', color='#3b82f6', fontsize=11)
        ax1_twin.set_ylabel('Cumulative Percentage (%)', color='red', fontsize=11)
        ax1.set_title('ABC Analysis - Pareto Chart', fontsize=13, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#3b82f6')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        ax1_twin.set_ylim(0, 105)
        ax1_twin.legend(loc='center right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Pie Chart - Revenue Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        pie_colors = [colors[cat] for cat in abc_summary['Category']]
        wedges, texts, autotexts = ax2.pie(
            abc_summary['Revenue_Percentage'],
            labels=abc_summary['Category'],
            autopct='%1.1f%%',
            colors=pie_colors,
            startangle=90,
            textprops={'fontsize': 11, 'weight': 'bold'}
        )
        ax2.set_title('Revenue Distribution by ABC Category', 
                     fontsize=12, fontweight='bold')
        
        # 3. Pie Chart - Item Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        wedges, texts, autotexts = ax3.pie(
            abc_summary['Number_of_Items'],
            labels=abc_summary['Category'],
            autopct='%1.1f%%',
            colors=pie_colors,
            startangle=90,
            textprops={'fontsize': 11, 'weight': 'bold'}
        )
        ax3.set_title('Item Count Distribution by ABC Category', 
                     fontsize=12, fontweight='bold')
        
        # 4. Bar Chart - Items vs Revenue
        ax4 = fig.add_subplot(gs[2, 0])
        x_pos = range(len(abc_summary))
        width = 0.35
        
        ax4.bar([i - width/2 for i in x_pos], abc_summary['Number_of_Items'], 
               width, label='Number of Items', color='#3b82f6', alpha=0.7)
        
        ax4_twin = ax4.twinx()
        ax4_twin.bar([i + width/2 for i in x_pos], abc_summary['Total_Revenue']/1000, 
                    width, label='Revenue ($K)', color='#10b981', alpha=0.7)
        
        ax4.set_xlabel('ABC Category', fontsize=11)
        ax4.set_ylabel('Number of Items', color='#3b82f6', fontsize=11)
        ax4_twin.set_ylabel('Revenue ($1000s)', color='#10b981', fontsize=11)
        ax4.set_title('Items vs Revenue by Category', fontsize=12, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(abc_summary['Category'])
        ax4.tick_params(axis='y', labelcolor='#3b82f6')
        ax4_twin.tick_params(axis='y', labelcolor='#10b981')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        
        # 5. Summary Table
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('tight')
        ax5.axis('off')
        
        table_data = []
        for _, row in abc_summary.iterrows():
            table_data.append([
                row['Category'],
                f"{row['Number_of_Items']:,}\n({row['Item_Percentage']:.1f}%)",
                f"${row['Total_Revenue']/1000:.0f}K\n({row['Revenue_Percentage']:.1f}%)"
            ])
        
        table = ax5.table(cellText=table_data,
                         colLabels=['Category', 'Items\n(% of Total)', 'Revenue\n(% of Total)'],
                         cellLoc='center',
                         loc='center',
                         colColours=['#f0f0f0']*3)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color code rows
        for i, cat in enumerate(abc_summary['Category'], start=1):
            table[(i, 0)].set_facecolor(colors[cat])
            table[(i, 1)].set_facecolor(colors[cat])
            table[(i, 2)].set_facecolor(colors[cat])
        
        ax5.set_title('ABC Summary Statistics', fontsize=12, fontweight='bold', pad=20)
        
        save_plot('abc_analysis.png')
    
    def get_category_recommendations(self):
        """Get management recommendations for each ABC category"""
        print_section_header("💡 ABC CATEGORY RECOMMENDATIONS")
        
        if self.abc_results is None:
            print("⚠️  No ABC analysis results available")
            return
        
        recommendations = {
            'A': {
                'focus': 'MAXIMUM',
                'review_frequency': 'Daily',
                'safety_stock': '2-3 weeks',
                'forecasting': 'Advanced time series models',
                'monitoring': 'Real-time inventory tracking',
                'ordering': 'Frequent small orders (EOQ)',
                'service_level': '98-99%'
            },
            'B': {
                'focus': 'MODERATE',
                'review_frequency': 'Weekly',
                'safety_stock': '3-4 weeks',
                'forecasting': 'Moving averages',
                'monitoring': 'Weekly review meetings',
                'ordering': 'Periodic review system',
                'service_level': '95-97%'
            },
            'C': {
                'focus': 'MINIMUM',
                'review_frequency': 'Monthly',
                'safety_stock': '4-6 weeks',
                'forecasting': 'Simple methods',
                'monitoring': 'Exception-based (stockouts)',
                'ordering': 'Bulk orders to reduce cost',
                'service_level': '90-93%'
            }
        }
        
        for category, recs in recommendations.items():
            count = (self.abc_results['abc_category'] == category).sum()
            revenue_pct = self.abc_results[self.abc_results['abc_category'] == category]['revenue_percentage'].sum()
            
            print(f"\n{'='*60}")
            print(f"CATEGORY {category} - {recs['focus']} PRIORITY")
            print(f"{'='*60}")
            print(f"Items: {count:,} | Revenue Contribution: {revenue_pct:.1f}%")
            print(f"\nRecommended Strategy:")
            for key, value in recs.items():
                if key != 'focus':
                    print(f"  • {key.replace('_', ' ').title()}: {value}")
        
        print(f"\n{'='*60}")
        print("KEY INSIGHT:")
        print("Focus 80% of inventory management effort on Category A items")
        print("These generate 70% of revenue but are only a small % of total SKUs")
        print(f"{'='*60}")


def main(sales_df, inventory_df=None):
    """Run ABC analysis"""
    
    analyzer = ABCAnalyzer(sales_df, inventory_df)
    
    # Perform analysis
    abc_results, abc_summary = analyzer.perform_abc_analysis(
        group_by='brand' if 'brand' in sales_df.columns else 'description',
        value_col='total_revenue'
    )
    
    # Get recommendations
    analyzer.get_category_recommendations()
    
    return analyzer


if __name__ == "__main__":
    print("ABC Analysis module loaded")
    print("Run from main.py to execute ABC analysis")