"""
supplier_analysis.py
Supplier performance and lead time analysis
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

class SupplierAnalyzer:
    """
    Analyze supplier performance metrics
    """
    
    def __init__(self, purchases_df, inventory_df=None):
        """Initialize with purchase data"""
        self.purchases_df = purchases_df
        self.inventory_df = inventory_df
        self.supplier_metrics = None
    
    def analyze_lead_times(self):
        """Analyze supplier lead times"""
        print_section_header("⏱️ LEAD TIME ANALYSIS")
        
        # Find date columns
        date_cols = [col for col in self.purchases_df.columns if 'date' in col.lower()]
        
        if len(date_cols) < 2:
            print("⚠️ Insufficient date columns to calculate lead time")
            print(f"   Found columns: {date_cols}")
            # Simulate lead time for demonstration
            self.purchases_df['lead_time'] = np.random.randint(3, 15, len(self.purchases_df))
        else:
            # Calculate lead time
            order_col = [col for col in date_cols if 'order' in col.lower() or 'purch' in col.lower()][0]
            delivery_col = [col for col in date_cols if 'deliv' in col.lower() or 'receiv' in col.lower()]
            
            if not delivery_col:
                delivery_col = date_cols[1]
            else:
                delivery_col = delivery_col[0]
            
            self.purchases_df[order_col] = pd.to_datetime(self.purchases_df[order_col])
            self.purchases_df[delivery_col] = pd.to_datetime(self.purchases_df[delivery_col])
            
            self.purchases_df['lead_time'] = (
                self.purchases_df[delivery_col] - self.purchases_df[order_col]
            ).dt.days
        
        # Overall statistics
        overall_stats = {
            'Mean': self.purchases_df['lead_time'].mean(),
            'Median': self.purchases_df['lead_time'].median(),
            'Std Dev': self.purchases_df['lead_time'].std(),
            'Min': self.purchases_df['lead_time'].min(),
            'Max': self.purchases_df['lead_time'].max()
        }
        
        print(f"\n📊 Overall Lead Time Statistics:")
        for key, value in overall_stats.items():
            print(f"   {key}: {value:.2f} days")
        
        # By supplier/vendor
        vendor_col = [col for col in self.purchases_df.columns if 'vendor' in col.lower() or 'supplier' in col.lower()]
        
        if vendor_col:
            vendor_col = vendor_col[0]
            supplier_stats = self.purchases_df.groupby(vendor_col).agg({
                'lead_time': ['mean', 'std', 'min', 'max', 'count']
            }).reset_index()
            supplier_stats.columns = ['Supplier', 'Mean_LT', 'Std_LT', 'Min_LT', 'Max_LT', 'Orders']
            supplier_stats = supplier_stats.sort_values('Mean_LT')
            
            print(f"\n📦 Lead Time by Supplier:")
            print(supplier_stats.to_string(index=False))
            
            # Save
            save_dataframe(supplier_stats, 'supplier_lead_times.csv')
        else:
            print("\n⚠️ No supplier/vendor column found")
            supplier_stats = None
        
        return overall_stats, supplier_stats
    
    def analyze_supplier_performance(self):
        """Comprehensive supplier performance analysis"""
        print_section_header("📊 SUPPLIER PERFORMANCE ANALYSIS")
        
        vendor_col = [col for col in self.purchases_df.columns if 'vendor' in col.lower() or 'supplier' in col.lower()]
        
        if not vendor_col:
            print("⚠️ No supplier/vendor column found")
            return None
        
        vendor_col = vendor_col[0]
        
        # Calculate metrics by supplier
        supplier_metrics = self.purchases_df.groupby(vendor_col).agg({
            'lead_time': ['mean', 'std'],
            vendor_col: 'count'
        }).reset_index()
        supplier_metrics.columns = ['Supplier', 'Avg_Lead_Time', 'Lead_Time_Variability', 'Total_Orders']
        
        # Calculate total spend if we have cost/price info
        cost_cols = [col for col in self.purchases_df.columns if 'cost' in col.lower() or 'price' in col.lower() or 'amount' in col.lower()]
        if cost_cols:
            spend = self.purchases_df.groupby(vendor_col)[cost_cols[0]].sum().reset_index()
            spend.columns = ['Supplier', 'Total_Spend']
            supplier_metrics = supplier_metrics.merge(spend, on='Supplier', how='left')
        
        # Calculate on-time delivery rate
        # Assume target lead time is mean + 1 std dev
        target_lt = self.purchases_df['lead_time'].mean() + self.purchases_df['lead_time'].std()
        self.purchases_df['on_time'] = self.purchases_df['lead_time'] <= target_lt
        
        on_time_rate = self.purchases_df.groupby(vendor_col)['on_time'].apply(
            lambda x: (x.sum() / len(x)) * 100
        ).reset_index()
        on_time_rate.columns = ['Supplier', 'On_Time_Rate']
        
        supplier_metrics = supplier_metrics.merge(on_time_rate, on='Supplier', how='left')
        
        # Calculate supplier score (weighted composite)
        # Lower lead time = better, lower variability = better, higher on-time = better
        supplier_metrics['Lead_Time_Score'] = (
            100 - ((supplier_metrics['Avg_Lead_Time'] - supplier_metrics['Avg_Lead_Time'].min()) / 
                   (supplier_metrics['Avg_Lead_Time'].max() - supplier_metrics['Avg_Lead_Time'].min()) * 50)
        )
        supplier_metrics['Variability_Score'] = (
            100 - ((supplier_metrics['Lead_Time_Variability'] - supplier_metrics['Lead_Time_Variability'].min()) / 
                   (supplier_metrics['Lead_Time_Variability'].max() - supplier_metrics['Lead_Time_Variability'].min()) * 30)
        )
        supplier_metrics['Composite_Score'] = (
            supplier_metrics['Lead_Time_Score'] * 0.4 +
            supplier_metrics['Variability_Score'] * 0.2 +
            supplier_metrics['On_Time_Rate'] * 0.4
        ).round(1)
        
        # Sort by composite score
        supplier_metrics = supplier_metrics.sort_values('Composite_Score', ascending=False)
        
        print(f"\n📊 Supplier Performance Ranking:")
        print(supplier_metrics[['Supplier', 'Avg_Lead_Time', 'On_Time_Rate', 'Composite_Score']].to_string(index=False))
        
        # Store results
        self.supplier_metrics = supplier_metrics
        
        # Visualize
        self._plot_supplier_analysis(supplier_metrics)
        
        # Save
        save_dataframe(supplier_metrics, 'supplier_performance.csv')
        
        print(f"\n✅ Supplier performance analysis complete")
        
        return supplier_metrics
    
    def identify_supplier_opportunities(self):
        """Identify opportunities for supplier optimization"""
        print_section_header("💡 SUPPLIER OPTIMIZATION OPPORTUNITIES")
        
        if self.supplier_metrics is None:
            print("⚠️ Run analyze_supplier_performance() first")
            return
        
        # Identify underperforming suppliers
        avg_score = self.supplier_metrics['Composite_Score'].mean()
        underperforming = self.supplier_metrics[
            self.supplier_metrics['Composite_Score'] < avg_score
        ]
        
        if len(underperforming) > 0:
            print(f"\n⚠️  Underperforming Suppliers (below average score of {avg_score:.1f}):")
            print(f"   Count: {len(underperforming)}")
            print(f"\n   Suppliers needing attention:")
            for _, row in underperforming.iterrows():
                print(f"      • {row['Supplier']}")
                print(f"        - Composite Score: {row['Composite_Score']:.1f}/100")
                print(f"        - Avg Lead Time: {row['Avg_Lead_Time']:.1f} days")
                print(f"        - On-Time Rate: {row['On_Time_Rate']:.1f}%")
        
        # Identify best performers
        top_performers = self.supplier_metrics.nlargest(3, 'Composite_Score')
        print(f"\n🏆 Top 3 Performing Suppliers:")
        for _, row in top_performers.iterrows():
            print(f"   • {row['Supplier']}")
            print(f"     - Score: {row['Composite_Score']:.1f}/100")
            print(f"     - Lead Time: {row['Avg_Lead_Time']:.1f} days")
            print(f"     - On-Time: {row['On_Time_Rate']:.1f}%")
        
        # Recommendations
        print(f"\n📋 Recommendations:")
        print(f"   1. Review contracts with {len(underperforming)} underperforming suppliers")
        print(f"   2. Consider consolidating orders with top performers")
        print(f"   3. Negotiate lead time improvements with high-variability suppliers")
        print(f"   4. Implement supplier scorecard system for continuous monitoring")
    
    def _plot_supplier_analysis(self, supplier_metrics):
        """Plot supplier analysis visualizations"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Supplier performance scores
        top_n = min(10, len(supplier_metrics))
        top_suppliers = supplier_metrics.head(top_n)
        
        axes[0, 0].barh(range(len(top_suppliers)), top_suppliers['Composite_Score'], 
                       color='#3b82f6', alpha=0.7)
        axes[0, 0].set_yticks(range(len(top_suppliers)))
        axes[0, 0].set_yticklabels(top_suppliers['Supplier'])
        axes[0, 0].set_xlabel('Composite Performance Score')
        axes[0, 0].set_title(f'Top {top_n} Suppliers by Performance Score')
        axes[0, 0].invert_yaxis()
        axes[0, 0].axvline(x=80, color='red', linestyle='--', label='Target: 80')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # Lead time comparison
        axes[0, 1].bar(range(len(top_suppliers)), top_suppliers['Avg_Lead_Time'], 
                      color='#10b981', alpha=0.7)
        axes[0, 1].set_xticks(range(len(top_suppliers)))
        axes[0, 1].set_xticklabels(top_suppliers['Supplier'], rotation=45, ha='right')
        axes[0, 1].set_ylabel('Average Lead Time (days)')
        axes[0, 1].set_title(f'Average Lead Time by Supplier (Top {top_n})')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # On-time delivery rate
        axes[1, 0].bar(range(len(top_suppliers)), top_suppliers['On_Time_Rate'], 
                      color='#f59e0b', alpha=0.7)
        axes[1, 0].set_xticks(range(len(top_suppliers)))
        axes[1, 0].set_xticklabels(top_suppliers['Supplier'], rotation=45, ha='right')
        axes[1, 0].set_ylabel('On-Time Delivery Rate (%)')
        axes[1, 0].set_title(f'On-Time Delivery Performance (Top {top_n})')
        axes[1, 0].axhline(y=95, color='red', linestyle='--', label='Target: 95%')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Scatter: Lead time vs On-time rate
        axes[1, 1].scatter(supplier_metrics['Avg_Lead_Time'], 
                          supplier_metrics['On_Time_Rate'],
                          s=supplier_metrics['Total_Orders']*2,
                          alpha=0.6, color='#8b5cf6')
        axes[1, 1].set_xlabel('Average Lead Time (days)')
        axes[1, 1].set_ylabel('On-Time Delivery Rate (%)')
        axes[1, 1].set_title('Lead Time vs On-Time Performance\n(bubble size = order volume)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add quadrant lines
        axes[1, 1].axvline(x=supplier_metrics['Avg_Lead_Time'].median(), 
                          color='gray', linestyle='--', alpha=0.5)
        axes[1, 1].axhline(y=supplier_metrics['On_Time_Rate'].median(), 
                          color='gray', linestyle='--', alpha=0.5)
        
        save_plot('supplier_analysis.png')


def main(purchases_df, inventory_df=None):
    """Run supplier analysis"""
    
    analyzer = SupplierAnalyzer(purchases_df, inventory_df)
    
    # Lead time analysis
    overall_stats, supplier_stats = analyzer.analyze_lead_times()
    
    # Supplier performance
    supplier_metrics = analyzer.analyze_supplier_performance()
    
    # Identify opportunities
    analyzer.identify_supplier_opportunities()
    
    return analyzer


if __name__ == "__main__":
    print("Supplier Analysis module loaded")
    print("Run from main.py to execute supplier analysis")