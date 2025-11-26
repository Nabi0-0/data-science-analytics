"""
ABC Analysis Module
Classifies inventory using the Pareto principle (80/20 rule)
A items: ~80% of value, ~20% of items
B items: ~15% of value, ~30% of items
C items: ~5% of value, ~50% of items
"""

import pandas as pd
import numpy as np


class ABCAnalyzer:
    """Performs ABC analysis on inventory"""
    
    def __init__(self, sales_df, inventory_df):
        self.sales_df = sales_df.copy()
        self.inventory_df = inventory_df.copy()
    
    def calculate_product_value(self):
        """Calculate total value for each product"""
        # Aggregate sales by product
        product_sales = self.sales_df.groupby('Brand').agg({
            'SalesDollars': 'sum',
            'SalesQuantity': 'sum'
        }).reset_index()
        
        product_sales.columns = ['Product', 'TotalRevenue', 'TotalQuantity']
        
        # Sort by revenue
        product_sales = product_sales.sort_values('TotalRevenue', ascending=False)
        
        # Calculate cumulative percentage
        product_sales['CumulativeRevenue'] = product_sales['TotalRevenue'].cumsum()
        total_revenue = product_sales['TotalRevenue'].sum()
        product_sales['CumulativePercentage'] = (product_sales['CumulativeRevenue'] / total_revenue) * 100
        
        # Calculate individual percentage
        product_sales['RevenuePercentage'] = (product_sales['TotalRevenue'] / total_revenue) * 100
        
        return product_sales
    
    def assign_abc_class(self, cumulative_pct):
        """Assign ABC class based on cumulative percentage"""
        if cumulative_pct <= 80:
            return 'A'
        elif cumulative_pct <= 95:
            return 'B'
        else:
            return 'C'
    
    def classify_inventory(self):
        """Perform ABC classification"""
        # Calculate product values
        product_data = self.calculate_product_value()
        
        # Assign ABC classes
        product_data['ABC_Class'] = product_data['CumulativePercentage'].apply(self.assign_abc_class)
        
        # Calculate statistics by class
        class_stats = product_data.groupby('ABC_Class').agg({
            'Product': 'count',
            'TotalRevenue': 'sum',
            'TotalQuantity': 'sum'
        }).reset_index()
        
        class_stats.columns = ['Class', 'ProductCount', 'TotalRevenue', 'TotalQuantity']
        
        # Calculate percentages
        total_products = class_stats['ProductCount'].sum()
        total_revenue = class_stats['TotalRevenue'].sum()
        
        class_stats['ProductPercentage'] = (class_stats['ProductCount'] / total_products) * 100
        class_stats['RevenuePercentage'] = (class_stats['TotalRevenue'] / total_revenue) * 100
        
        # Top products in each class
        top_a = product_data[product_data['ABC_Class'] == 'A'].head(10)
        top_b = product_data[product_data['ABC_Class'] == 'B'].head(10)
        top_c = product_data[product_data['ABC_Class'] == 'C'].head(10)
        
        return {
            'classification_summary': class_stats.to_dict('records'),
            'all_products': product_data.to_dict('records'),
            'top_a_products': top_a.to_dict('records'),
            'top_b_products': top_b.to_dict('records'),
            'top_c_products': top_c.to_dict('records'),
            'total_products': int(total_products),
            'total_revenue': float(total_revenue),
            'recommendations': self.generate_recommendations(class_stats)
        }
    
    def generate_recommendations(self, class_stats):
        """Generate inventory management recommendations"""
        recommendations = []
        
        for _, row in class_stats.iterrows():
            cls = row['Class']
            
            if cls == 'A':
                recommendations.append({
                    'class': 'A',
                    'priority': 'HIGH',
                    'strategy': 'Tight inventory control',
                    'actions': [
                        'Monitor daily',
                        'Accurate demand forecasting',
                        'Negotiate better supplier terms',
                        'Minimize stockouts',
                        'Consider JIT (Just-In-Time) delivery'
                    ],
                    'review_frequency': 'Daily/Weekly'
                })
            elif cls == 'B':
                recommendations.append({
                    'class': 'B',
                    'priority': 'MEDIUM',
                    'strategy': 'Moderate inventory control',
                    'actions': [
                        'Monitor weekly',
                        'Standard forecasting methods',
                        'Maintain safety stock',
                        'Regular reviews',
                        'Balance cost vs. availability'
                    ],
                    'review_frequency': 'Weekly/Bi-weekly'
                })
            else:  # C
                recommendations.append({
                    'class': 'C',
                    'priority': 'LOW',
                    'strategy': 'Basic inventory control',
                    'actions': [
                        'Monitor monthly',
                        'Simple forecasting',
                        'Higher safety stock acceptable',
                        'Bulk ordering to reduce costs',
                        'Focus on cost minimization'
                    ],
                    'review_frequency': 'Monthly/Quarterly'
                })
        
        return recommendations
    
    def analyze_by_category(self):
        """Perform ABC analysis by product category"""
        if 'Classification' not in self.sales_df.columns:
            return None
        
        category_analysis = {}
        
        for category in self.sales_df['Classification'].unique():
            if pd.isna(category):
                continue
            
            category_df = self.sales_df[self.sales_df['Classification'] == category]
            
            # Calculate product values within category
            product_sales = category_df.groupby('Brand').agg({
                'SalesDollars': 'sum',
                'SalesQuantity': 'sum'
            }).reset_index()
            
            product_sales.columns = ['Product', 'TotalRevenue', 'TotalQuantity']
            product_sales = product_sales.sort_values('TotalRevenue', ascending=False)
            
            # Calculate cumulative percentage
            product_sales['CumulativeRevenue'] = product_sales['TotalRevenue'].cumsum()
            total_revenue = product_sales['TotalRevenue'].sum()
            
            if total_revenue > 0:
                product_sales['CumulativePercentage'] = (product_sales['CumulativeRevenue'] / total_revenue) * 100
                product_sales['ABC_Class'] = product_sales['CumulativePercentage'].apply(self.assign_abc_class)
                
                category_analysis[category] = {
                    'total_revenue': float(total_revenue),
                    'product_count': len(product_sales),
                    'top_products': product_sales.head(5).to_dict('records')
                }
        
        return category_analysis
    
    def get_inventory_turnover(self):
        """Calculate inventory turnover metrics"""
        # Merge sales with inventory
        if 'InventoryId' not in self.sales_df.columns or 'InventoryId' not in self.inventory_df.columns:
            return None
        
        # Calculate average inventory
        avg_inventory = self.inventory_df.groupby('Brand')['onHand'].mean().reset_index()
        avg_inventory.columns = ['Product', 'AvgInventory']
        
        # Calculate COGS (using sales dollars as proxy)
        cogs = self.sales_df.groupby('Brand')['SalesDollars'].sum().reset_index()
        cogs.columns = ['Product', 'COGS']
        
        # Merge and calculate turnover
        turnover_data = pd.merge(cogs, avg_inventory, on='Product', how='inner')
        
        # Avoid division by zero
        turnover_data['AvgInventory'] = turnover_data['AvgInventory'].replace(0, np.nan)
        turnover_data['TurnoverRatio'] = turnover_data['COGS'] / (turnover_data['AvgInventory'] * turnover_data['COGS'].mean() / turnover_data['AvgInventory'].mean())
        
        # Sort by turnover
        turnover_data = turnover_data.sort_values('TurnoverRatio', ascending=False)
        
        return {
            'high_turnover': turnover_data.head(10).to_dict('records'),
            'low_turnover': turnover_data.tail(10).to_dict('records'),
            'avg_turnover': float(turnover_data['TurnoverRatio'].mean()) if len(turnover_data) > 0 else 0
        }


if __name__ == '__main__':
    print("ABC Analysis Module")
    print("Classifies inventory using the Pareto principle")
    print("Import and use with: from abc_analysis import ABCAnalyzer")