"""
Reorder Point Analysis Module
Calculates when to reorder inventory to avoid stockouts
ROP = (Average Daily Demand × Lead Time) + Safety Stock
"""

import pandas as pd
import numpy as np
from scipy import stats


class ReorderPointCalculator:
    """Calculates reorder points for inventory management"""
    
    def __init__(self, sales_df, inventory_df, service_level=0.95, default_lead_time=7):
        """
        Initialize Reorder Point Calculator
        
        Parameters:
        - service_level: Desired service level (default: 95%)
        - default_lead_time: Default lead time in days (default: 7)
        """
        self.sales_df = sales_df.copy()
        self.inventory_df = inventory_df.copy()
        self.service_level = service_level
        self.default_lead_time = default_lead_time
        self.z_score = stats.norm.ppf(service_level)
    
    def calculate_daily_demand(self):
        """Calculate average daily demand for each product"""
        # Ensure date column is datetime
        if 'SalesDate' in self.sales_df.columns:
            self.sales_df['SalesDate'] = pd.to_datetime(self.sales_df['SalesDate'], errors='coerce')
        
        # Calculate daily demand
        daily_demand = self.sales_df.groupby(['Brand', 'SalesDate']).agg({
            'SalesQuantity': 'sum'
        }).reset_index()
        
        # Calculate statistics
        demand_stats = daily_demand.groupby('Brand')['SalesQuantity'].agg([
            ('AvgDailyDemand', 'mean'),
            ('StdDailyDemand', 'std'),
            ('MaxDailyDemand', 'max'),
            ('MinDailyDemand', 'min')
        ]).reset_index()
        
        # Fill NaN std with 0
        demand_stats['StdDailyDemand'] = demand_stats['StdDailyDemand'].fillna(0)
        
        return demand_stats
    
    def calculate_safety_stock(self, avg_demand, std_demand, lead_time):
        """
        Calculate safety stock
        Safety Stock = Z × σ × √Lead Time
        """
        if std_demand <= 0:
            return 0
        
        safety_stock = self.z_score * std_demand * np.sqrt(lead_time)
        return max(0, safety_stock)
    
    def calculate_reorder_point(self, avg_demand, std_demand, lead_time):
        """
        Calculate Reorder Point
        ROP = (Average Daily Demand × Lead Time) + Safety Stock
        """
        lead_time_demand = avg_demand * lead_time
        safety_stock = self.calculate_safety_stock(avg_demand, std_demand, lead_time)
        
        rop = lead_time_demand + safety_stock
        
        return {
            'reorder_point': rop,
            'lead_time_demand': lead_time_demand,
            'safety_stock': safety_stock
        }
    
    def calculate_reorder_points(self):
        """Calculate reorder points for all products"""
        # Get demand statistics
        demand_stats = self.calculate_daily_demand()
        
        # Get current inventory levels
        current_inventory = self.inventory_df.groupby('Brand')['onHand'].sum().reset_index()
        current_inventory.columns = ['Brand', 'CurrentStock']
        
        # Merge data
        reorder_data = pd.merge(demand_stats, current_inventory, on='Brand', how='left')
        reorder_data['CurrentStock'] = reorder_data['CurrentStock'].fillna(0)
        
        # Calculate reorder points
        results = []
        
        for _, row in reorder_data.iterrows():
            product = row['Brand']
            avg_demand = row['AvgDailyDemand']
            std_demand = row['StdDailyDemand']
            current_stock = row['CurrentStock']
            
            # Calculate ROP
            rop_data = self.calculate_reorder_point(
                avg_demand, std_demand, self.default_lead_time
            )
            
            # Calculate days of stock remaining
            if avg_demand > 0:
                days_of_stock = current_stock / avg_demand
            else:
                days_of_stock = 999
            
            # Determine if reorder is needed
            needs_reorder = current_stock <= rop_data['reorder_point']
            
            # Calculate order quantity suggestion (if needed)
            if needs_reorder:
                # Order enough for 30 days
                suggested_order = max(0, (avg_demand * 30) - current_stock)
            else:
                suggested_order = 0
            
            results.append({
                'Product': product,
                'CurrentStock': float(current_stock),
                'AvgDailyDemand': float(avg_demand),
                'StdDailyDemand': float(std_demand),
                'ReorderPoint': float(rop_data['reorder_point']),
                'SafetyStock': float(rop_data['safety_stock']),
                'LeadTimeDays': self.default_lead_time,
                'DaysOfStock': float(days_of_stock),
                'NeedsReorder': bool(needs_reorder),
                'SuggestedOrderQty': float(suggested_order),
                'ServiceLevel': self.service_level,
                'StockoutRisk': 'HIGH' if current_stock < rop_data['safety_stock'] else 
                               'MEDIUM' if needs_reorder else 'LOW'
            })
        
        return pd.DataFrame(results)
    
    def get_reorder_alerts(self):
        """Get products that need immediate reordering"""
        rop_df = self.calculate_reorder_points()
        
        # Filter products that need reorder
        alerts = rop_df[rop_df['NeedsReorder'] == True].copy()
        
        # Sort by urgency (days of stock)
        alerts = alerts.sort_values('DaysOfStock')
        
        return {
            'critical_items': alerts[alerts['DaysOfStock'] < 3].to_dict('records'),
            'warning_items': alerts[(alerts['DaysOfStock'] >= 3) & (alerts['DaysOfStock'] < 7)].to_dict('records'),
            'all_reorder_items': alerts.to_dict('records'),
            'total_items_needing_reorder': len(alerts),
            'total_suggested_order_value': float(alerts['SuggestedOrderQty'].sum())
        }
    
    def calculate_optimal_service_level(self, product_name, stockout_cost, holding_cost_per_unit):
        """
        Calculate optimal service level based on costs
        Balances stockout cost vs holding cost
        """
        demand_stats = self.calculate_daily_demand()
        product_data = demand_stats[demand_stats['Brand'] == product_name]
        
        if len(product_data) == 0:
            return None
        
        avg_demand = product_data.iloc[0]['AvgDailyDemand']
        std_demand = product_data.iloc[0]['StdDailyDemand']
        
        # Calculate optimal service level
        # This is a simplified model
        ratio = stockout_cost / (stockout_cost + holding_cost_per_unit)
        optimal_service_level = min(0.99, max(0.80, ratio))
        
        # Calculate ROP with optimal service level
        z = stats.norm.ppf(optimal_service_level)
        safety_stock = z * std_demand * np.sqrt(self.default_lead_time)
        rop = (avg_demand * self.default_lead_time) + safety_stock
        
        return {
            'product': product_name,
            'optimal_service_level': float(optimal_service_level),
            'recommended_safety_stock': float(safety_stock),
            'recommended_rop': float(rop),
            'stockout_cost': float(stockout_cost),
            'holding_cost': float(holding_cost_per_unit)
        }
    
    def forecast_stockout_probability(self, product_name):
        """Calculate probability of stockout"""
        rop_df = self.calculate_reorder_points()
        product_data = rop_df[rop_df['Product'] == product_name]
        
        if len(product_data) == 0:
            return None
        
        current_stock = product_data.iloc[0]['CurrentStock']
        avg_demand = product_data.iloc[0]['AvgDailyDemand']
        std_demand = product_data.iloc[0]['StdDailyDemand']
        
        # Calculate stockout probability over lead time
        lead_time_demand = avg_demand * self.default_lead_time
        lead_time_std = std_demand * np.sqrt(self.default_lead_time)
        
        if lead_time_std > 0:
            z = (current_stock - lead_time_demand) / lead_time_std
            stockout_prob = 1 - stats.norm.cdf(z)
        else:
            stockout_prob = 0 if current_stock >= lead_time_demand else 1
        
        return {
            'product': product_name,
            'current_stock': float(current_stock),
            'stockout_probability': float(stockout_prob),
            'risk_level': 'HIGH' if stockout_prob > 0.20 else 
                         'MEDIUM' if stockout_prob > 0.05 else 'LOW',
            'expected_lead_time_demand': float(lead_time_demand)
        }
    
    def generate_inventory_report(self):
        """Generate comprehensive inventory status report"""
        rop_df = self.calculate_reorder_points()
        alerts = self.get_reorder_alerts()
        
        # Calculate summary statistics
        total_products = len(rop_df)
        products_needing_reorder = len(rop_df[rop_df['NeedsReorder'] == True])
        critical_items = len(rop_df[rop_df['StockoutRisk'] == 'HIGH'])
        
        avg_days_of_stock = rop_df['DaysOfStock'].mean()
        
        # Top 10 products by urgency
        urgent_products = rop_df[rop_df['NeedsReorder'] == True].nsmallest(10, 'DaysOfStock')
        
        return {
            'summary': {
                'total_products': total_products,
                'products_needing_reorder': products_needing_reorder,
                'critical_items': critical_items,
                'avg_days_of_stock': float(avg_days_of_stock),
                'service_level': self.service_level
            },
            'alerts': alerts,
            'urgent_products': urgent_products.to_dict('records'),
            'inventory_health_score': float((1 - (critical_items / total_products)) * 100) if total_products > 0 else 100
        }


if __name__ == '__main__':
    print("Reorder Point Analysis Module")
    print("Calculates when to reorder inventory to avoid stockouts")
    print("Import and use with: from reorder_points import ReorderPointCalculator")