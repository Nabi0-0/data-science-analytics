"""
Economic Order Quantity (EOQ) Optimization Module
Calculates optimal order quantities to minimize total inventory costs
EOQ Formula: √(2DS/H)
where D = annual demand, S = ordering cost, H = holding cost
"""

import pandas as pd
import numpy as np


class EOQOptimizer:
    """Calculates Economic Order Quantity for inventory optimization"""
    
    def __init__(self, sales_df, purchases_df, ordering_cost=100, holding_cost_rate=0.25):
        """
        Initialize EOQ Optimizer
        
        Parameters:
        - ordering_cost: Fixed cost per order (default: $100)
        - holding_cost_rate: Annual holding cost as % of item value (default: 25%)
        """
        self.sales_df = sales_df.copy()
        self.purchases_df = purchases_df.copy()
        self.ordering_cost = ordering_cost
        self.holding_cost_rate = holding_cost_rate
    
    def calculate_annual_demand(self):
        """Calculate annual demand for each product"""
        # Aggregate sales by product
        demand = self.sales_df.groupby('Brand').agg({
            'SalesQuantity': 'sum',
            'SalesDollars': 'sum',
            'SalesPrice': 'mean'
        }).reset_index()
        
        demand.columns = ['Product', 'AnnualDemand', 'AnnualRevenue', 'AvgPrice']
        
        return demand
    
    def calculate_eoq(self, annual_demand, unit_cost):
        """
        Calculate Economic Order Quantity
        EOQ = √(2DS/H)
        """
        holding_cost = unit_cost * self.holding_cost_rate
        
        if holding_cost <= 0:
            return None
        
        eoq = np.sqrt((2 * annual_demand * self.ordering_cost) / holding_cost)
        
        return eoq
    
    def calculate_total_cost(self, annual_demand, order_quantity, unit_cost):
        """Calculate total inventory cost"""
        if order_quantity <= 0:
            return None
        
        # Ordering cost
        num_orders = annual_demand / order_quantity
        ordering_cost_total = num_orders * self.ordering_cost
        
        # Holding cost
        avg_inventory = order_quantity / 2
        holding_cost = avg_inventory * unit_cost * self.holding_cost_rate
        
        # Total cost
        total_cost = ordering_cost_total + holding_cost
        
        return {
            'ordering_cost': ordering_cost_total,
            'holding_cost': holding_cost,
            'total_cost': total_cost,
            'num_orders_per_year': num_orders
        }
    
    def calculate_eoq_all_products(self):
        """Calculate EOQ for all products"""
        # Get annual demand
        demand_data = self.calculate_annual_demand()
        
        # Calculate EOQ for each product
        eoq_results = []
        
        for _, row in demand_data.iterrows():
            product = row['Product']
            annual_demand = row['AnnualDemand']
            unit_cost = row['AvgPrice']
            
            if annual_demand <= 0 or unit_cost <= 0:
                continue
            
            # Calculate EOQ
            eoq = self.calculate_eoq(annual_demand, unit_cost)
            
            if eoq is None:
                continue
            
            # Calculate costs with EOQ
            eoq_costs = self.calculate_total_cost(annual_demand, eoq, unit_cost)
            
            # Calculate current costs (assuming order once per month)
            current_order_qty = annual_demand / 12
            current_costs = self.calculate_total_cost(annual_demand, current_order_qty, unit_cost)
            
            # Calculate savings
            if current_costs and eoq_costs:
                savings = current_costs['total_cost'] - eoq_costs['total_cost']
                savings_pct = (savings / current_costs['total_cost']) * 100
            else:
                savings = 0
                savings_pct = 0
            
            eoq_results.append({
                'Product': product,
                'AnnualDemand': float(annual_demand),
                'UnitCost': float(unit_cost),
                'EOQ': float(eoq),
                'OrdersPerYear': float(eoq_costs['num_orders_per_year']),
                'OrderingCost': float(eoq_costs['ordering_cost']),
                'HoldingCost': float(eoq_costs['holding_cost']),
                'TotalCost': float(eoq_costs['total_cost']),
                'CurrentOrderQty': float(current_order_qty),
                'CurrentTotalCost': float(current_costs['total_cost']) if current_costs else 0,
                'PotentialSavings': float(savings),
                'SavingsPercentage': float(savings_pct)
            })
        
        return pd.DataFrame(eoq_results)
    
    def get_top_savings_opportunities(self, top_n=20):
        """Get products with highest potential savings"""
        eoq_df = self.calculate_eoq_all_products()
        
        if len(eoq_df) == 0:
            return None
        
        # Sort by potential savings
        top_savings = eoq_df.nlargest(top_n, 'PotentialSavings')
        
        return top_savings.to_dict('records')
    
    def calculate_reorder_frequency(self, eoq, annual_demand):
        """Calculate how often to reorder"""
        if eoq <= 0:
            return None
        
        orders_per_year = annual_demand / eoq
        days_between_orders = 365 / orders_per_year
        
        return {
            'orders_per_year': float(orders_per_year),
            'days_between_orders': float(days_between_orders),
            'orders_per_month': float(orders_per_year / 12)
        }
    
    def generate_order_schedule(self, product_name, start_date='2017-01-01'):
        """Generate an optimal order schedule for a product"""
        demand_data = self.calculate_annual_demand()
        product_data = demand_data[demand_data['Product'] == product_name]
        
        if len(product_data) == 0:
            return None
        
        annual_demand = product_data.iloc[0]['AnnualDemand']
        unit_cost = product_data.iloc[0]['AvgPrice']
        
        eoq = self.calculate_eoq(annual_demand, unit_cost)
        
        if eoq is None:
            return None
        
        frequency = self.calculate_reorder_frequency(eoq, annual_demand)
        
        # Generate order dates
        start = pd.to_datetime(start_date)
        num_orders = int(frequency['orders_per_year'])
        days_between = int(frequency['days_between_orders'])
        
        order_dates = [start + pd.Timedelta(days=i * days_between) for i in range(num_orders)]
        
        return {
            'product': product_name,
            'eoq': float(eoq),
            'orders_per_year': frequency['orders_per_year'],
            'order_dates': [d.strftime('%Y-%m-%d') for d in order_dates],
            'order_quantity': float(eoq),
            'annual_demand': float(annual_demand)
        }
    
    def sensitivity_analysis(self, product_name):
        """Perform sensitivity analysis on EOQ"""
        demand_data = self.calculate_annual_demand()
        product_data = demand_data[demand_data['Product'] == product_name]
        
        if len(product_data) == 0:
            return None
        
        annual_demand = product_data.iloc[0]['AnnualDemand']
        unit_cost = product_data.iloc[0]['AvgPrice']
        
        # Vary ordering cost
        ordering_costs = [50, 100, 150, 200, 250]
        sensitivity_data = []
        
        for oc in ordering_costs:
            temp_optimizer = EOQOptimizer(self.sales_df, self.purchases_df, 
                                         ordering_cost=oc, 
                                         holding_cost_rate=self.holding_cost_rate)
            eoq = temp_optimizer.calculate_eoq(annual_demand, unit_cost)
            costs = temp_optimizer.calculate_total_cost(annual_demand, eoq, unit_cost)
            
            sensitivity_data.append({
                'ordering_cost': oc,
                'eoq': float(eoq),
                'total_cost': float(costs['total_cost'])
            })
        
        return {
            'product': product_name,
            'sensitivity_data': sensitivity_data
        }


if __name__ == '__main__':
    print("Economic Order Quantity (EOQ) Optimization Module")
    print("Calculates optimal order quantities to minimize costs")
    print("Import and use with: from eoq_opt import EOQOptimizer")