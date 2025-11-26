"""
Supplier Analysis Module
Analyzes supplier performance and reliability
"""

import pandas as pd
import numpy as np


class SupplierAnalyzer:
    """Analyzes supplier performance metrics"""
    
    def __init__(self, purchases_df, invoice_df):
        self.purchases_df = purchases_df.copy()
        self.invoice_df = invoice_df.copy()
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for analysis"""
        # Parse dates
        date_columns = ['InvoiceDate', 'PODate', 'PayDate']
        for col in date_columns:
            if col in self.invoice_df.columns:
                self.invoice_df[col] = pd.to_datetime(self.invoice_df[col], errors='coerce')
    
    def calculate_lead_times(self):
        """Calculate average lead times by supplier"""
        # Calculate lead time (PO to Invoice)
        self.invoice_df['LeadTime'] = (self.invoice_df['InvoiceDate'] - self.invoice_df['PODate']).dt.days
        
        # Remove negative and extreme values
        self.invoice_df['LeadTime'] = self.invoice_df['LeadTime'].clip(0, 365)
        
        # Aggregate by supplier
        lead_times = self.invoice_df.groupby('VendorName')['LeadTime'].agg([
            ('AvgLeadTime', 'mean'),
            ('StdLeadTime', 'std'),
            ('MinLeadTime', 'min'),
            ('MaxLeadTime', 'max'),
            ('MedianLeadTime', 'median')
        ]).reset_index()
        
        lead_times = lead_times.fillna(0)
        
        return lead_times
    
    def calculate_on_time_delivery(self, tolerance_days=2):
        """Calculate on-time delivery percentage"""
        # Assume expected delivery is PO date + average lead time
        avg_lead_time = self.invoice_df['LeadTime'].mean()
        
        self.invoice_df['ExpectedDate'] = self.invoice_df['PODate'] + pd.Timedelta(days=avg_lead_time)
        self.invoice_df['OnTime'] = (self.invoice_df['InvoiceDate'] - self.invoice_df['ExpectedDate']).dt.days <= tolerance_days
        
        # Calculate by supplier
        on_time = self.invoice_df.groupby('VendorName')['OnTime'].agg([
            ('TotalOrders', 'count'),
            ('OnTimeOrders', 'sum')
        ]).reset_index()
        
        on_time['OnTimePercentage'] = (on_time['OnTimeOrders'] / on_time['TotalOrders']) * 100
        
        return on_time
    
    def calculate_cost_metrics(self):
        """Calculate cost-related metrics by supplier"""
        # Aggregate spending by supplier
        spending = self.invoice_df.groupby('VendorName').agg({
            'Dollars': 'sum',
            'Freight': 'sum',
            'Quantity': 'sum'
        }).reset_index()
        
        spending.columns = ['VendorName', 'TotalSpend', 'TotalFreight', 'TotalQuantity']
        
        # Calculate unit cost
        spending['AvgUnitCost'] = spending['TotalSpend'] / spending['TotalQuantity'].replace(0, np.nan)
        spending['FreightPercentage'] = (spending['TotalFreight'] / spending['TotalSpend']) * 100
        
        spending = spending.fillna(0)
        
        return spending
    
    def calculate_reliability_score(self):
        """Calculate overall supplier reliability score (0-100)"""
        # Get metrics
        lead_times = self.calculate_lead_times()
        on_time = self.calculate_on_time_delivery()
        costs = self.calculate_cost_metrics()
        
        # Merge all metrics
        supplier_metrics = pd.merge(lead_times, on_time, on='VendorName', how='outer')
        supplier_metrics = pd.merge(supplier_metrics, costs, on='VendorName', how='outer')
        
        supplier_metrics = supplier_metrics.fillna(0)
        
        # Calculate scores (0-100)
        # 1. Lead Time Score (lower is better)
        if supplier_metrics['AvgLeadTime'].max() > 0:
            supplier_metrics['LeadTimeScore'] = 100 * (1 - (supplier_metrics['AvgLeadTime'] / supplier_metrics['AvgLeadTime'].max()))
        else:
            supplier_metrics['LeadTimeScore'] = 100
        
        # 2. On-Time Delivery Score
        supplier_metrics['DeliveryScore'] = supplier_metrics['OnTimePercentage']
        
        # 3. Consistency Score (lower std is better)
        if supplier_metrics['StdLeadTime'].max() > 0:
            supplier_metrics['ConsistencyScore'] = 100 * (1 - (supplier_metrics['StdLeadTime'] / supplier_metrics['StdLeadTime'].max()))
        else:
            supplier_metrics['ConsistencyScore'] = 100
        
        # Overall Reliability Score (weighted average)
        supplier_metrics['ReliabilityScore'] = (
            supplier_metrics['LeadTimeScore'] * 0.3 +
            supplier_metrics['DeliveryScore'] * 0.4 +
            supplier_metrics['ConsistencyScore'] * 0.3
        )
        
        # Assign rating
        supplier_metrics['Rating'] = supplier_metrics['ReliabilityScore'].apply(
            lambda x: 'Excellent' if x >= 90 else
                     'Good' if x >= 75 else
                     'Fair' if x >= 60 else
                     'Poor'
        )
        
        return supplier_metrics
    
    def analyze_suppliers(self):
        """Generate comprehensive supplier analysis"""
        supplier_metrics = self.calculate_reliability_score()
        
        # Sort by reliability score
        supplier_metrics = supplier_metrics.sort_values('ReliabilityScore', ascending=False)
        
        # Top performers
        top_suppliers = supplier_metrics.head(10)
        
        # Poor performers
        poor_suppliers = supplier_metrics[supplier_metrics['Rating'] == 'Poor']
        
        # Calculate summary statistics
        total_suppliers = len(supplier_metrics)
        avg_reliability = supplier_metrics['ReliabilityScore'].mean()
        total_spend = supplier_metrics['TotalSpend'].sum()
        
        # Spending concentration (top 5 suppliers)
        top_5_spend = supplier_metrics.head(5)['TotalSpend'].sum()
        spend_concentration = (top_5_spend / total_spend) * 100 if total_spend > 0 else 0
        
        return {
            'summary': {
                'total_suppliers': int(total_suppliers),
                'avg_reliability_score': float(avg_reliability),
                'total_spend': float(total_spend),
                'spend_concentration_top5': float(spend_concentration),
                'excellent_suppliers': int(len(supplier_metrics[supplier_metrics['Rating'] == 'Excellent'])),
                'poor_suppliers': int(len(poor_suppliers))
            },
            'top_suppliers': top_suppliers.to_dict('records'),
            'poor_suppliers': poor_suppliers.to_dict('records'),
            'all_suppliers': supplier_metrics.to_dict('records'),
            'recommendations': self.generate_recommendations(supplier_metrics)
        }
    
    def generate_recommendations(self, supplier_metrics):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Check for poor performers
        poor = supplier_metrics[supplier_metrics['Rating'] == 'Poor']
        if len(poor) > 0:
            recommendations.append({
                'type': 'WARNING',
                'category': 'Poor Performance',
                'message': f'{len(poor)} supplier(s) rated as Poor',
                'action': 'Review contracts and consider alternative suppliers',
                'suppliers': poor['VendorName'].tolist()[:5]
            })
        
        # Check for high lead times
        high_lead_time = supplier_metrics[supplier_metrics['AvgLeadTime'] > 14]
        if len(high_lead_time) > 0:
            recommendations.append({
                'type': 'INFO',
                'category': 'Long Lead Times',
                'message': f'{len(high_lead_time)} supplier(s) with lead time > 14 days',
                'action': 'Negotiate faster delivery terms or increase safety stock',
                'suppliers': high_lead_time['VendorName'].tolist()[:5]
            })
        
        # Check for low on-time delivery
        low_on_time = supplier_metrics[supplier_metrics['OnTimePercentage'] < 80]
        if len(low_on_time) > 0:
            recommendations.append({
                'type': 'WARNING',
                'category': 'Delivery Issues',
                'message': f'{len(low_on_time)} supplier(s) with <80% on-time delivery',
                'action': 'Implement delivery performance penalties in contracts',
                'suppliers': low_on_time['VendorName'].tolist()[:5]
            })
        
        # Identify excellent suppliers for partnership
        excellent = supplier_metrics[supplier_metrics['Rating'] == 'Excellent'].head(5)
        if len(excellent) > 0:
            recommendations.append({
                'type': 'SUCCESS',
                'category': 'Strategic Partnerships',
                'message': f'{len(excellent)} excellent supplier(s) identified',
                'action': 'Consider volume commitments for better pricing',
                'suppliers': excellent['VendorName'].tolist()
            })
        
        # Check spending concentration
        total_spend = supplier_metrics['TotalSpend'].sum()
        if total_spend > 0:
            top_supplier_spend = supplier_metrics.iloc[0]['TotalSpend']
            concentration = (top_supplier_spend / total_spend) * 100
            
            if concentration > 40:
                recommendations.append({
                    'type': 'WARNING',
                    'category': 'Supplier Concentration Risk',
                    'message': f'Top supplier represents {concentration:.1f}% of spending',
                    'action': 'Diversify supplier base to reduce risk',
                    'suppliers': [supplier_metrics.iloc[0]['VendorName']]
                })
        
        return recommendations
    
    def get_supplier_scorecard(self, vendor_name):
        """Get detailed scorecard for a specific supplier"""
        supplier_metrics = self.calculate_reliability_score()
        supplier_data = supplier_metrics[supplier_metrics['VendorName'] == vendor_name]
        
        if len(supplier_data) == 0:
            return None
        
        supplier = supplier_data.iloc[0]
        
        return {
            'vendor_name': vendor_name,
            'overall_rating': supplier['Rating'],
            'reliability_score': float(supplier['ReliabilityScore']),
            'metrics': {
                'lead_time': {
                    'average': float(supplier['AvgLeadTime']),
                    'std_dev': float(supplier['StdLeadTime']),
                    'min': float(supplier['MinLeadTime']),
                    'max': float(supplier['MaxLeadTime']),
                    'score': float(supplier['LeadTimeScore'])
                },
                'delivery': {
                    'on_time_percentage': float(supplier['OnTimePercentage']),
                    'total_orders': int(supplier['TotalOrders']),
                    'on_time_orders': int(supplier['OnTimeOrders']),
                    'score': float(supplier['DeliveryScore'])
                },
                'cost': {
                    'total_spend': float(supplier['TotalSpend']),
                    'avg_unit_cost': float(supplier['AvgUnitCost']),
                    'freight_percentage': float(supplier['FreightPercentage'])
                },
                'consistency': {
                    'score': float(supplier['ConsistencyScore'])
                }
            }
        }


if __name__ == '__main__':
    print("Supplier Analysis Module")
    print("Analyzes supplier performance and reliability")
    print("Import and use with: from supplier_analysis import SupplierAnalyzer")