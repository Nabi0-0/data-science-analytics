"""
data_loader.py
Load and clean all Slooze inventory datasets
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from utils import (
    print_section_header, 
    clean_currency_column,
    clean_numeric_column,
    standardize_column_names,
    validate_data_quality,
    log_analysis_step
)

class SloozeDataLoader:
    """
    Load and process all Slooze inventory data files
    """
    
    def __init__(self, data_dir='Data'):
        """Initialize with data directory path"""
        self.data_dir = data_dir
        self.sales_df = None
        self.inventory_beg_df = None
        self.inventory_end_df = None
        self.purchases_df = None
        self.invoice_purchases_df = None
        self.purchase_prices_df = None
        
    def load_all_data(self):
        """Load all data files"""
        print_section_header("📂 LOADING DATA FILES")
        
        try:
            # Load sales data
            log_analysis_step("Loading Sales Data")
            self.sales_df = self._load_sales()
            
            # Load beginning inventory
            log_analysis_step("Loading Beginning Inventory")
            self.inventory_beg_df = self._load_beginning_inventory()
            
            # Load ending inventory
            log_analysis_step("Loading Ending Inventory")
            self.inventory_end_df = self._load_ending_inventory()
            
            # Load purchases
            log_analysis_step("Loading Purchases")
            self.purchases_df = self._load_purchases()
            
            # Load invoice purchases
            log_analysis_step("Loading Invoice Purchases")
            self.invoice_purchases_df = self._load_invoice_purchases()
            
            # Load purchase prices
            log_analysis_step("Loading Purchase Prices")
            self.purchase_prices_df = self._load_purchase_prices()
            
            print("\n✅ All data files loaded successfully!")
            
            return True
            
        except FileNotFoundError as e:
            print(f"\n❌ Error loading data: {e}")
            print("Please ensure all CSV files are in the 'Data/' directory")
            return False
    
    def _load_sales(self):
        """Load and clean sales data"""
        df = pd.read_csv(f'{self.data_dir}/SalesFINAL12312016.csv')
        
        # Standardize column names
        df = standardize_column_names(df)
        
        # Parse dates
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Clean currency columns
        currency_columns = [col for col in df.columns if any(x in col.lower() for x in ['price', 'sales', 'revenue', 'cost'])]
        for col in currency_columns:
            if col in df.columns:
                df[col] = clean_currency_column(df[col])
        
        # Clean numeric columns
        numeric_columns = df.select_dtypes(include=['object']).columns
        for col in numeric_columns:
            if col not in date_columns:
                try:
                    df[col] = clean_numeric_column(df[col])
                except:
                    pass
        
        # Validate
        validate_data_quality(df, "Sales Data")
        
        return df
    
    def _load_beginning_inventory(self):
        """Load and clean beginning inventory"""
        df = pd.read_csv(f'{self.data_dir}/BegInvFINAL12312016.csv')
        df = standardize_column_names(df)
        
        # Clean currency/numeric columns
        for col in df.columns:
            if df[col].dtype == 'object' and col not in ['brand', 'description', 'vendor', 'classification']:
                df[col] = clean_currency_column(df[col])
        
        validate_data_quality(df, "Beginning Inventory")
        
        return df
    
    def _load_ending_inventory(self):
        """Load and clean ending inventory"""
        df = pd.read_csv(f'{self.data_dir}/EndInvFINAL12312016.csv')
        df = standardize_column_names(df)
        
        # Clean currency/numeric columns
        for col in df.columns:
            if df[col].dtype == 'object' and col not in ['brand', 'description', 'vendor', 'classification']:
                df[col] = clean_currency_column(df[col])
        
        validate_data_quality(df, "Ending Inventory")
        
        return df
    
    def _load_purchases(self):
        """Load and clean purchase data"""
        df = pd.read_csv(f'{self.data_dir}/PurchasesFINAL12312016.csv')
        df = standardize_column_names(df)
        
        # Parse dates
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Clean currency columns
        for col in df.columns:
            if df[col].dtype == 'object' and col not in date_columns:
                if any(x in col.lower() for x in ['price', 'cost', 'amount', 'dollars']):
                    df[col] = clean_currency_column(df[col])
        
        validate_data_quality(df, "Purchases Data")
        
        return df
    
    def _load_invoice_purchases(self):
        """Load and clean invoice purchase data"""
        df = pd.read_csv(f'{self.data_dir}/InvoicePurchases12312016.csv')
        df = standardize_column_names(df)
        
        # Parse dates
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Clean currency columns
        for col in df.columns:
            if df[col].dtype == 'object' and col not in date_columns:
                try:
                    df[col] = clean_currency_column(df[col])
                except:
                    pass
        
        validate_data_quality(df, "Invoice Purchases")
        
        return df
    
    def _load_purchase_prices(self):
        """Load and clean purchase prices"""
        df = pd.read_csv(f'{self.data_dir}/2017PurchasePricesDec.csv')
        df = standardize_column_names(df)
        
        # Clean currency columns
        for col in df.columns:
            if 'price' in col.lower() or 'cost' in col.lower():
                df[col] = clean_currency_column(df[col])
        
        validate_data_quality(df, "Purchase Prices")
        
        return df
    
    def create_unified_sales_dataset(self):
        """Create unified sales dataset with all necessary information"""
        print_section_header("🔧 CREATING UNIFIED SALES DATASET")
        
        sales = self.sales_df.copy()
        
        # Add calculated metrics
        if 'quantity' in sales.columns and 'price' in sales.columns:
            sales['total_revenue'] = sales['quantity'] * sales['price']
        
        # Add temporal features
        if 'date' in sales.columns or any('date' in col for col in sales.columns):
            date_col = [col for col in sales.columns if 'date' in col][0]
            sales['year'] = sales[date_col].dt.year
            sales['month'] = sales[date_col].dt.month
            sales['quarter'] = sales[date_col].dt.quarter
            sales['day_of_week'] = sales[date_col].dt.dayofweek
            sales['week_of_year'] = sales[date_col].dt.isocalendar().week
        
        print(f"✅ Unified sales dataset created: {len(sales):,} records")
        
        return sales
    
    def create_unified_inventory_dataset(self):
        """Create unified inventory dataset"""
        print_section_header("🔧 CREATING UNIFIED INVENTORY DATASET")
        
        # Use ending inventory as base
        inventory = self.inventory_end_df.copy()
        
        # Add identifiers if missing
        if 'product_id' not in inventory.columns:
            inventory['product_id'] = 'P' + (inventory.index + 1).astype(str).str.zfill(5)
        
        # Ensure we have key columns
        required_columns = ['product_id', 'brand', 'description']
        missing_cols = [col for col in required_columns if col not in inventory.columns]
        
        if missing_cols:
            print(f"⚠️  Missing columns: {missing_cols}")
            # Try to find alternative column names
            for col in missing_cols:
                alternatives = [c for c in inventory.columns if col[:3] in c.lower()]
                if alternatives:
                    print(f"   Using '{alternatives[0]}' for '{col}'")
        
        print(f"✅ Unified inventory dataset created: {len(inventory):,} records")
        
        return inventory
    
    def create_unified_purchase_dataset(self):
        """Create unified purchase dataset"""
        print_section_header("🔧 CREATING UNIFIED PURCHASE DATASET")
        
        purchases = self.purchases_df.copy()
        
        # Merge with purchase prices if available
        if self.purchase_prices_df is not None and not self.purchase_prices_df.empty:
            # Try to find common column for merging
            common_cols = set(purchases.columns) & set(self.purchase_prices_df.columns)
            if common_cols:
                merge_col = list(common_cols)[0]
                print(f"   Merging purchases with prices on: {merge_col}")
                purchases = purchases.merge(
                    self.purchase_prices_df,
                    on=merge_col,
                    how='left',
                    suffixes=('', '_price')
                )
        
        # Calculate lead time if we have order and delivery dates
        date_cols = [col for col in purchases.columns if 'date' in col.lower()]
        if len(date_cols) >= 2:
            order_col = [col for col in date_cols if 'order' in col.lower() or 'purchase' in col.lower()]
            delivery_col = [col for col in date_cols if 'delivery' in col.lower() or 'receive' in col.lower()]
            
            if order_col and delivery_col:
                purchases['lead_time_days'] = (
                    purchases[delivery_col[0]] - purchases[order_col[0]]
                ).dt.days
                print(f"   ✅ Lead time calculated")
        
        print(f"✅ Unified purchase dataset created: {len(purchases):,} records")
        
        return purchases
    
    def get_data_summary(self):
        """Get summary of all loaded datasets"""
        print_section_header("📊 DATA SUMMARY")
        
        datasets = {
            'Sales': self.sales_df,
            'Beginning Inventory': self.inventory_beg_df,
            'Ending Inventory': self.inventory_end_df,
            'Purchases': self.purchases_df,
            'Invoice Purchases': self.invoice_purchases_df,
            'Purchase Prices': self.purchase_prices_df
        }
        
        summary = []
        for name, df in datasets.items():
            if df is not None:
                summary.append({
                    'Dataset': name,
                    'Records': len(df),
                    'Columns': len(df.columns),
                    'Memory (MB)': df.memory_usage(deep=True).sum() / 1024 / 1024
                })
        
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    def export_cleaned_data(self, output_dir='output/csv'):
        """Export cleaned datasets"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print_section_header("💾 EXPORTING CLEANED DATA")
        
        exports = {
            'sales_cleaned.csv': self.sales_df,
            'inventory_beginning_cleaned.csv': self.inventory_beg_df,
            'inventory_ending_cleaned.csv': self.inventory_end_df,
            'purchases_cleaned.csv': self.purchases_df,
            'invoice_purchases_cleaned.csv': self.invoice_purchases_df,
            'purchase_prices_cleaned.csv': self.purchase_prices_df
        }
        
        for filename, df in exports.items():
            if df is not None:
                filepath = os.path.join(output_dir, filename)
                df.to_csv(filepath, index=False)
                print(f"✅ Exported: {filepath}")


def main():
    """Test data loader"""
    loader = SloozeDataLoader()
    
    if loader.load_all_data():
        loader.get_data_summary()
        
        # Create unified datasets
        sales = loader.create_unified_sales_dataset()
        inventory = loader.create_unified_inventory_dataset()
        purchases = loader.create_unified_purchase_dataset()
        
        # Export cleaned data
        loader.export_cleaned_data()
        
        print("\n✅ Data loading complete!")
        return loader
    else:
        print("\n❌ Data loading failed!")
        return None


if __name__ == "__main__":
    loader = main()