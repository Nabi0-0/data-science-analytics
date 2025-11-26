"""
Data Loader Module
Safely loads and validates all CSV files
"""

import pandas as pd
import os
from datetime import datetime


class DataLoader:
    """Handles loading and validation of all CSV data files"""
    
    def __init__(self, data_dir='Data'):
        self.data_dir = data_dir
        self.required_files = {
            'sales': 'SalesFINAL12312016.csv',
            'purchases': 'PurchasesFINAL12312016.csv',
            'inventory_begin': 'BegInvFINAL12312016.csv',
            'inventory_end': 'EndInvFINAL12312016.csv',
            'invoice_purchases': 'InvoicePurchases12312016.csv',
            'purchase_prices': '2017PurchasePricesDec.csv'
        }
    
    def load_csv_safe(self, filename):
        """Safely load CSV with error handling"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
                    print(f"✓ Loaded {filename} ({len(df)} rows, encoding: {encoding})")
                    return df
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, try with error handling
            df = pd.read_csv(filepath, encoding='utf-8', errors='ignore', low_memory=False)
            print(f"⚠ Loaded {filename} with errors ignored ({len(df)} rows)")
            return df
            
        except Exception as e:
            print(f"✗ Error loading {filename}: {e}")
            raise
    
    def clean_column_names(self, df):
        """Clean and standardize column names"""
        df.columns = df.columns.str.strip()
        return df
    
    def parse_dates(self, df, date_columns):
        """Parse date columns safely"""
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception as e:
                    print(f"Warning: Could not parse dates in {col}: {e}")
        return df
    
    def load_sales_data(self):
        """Load and clean sales data"""
        df = self.load_csv_safe(self.required_files['sales'])
        df = self.clean_column_names(df)
        df = self.parse_dates(df, ['SalesDate'])
        
        # Clean numeric columns
        numeric_cols = ['SalesQuantity', 'SalesDollars', 'SalesPrice', 'Volume', 'ExciseTax']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def load_purchases_data(self):
        """Load and clean purchases data"""
        df = self.load_csv_safe(self.required_files['purchases'])
        df = self.clean_column_names(df)
        
        # Parse dates if present
        date_cols = [col for col in df.columns if 'Date' in col or 'date' in col]
        df = self.parse_dates(df, date_cols)
        
        return df
    
    def load_inventory_data(self, inv_type='begin'):
        """Load inventory data (begin or end)"""
        file_key = f'inventory_{inv_type}'
        df = self.load_csv_safe(self.required_files[file_key])
        df = self.clean_column_names(df)
        df = self.parse_dates(df, ['startDate', 'endDate'])
        
        # Clean numeric columns
        if 'onHand' in df.columns:
            df['onHand'] = pd.to_numeric(df['onHand'], errors='coerce')
        if 'Price' in df.columns:
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        
        return df
    
    def load_invoice_purchases(self):
        """Load invoice purchases data"""
        df = self.load_csv_safe(self.required_files['invoice_purchases'])
        df = self.clean_column_names(df)
        df = self.parse_dates(df, ['InvoiceDate', 'PODate', 'PayDate'])
        
        # Clean numeric columns
        numeric_cols = ['Quantity', 'Dollars', 'Freight']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def load_purchase_prices(self):
        """Load purchase prices data"""
        df = self.load_csv_safe(self.required_files['purchase_prices'])
        df = self.clean_column_names(df)
        
        # Clean numeric columns
        if 'Price' in df.columns:
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        if 'PurchasePrice' in df.columns:
            df['PurchasePrice'] = pd.to_numeric(df['PurchasePrice'], errors='coerce')
        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        
        return df
    
    def validate_data(self, data_dict):
        """Validate loaded data"""
        validation_report = {}
        
        for name, df in data_dict.items():
            validation_report[name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
                'duplicate_rows': df.duplicated().sum()
            }
        
        return validation_report
    
    def load_all_data(self):
        """Load all data files"""
        print("\n" + "="*60)
        print("Loading Slooze Inventory Data")
        print("="*60)
        
        data = {}
        
        try:
            data['sales'] = self.load_sales_data()
            data['purchases'] = self.load_purchases_data()
            data['inventory_begin'] = self.load_inventory_data('begin')
            data['inventory_end'] = self.load_inventory_data('end')
            data['invoice_purchases'] = self.load_invoice_purchases()
            data['purchase_prices'] = self.load_purchase_prices()
            
            print("\n" + "="*60)
            print("Data Loading Summary")
            print("="*60)
            
            validation = self.validate_data(data)
            for name, stats in validation.items():
                print(f"\n{name.upper()}:")
                print(f"  Rows: {stats['rows']:,}")
                print(f"  Columns: {stats['columns']}")
                print(f"  Missing: {stats['missing_percentage']:.2f}%")
                print(f"  Duplicates: {stats['duplicate_rows']}")
            
            print("\n" + "="*60)
            print("✓ All data loaded successfully!")
            print("="*60 + "\n")
            
            return data
            
        except Exception as e:
            print(f"\n✗ Error loading data: {e}")
            raise


if __name__ == '__main__':
    # Test data loading
    loader = DataLoader()
    data = loader.load_all_data()
    
    # Display sample from each dataset
    print("\n" + "="*60)
    print("Sample Data Preview")
    print("="*60)
    
    for name, df in data.items():
        print(f"\n{name.upper()} (first 3 rows):")
        print(df.head(3).to_string())