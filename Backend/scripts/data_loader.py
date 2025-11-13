"""
data_loader.py
Load and clean all Slooze inventory datasets
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from scripts.utils import (
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
        self.data_dir = data_dir
        self.sales_df = None
        self.inventory_beg_df = None
        self.inventory_end_df = None
        self.purchases_df = None
        self.invoice_purchases_df = None
        self.purchase_prices_df = None

    # --------------------------------------------------------
    #  LOAD ALL DATASETS
    # --------------------------------------------------------
    def load_all_data(self):
        print_section_header("📂 LOADING DATA FILES")

        try:
            log_analysis_step("Loading Sales Data")
            self.sales_df = self._load_sales()

            log_analysis_step("Loading Beginning Inventory")
            self.inventory_beg_df = self._load_beginning_inventory()

            log_analysis_step("Loading Ending Inventory")
            self.inventory_end_df = self._load_ending_inventory()

            log_analysis_step("Loading Purchases")
            self.purchases_df = self._load_purchases()

            log_analysis_step("Loading Invoice Purchases")
            self.invoice_purchases_df = self._load_invoice_purchases()

            log_analysis_step("Loading Purchase Prices")
            self.purchase_prices_df = self._load_purchase_prices()

            print("\n✅ All data files loaded successfully!")
            return True

        except FileNotFoundError as e:
            print(f"\n❌ Error loading data: {e}")
            print("Please ensure all CSV files are in the 'Data/' directory")
            return False

    # --------------------------------------------------------
    #  INDIVIDUAL LOADERS
    # --------------------------------------------------------
    def _load_sales(self):
        df = pd.read_csv(f'{self.data_dir}/SalesFINAL12312016.csv')
        df = standardize_column_names(df)

        # Rename known columns to standard ones
        rename_map = {
            'salesquantity': 'quantity',
            'salesdollars': 'total_revenue',
            'salesprice': 'price',
            'salesdate': 'date'
        }
        df.rename(columns=rename_map, inplace=True)

        # Parse date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Clean currency columns
        for col in ['price', 'total_revenue']:
            if col in df.columns:
                df[col] = clean_currency_column(df[col])

        validate_data_quality(df, "Sales Data")
        return df

    def _load_beginning_inventory(self):
        df = pd.read_csv(f'{self.data_dir}/BegInvFINAL12312016.csv')
        df = standardize_column_names(df)
        # No critical renames here, only cleaning
        self._clean_numeric_text_mix(df, "Beginning Inventory")
        validate_data_quality(df, "Beginning Inventory")
        return df

    def _load_ending_inventory(self):
        df = pd.read_csv(f'{self.data_dir}/EndInvFINAL12312016.csv')
        df = standardize_column_names(df)

        # Rename to consistent field names
        rename_map = {
            'onhand': 'stock_qty',
            'price': 'unit_price',
            'enddate': 'date'
        }
        df.rename(columns=rename_map, inplace=True)

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        self._clean_numeric_text_mix(df, "Ending Inventory")
        validate_data_quality(df, "Ending Inventory")
        return df

    def _load_purchases(self):
        df = pd.read_csv(f'{self.data_dir}/PurchasesFINAL12312016.csv')
        df = standardize_column_names(df)

        # Rename columns
        rename_map = {
            'podate': 'order_date',
            'receivingdate': 'delivery_date',
            'purchaseprice': 'unit_cost',
            'quantity': 'quantity',
            'dollars': 'total_cost'
        }
        df.rename(columns=rename_map, inplace=True)

        # Parse dates
        for col in ['order_date', 'delivery_date', 'invoicedate', 'paydate']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Clean numeric / currency fields
        for col in ['unit_cost', 'total_cost']:
            if col in df.columns:
                df[col] = clean_currency_column(df[col])

        validate_data_quality(df, "Purchases Data")
        return df

    def _load_invoice_purchases(self):
        df = pd.read_csv(f'{self.data_dir}/InvoicePurchases12312016.csv')
        df = standardize_column_names(df)
        self._clean_numeric_text_mix(df, "Invoice Purchases")
        validate_data_quality(df, "Invoice Purchases")
        return df

    def _load_purchase_prices(self):
        df = pd.read_csv(f'{self.data_dir}/2017PurchasePricesDec.csv')
        df = standardize_column_names(df)
        for col in df.columns:
            if 'price' in col or 'cost' in col:
                df[col] = clean_currency_column(df[col])
        validate_data_quality(df, "Purchase Prices")
        return df

    # --------------------------------------------------------
    #  DATA UNIFICATION
    # --------------------------------------------------------
    def create_unified_sales_dataset(self):
        print_section_header("🔧 CREATING UNIFIED SALES DATASET")
        sales = self.sales_df.copy()

        # Ensure revenue exists
        if 'total_revenue' not in sales.columns and {'quantity', 'price'} <= set(sales.columns):
            sales['total_revenue'] = sales['quantity'] * sales['price']

        # Add date-based breakdowns
        if 'date' in sales.columns:
            sales['year'] = sales['date'].dt.year
            sales['month'] = sales['date'].dt.month
            sales['quarter'] = sales['date'].dt.quarter
            sales['day_of_week'] = sales['date'].dt.dayofweek
            sales['week_of_year'] = sales['date'].dt.isocalendar().week

        print(f"✅ Unified sales dataset created: {len(sales):,} records")
        return sales

    def create_unified_inventory_dataset(self):
        print_section_header("🔧 CREATING UNIFIED INVENTORY DATASET")
        inventory = self.inventory_end_df.copy()

        if 'product_id' not in inventory.columns:
            inventory['product_id'] = 'P' + (inventory.index + 1).astype(str).str.zfill(5)

        print(f"✅ Unified inventory dataset created: {len(inventory):,} records")
        return inventory

    def create_unified_purchase_dataset(self):
        print_section_header("🔧 CREATING UNIFIED PURCHASE DATASET")
        purchases = self.purchases_df.copy()

        # Calculate lead time
        if {'order_date', 'delivery_date'} <= set(purchases.columns):
            purchases["lead_time_days"] = (
                purchases["delivery_date"] - purchases["order_date"]
            ).dt.days
            print("   ✅ Lead time calculated")

        print(f"✅ Unified purchase dataset created: {len(purchases):,} records")
        return purchases

    # --------------------------------------------------------
    #  UTILITIES
    # --------------------------------------------------------
    def _clean_numeric_text_mix(self, df, label):
        """Clean columns that may mix numbers and text"""
        for col in df.columns:
            if df[col].dtype == 'object' and col not in ['brand', 'description', 'vendor', 'classification']:
                ratio = df[col].astype(str).str.replace(r'[^0-9.\-]', '', regex=True).str.len().gt(0).mean()
                if ratio > 0.5:
                    try:
                        df[col] = clean_currency_column(df[col])
                    except Exception:
                        print(f"⚠️  Skipped non-numeric column during cleaning: {col}")
                else:
                    print(f"⚠️  Detected text column, skipping: {col}")

    def get_data_summary(self):
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
        print(pd.DataFrame(summary).to_string(index=False))

    def export_cleaned_data(self, output_dir='output/csv'):
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
                path = os.path.join(output_dir, filename)
                df.to_csv(path, index=False)
                print(f"✅ Exported: {path}")


# --------------------------------------------------------
# MAIN TEST RUNNER
# --------------------------------------------------------
def main():
    loader = SloozeDataLoader()
    if loader.load_all_data():
        loader.get_data_summary()
        loader.create_unified_sales_dataset()
        loader.create_unified_inventory_dataset()
        loader.create_unified_purchase_dataset()
        loader.export_cleaned_data()
        print("\n✅ Data loading complete!")
        return loader
    else:
        print("\n❌ Data loading failed!")
        return None


if __name__ == "__main__":
    loader = main()
