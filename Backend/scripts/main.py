"""
main.py
Main script to run all Slooze inventory analyses
"""

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import pandas as pd

# Import all modules
from utils import setup_output_directories, print_section_header, create_business_summary
from data_loader import SloozeDataLoader
import forecasting
import abc_analysis
import eoq_reorder
import supplier_analysis
from reorder_analysis import generate_reorder_points

def print_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🍷 SLOOZE INVENTORY ANALYSIS SYSTEM 🍷                    ║
║                                                                              ║
║              Wine & Spirits Retail Optimization & Analytics                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)
    print(f"\n📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Data Source: ../Data/")
    print("\n" + "="*80)

def main():
    """
    Main execution function - runs all analyses
    """
    
    # Print banner
    print_banner()
    
    # Setup output directories
    setup_output_directories()
    
    # ====================
    # STEP 1: LOAD DATA
    # ====================
    print_section_header("STEP 1: DATA LOADING & PREPARATION")
    
    loader = SloozeDataLoader(data_dir='Data')
    
    if not loader.load_all_data():
        print("\n❌ Failed to load data. Exiting...")
        return
    
    # Get data summary
    loader.get_data_summary()
    
    # Create unified datasets
    print("\n" + "-"*80)
    sales_df = loader.create_unified_sales_dataset()
    inventory_df = loader.create_unified_inventory_dataset()
    purchases_df = loader.create_unified_purchase_dataset()
    
    # Export cleaned data
    loader.export_cleaned_data()
    
    # ====================
    # STEP 2: DEMAND FORECASTING
    # ====================
    print_section_header("STEP 2: DEMAND FORECASTING")
    
    try:
        forecaster = forecasting.main(sales_df)
        print("✅ Demand forecasting completed successfully")
    except Exception as e:
        print(f"⚠️ Demand forecasting encountered an issue: {e}")
        forecaster = None
    
    # ====================
    # STEP 3: ABC ANALYSIS
    # ====================
    print_section_header("STEP 3: ABC ANALYSIS")
    
    try:
        abc_analyzer = abc_analysis.main(sales_df, inventory_df)
        print("✅ ABC analysis completed successfully")
    except Exception as e:
        print(f"⚠️ ABC analysis encountered an issue: {e}")
        abc_analyzer = None
    
    # ====================
    # STEP 4: EOQ & REORDER POINTS
    # ====================
    print_section_header("STEP 4: EOQ & REORDER POINT ANALYSIS")
    
    try:
        eoq_analyzer = eoq_reorder.main(sales_df, inventory_df, purchases_df)
        print("✅ EOQ and reorder point analysis completed successfully")
    except Exception as e:
        print(f"⚠️ EOQ analysis encountered an issue: {e}")
        eoq_analyzer = None
    
    # ====================
    # STEP 5: SUPPLIER ANALYSIS
    # ====================
    print_section_header("STEP 5: SUPPLIER PERFORMANCE ANALYSIS")
    
    try:
        supplier_analyzer = supplier_analysis.main(purchases_df, inventory_df)
        print("✅ Supplier analysis completed successfully")
    except Exception as e:
        print(f"⚠️ Supplier analysis encountered an issue: {e}")
        supplier_analyzer = None
    
    # ====================
    # STEP 6: EXECUTIVE SUMMARY
    # ====================
    print_section_header("STEP 6: GENERATING EXECUTIVE SUMMARY")
    
    generate_executive_summary(
        sales_df, inventory_df, 
        forecaster, abc_analyzer, eoq_analyzer, supplier_analyzer
    )
    
    # ====================
    # COMPLETION
    # ====================
    print_section_header("✅ ANALYSIS COMPLETE!")
    
    print("\n📁 Generated Output Files:")
    print("\n   CSV Files (data/):")
    print("      • abc_classification.csv")
    print("      • abc_summary.csv")
    print("      • demand_forecast_sarima.csv")
    print("      • demand_forecast_exp_smoothing.csv")
    print("      • eoq_analysis.csv")
    print("      • reorder_points.csv")
    print("      • supplier_lead_times.csv")
    print("      • supplier_performance.csv")
    print("      • All cleaned data files")
    
    print("\n   Visualizations (png/):")
    print("      • time_series_decomposition.png")
    print("      • demand_forecast.png")
    print("      • abc_analysis.png")
    print("      • eoq_analysis.png")
    print("      • reorder_point_analysis.png")
    print("      • supplier_analysis.png")
    
    print("\n   Reports (reports/):")
    print("      • executive_summary.txt")
    print("      • business_metrics.txt")
    
    print("\n🎯 Next Steps & Recommendations:")
    print_next_steps(eoq_analyzer, abc_analyzer, supplier_analyzer)
    
    print("\n" + "="*80)
    print("Thank you for using Slooze Inventory Analysis System!")
    print("For questions: careers@slooze.xyz")
    print("="*80 + "\n")


def generate_executive_summary(sales_df, inventory_df, forecaster, 
                               abc_analyzer, eoq_analyzer, supplier_analyzer):
    """
    Generate comprehensive executive summary
    """
    
    summary_lines = []
    summary_lines.append("="*80)
    summary_lines.append("SLOOZE INVENTORY ANALYSIS - EXECUTIVE SUMMARY")
    summary_lines.append("="*80)
    summary_lines.append(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Sales Summary
    if sales_df is not None and len(sales_df) > 0:
        date_col = [col for col in sales_df.columns if 'date' in col.lower()]
        if date_col:
            date_col = date_col[0]
            start_date = sales_df[date_col].min()
            end_date = sales_df[date_col].max()
            summary_lines.append(f"Analysis Period: {start_date} to {end_date}")
        
        summary_lines.append(f"\n{'='*80}")
        summary_lines.append("KEY PERFORMANCE INDICATORS")
        summary_lines.append(f"{'='*80}")
        
        total_records = len(sales_df)
        summary_lines.append(f"Total Transactions: {total_records:,}")
        
        if 'quantity' in sales_df.columns:
            total_quantity = sales_df['quantity'].sum()
            summary_lines.append(f"Total Units Sold: {total_quantity:,.0f}")
        
        if 'total_revenue' in sales_df.columns:
            total_revenue = sales_df['total_revenue'].sum()
            summary_lines.append(f"Total Revenue: ${total_revenue:,.2f}")
        
        # Unique products
        product_cols = [col for col in sales_df.columns if 'brand' in col.lower() or 'product' in col.lower()]
        if product_cols:
            unique_products = sales_df[product_cols[0]].nunique()
            summary_lines.append(f"Unique Products: {unique_products:,}")
    
    # ABC Analysis Results
    if abc_analyzer and abc_analyzer.abc_results is not None:
        summary_lines.append(f"\n{'='*80}")
        summary_lines.append("ABC ANALYSIS INSIGHTS")
        summary_lines.append(f"{'='*80}")
        
        abc_summary = abc_analyzer.abc_results.groupby('abc_category').agg({
            abc_analyzer.abc_results.columns[0]: 'count',
            'total_revenue': 'sum'
        }).reset_index()
        
        for _, row in abc_summary.iterrows():
            cat = row['abc_category']
            count = row[abc_analyzer.abc_results.columns[0]]
            revenue = row['total_revenue']
            summary_lines.append(f"\nCategory {cat}:")
            summary_lines.append(f"  Products: {count:,}")
            summary_lines.append(f"  Revenue: ${revenue:,.2f}")
    
    # EOQ Results
    if eoq_analyzer and eoq_analyzer.eoq_results is not None:
        summary_lines.append(f"\n{'='*80}")
        summary_lines.append("EOQ OPTIMIZATION POTENTIAL")
        summary_lines.append(f"{'='*80}")
        
        total_savings = eoq_analyzer.eoq_results['annual_savings'].sum()
        summary_lines.append(f"\nTotal Annual Savings Potential: ${total_savings:,.2f}")
        summary_lines.append(f"Average Savings per Product: ${eoq_analyzer.eoq_results['annual_savings'].mean():,.2f}")
    
    # Reorder Point Critical Items
    if eoq_analyzer and eoq_analyzer.reorder_results is not None:
        summary_lines.append(f"\n{'='*80}")
        summary_lines.append("INVENTORY STATUS")
        summary_lines.append(f"{'='*80}")
        
        if 'status' in eoq_analyzer.reorder_results.columns:
            status_counts = eoq_analyzer.reorder_results['status'].value_counts()
            summary_lines.append("\nCurrent Status:")
            for status, count in status_counts.items():
                summary_lines.append(f"  {status}: {count:,} items")
    
    # Supplier Performance
    if supplier_analyzer and supplier_analyzer.supplier_metrics is not None:
        summary_lines.append(f"\n{'='*80}")
        summary_lines.append("SUPPLIER PERFORMANCE")
        summary_lines.append(f"{'='*80}")
        
        avg_lead_time = supplier_analyzer.supplier_metrics['Avg_Lead_Time'].mean()
        avg_on_time = supplier_analyzer.supplier_metrics['On_Time_Rate'].mean()
        
        summary_lines.append(f"\nAverage Lead Time: {avg_lead_time:.1f} days")
        summary_lines.append(f"Average On-Time Rate: {avg_on_time:.1f}%")
        
        top_supplier = supplier_analyzer.supplier_metrics.iloc[0]
        summary_lines.append(f"\nTop Performing Supplier: {top_supplier['Supplier']}")
        summary_lines.append(f"  Score: {top_supplier['Composite_Score']:.1f}/100")
    
    # Forecasting Results
    if forecaster and forecaster.forecast_results:
        summary_lines.append(f"\n{'='*80}")
        summary_lines.append("DEMAND FORECAST")
        summary_lines.append(f"{'='*80}")
        
        if 'sarima' in forecaster.forecast_results:
            metrics = forecaster.forecast_results['sarima']['metrics']
            summary_lines.append(f"\nForecast Accuracy (MAPE): {metrics['mape']:.2f}%")
            summary_lines.append(f"Model: SARIMA")
    
    # Write to file
    summary_text = "\n".join(summary_lines)
    
    with open('output/reports/executive_summary.txt', 'w') as f:
        f.write(summary_text)
    
    print("\n📄 Executive summary generated")
    print(summary_text)


def print_next_steps(eoq_analyzer, abc_analyzer, supplier_analyzer):
    """Print actionable next steps"""
    
    print("\n   IMMEDIATE ACTIONS (0-7 days):")
    
    # Critical inventory items
    if eoq_analyzer and eoq_analyzer.reorder_results is not None:
        critical = eoq_analyzer.reorder_results[
            eoq_analyzer.reorder_results['status'].isin(['CRITICAL', 'OUT_OF_STOCK'])
        ]
        if len(critical) > 0:
            print(f"      1. ⚠️  URGENT: Place orders for {len(critical)} critical items")
            print(f"         Top priority: {critical.iloc[0]['brand'] if 'brand' in critical.columns else 'Check reorder_points.csv'}")
    
    # EOQ implementation
    if eoq_analyzer and eoq_analyzer.eoq_results is not None:
        top_saving = eoq_analyzer.eoq_results.nlargest(1, 'annual_savings').iloc[0]
        print(f"      2. 💰 Implement EOQ for top product (saves ${top_saving['annual_savings']:,.0f}/year)")
    
    print("\n   SHORT-TERM ACTIONS (1-4 weeks):")
    print("      3. Review and renegotiate with underperforming suppliers")
    print("      4. Implement daily monitoring for Category A items")
    print("      5. Set up automated reorder point alerts")
    
    print("\n   LONG-TERM INITIATIVES (1-6 months):")
    print("      6. Transition to JIT ordering for high-turnover items")
    print("      7. Develop supplier scorecard and SLA system")
    print("      8. Consider liquidation strategy for slow-moving inventory")
    print("      9. Implement demand sensing and advanced forecasting")
    print("      10. Optimize warehouse layout based on ABC classification")

    import pandas as pd
import os

def generate_reorder_points(data_dir='output/csv'):
    """
    Generate reorder_points.csv using sales, inventory, and supplier data.
    """

    # --- Load required CSVs ---
    try:
        sales_df = pd.read_csv(os.path.join(data_dir, 'sales_cleaned.csv'))
        inventory_df = pd.read_csv(os.path.join(data_dir, 'inventory_ending_cleaned.csv'))
        supplier_df = pd.read_csv(os.path.join(data_dir, 'supplier_lead_times.csv'))
    except FileNotFoundError as e:
        print("Missing required CSV:", e)
        return None

    # --- Prepare sales summary ---
    sales_summary = (
        sales_df.groupby('brand')['quantity']
        .sum()
        .reset_index()
        .rename(columns={'quantity': 'total_sold'})
    )

    # --- Compute average daily demand (assuming 365 days) ---
    sales_summary['avg_daily_demand'] = sales_summary['total_sold'] / 365

    # --- Lead time (default 7 days if missing) ---
    lead_time = supplier_df.groupby('Supplier')['Avg_Lead_Time'].mean().reset_index()
    avg_lead_time = lead_time['Avg_Lead_Time'].mean() if not lead_time.empty else 7

    # --- Merge with inventory to get stock on hand ---
    if 'brand' not in inventory_df.columns:
        inventory_df['brand'] = range(len(inventory_df))
    stock_df = inventory_df.groupby('brand')['stock_on_hand'].sum().reset_index()

    df = pd.merge(sales_summary, stock_df, on='brand', how='left')
    df['lead_time'] = avg_lead_time

    # --- Calculate reorder point ---
    df['reorder_point'] = (df['avg_daily_demand'] * df['lead_time'] * 1.2).round(2)
    df['on_hand'] = df['stock_on_hand']

    # --- Determine stock status ---
    def classify_status(row):
        if row['on_hand'] <= 0:
            return 'OUT_OF_STOCK'
        elif row['on_hand'] < 0.5 * row['reorder_point']:
            return 'CRITICAL'
        elif row['on_hand'] < row['reorder_point']:
            return 'WARNING'
        else:
            return 'HEALTHY'

    df['status'] = df.apply(classify_status, axis=1)

    # --- Estimate stockout risk ---
    df['stockout_risk'] = (1 - (df['on_hand'] / (df['reorder_point'] + 1))) * 100
    df['stockout_risk'] = df['stockout_risk'].clip(lower=0, upper=100).round(1)

    # --- Select relevant columns ---
    final_df = df[['brand', 'on_hand', 'reorder_point', 'status', 'stockout_risk']]

    # --- Save output ---
    output_path = os.path.join(data_dir, 'reorder_points.csv')
    final_df.to_csv(output_path, index=False)
    print(f"✅ Reorder points file generated: {output_path}")

    return final_df



if __name__ == "__main__":
    main()