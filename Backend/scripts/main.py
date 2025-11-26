"""
Main Analysis Script
Run all analytics locally without server
"""

import os
import json
from datetime import datetime

from data_loader import DataLoader
from forecasting import DemandForecaster
from abc_analysis import ABCAnalyzer
from eoq_opt import EOQOptimizer
from reorder_points import ReorderPointCalculator
from supplier_analysis import SupplierAnalyzer


def create_output_dirs():
    """Create output directories if they don't exist"""
    dirs = ['output', 'output/forecasts', 'output/abc_analysis', 
            'output/eoq', 'output/reorder_points', 'output/supplier_analysis']
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def save_json(data, filepath):
    """Save data as JSON"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"✓ Saved: {filepath}")


def run_all_analytics():
    """Run all analytics modules and save results"""
    print("\n" + "="*70)
    print("SLOOZE INVENTORY ANALYTICS - COMPLETE ANALYSIS")
    print("="*70)
    
    # Create output directories
    create_output_dirs()
    
    # Load data
    print("\n[1/6] Loading Data...")
    loader = DataLoader('Data')
    data = loader.load_all_data()
    
    # 1. Demand Forecasting
    print("\n[2/6] Running Demand Forecasting...")
    forecaster = DemandForecaster(data['sales'])
    
    # Get top 10 products
    top_products = data['sales'].groupby('Brand')['SalesDollars'].sum().nlargest(10).index.tolist()
    
    forecast_results = []
    for i, product in enumerate(top_products[:5], 1):
        print(f"  Forecasting {i}/5: {product}")
        try:
            forecast = forecaster.forecast_product(product, periods=30)
            if forecast:
                forecast_results.append(forecast)
        except Exception as e:
            print(f"  Warning: Could not forecast {product}: {e}")
    
    save_json(forecast_results, 'output/forecasts/demand_forecasts.json')
    
    # 2. ABC Analysis
    print("\n[3/6] Running ABC Analysis...")
    abc_analyzer = ABCAnalyzer(data['sales'], data['inventory_end'])
    abc_results = abc_analyzer.classify_inventory()
    save_json(abc_results, 'output/abc_analysis/classification.json')
    
    # 3. EOQ Optimization
    print("\n[4/6] Running EOQ Optimization...")
    eoq_optimizer = EOQOptimizer(data['sales'], data['purchases'])
    
    # Get all EOQ calculations
    eoq_df = eoq_optimizer.calculate_eoq_all_products()
    
    # Get top savings opportunities
    top_savings = eoq_optimizer.get_top_savings_opportunities(top_n=20)
    
    eoq_results = {
        'top_savings_opportunities': top_savings,
        'total_potential_savings': float(eoq_df['PotentialSavings'].sum()) if len(eoq_df) > 0 else 0,
        'total_products_analyzed': len(eoq_df)
    }
    save_json(eoq_results, 'output/eoq/optimization_results.json')
    
    # Save full EOQ data
    if len(eoq_df) > 0:
        eoq_df.to_csv('output/eoq/eoq_all_products.csv', index=False)
        print("✓ Saved: output/eoq/eoq_all_products.csv")
    
    # 4. Reorder Point Analysis
    print("\n[5/6] Running Reorder Point Analysis...")
    reorder_calc = ReorderPointCalculator(data['sales'], data['inventory_end'])
    
    # Get reorder alerts
    reorder_alerts = reorder_calc.get_reorder_alerts()
    
    # Get full inventory report
    inventory_report = reorder_calc.generate_inventory_report()
    
    reorder_results = {
        'alerts': reorder_alerts,
        'inventory_report': inventory_report
    }
    save_json(reorder_results, 'output/reorder_points/reorder_analysis.json')
    
    # Save full reorder points data
    rop_df = reorder_calc.calculate_reorder_points()
    rop_df.to_csv('output/reorder_points/reorder_points_all.csv', index=False)
    print("✓ Saved: output/reorder_points/reorder_points_all.csv")
    
    # 5. Supplier Analysis
    print("\n[6/6] Running Supplier Analysis...")
    supplier_analyzer = SupplierAnalyzer(data['purchases'], data['invoice_purchases'])
    supplier_results = supplier_analyzer.analyze_suppliers()
    save_json(supplier_results, 'output/supplier_analysis/performance_metrics.json')
    
    # Generate Summary Report
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE - SUMMARY REPORT")
    print("="*70)
    
    print("\n📊 DEMAND FORECASTING")
    print(f"  • Products forecasted: {len(forecast_results)}")
    if forecast_results:
        avg_forecast = sum(f['total_forecast_demand'] for f in forecast_results) / len(forecast_results)
        print(f"  • Average 30-day forecast: {avg_forecast:.0f} units")
    
    print("\n📦 ABC ANALYSIS")
    if abc_results and 'classification_summary' in abc_results:
        for cls in abc_results['classification_summary']:
            print(f"  • Class {cls['Class']}: {cls['ProductCount']} products, "
                  f"{cls['RevenuePercentage']:.1f}% revenue")
    
    print("\n💰 EOQ OPTIMIZATION")
    print(f"  • Products analyzed: {eoq_results['total_products_analyzed']}")
    print(f"  • Total potential savings: ${eoq_results['total_potential_savings']:,.0f}")
    
    print("\n🔔 REORDER POINT ANALYSIS")
    print(f"  • Products needing reorder: {reorder_alerts.get('total_items_needing_reorder', 0)}")
    print(f"  • Critical items: {len(reorder_alerts.get('critical_items', []))}")
    print(f"  • Warning items: {len(reorder_alerts.get('warning_items', []))}")
    print(f"  • Inventory health score: {inventory_report.get('inventory_health_score', 0):.0f}%")
    
    print("\n🏢 SUPPLIER ANALYSIS")
    if supplier_results and 'summary' in supplier_results:
        summary = supplier_results['summary']
        print(f"  • Total suppliers: {summary.get('total_suppliers', 0)}")
        print(f"  • Average reliability: {summary.get('avg_reliability_score', 0):.0f}%")
        print(f"  • Excellent suppliers: {summary.get('excellent_suppliers', 0)}")
        print(f"  • Poor performers: {summary.get('poor_suppliers', 0)}")
    
    print("\n" + "="*70)
    print("✓ All results saved to output/ directory")
    print(f"✓ Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    return {
        'forecast': forecast_results,
        'abc': abc_results,
        'eoq': eoq_results,
        'reorder': reorder_results,
        'supplier': supplier_results
    }


if __name__ == '__main__':
    try:
        results = run_all_analytics()
        
        # Save complete results
        save_json(results, 'output/complete_analysis_results.json')
        
        print("\n✅ SUCCESS: Complete analysis finished!")
        print("📁 Check the 'output/' directory for all results\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: Analysis failed")
        print(f"Error details: {e}")
        print("\nPlease ensure:")
        print("  1. All CSV files are in the Data/ directory")
        print("  2. Virtual environment is activated")
        print("  3. All dependencies are installed (pip install -r requirements.txt)\n")
        raise