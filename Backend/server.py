"""
Slooze Inventory Analytics - Main Backend Server
Flask API for serving analytics endpoints
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import sys

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from data_loader import DataLoader
from forecasting import DemandForecaster
from abc_analysis import ABCAnalyzer
from eoq_opt import EOQOptimizer
from reorder_points import ReorderPointCalculator
from supplier_analysis import SupplierAnalyzer

app = Flask(__name__)
CORS(app)

# Initialize data loader
data_loader = DataLoader('Data')

# Cache for loaded data
_data_cache = None

def get_data():
    """Load and cache data"""
    global _data_cache
    if _data_cache is None:
        _data_cache = data_loader.load_all_data()
    return _data_cache


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Slooze Analytics API is running'
    })


@app.route('/api/overview', methods=['GET'])
def get_overview():
    """Get dashboard overview statistics"""
    try:
        data = get_data()
        sales_df = data['sales']
        purchases_df = data['purchases']
        inventory_begin = data['inventory_begin']
        inventory_end = data['inventory_end']
        
        overview = {
            'total_sales': float(sales_df['SalesDollars'].sum()),
            'total_revenue': float(sales_df['SalesDollars'].sum()),
            'total_units_sold': int(sales_df['SalesQuantity'].sum()),
            'total_purchases': float(purchases_df['Dollars'].sum()) if 'Dollars' in purchases_df.columns else 0,
            'unique_products': int(sales_df['Brand'].nunique()),
            'unique_stores': int(sales_df['Store'].nunique()),
            'inventory_value_begin': float((inventory_begin['onHand'] * inventory_begin['Price']).sum()),
            'inventory_value_end': float((inventory_end['onHand'] * inventory_end['Price']).sum()),
            'avg_transaction_value': float(sales_df['SalesDollars'].mean()),
            'top_categories': sales_df.groupby('Classification')['SalesDollars'].sum().nlargest(5).to_dict()
        }
        
        return jsonify(overview)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    """Get demand forecasts"""
    try:
        data = get_data()
        forecaster = DemandForecaster(data['sales'])
        
        # Get top products by revenue
        top_products = data['sales'].groupby('Brand')['SalesDollars'].sum().nlargest(10).index.tolist()
        
        forecasts = []
        for product in top_products[:5]:  # Forecast top 5 products
            try:
                forecast_data = forecaster.forecast_product(product, periods=30)
                if forecast_data:
                    forecasts.append(forecast_data)
            except Exception as e:
                print(f"Error forecasting {product}: {e}")
                continue
        
        return jsonify({
            'forecasts': forecasts,
            'summary': {
                'products_forecasted': len(forecasts),
                'forecast_horizon': 30
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/abc-analysis', methods=['GET'])
def get_abc_analysis():
    """Get ABC inventory classification"""
    try:
        data = get_data()
        analyzer = ABCAnalyzer(data['sales'], data['inventory_end'])
        
        abc_result = analyzer.classify_inventory()
        
        return jsonify(abc_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/eoq', methods=['GET'])
def get_eoq():
    """Get Economic Order Quantity analysis"""
    try:
        data = get_data()
        optimizer = EOQOptimizer(data['sales'], data['purchases'])
        
        eoq_result = optimizer.calculate_eoq()
        
        return jsonify(eoq_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reorder-points', methods=['GET'])
def get_reorder_points():
    """Get reorder point calculations"""
    try:
        data = get_data()
        calculator = ReorderPointCalculator(data['sales'], data['inventory_end'])
        
        reorder_result = calculator.calculate_reorder_points()
        
        return jsonify(reorder_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/supplier-analysis', methods=['GET'])
def get_supplier_analysis():
    """Get supplier performance analysis"""
    try:
        data = get_data()
        analyzer = SupplierAnalyzer(data['purchases'], data['invoice_purchases'])
        
        supplier_result = analyzer.analyze_suppliers()
        
        return jsonify(supplier_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/data-quality', methods=['GET'])
def get_data_quality():
    """Get data quality metrics"""
    try:
        data = get_data()
        
        quality_metrics = {}
        for name, df in data.items():
            quality_metrics[name] = {
                'total_records': len(df),
                'missing_values': df.isnull().sum().to_dict(),
                'columns': df.columns.tolist(),
                'memory_usage': float(df.memory_usage(deep=True).sum() / 1024 / 1024)  # MB
            }
        
        return jsonify(quality_metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Slooze Analytics Server...")
    print("Server running on http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  GET /api/health - Health check")
    print("  GET /api/overview - Dashboard overview")
    print("  GET /api/forecast - Demand forecasts")
    print("  GET /api/abc-analysis - ABC classification")
    print("  GET /api/eoq - EOQ optimization")
    print("  GET /api/reorder-points - Reorder calculations")
    print("  GET /api/supplier-analysis - Supplier metrics")
    print("  GET /api/data-quality - Data quality report")
    
    app.run(debug=True, host='0.0.0.0', port=5000)