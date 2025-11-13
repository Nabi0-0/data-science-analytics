"""
Flask API Server for Slooze Inventory Analysis
Connects Python analytics to React dashboard
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import os
import sys
import json
from datetime import datetime
import subprocess

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.data_loader import SloozeDataLoader
from scripts import forecasting 
from scripts import abc_analysis
from scripts import eoq_reorder
from scripts import supplier_analysis

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables for data caching
cached_data = {
    'loader': None,
    'sales_df': None,
    'inventory_df': None,
    'purchases_df': None,
    'last_updated': None
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_data():
    """Load data if not already loaded"""
    if cached_data['loader'] is None or cached_data['last_updated'] is None:
        print("Loading data...")
        loader = SloozeDataLoader(data_dir='Data')
        if loader.load_all_data():
            cached_data['loader'] = loader
            cached_data['sales_df'] = loader.create_unified_sales_dataset()
            cached_data['inventory_df'] = loader.create_unified_inventory_dataset()
            cached_data['purchases_df'] = loader.create_unified_purchase_dataset()
            cached_data['last_updated'] = datetime.now().isoformat()
            return True
        return False
    return True

def read_csv_safe(filename):
    """Safely read CSV file from output directory"""
    filepath = os.path.join('output', 'csv', filename)
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return None
    return None

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'data_loaded': cached_data['last_updated'] is not None
    })

@app.route('/api/data/load', methods=['POST'])
def load_data_endpoint():
    """Load or reload data"""
    try:
        success = load_data()
        if success:
            return jsonify({
                'success': True,
                'message': 'Data loaded successfully',
                'timestamp': cached_data['last_updated']
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to load data'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/analysis/run-all', methods=['POST'])
def run_all_analyses():
    """Run all analyses"""
    try:
        # Run main.py script
        result = subprocess.run(
            ['python', 'scripts/main.py'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)
        )
        
        if result.returncode == 0:
            return jsonify({
                'success': True,
                'message': 'All analyses completed successfully',
                'output': result.stdout
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Analysis failed',
                'error': result.stderr
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/overview', methods=['GET'])
def get_overview():
    """Get overview dashboard data"""
    try:
        if not load_data():
            return jsonify({'error': 'Failed to load data'}), 500
        
        sales_df = cached_data['sales_df']
        inventory_df = cached_data['inventory_df']
        
        # Calculate KPIs
        total_revenue = sales_df['total_revenue'].sum() if 'total_revenue' in sales_df.columns else 0
        total_quantity = sales_df['quantity'].sum() if 'quantity' in sales_df.columns else 0
        unique_products = sales_df['brand'].nunique() if 'brand' in sales_df.columns else 0
        
        # Get reorder status
        reorder_df = read_csv_safe('reorder_points.csv')
        critical_items = 0
        if reorder_df is not None and 'status' in reorder_df.columns:
            critical_items = len(reorder_df[reorder_df['status'].isin(['CRITICAL', 'OUT_OF_STOCK'])])
        
        # Get supplier stats
        supplier_df = read_csv_safe('supplier_performance.csv')
        avg_lead_time = 0
        if supplier_df is not None and 'Avg_Lead_Time' in supplier_df.columns:
            avg_lead_time = supplier_df['Avg_Lead_Time'].mean()
        
        return jsonify({
            'kpis': {
                'total_revenue': float(total_revenue),
                'total_quantity': int(total_quantity),
                'unique_products': int(unique_products),
                'critical_items': int(critical_items),
                'avg_lead_time': float(avg_lead_time)
            },
            'last_updated': cached_data['last_updated']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    """Get demand forecast data"""
    try:
        df = read_csv_safe('demand_forecast_sarima.csv')
        if df is None:
            return jsonify({'error': 'Forecast data not found'}), 404
        
        # Convert to dict for JSON
        forecast_data = df.to_dict('records')
        
        # Get accuracy metrics if available
        accuracy_df = read_csv_safe('forecast_accuracy_comparison.csv')
        accuracy = None
        if accuracy_df is not None:
            accuracy = accuracy_df.to_dict('records')
        
        return jsonify({
            'forecast': forecast_data,
            'accuracy': accuracy
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/abc-analysis', methods=['GET'])
def get_abc_analysis():
    """Get ABC analysis results"""
    try:
        classification_df = read_csv_safe('abc_classification.csv')
        summary_df = read_csv_safe('abc_summary.csv')
        
        if classification_df is None or summary_df is None:
            return jsonify({'error': 'ABC analysis data not found'}), 404
        
        return jsonify({
            'classification': classification_df.head(100).to_dict('records'),  # Limit for performance
            'summary': summary_df.to_dict('records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/eoq', methods=['GET'])
def get_eoq_analysis():
    """Get EOQ analysis results"""
    try:
        df = read_csv_safe('eoq_analysis.csv')
        if df is None:
            return jsonify({'error': 'EOQ data not found'}), 404
        
        # Get top savings opportunities
        top_savings = df.nlargest(10, 'annual_savings').to_dict('records')
        
        # Calculate totals
        total_savings = df['annual_savings'].sum()
        total_products = len(df)
        
        return jsonify({
            'summary': {
                'total_savings': float(total_savings),
                'total_products': int(total_products),
                'avg_savings': float(total_savings / total_products) if total_products > 0 else 0
            },
            'top_savings': top_savings,
            'all_products': df.to_dict('records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reorder-points', methods=['GET'])
def get_reorder_points():
    """Get reorder point analysis"""
    try:
        df = read_csv_safe('reorder_points.csv')
        if df is None:
            return jsonify({'error': 'Reorder point data not found'}), 404
        
        # Get status distribution
        status_dist = df['status'].value_counts().to_dict() if 'status' in df.columns else {}
        
        # Get critical items
        critical = df[df['status'].isin(['CRITICAL', 'OUT_OF_STOCK'])].to_dict('records') if 'status' in df.columns else []
        
        return jsonify({
            'status_distribution': status_dist,
            'critical_items': critical,
            'all_items': df.to_dict('records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/suppliers', methods=['GET'])
def get_supplier_performance():
    """Get supplier performance data"""
    try:
        performance_df = read_csv_safe('supplier_performance.csv')
        lead_times_df = read_csv_safe('supplier_lead_times.csv')
        
        if performance_df is None:
            return jsonify({'error': 'Supplier data not found'}), 404
        
        return jsonify({
            'performance': performance_df.to_dict('records'),
            'lead_times': lead_times_df.to_dict('records') if lead_times_df is not None else []
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/top-products', methods=['GET'])
def get_top_products():
    """Get top performing products"""
    try:
        if not load_data():
            return jsonify({'error': 'Failed to load data'}), 500
        
        sales_df = cached_data['sales_df']
        
        # Get top products by revenue
        if 'total_revenue' in sales_df.columns:
            top_products = sales_df.groupby('brand').agg({
                'total_revenue': 'sum',
                'quantity': 'sum'
            }).nlargest(10, 'total_revenue').reset_index()
            
            return jsonify({
                'top_products': top_products.to_dict('records')
            })
        else:
            return jsonify({'error': 'Revenue data not available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/seasonal-trends', methods=['GET'])
def get_seasonal_trends():
    """Get seasonal trend data"""
    try:
        if not load_data():
            return jsonify({'error': 'Failed to load data'}), 500
        
        sales_df = cached_data['sales_df']
        
        # Aggregate by month
        if 'month' in sales_df.columns:
            monthly = sales_df.groupby('month').agg({
                'quantity': 'sum'
            }).reset_index()
            monthly['month_name'] = monthly['month'].apply(
                lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][int(x)-1]
            )
            
            return jsonify({
                'monthly_trends': monthly.to_dict('records')
            })
        else:
            return jsonify({'error': 'Date data not available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/<filename>', methods=['GET'])
def export_file(filename):
    """Export CSV or PNG file"""
    try:
        # Check in csv directory
        csv_path = os.path.join('output', 'csv', filename)
        if os.path.exists(csv_path):
            return send_file(csv_path, as_attachment=True)
        
        # Check in png directory
        png_path = os.path.join('output', 'png', filename)
        if os.path.exists(png_path):
            return send_file(png_path, as_attachment=True)
        
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/executive-summary', methods=['GET'])
def get_executive_summary():
    """Get executive summary"""
    try:
        summary_path = os.path.join('output', 'reports', 'executive_summary.txt')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                content = f.read()
            return jsonify({
                'summary': content
            })
        else:
            return jsonify({'error': 'Executive summary not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("🍷 Slooze Inventory Analysis API Server")
    print("="*80)
    print("\nStarting Flask server...")
    print("API will be available at: http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  GET  /api/health              - Health check")
    print("  POST /api/data/load           - Load data")
    print("  POST /api/analysis/run-all    - Run all analyses")
    print("  GET  /api/overview            - Dashboard overview")
    print("  GET  /api/forecast            - Demand forecast")
    print("  GET  /api/abc-analysis        - ABC analysis")
    print("  GET  /api/eoq                 - EOQ analysis")
    print("  GET  /api/reorder-points      - Reorder points")
    print("  GET  /api/suppliers           - Supplier performance")
    print("  GET  /api/top-products        - Top products")
    print("  GET  /api/seasonal-trends     - Seasonal trends")
    print("  GET  /api/executive-summary   - Executive summary")
    print("  GET  /api/export/<filename>   - Export files")
    print("\n" + "="*80)
    
    app.run(debug=True, host='0.0.0.0', port=5000)