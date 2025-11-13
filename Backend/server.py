"""
Backend server for Slooze Inventory Analysis
Exposes REST endpoints that call into your existing scripts in scripts/
"""

import os
import traceback
from flask import Flask, jsonify, request
from flask_cors import CORS

# --- adjust path if needed ---
# If Backend/server.py is in Backend/ and your scripts are in Backend/scripts/,
# python package import style should be: from scripts.data_loader import ...
# Make sure Backend is run as working dir or add to sys.path if needed.

app = Flask(__name__)
CORS(app)

# Lazy-loaded cached objects
_state = {
    "loader": None,
    "sales_df": None,
    "inventory_df": None,
    "purchases_df": None,
    "forecast_result": None,
    "abc_result": None,
    "eoq_result": None,
    "reorder_result": None,
    "supplier_result": None
}

# Import your modules (deferred to avoid import-time errors when dependencies missing)
def safe_imports():
    global SloozeDataLoader, forecasting, abc_analysis, eoq_reorder, supplier_analysis, main_script
    try:
        from scripts.data_loader import SloozeDataLoader
        import scripts.forecasting as forecasting
        import scripts.abc_analysis as abc_analysis
        import scripts.eoq_reorder as eoq_reorder
        import scripts.supplier_analysis as supplier_analysis
        import scripts.main as main_script
    except Exception as e:
        # re-raise so calling code sees it
        raise

# Utility to convert pandas objects to JSON-serializable forms
def to_records(obj):
    try:
        import pandas as pd
    except Exception:
        return obj
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return obj.fillna("").to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.fillna("").to_dict()
    # fallback for lists/dicts already JSON friendly
    return obj

@app.route("/api/load", methods=["POST", "GET"])
def api_load_data():
    """
    Load data using SloozeDataLoader (idempotent). 
    GET returns status, POST forces reload.
    """
    try:
        safe_imports()
        force = request.method == "POST" or request.args.get("force", "0") == "1"

        if _state["loader"] is None or force:
            loader = SloozeDataLoader(data_dir=os.path.join(os.path.dirname(__file__), "..", "Data"))
            ok = loader.load_all_data()
            if not ok:
                return jsonify({"error": "data load failed - check Data/ files and logs"}), 500

            # store loaded dataframes & loader
            _state["loader"] = loader
            _state["sales_df"] = loader.sales_df
            _state["inventory_df"] = loader.inventory_end_df or loader.inventory_beg_df
            _state["purchases_df"] = loader.purchases_df

            # clear previous derived results
            _state.update({
                "forecast_result": None,
                "abc_result": None,
                "eoq_result": None,
                "reorder_result": None,
                "supplier_result": None
            })

            return jsonify({
                "status": "loaded",
                "sales_records": int(len(_state["sales_df"])) if _state["sales_df"] is not None else 0,
                "inventory_records": int(len(_state["inventory_df"])) if _state["inventory_df"] is not None else 0,
                "purchases_records": int(len(_state["purchases_df"])) if _state["purchases_df"] is not None else 0
            })
        else:
            return jsonify({"status": "already_loaded"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/summary")
def api_summary():
    """Lightweight summary for dashboard top cards"""
    loader = _state.get("loader")
    if loader is None:
        return jsonify({"error": "data not loaded. Call /api/load (GET) or /api/load?force=1 (POST)"}), 400

    sales = _state["sales_df"]
    inventory = _state["inventory_df"]
    purchases = _state["purchases_df"]

    resp = {
        "total_revenue": float(sales["total_revenue"].sum()) if sales is not None and "total_revenue" in sales.columns else None,
        "total_transactions": int(len(sales)) if sales is not None else 0,
        "active_skus": int(sales["brand"].nunique()) if sales is not None and "brand" in sales.columns else None,
        "inventory_count": int(len(inventory)) if inventory is not None else 0,
        "avg_lead_time_days": float(purchases["delivery_date"].sub(purchases["order_date"]).dt.days.mean()) if purchases is not None and {"delivery_date","order_date"} <= set(purchases.columns) else None
    }
    return jsonify(resp)


@app.route("/api/forecast")
def api_forecast():
    """Run forecasting.main(sales_df) or return cached"""
    try:
        safe_imports()
        if _state["sales_df"] is None:
            return jsonify({"error": "sales data not loaded. Call /api/load first."}), 400

        if _state["forecast_result"] is None:
            # your forecasting.main should accept sales_df and return an object or dict
            try:
                res = forecasting.main(_state["sales_df"])
            except TypeError:
                # if forecasting.main expects path/file instead try calling other interface
                res = forecasting.main()
            # Try to extract reasonable output
            if hasattr(res, "forecast_results"):
                _state["forecast_result"] = res.forecast_results
            else:
                _state["forecast_result"] = res

        return jsonify(to_records(_state["forecast_result"]))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/abc")
def api_abc():
    """Run ABC analysis and return classification and summary"""
    try:
        safe_imports()
        if _state["sales_df"] is None:
            return jsonify({"error": "sales data not loaded. Call /api/load first."}), 400

        if _state["abc_result"] is None:
            res = abc_analysis.main(_state["sales_df"], _state["inventory_df"])
            # many abc modules return an object with abc_results DataFrame
            if hasattr(res, "abc_results"):
                _state["abc_result"] = {
                    "classification": res.abc_results.to_dict(orient="records"),
                }
            else:
                _state["abc_result"] = res

        return jsonify(_state["abc_result"])
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/top_products")
def api_top_products():
    """Return top products by revenue from sales_df (fast)"""
    try:
        if _state["sales_df"] is None:
            return jsonify({"error": "sales data not loaded. Call /api/load first."}), 400
        df = _state["sales_df"]
        # try common columns: brand/description/total_revenue
        key_col = "brand" if "brand" in df.columns else (df.columns[0] if len(df.columns) else None)
        if key_col is None or "total_revenue" not in df.columns:
            return jsonify([])

        top = df.groupby(key_col)["total_revenue"].sum().sort_values(ascending=False).head(10).reset_index()
        top = top.rename(columns={key_col: "name", "total_revenue": "revenue"})
        return jsonify(to_records(top))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/eoq")
def api_eoq():
    """Run EOQ & reorder calculations (calls your eoq_reorder.main)"""
    try:
        safe_imports()
        if _state["eoq_result"] is None:
            res = eoq_reorder.main(_state["sales_df"], _state["inventory_df"], _state["purchases_df"])
            # expect res to have eoq_results, reorder_results attributes (DataFrames)
            out = {}
            if hasattr(res, "eoq_results"):
                out["eoq_results"] = res.eoq_results.to_dict(orient="records")
            if hasattr(res, "reorder_results"):
                out["reorder_results"] = res.reorder_results.to_dict(orient="records")
            _state["eoq_result"] = out
        return jsonify(_state["eoq_result"])
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/reorder")
def api_reorder():
    """Return reorder points (if available from EOQ or separate module)"""
    try:
        # serve cached reorder if present
        if _state.get("eoq_result") and _state["eoq_result"].get("reorder_results"):
            return jsonify(_state["eoq_result"]["reorder_results"])
        # else run eoq
        return api_eoq()
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/supplier")
def api_supplier():
    """Run supplier performance analysis"""
    try:
        safe_imports()
        if _state["supplier_result"] is None:
            res = supplier_analysis.main(_state["purchases_df"], _state["inventory_df"])
            if hasattr(res, "supplier_metrics"):
                _state["supplier_result"] = res.supplier_metrics.to_dict(orient="records")
            else:
                _state["supplier_result"] = res
        return jsonify(_state["supplier_result"])
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/run_all", methods=["POST"])
def api_run_all():
    """
    Run the full pipeline (like main.py orchestration).
    This can be heavy — it's provided as POST so frontend can trigger it intentionally.
    """
    try:
        safe_imports()
        # call your main orchestration if available
        try:
            main_script.main()
            return jsonify({"status": "ran_main"})
        except Exception as e:
            # fallback: run modules stepwise
            api_load_data()
            api_forecast()
            api_abc()
            api_eoq()
            api_supplier()
            return jsonify({"status": "ran_individual_steps"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/")
def health():
    return jsonify({"status": "ok", "message": "Slooze backend up"})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
