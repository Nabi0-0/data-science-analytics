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
