# 🍷 Slooze Inventory Analysis - Data Science Solution

## Wine & Spirits Retail Optimization System

A complete analytics and automation system for inventory optimization, sales forecasting, supplier performance tracking, and cost reduction — designed for a large-scale retail environment.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Format](#data-format)
- [Methodology](#methodology)
- [Output Files](#output-files)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Author](#author)

---

## 🎯 Overview

This project streamlines end-to-end **retail inventory management** using advanced data analytics.  
It replaces manual spreadsheet-based analysis with automated, scalable Python workflows for large datasets.

### Business Objectives

- **Inventory Optimization:** Determine ideal stock levels and reorder quantities  
- **Demand Forecasting:** Predict future sales to prevent overstocking or stockouts  
- **Supplier Evaluation:** Analyze performance, consistency, and lead times  
- **Operational Efficiency:** Automate reports and visual insights for decision-makers  

---

## ✨ Features

### 1. 📈 Demand Forecasting
- SARIMA-based **time series forecasting**
- Trend & seasonality decomposition
- Rolling 6-month prediction with 95% confidence intervals
- Visual time series forecast charts and evaluation metrics

### 2. 📦 ABC Classification
- Product segmentation based on **cumulative revenue contribution**
- A/B/C categories for strategic stock management
- Pareto visualization for high-revenue SKUs

### 3. 💰 EOQ (Economic Order Quantity)
- Optimal order calculation to **minimize total cost**
- Comparison of **current vs optimized** purchasing strategy
- Estimated savings per SKU

### 4. 🔔 Reorder Point Analysis
- Calculates reorder points + safety stock
- Integrates supplier **lead time variability**
- Prevents stockouts and lost revenue

### 5. ⏱️ Lead Time & Supplier Performance
- Measures average, median, and variability in lead time
- Tracks on-time delivery percentage
- Flags unreliable suppliers

### 6. 🧾 Automated Data Cleaning & Export
- Unified data loader automatically:
  - Standardizes column names  
  - Cleans currencies, numerics, and dates  
  - Exports all cleaned datasets for downstream analysis  

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- `pip` package manager

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Nabi0-0/data-science-analytics.git
cd data-science-analytics
2️⃣ Create Virtual Environment
bash
Copy code
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Add Data
Place all .csv files inside the Data/ folder:

Copy code
Data/
 ├── SalesFINAL12312016.csv
 ├── PurchasesFINAL12312016.csv
 ├── EndInvFINAL12312016.csv
 ├── BegInvFINAL12312016.csv
 ├── InvoicePurchases12312016.csv
 └── 2017PurchasePricesDec.csv
💻 Usage
Run the Full Analysis
bash
Copy code
python scripts/main.py
Or Use Individual Modules
python
Copy code
from scripts.data_loader import SloozeDataLoader

loader = SloozeDataLoader()
loader.load_all_data()
summary = loader.get_data_summary()
sales = loader.create_unified_sales_dataset()
purchases = loader.create_unified_purchase_dataset()
🧮 Data Format
Sales Data (SalesFINAL12312016.csv)
Column	Description
date	Date of sale
quantity	Units sold
price	Selling price per unit
total_revenue	Computed: quantity × price
brand / store / vendor	Product and vendor metadata

Purchases Data (PurchasesFINAL12312016.csv)
Column	Description
order_date	Purchase order date
delivery_date	Goods receiving date
quantity	Purchased units
unit_cost	Cost per unit
total_cost	Total purchase cost
vendor_name	Supplier details

Ending Inventory (EndInvFINAL12312016.csv)
Column	Description
date	End-of-year date
stock_qty	Units in stock
unit_price	Selling price
brand / description	Product info

📁 Output Files
📊 CSV Exports
Located in output/csv/

sales_cleaned.csv

purchases_cleaned.csv

inventory_ending_cleaned.csv

abc_classification.csv

eoq_results.csv

reorder_points.csv

supplier_lead_times.csv

🖼️ PNG Visualizations
Located in output/png/

demand_forecast.png

abc_analysis.png

eoq_analysis.png

reorder_point_analysis.png

lead_time_analysis.png

📑 Reports
Located in output/reports/

executive_summary.txt (insights and recommendations)

🧠 Methodology
Module	Technique	Output
Forecasting	SARIMA (1,1,1)(1,1,1,7)	6-month demand prediction
ABC Analysis	Pareto (70/21/9 rule)	Product classification
EOQ	√(2DS/H)	Optimal order quantity
Reorder Point	Safety Stock + Lead Time	Stock alert threshold
Supplier Lead Time	Mean, SD, On-time %	Reliability score

🧾 Technologies Used
Python 3.8+

Pandas, NumPy

Matplotlib, Seaborn

Statsmodels, Scikit-learn

OpenPyXL, SciPy

📂 Project Structure
bash
Copy code
data-science-analytics/
│
├── Data/                     # Raw datasets (CSV)
├── scripts/                  # Main analytical modules
│   ├── main.py
│   ├── data_loader.py
│   ├── forecasting.py
│   ├── abc_analysis.py
│   ├── eoq_reorder.py
│   ├── supplier_analysis.py
│   └── utils.py
├── output/
│   ├── csv/
│   ├── png/
│   └── reports/
├── requirements.txt
├── README.md
└── SloozeInventoryDashboard/
👩‍💻 Author
Vedanshi Goswami

GitHub: @Nabi0-0

LinkedIn: Vedanshi Goswami

Email: vedanshigoswami0@gmail.com

📝 Notes
All cleaned datasets are automatically exported in /output/csv

Visuals are saved in /output/png (300 DPI)

Designed to handle large datasets efficiently

Modular architecture — easy to extend with ML or BI dashboards

Last Updated: 11 Nov 2025
Version: 1.1.0