import React, { useState, useEffect, useCallback } from 'react';
import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { TrendingUp, Package, AlertTriangle, DollarSign, Users, ShoppingCart, Truck, Clock } from 'lucide-react';

const API_BASE = 'http://localhost:5000/api';
const COLORS = ['#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#3b82f6', '#ef4444'];

const styles = {
  container: {
    minHeight: '100vh',
    background: 'linear-gradient(to bottom right, #0f172a, #581c87, #0f172a)',
    padding: '2rem',
  },
  maxWidth: {
    maxWidth: '1280px',
    margin: '0 auto',
  },
  header: {
    textAlign: 'center',
    marginBottom: '2rem',
  },
  headerBadge: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '1rem',
    backgroundColor: 'rgba(139, 92, 246, 0.2)',
    backdropFilter: 'blur(10px)',
    padding: '1rem 2rem',
    borderRadius: '9999px',
    border: '1px solid rgba(139, 92, 246, 0.3)',
    marginBottom: '1rem',
    boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.3)',
  },
  title: {
    fontSize: '2rem',
    fontWeight: 'bold',
    color: 'white',
    margin: 0,
  },
  subtitle: {
    color: '#d8b4fe',
    fontSize: '1.125rem',
  },
  navContainer: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '0.75rem',
    backgroundColor: 'rgba(30, 41, 59, 0.5)',
    backdropFilter: 'blur(10px)',
    padding: '1rem',
    borderRadius: '0.75rem',
    border: '1px solid #334155',
    marginBottom: '2rem',
    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.3)',
  },
  navButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    padding: '0.75rem 1.25rem',
    borderRadius: '0.5rem',
    border: 'none',
    cursor: 'pointer',
    fontSize: '0.875rem',
    fontWeight: '500',
    transition: 'all 0.2s',
  },
  navButtonActive: {
    backgroundColor: '#8b5cf6',
    color: 'white',
    boxShadow: '0 10px 15px -3px rgba(139, 92, 246, 0.5)',
    transform: 'scale(1.05)',
  },
  navButtonInactive: {
    backgroundColor: 'rgba(51, 65, 85, 0.5)',
    color: '#cbd5e1',
  },
  contentCard: {
    backgroundColor: 'rgba(30, 41, 59, 0.5)',
    backdropFilter: 'blur(10px)',
    borderRadius: '0.75rem',
    border: '1px solid #334155',
    padding: '2rem',
    boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
    minHeight: '600px',
  },
  sectionTitle: {
    fontSize: '2rem',
    fontWeight: 'bold',
    color: 'white',
    marginBottom: '2rem',
  },
  statsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
    gap: '1.5rem',
    marginBottom: '2rem',
  },
  statCard: {
    backgroundColor: 'rgba(15, 23, 42, 0.5)',
    padding: '1.5rem',
    borderRadius: '0.5rem',
    border: '1px solid #334155',
    transition: 'border-color 0.2s',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'space-between',
    minHeight: '120px',
  },
  statLabel: {
    color: '#94a3b8',
    fontSize: '0.875rem',
    fontWeight: '500',
    marginBottom: '0.75rem',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  statValue: {
    fontSize: '2rem',
    fontWeight: 'bold',
    color: 'white',
    lineHeight: '1.2',
  },
  card: {
    backgroundColor: 'rgba(15, 23, 42, 0.5)',
    padding: '2rem',
    borderRadius: '0.5rem',
    border: '1px solid #334155',
    marginBottom: '2rem',
  },
  cardTitle: {
    fontSize: '1.25rem',
    fontWeight: '600',
    color: 'white',
    marginBottom: '1.5rem',
  },
  loading: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '3rem',
    color: '#94a3b8',
  },
  spinner: {
    width: '3rem',
    height: '3rem',
    border: '4px solid #8b5cf6',
    borderTopColor: 'transparent',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
  },
};

export default function Dashboard() {
  const [activeView, setActiveView] = useState('overview');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState({
    overview: null,
    forecast: null,
    abc: null,
    eoq: null,
    reorder: null,
    suppliers: null
  });

  const loadData = useCallback(async (view) => {
    if (data[view]) return;
    
    setLoading(true);
    try {
      const endpoint = {
        overview: 'overview',
        forecast: 'forecast',
        abc: 'abc-analysis',
        eoq: 'eoq',
        reorder: 'reorder-points',
        suppliers: 'supplier-analysis'
      }[view];
      
      const response = await fetch(`${API_BASE}/${endpoint}`);
      const result = await response.json();
      setData(prev => ({ ...prev, [view]: result }));
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
    }
  }, [data]);

  useEffect(() => {
    loadData(activeView);
  }, [activeView, loadData]);

  const views = [
    { id: 'overview', label: 'Overview', icon: BarChart },
    { id: 'forecast', label: 'Demand Forecast', icon: TrendingUp },
    { id: 'abc', label: 'ABC Analysis', icon: Package },
    { id: 'eoq', label: 'EOQ Optimization', icon: ShoppingCart },
    { id: 'reorder', label: 'Reorder Points', icon: AlertTriangle },
    { id: 'suppliers', label: 'Suppliers', icon: Truck }
  ];

  return (
    <div style={styles.container}>
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
      <div style={styles.maxWidth}>
        {/* Header */}
        <div style={styles.header}>
          <div style={styles.headerBadge}>
            <Package size={28} color="#c084fc" />
            <h1 style={styles.title}>Slooze Inventory Analytics</h1>
          </div>
          <p style={styles.subtitle}>Wine & Spirits Retail Intelligence Platform</p>
        </div>

        {/* Navigation */}
        <div style={styles.navContainer}>
          {views.map(view => {
            const Icon = view.icon;
            const isActive = activeView === view.id;
            return (
              <button
                key={view.id}
                onClick={() => setActiveView(view.id)}
                style={{
                  ...styles.navButton,
                  ...(isActive ? styles.navButtonActive : styles.navButtonInactive)
                }}
                onMouseEnter={(e) => {
                  if (!isActive) {
                    e.target.style.backgroundColor = 'rgba(51, 65, 85, 1)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isActive) {
                    e.target.style.backgroundColor = 'rgba(51, 65, 85, 0.5)';
                  }
                }}
              >
                <Icon size={20} />
                <span>{view.label}</span>
              </button>
            );
          })}
        </div>

        {/* Content */}
        <div style={styles.contentCard}>
          {loading && <LoadingSpinner />}
          {!loading && activeView === 'overview' && <OverviewView data={data.overview} />}
          {!loading && activeView === 'forecast' && <ForecastView data={data.forecast} />}
          {!loading && activeView === 'abc' && <ABCView data={data.abc} />}
          {!loading && activeView === 'eoq' && <EOQView data={data.eoq} />}
          {!loading && activeView === 'reorder' && <ReorderView data={data.reorder} />}
          {!loading && activeView === 'suppliers' && <SuppliersView data={data.suppliers} />}
        </div>
      </div>
    </div>
  );
}

function LoadingSpinner() {
  return (
    <div style={styles.loading}>
      <div style={styles.spinner}></div>
    </div>
  );
}

function OverviewView({ data }) {
  if (!data) return <div style={styles.loading}>Loading overview...</div>;

  const stats = [
    { label: 'Total Revenue', value: `${(data.total_revenue / 1000000).toFixed(2)}M`, icon: DollarSign, color: '#10b981' },
    { label: 'Units Sold', value: data.total_units_sold?.toLocaleString() || '0', icon: Package, color: '#3b82f6' },
    { label: 'Unique Products', value: data.unique_products, icon: ShoppingCart, color: '#8b5cf6' },
    { label: 'Active Stores', value: data.unique_stores, icon: Users, color: '#ec4899' }
  ];

  const categoryData = data.top_categories ? Object.entries(data.top_categories).map(([name, value]) => ({
    name,
    value: value
  })) : [];

  return (
    <div>
      <h2 style={styles.sectionTitle}>Business Overview</h2>
      
      <div style={styles.statsGrid}>
        {stats.map((stat, i) => {
          const Icon = stat.icon;
          return (
            <div key={i} style={styles.statCard}>
              <div style={styles.statLabel}>
                <Icon size={20} color={stat.color} />
                {stat.label}
              </div>
              <div style={{ ...styles.statValue, color: stat.color }}>{stat.value}</div>
            </div>
          );
        })}
      </div>

      <div style={styles.card}>
        <h3 style={styles.cardTitle}>Revenue by Category</h3>
        <div style={{ width: '100%', height: '400px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={categoryData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={130}
                label={entry => `${entry.name}: ${(entry.value / 1000).toFixed(0)}K`}
                labelLine={{ stroke: '#94a3b8' }}
              >
                {categoryData.map((entry, index) => (
                  <Cell key={index} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip 
                formatter={(value) => `${(value / 1000).toFixed(2)}K`}
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px', padding: '12px' }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
        gap: '1.5rem',
        marginTop: '2rem'
      }}>
        <div style={styles.statCard}>
          <div style={styles.statLabel}>Inventory Value (Begin)</div>
          <div style={{ ...styles.statValue, color: '#3b82f6' }}>
            ${(data.inventory_value_begin / 1000000).toFixed(2)}M
          </div>
        </div>
        <div style={styles.statCard}>
          <div style={styles.statLabel}>Inventory Value (End)</div>
          <div style={{ ...styles.statValue, color: '#10b981' }}>
            ${(data.inventory_value_end / 1000000).toFixed(2)}M
          </div>
        </div>
      </div>
    </div>
  );
}

function ForecastView({ data }) {
  if (!data || !data.forecasts) return <div style={styles.loading}>Loading forecasts...</div>;

  return (
    <div>
      <h2 style={styles.sectionTitle}>Demand Forecasting</h2>
      
      {data.forecasts.slice(0, 3).map((forecast, i) => (
        <div key={i} style={{ ...styles.card, marginBottom: '2rem' }}>
          <h3 style={{ ...styles.cardTitle, color: '#c084fc' }}>{forecast.product}</h3>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
            <div>
              <div style={{ color: '#94a3b8', fontSize: '0.875rem' }}>Avg Daily Sales</div>
              <div style={{ color: 'white', fontSize: '1.5rem', fontWeight: 'bold' }}>{forecast.avg_daily_sales.toFixed(1)}</div>
            </div>
            <div>
              <div style={{ color: '#94a3b8', fontSize: '0.875rem' }}>Forecast Period</div>
              <div style={{ color: 'white', fontSize: '1.5rem', fontWeight: 'bold' }}>{forecast.forecast_periods} days</div>
            </div>
            <div>
              <div style={{ color: '#94a3b8', fontSize: '0.875rem' }}>Total Forecast</div>
              <div style={{ color: 'white', fontSize: '1.5rem', fontWeight: 'bold' }}>{forecast.total_forecast_demand.toFixed(0)}</div>
            </div>
          </div>

          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={forecast.forecast_values.map((val, idx) => ({ 
              day: idx + 1, 
              forecast: val
            }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="day" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip />
              <Area type="monotone" dataKey="forecast" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.3} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      ))}
    </div>
  );
}

function ABCView({ data }) {
  if (!data || !data.classification_summary) return <div style={styles.loading}>Loading ABC analysis...</div>;

  const chartData = data.classification_summary.map(item => ({
    class: `Class ${item.Class}`,
    revenue: item.TotalRevenue / 1000000
  }));

  return (
    <div>
      <h2 style={styles.sectionTitle}>ABC Inventory Classification</h2>
      
      <div style={styles.statsGrid}>
        {data.classification_summary.map((cls, i) => (
          <div key={i} style={styles.statCard}>
            <div style={{ fontSize: '2.5rem', fontWeight: 'bold', color: '#c084fc', marginBottom: '0.5rem' }}>
              Class {cls.Class}
            </div>
            <div style={{ color: '#cbd5e1', marginBottom: '0.25rem' }}>
              {cls.ProductCount} products ({cls.ProductPercentage.toFixed(1)}%)
            </div>
            <div style={{ color: '#cbd5e1', marginBottom: '0.25rem' }}>
              ${(cls.TotalRevenue / 1000000).toFixed(2)}M revenue
            </div>
            <div style={{ color: '#10b981', fontWeight: 'bold', fontSize: '1.125rem' }}>
              {cls.RevenuePercentage.toFixed(1)}% of total
            </div>
          </div>
        ))}
      </div>

      <div style={styles.card}>
        <h3 style={styles.cardTitle}>Revenue Distribution</h3>
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="class" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip />
            <Bar dataKey="revenue" fill="#8b5cf6" name="Revenue ($M)" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function EOQView({ data }) {
  if (!data) return <div style={styles.loading}>Loading EOQ analysis...</div>;

  const topSavings = data.top_savings?.slice(0, 10) || [];

  return (
    <div>
      <h2 style={styles.sectionTitle}>Economic Order Quantity (EOQ)</h2>
      
      <div style={{ ...styles.card, marginBottom: '2rem' }}>
        <div style={{ color: '#94a3b8', marginBottom: '0.5rem' }}>Total Potential Savings</div>
        <div style={{ fontSize: '3rem', fontWeight: 'bold', color: '#10b981' }}>
          ${topSavings.reduce((sum, item) => sum + item.PotentialSavings, 0).toLocaleString()}
        </div>
      </div>

      <h3 style={{ ...styles.cardTitle, marginBottom: '1rem' }}>Top Optimization Opportunities</h3>
      {topSavings.map((item, i) => (
        <div key={i} style={{ ...styles.card, padding: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem', flexWrap: 'wrap', gap: '1rem' }}>
            <div style={{ color: '#c084fc', fontWeight: '600', fontSize: '1.125rem' }}>{item.Product}</div>
            <div style={{ color: '#10b981', fontWeight: 'bold', fontSize: '1.25rem' }}>
              ${item.PotentialSavings.toFixed(0)} savings
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1rem' }}>
            <div>
              <div style={{ color: '#64748b', fontSize: '0.875rem' }}>Current Order</div>
              <div style={{ color: 'white', fontWeight: '600' }}>{item.CurrentOrderQty.toFixed(0)} units</div>
            </div>
            <div>
              <div style={{ color: '#64748b', fontSize: '0.875rem' }}>Optimal EOQ</div>
              <div style={{ color: 'white', fontWeight: '600' }}>{item.EOQ.toFixed(0)} units</div>
            </div>
            <div>
              <div style={{ color: '#64748b', fontSize: '0.875rem' }}>Orders/Year</div>
              <div style={{ color: 'white', fontWeight: '600' }}>{item.OrdersPerYear.toFixed(1)}</div>
            </div>
            <div>
              <div style={{ color: '#64748b', fontSize: '0.875rem' }}>Savings</div>
              <div style={{ color: '#10b981', fontWeight: 'bold' }}>{item.SavingsPercentage.toFixed(1)}%</div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function ReorderView({ data }) {
  if (!data || !data.summary) return <div style={styles.loading}>Loading reorder analysis...</div>;

  const alerts = data.alerts || {};
  const critical = alerts.critical_items || [];

  return (
    <div>
      <h2 style={styles.sectionTitle}>Reorder Point Analysis</h2>
      
      <div style={styles.statsGrid}>
        <div style={styles.statCard}>
          <div style={styles.statLabel}>Total Products</div>
          <div style={{ ...styles.statValue, color: '#3b82f6' }}>{data.summary.total_products}</div>
        </div>
        <div style={styles.statCard}>
          <div style={styles.statLabel}>Need Reorder</div>
          <div style={{ ...styles.statValue, color: '#f59e0b' }}>{data.summary.products_needing_reorder}</div>
        </div>
        <div style={styles.statCard}>
          <div style={styles.statLabel}>Critical Items</div>
          <div style={{ ...styles.statValue, color: '#ef4444' }}>{data.summary.critical_items}</div>
        </div>
        <div style={styles.statCard}>
          <div style={styles.statLabel}>Inventory Health</div>
          <div style={{ ...styles.statValue, color: '#10b981' }}>{data.inventory_health_score?.toFixed(0)}%</div>
        </div>
      </div>

      {critical.length > 0 && (
        <div style={{ 
          backgroundColor: 'rgba(127, 29, 29, 0.2)', 
          border: '2px solid rgba(239, 68, 68, 0.3)',
          borderRadius: '0.5rem',
          padding: '1.5rem',
          marginTop: '2rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
            <AlertTriangle size={24} color="#f87171" />
            <h3 style={{ fontSize: '1.25rem', fontWeight: '600', color: '#fca5a5' }}>Critical Stock Alerts</h3>
          </div>
          {critical.slice(0, 5).map((item, i) => (
            <div key={i} style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              padding: '1rem',
              backgroundColor: 'rgba(15, 23, 42, 0.3)',
              borderRadius: '0.5rem',
              marginBottom: '0.75rem',
              flexWrap: 'wrap',
              gap: '1rem'
            }}>
              <span style={{ color: 'white', fontWeight: '600' }}>{item.Product}</span>
              <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
                <span style={{ color: '#94a3b8' }}>Stock: <span style={{ color: 'white' }}>{item.CurrentStock.toFixed(0)}</span></span>
                <span style={{ color: '#f87171', fontWeight: '600' }}>{item.DaysOfStock.toFixed(1)} days left</span>
                <span style={{ color: '#10b981' }}>Order: <span style={{ fontWeight: 'bold' }}>{item.SuggestedOrderQty.toFixed(0)}</span></span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SuppliersView({ data }) {
  if (!data || !data.summary) return <div style={styles.loading}>Loading supplier analysis...</div>;

  const top = data.top_suppliers?.slice(0, 10) || [];

  return (
    <div>
      <h2 style={styles.sectionTitle}>Supplier Performance</h2>
      
      <div style={styles.statsGrid}>
        <div style={styles.statCard}>
          <div style={styles.statLabel}>Total Suppliers</div>
          <div style={{ ...styles.statValue, color: '#3b82f6' }}>{data.summary.total_suppliers}</div>
        </div>
        <div style={styles.statCard}>
          <div style={styles.statLabel}>Avg Reliability</div>
          <div style={{ ...styles.statValue, color: '#10b981' }}>{data.summary.avg_reliability_score?.toFixed(0)}%</div>
        </div>
        <div style={styles.statCard}>
          <div style={styles.statLabel}>Total Spend</div>
          <div style={{ ...styles.statValue, color: '#8b5cf6' }}>${(data.summary.total_spend / 1000000).toFixed(2)}M</div>
        </div>
        <div style={styles.statCard}>
          <div style={styles.statLabel}>Top 5 Concentration</div>
          <div style={{ ...styles.statValue, color: '#f59e0b' }}>{data.summary.spend_concentration_top5?.toFixed(0)}%</div>
        </div>
      </div>

      <h3 style={{ ...styles.cardTitle, marginTop: '2rem', marginBottom: '1rem' }}>Top Performing Suppliers</h3>
      {top.map((supplier, i) => (
        <div key={i} style={{ ...styles.card, padding: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem', flexWrap: 'wrap', gap: '1rem' }}>
            <div>
              <div style={{ color: 'white', fontWeight: '600', fontSize: '1.125rem', marginBottom: '0.25rem' }}>
                {supplier.VendorName}
              </div>
              <div style={{ color: '#94a3b8' }}>
                Reliability Score: <span style={{ color: '#c084fc', fontWeight: '600' }}>{supplier.ReliabilityScore.toFixed(1)}/100</span>
              </div>
            </div>
            <div style={{
              padding: '0.5rem 1rem',
              borderRadius: '9999px',
              fontSize: '0.875rem',
              fontWeight: 'bold',
              backgroundColor: supplier.Rating === 'Excellent' ? 'rgba(16, 185, 129, 0.2)' :
                             supplier.Rating === 'Good' ? 'rgba(59, 130, 246, 0.2)' :
                             supplier.Rating === 'Fair' ? 'rgba(245, 158, 11, 0.2)' : 
                             'rgba(239, 68, 68, 0.2)',
              color: supplier.Rating === 'Excellent' ? '#6ee7b7' :
                    supplier.Rating === 'Good' ? '#93c5fd' :
                    supplier.Rating === 'Fair' ? '#fcd34d' : '#fca5a5'
            }}>
              {supplier.Rating}
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1rem' }}>
            <div>
              <div style={{ color: '#64748b', fontSize: '0.875rem' }}>Lead Time</div>
              <div style={{ color: 'white', fontWeight: '600' }}>{supplier.AvgLeadTime?.toFixed(1)} days</div>
            </div>
            <div>
              <div style={{ color: '#64748b', fontSize: '0.875rem' }}>On-Time</div>
              <div style={{ color: 'white', fontWeight: '600' }}>{supplier.OnTimePercentage?.toFixed(1)}%</div>
            </div>
            <div>
              <div style={{ color: '#64748b', fontSize: '0.875rem' }}>Total Orders</div>
              <div style={{ color: 'white', fontWeight: '600' }}>{supplier.TotalOrders}</div>
            </div>
            <div>
              <div style={{ color: '#64748b', fontSize: '0.875rem' }}>Total Spend</div>
              <div style={{ color: 'white', fontWeight: '600' }}>${(supplier.TotalSpend / 1000).toFixed(0)}K</div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}