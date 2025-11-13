import React, { useState, useEffect } from 'react';
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import {
  TrendingUp, Package, DollarSign, AlertTriangle,
  Calendar, Clock, RefreshCw, PlayCircle
} from 'lucide-react';

const API_URL = 'http://localhost:5000/api';

const SloozeInventoryDashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  // Data states
  const [overviewData, setOverviewData] = useState(null);
  const [forecastData, setForecastData] = useState(null);
  const [abcData, setAbcData] = useState(null);
  const [eoqData, setEoqData] = useState(null);
  const [reorderData, setReorderData] = useState(null);
  const [supplierData, setSupplierData] = useState(null);
  const [topProducts, setTopProducts] = useState(null);
  const [seasonalTrends, setSeasonalTrends] = useState(null);

  const COLORS = ['#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#8b5cf6'];

  // Fetch helper
  const fetchData = async (endpoint, setter) => {
    try {
      const response = await fetch(`${API_URL}/${endpoint}`);
      if (!response.ok) throw new Error(`Failed to fetch ${endpoint}`);
      const data = await response.json();
      setter(data);
      return data;
    } catch (err) {
      console.error(`Error fetching ${endpoint}:`, err);
      setError(`Failed to load ${endpoint}`);
      return null;
    }
  };

  // Load all API endpoints
  const loadAllData = async () => {
    setLoading(true);
    setError(null);
    try {
      await Promise.all([
        fetchData('overview', setOverviewData),
        fetchData('forecast', setForecastData),
        fetchData('abc-analysis', setAbcData),
        fetchData('eoq', setEoqData),
        fetchData('reorder-points', setReorderData),
        fetchData('suppliers', setSupplierData),
        fetchData('top-products', setTopProducts),
        fetchData('seasonal-trends', setSeasonalTrends)
      ]);
      setLastUpdated(new Date().toLocaleString());
    } catch (err) {
      setError('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  // Run full backend analysis
  const runAnalysis = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_URL}/analysis/run-all`, { method: 'POST' });
      if (!response.ok) throw new Error('Analysis failed');
      alert('Analysis completed successfully!');
      await loadAllData();
    } catch (err) {
      alert('Analysis failed: ' + err.message);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAllData();
  }, []);

  const tabs = [
    { id: 'overview', label: 'Overview', icon: TrendingUp },
    { id: 'forecast', label: 'Demand Forecast', icon: Calendar },
    { id: 'abc', label: 'ABC Analysis', icon: Package },
    { id: 'eoq', label: 'EOQ Optimization', icon: DollarSign },
    { id: 'reorder', label: 'Reorder Points', icon: AlertTriangle },
    { id: 'supplier', label: 'Supplier Analysis', icon: Clock }
  ];

  // === Overview Tab ===
  const renderOverview = () => {
    if (!overviewData || !topProducts || !seasonalTrends)
      return <div className="text-center py-8">Loading overview data...</div>;

    const kpis = overviewData.kpis || {};

    return (
      <div className="space-y-6">
        {/* KPI Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <KpiCard title="Total Revenue" value={`$${(kpis.total_revenue || 0).toLocaleString()}`} color="green" icon={<DollarSign size={32} />} subtitle="Year 2016" />
          <KpiCard title="Active Products" value={kpis.unique_products || 0} color="blue" icon={<Package size={32} />} subtitle="Unique brands" />
          <KpiCard title="Stockout Risk" value={kpis.critical_items || 0} color="orange" icon={<AlertTriangle size={32} />} subtitle="Items need attention" />
          <KpiCard title="Avg Lead Time" value={`${(kpis.avg_lead_time || 0).toFixed(1)} days`} color="purple" icon={<Clock size={32} />} subtitle="Supplier average" />
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ChartCard title="Top 10 Products by Revenue">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={topProducts.top_products?.slice(0, 10) || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="brand" angle={-15} textAnchor="end" height={80} />
                <YAxis />
                <Tooltip formatter={(v) => `$${v.toLocaleString()}`} />
                <Bar dataKey="total_revenue" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Monthly Sales Trend">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={seasonalTrends.monthly_trends || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month_name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="quantity" stroke="#10b981" strokeWidth={2} name="Units Sold" />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>
      </div>
    );
  };

  // === Forecast ===
  const renderForecast = () =>
    !forecastData ? (
      <div className="text-center py-8">Loading forecast data...</div>
    ) : (
      <ChartCard title="180-Day Demand Forecast">
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={forecastData.forecast?.slice(0, 90) || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="forecast" stroke="#10b981" strokeWidth={3} />
            <Line type="monotone" dataKey="lower_bound" stroke="#ef4444" strokeDasharray="3 3" />
            <Line type="monotone" dataKey="upper_bound" stroke="#ef4444" strokeDasharray="3 3" />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>
    );

  // === ABC ===
  const renderABC = () =>
    !abcData ? (
      <div className="text-center py-8">Loading ABC analysis...</div>
    ) : (
      <div className="space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ChartCard title="Revenue Distribution">
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={abcData.summary || []}
                  dataKey="Revenue_Percentage"
                  outerRadius={100}
                  label={({ Category, Revenue_Percentage }) => `${Category}: ${Revenue_Percentage?.toFixed(1)}%`}
                >
                  {(abcData.summary || []).map((entry, idx) => (
                    <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </ChartCard>

          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <h3 className="text-lg font-semibold mb-4">ABC Summary</h3>
            <div className="space-y-4">
              {(abcData.summary || []).map((cat, i) => (
                <div key={i} className={`p-4 rounded-lg border-2 ${
                  cat.Category === 'A' ? 'border-green-300 bg-green-50' :
                  cat.Category === 'B' ? 'border-orange-300 bg-orange-50' :
                  'border-red-300 bg-red-50'
                }`}>
                  <div className="flex justify-between">
                    <div>
                      <h4 className="font-bold text-lg">Category {cat.Category}</h4>
                      <p className="text-sm">Items: {cat.Number_of_Items} ({cat.Item_Percentage?.toFixed(1)}%)</p>
                    </div>
                    <div className="text-right">
                      <p className="font-bold text-lg">{cat.Revenue_Percentage?.toFixed(1)}%</p>
                      <p className="text-sm">of Revenue</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );

  // === EOQ ===
  const renderEOQ = () =>
    !eoqData ? (
      <div className="text-center py-8">Loading EOQ data...</div>
    ) : (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <KpiCard title="Total Savings Potential" value={`$${(eoqData.summary?.total_savings || 0).toLocaleString()}`} color="green" />
          <KpiCard title="Products Analyzed" value={eoqData.summary?.total_products || 0} color="blue" />
          <KpiCard title="Avg Savings per Product" value={`$${(eoqData.summary?.avg_savings || 0).toFixed(0)}`} color="purple" />
        </div>
      </div>
    );

  // === Reorder ===
  const renderReorder = () =>
    !reorderData ? (
      <div className="text-center py-8">Loading reorder data...</div>
    ) : (
      <div className="space-y-6">
        <ChartCard title="Inventory Status Overview">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(reorderData.status_distribution || {}).map(([status, count]) => (
              <div key={status} className="p-4 rounded-lg border bg-gray-50 text-center">
                <p className="text-2xl font-bold">{count}</p>
                <p className="text-sm">{status.replace('_', ' ')}</p>
              </div>
            ))}
          </div>
        </ChartCard>
      </div>
    );

  // === Supplier ===
  const renderSupplier = () =>
    !supplierData ? (
      <div className="text-center py-8">Loading supplier data...</div>
    ) : (
      <ChartCard title="Supplier Performance Ranking">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={supplierData.performance?.slice(0, 10)}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="Supplier" angle={-15} textAnchor="end" height={80} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="Composite_Score" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
    );

  // === UI Components ===
  const KpiCard = ({ title, value, subtitle, color, icon }) => (
    <div className={`bg-${color}-50 border border-${color}-200 p-6 rounded-lg`}>
      <div className="flex items-center justify-between">
        <div>
          <p className={`text-${color}-600 text-sm font-medium`}>{title}</p>
          <p className={`text-3xl font-bold text-${color}-900`}>{value}</p>
          {subtitle && <p className={`text-${color}-600 text-sm mt-1`}>{subtitle}</p>}
        </div>
        {icon}
      </div>
    </div>
  );

  const ChartCard = ({ title, children }) => (
    <div className="bg-white p-6 rounded-lg border border-gray-200">
      <h3 className="text-lg font-semibold mb-4">{title}</h3>
      {children}
    </div>
  );

  const renderContent = () => {
    if (loading)
      return (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <RefreshCw className="animate-spin mx-auto mb-4" size={48} />
            <p>Loading data...</p>
          </div>
        </div>
      );
    if (error)
      return (
        <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
          <AlertTriangle className="mx-auto mb-4 text-red-500" size={48} />
          <p className="text-red-800 font-semibold mb-2">Error loading data</p>
          <p className="text-red-600 text-sm">{error}</p>
          <button
            onClick={loadAllData}
            className="mt-4 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      );

    switch (activeTab) {
      case 'overview': return renderOverview();
      case 'forecast': return renderForecast();
      case 'abc': return renderABC();
      case 'eoq': return renderEOQ();
      case 'reorder': return renderReorder();
      case 'supplier': return renderSupplier();
      default: return renderOverview();
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8 flex justify-between items-center">
          <div>
            <h1 className="text-4xl font-bold text-gray-900 mb-2">🍷 Slooze Inventory Analytics</h1>
            <p className="text-gray-600">Wine & Spirits Retail Optimization Dashboard</p>
            {lastUpdated && (
              <p className="text-sm text-gray-500 mt-1">Last updated: {lastUpdated}</p>
            )}
          </div>
          <div className="flex gap-2">
            <button
              onClick={runAnalysis}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
            >
              <PlayCircle size={20} /> Run Analysis
            </button>
            <button
              onClick={loadAllData}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
            >
              <RefreshCw size={20} className={loading ? 'animate-spin' : ''} /> Refresh
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="bg-white rounded-lg border border-gray-200 mb-6 overflow-x-auto">
          <div className="flex">
            {tabs.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`flex items-center px-6 py-4 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                  activeTab === id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Icon size={18} className="mr-2" />
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Main Content */}
        {renderContent()}
      </div>
    </div>
  );
};

export default SloozeInventoryDashboard;
