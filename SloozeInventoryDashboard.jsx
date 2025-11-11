import React, { useState } from 'react';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, Package, DollarSign, AlertTriangle, Calendar, Clock } from 'lucide-react';

const SloozeInventoryDashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');

  const demandForecast = [
    { month: 'Jan', actual: 45000, forecast: 44500, upper: 47000, lower: 42000 },
    { month: 'Feb', actual: 48000, forecast: 48200, upper: 50500, lower: 45900 },
    { month: 'Mar', actual: 52000, forecast: 51800, upper: 54500, lower: 49100 },
    { month: 'Apr', actual: 49000, forecast: 50000, upper: 52500, lower: 47500 },
    { month: 'May', forecast: 53000, upper: 56000, lower: 50000 },
    { month: 'Jun', forecast: 55000, upper: 58000, lower: 52000 }
  ];

  const abcAnalysis = [
    { category: 'Category A', items: 150, revenue: 2500000, percentage: 70 },
    { category: 'Category B', items: 350, revenue: 750000, percentage: 21 },
    { category: 'Category C', items: 500, revenue: 320000, percentage: 9 }
  ];

  const topProducts = [
    { name: 'Premium Red Wine', sales: 15420, revenue: 385500 },
    { name: 'Craft Whiskey', sales: 12850, revenue: 578250 },
    { name: 'Premium Vodka', sales: 11200, revenue: 336000 },
    { name: 'Champagne Vintage', sales: 8950, revenue: 447500 },
    { name: 'Craft Beer Pack', sales: 8300, revenue: 207500 }
  ];

  const eoqMetrics = [
    { product: 'Red Wine', currentOrder: 500, optimalEOQ: 387, savings: 2850 },
    { product: 'Whiskey', currentOrder: 300, optimalEOQ: 245, savings: 1920 },
    { product: 'Vodka', currentOrder: 400, optimalEOQ: 328, savings: 1650 },
    { product: 'Champagne', currentOrder: 250, optimalEOQ: 198, savings: 1420 },
    { product: 'Beer', currentOrder: 600, optimalEOQ: 485, savings: 2100 }
  ];

  const reorderPoints = [
    { product: 'Premium Red Wine', current: 245, reorderPoint: 180, leadTime: 7, safetyStock: 50, status: 'healthy' },
    { product: 'Craft Whiskey', current: 89, reorderPoint: 120, leadTime: 10, safetyStock: 40, status: 'critical' },
    { product: 'Premium Vodka', current: 198, reorderPoint: 150, leadTime: 5, safetyStock: 35, status: 'healthy' },
    { product: 'Champagne', current: 142, reorderPoint: 130, leadTime: 14, safetyStock: 45, status: 'warning' },
    { product: 'Craft Beer', current: 315, reorderPoint: 200, leadTime: 3, safetyStock: 30, status: 'healthy' }
  ];

  const supplierMetrics = [
    { supplier: 'Wine Distributors Inc', leadTime: 7.2, onTimeRate: 94, defectRate: 1.2, score: 92 },
    { supplier: 'Premium Spirits Co', leadTime: 10.5, onTimeRate: 88, defectRate: 2.1, score: 85 },
    { supplier: 'Global Beverages', leadTime: 5.8, onTimeRate: 96, defectRate: 0.8, score: 95 },
    { supplier: 'Local Brewers', leadTime: 3.2, onTimeRate: 98, defectRate: 0.5, score: 98 }
  ];

  const seasonalTrends = [
    { month: 'Jan', wine: 3200, spirits: 4500, beer: 2800 },
    { month: 'Feb', wine: 3400, spirits: 4200, beer: 2600 },
    { month: 'Mar', wine: 3800, spirits: 4100, beer: 3200 },
    { month: 'Apr', wine: 4200, spirits: 3800, beer: 3800 },
    { month: 'May', wine: 4500, spirits: 3500, beer: 4500 },
    { month: 'Jun', wine: 4800, spirits: 3400, beer: 5200 }
  ];

  const COLORS = ['#10b981', '#f59e0b', '#ef4444'];

  const tabs = [
    { id: 'overview', label: 'Overview', icon: TrendingUp },
    { id: 'forecast', label: 'Demand Forecast', icon: Calendar },
    { id: 'abc', label: 'ABC Analysis', icon: Package },
    { id: 'eoq', label: 'EOQ Optimization', icon: DollarSign },
    { id: 'reorder', label: 'Reorder Points', icon: AlertTriangle },
    { id: 'supplier', label: 'Supplier Analysis', icon: Clock }
  ];

  const renderOverview = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-lg border border-green-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-green-600 text-sm font-medium">Total Revenue</p>
              <p className="text-3xl font-bold text-green-900">$3.57M</p>
              <p className="text-green-600 text-sm mt-1">↑ 12.5% vs last month</p>
            </div>
            <DollarSign className="text-green-500" size={32} />
          </div>
        </div>
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-lg border border-blue-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-blue-600 text-sm font-medium">Active SKUs</p>
              <p className="text-3xl font-bold text-blue-900">1,000</p>
              <p className="text-blue-600 text-sm mt-1">Across all categories</p>
            </div>
            <Package className="text-blue-500" size={32} />
          </div>
        </div>
        <div className="bg-gradient-to-br from-orange-50 to-orange-100 p-6 rounded-lg border border-orange-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-orange-600 text-sm font-medium">Stockout Risk</p>
              <p className="text-3xl font-bold text-orange-900">3 Items</p>
              <p className="text-orange-600 text-sm mt-1">Require immediate action</p>
            </div>
            <AlertTriangle className="text-orange-500" size={32} />
          </div>
        </div>
        <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-6 rounded-lg border border-purple-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-purple-600 text-sm font-medium">Avg Lead Time</p>
              <p className="text-3xl font-bold text-purple-900">8.2 days</p>
              <p className="text-purple-600 text-sm mt-1">↓ 1.3 days improved</p>
            </div>
            <Clock className="text-purple-500" size={32} />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-semibold mb-4">Top 5 Products by Revenue</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={topProducts}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-15} textAnchor="end" height={80} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="revenue" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-semibold mb-4">Seasonal Sales Trends</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={seasonalTrends}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="wine" stroke="#ef4444" strokeWidth={2} />
              <Line type="monotone" dataKey="spirits" stroke="#f59e0b" strokeWidth={2} />
              <Line type="monotone" dataKey="beer" stroke="#10b981" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );

  const renderForecast = () => (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-lg border border-gray-200">
        <h3 className="text-lg font-semibold mb-4">6-Month Demand Forecast</h3>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={demandForecast}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="actual" stroke="#3b82f6" strokeWidth={3} name="Actual Sales" />
            <Line type="monotone" dataKey="forecast" stroke="#10b981" strokeWidth={3} strokeDasharray="5 5" name="Forecast" />
            <Line type="monotone" dataKey="upper" stroke="#ef4444" strokeWidth={1} strokeDasharray="3 3" name="Upper Bound" />
            <Line type="monotone" dataKey="lower" stroke="#ef4444" strokeWidth={1} strokeDasharray="3 3" name="Lower Bound" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  const renderABC = () => (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-lg border border-gray-200">
        <h3 className="text-lg font-semibold mb-4">Revenue Distribution</h3>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={abcAnalysis}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ category, percentage }) => `${category}: ${percentage}%`}
              outerRadius={100}
              fill="#8884d8"
              dataKey="percentage"
            >
              {abcAnalysis.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  const renderEOQ = () => (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-lg border border-gray-200">
        <h3 className="text-lg font-semibold mb-4">EOQ vs Current Order Quantity</h3>
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={eoqMetrics}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="product" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="currentOrder" fill="#ef4444" name="Current Order" />
            <Bar dataKey="optimalEOQ" fill="#10b981" name="Optimal EOQ" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  const renderReorder = () => (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-lg border border-gray-200">
        <h3 className="text-lg font-semibold mb-4">Reorder Point Status</h3>
        <div className="space-y-3">
          {reorderPoints.map((item, idx) => {
            const percentOfReorder = (item.current / item.reorderPoint) * 100;
            const statusColors = {
              critical: { bg: 'bg-red-50', border: 'border-red-200', bar: 'bg-red-500', badge: 'bg-red-200 text-red-900' },
              warning: { bg: 'bg-orange-50', border: 'border-orange-200', bar: 'bg-orange-500', badge: 'bg-orange-200 text-orange-900' },
              healthy: { bg: 'bg-green-50', border: 'border-green-200', bar: 'bg-green-500', badge: 'bg-green-200 text-green-900' }
            };
            const colors = statusColors[item.status];
            
            return (
              <div key={idx} className={`${colors.bg} p-4 rounded-lg border ${colors.border}`}>
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h4 className="font-semibold">{item.product}</h4>
                    <p className="text-sm text-gray-600">
                      Current: {item.current} | Reorder: {item.reorderPoint}
                    </p>
                  </div>
                  <span className={`px-3 py-1 rounded text-sm font-semibold ${colors.badge}`}>
                    {item.status.toUpperCase()}
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div 
                    className={`${colors.bar} h-3 rounded-full`}
                    style={{ width: `${Math.min(percentOfReorder, 100)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );

  const renderSupplier = () => (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-lg border border-gray-200">
        <h3 className="text-lg font-semibold mb-4">Supplier Performance Scores</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={supplierMetrics}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="supplier" angle={-15} textAnchor="end" height={80} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="score" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  const renderContent = () => {
    switch (activeTab) {
      case 'overview':
        return renderOverview();
      case 'forecast':
        return renderForecast();
      case 'abc':
        return renderABC();
      case 'eoq':
        return renderEOQ();
      case 'reorder':
        return renderReorder();
      case 'supplier':
        return renderSupplier();
      default:
        return renderOverview();
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Slooze Inventory Analytics</h1>
          <p className="text-gray-600">Wine & Spirits Retail Optimization Dashboard</p>
        </div>

        <div className="bg-white rounded-lg border border-gray-200 mb-6">
          <div className="flex overflow-x-auto">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center px-6 py-4 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon size={18} className="mr-2" />
                  {tab.label}
                </button>
              );
            })}
          </div>
        </div>

        <div>{renderContent()}</div>
      </div>
    </div>
  );
};

export default SloozeInventoryDashboard;