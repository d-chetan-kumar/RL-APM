import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  TrendingUp,
  DollarSign,
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  Sparkles,
  ChevronRight,
  Cpu,
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { MetricCardSkeleton, ChartSkeleton } from '../components/ui/Skeleton';
import { ErrorCard } from '../components/ui/ErrorCard';
import { EquityCurveChart } from '../components/charts/EquityCurveChart';
import { AllocationDonutChart } from '../components/charts/AllocationDonutChart';
import { AllocationBarChart } from '../components/charts/AllocationBarChart';
import { api } from '../services/api';
import type {
  BacktestMetricsResponse,
  EquityCurveResponse,
  MarketSummaryResponse,
  PredictionResponse,
} from '../types';
import {
  formatCurrency,
  formatPercent,
  formatRatio,
} from '../lib/formatters';
import { ASSET_COLORS } from '../lib/utils';

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();

  const [backtest, setBacktest] = useState<BacktestMetricsResponse | null>(null);
  const [equityCurve, setEquityCurve] = useState<EquityCurveResponse | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [marketSummary, setMarketSummary] = useState<MarketSummaryResponse | null>(null);

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadDashboardData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [bt, eq, pred, mkt] = await Promise.all([
        api.getBacktest(),
        api.getEquityCurve(),
        api.predictPortfolio(),
        api.getMarketSummary(),
      ]);
      setBacktest(bt);
      setEquityCurve(eq);
      setPrediction(pred);
      setMarketSummary(mkt);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load dashboard data');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadDashboardData();
  }, []);

  if (error) {
    return (
      <div className="py-8">
        <ErrorCard
          title="Unable to load dashboard data"
          message={error}
          onRetry={loadDashboardData}
        />
      </div>
    );
  }

  // Derive model insights strictly from actual data
  const highestAllocAsset = prediction
    ? Object.entries(prediction.allocations).reduce((max, curr) => (curr[1] > max[1] ? curr : max))
    : null;

  const lowestAllocAsset = prediction
    ? Object.entries(prediction.allocations).reduce((min, curr) => (curr[1] < min[1] ? curr : min))
    : null;

  return (
    <div className="space-y-6 pb-12">
      {/* Hero Section */}
      <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-card flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <Badge variant="brand" size="sm">
              DDPG • 5 Assets • 50-Dimensional State
            </Badge>
            <Badge variant="outline" size="sm">
              Held-Out 2024 Evaluation
            </Badge>
          </div>
          <h2 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
            Portfolio Intelligence
          </h2>
          <p className="text-xs sm:text-sm text-slate-500 max-w-xl">
            Deep Deterministic Policy Gradient agent optimizing multi-asset capital allocation
            under transaction frictions and non-linear technical indicators.
          </p>
        </div>
        <div className="flex items-center gap-2.5">
          <button
            onClick={() => navigate('/portfolio')}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl text-xs font-semibold shadow-xs flex items-center gap-1.5 transition-colors cursor-pointer"
          >
            <Cpu className="w-4 h-4" />
            <span>Interactive Inference</span>
          </button>
          <button
            onClick={() => navigate('/methodology')}
            className="px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl text-xs font-semibold transition-colors cursor-pointer"
          >
            Methodology
          </button>
        </div>
      </div>

      {/* Metric Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {isLoading || !backtest ? (
          <>
            <MetricCardSkeleton />
            <MetricCardSkeleton />
            <MetricCardSkeleton />
            <MetricCardSkeleton />
          </>
        ) : (
          <>
            {/* 1. Final Portfolio Value */}
            <Card hoverable className="p-5">
              <div className="flex items-center justify-between text-slate-500 mb-2">
                <span className="text-xs font-medium">Test Portfolio Value</span>
                <div className="w-8 h-8 rounded-lg bg-indigo-50 text-indigo-600 flex items-center justify-center">
                  <DollarSign className="w-4 h-4" />
                </div>
              </div>
              <div className="text-2xl font-bold text-slate-900 tracking-tight">
                {formatCurrency(backtest.ddpg.final_portfolio_value)}
              </div>
              <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-100 text-[11px]">
                <span className="text-slate-500">2024 Held-Out Test</span>
                <span className="text-slate-600 font-medium">
                  EW: {formatCurrency(backtest.equal_weight.final_portfolio_value)}
                </span>
              </div>
            </Card>

            {/* 2. Cumulative Return */}
            <Card hoverable className="p-5">
              <div className="flex items-center justify-between text-slate-500 mb-2">
                <span className="text-xs font-medium">Cumulative Return</span>
                <div className="w-8 h-8 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center">
                  <TrendingUp className="w-4 h-4" />
                </div>
              </div>
              <div className="text-2xl font-bold text-emerald-600 tracking-tight">
                {formatPercent(backtest.ddpg.cumulative_return)}
              </div>
              <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-100 text-[11px]">
                <span className="text-slate-500">Annualized: {formatPercent(backtest.ddpg.annualized_return)}</span>
                <span className="text-slate-600 font-medium">
                  EW: {formatPercent(backtest.equal_weight.cumulative_return)}
                </span>
              </div>
            </Card>

            {/* 3. Sharpe Ratio */}
            <Card hoverable className="p-5">
              <div className="flex items-center justify-between text-slate-500 mb-2">
                <span className="text-xs font-medium">Sharpe Ratio</span>
                <div className="w-8 h-8 rounded-lg bg-indigo-50 text-indigo-600 flex items-center justify-center">
                  <Activity className="w-4 h-4" />
                </div>
              </div>
              <div className="text-2xl font-bold text-slate-900 tracking-tight">
                {formatRatio(backtest.ddpg.sharpe_ratio)}
              </div>
              <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-100 text-[11px]">
                <span className="text-slate-500">Vol: {formatPercent(backtest.ddpg.annualized_volatility, false)}</span>
                <span className="text-slate-600 font-medium">
                  EW: {formatRatio(backtest.equal_weight.sharpe_ratio)}
                </span>
              </div>
            </Card>

            {/* 4. Maximum Drawdown */}
            <Card hoverable className="p-5">
              <div className="flex items-center justify-between text-slate-500 mb-2">
                <span className="text-xs font-medium">Maximum Drawdown</span>
                <div className="w-8 h-8 rounded-lg bg-rose-50 text-rose-600 flex items-center justify-center">
                  <ArrowDownRight className="w-4 h-4" />
                </div>
              </div>
              <div className="text-2xl font-bold text-slate-900 tracking-tight">
                {formatPercent(backtest.ddpg.max_drawdown, false)}
              </div>
              <div className="flex items-center justify-between mt-2 pt-2 border-t border-slate-100 text-[11px]">
                <span className="text-slate-500">Peak-to-Trough</span>
                <span className="text-slate-600 font-medium">
                  EW: {formatPercent(backtest.equal_weight.max_drawdown, false)}
                </span>
              </div>
            </Card>
          </>
        )}
      </div>

      {/* Main Charts Row: Performance Equity Curve (2/3) + Real Allocation (1/3) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Equity Curve Chart */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <div>
              <CardTitle>Portfolio Performance vs. Benchmark</CardTitle>
              <p className="text-xs text-slate-500 mt-0.5">
                Out-of-sample test trajectory across 252 trading days in 2024
              </p>
            </div>
            <Badge variant="slate" size="sm">
              Held-Out 2024
            </Badge>
          </CardHeader>
          <CardContent>
            {isLoading || !equityCurve ? (
              <ChartSkeleton height="h-80" />
            ) : (
              <EquityCurveChart data={equityCurve.points} height={340} />
            )}
          </CardContent>
        </Card>

        {/* Real Model Allocation */}
        <Card className="flex flex-col justify-between">
          <CardHeader>
            <div>
              <CardTitle>Model Allocation</CardTitle>
              <p className="text-xs text-slate-500 mt-0.5">
                Latest Historical-Data Inference
              </p>
            </div>
            <Badge variant="brand" size="sm">
              {prediction ? prediction.date : 'Latest'}
            </Badge>
          </CardHeader>
          <CardContent className="space-y-4">
            {isLoading || !prediction ? (
              <div className="h-64 flex items-center justify-center">
                <MetricCardSkeleton />
              </div>
            ) : (
              <>
                <AllocationDonutChart allocations={prediction.allocations} height={190} />
                <div className="pt-2 border-t border-slate-100">
                  <AllocationBarChart allocations={prediction.allocations} />
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Market Snapshot Row */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-slate-800">Asset Universe Snapshot</h3>
            <p className="text-xs text-slate-500">
              Latest indicators and prices for all 5 portfolio equities
            </p>
          </div>
          <button
            onClick={() => navigate('/market')}
            className="text-xs font-semibold text-indigo-600 hover:text-indigo-700 flex items-center gap-1 cursor-pointer"
          >
            <span>Full Market Analysis</span>
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
          {isLoading || !marketSummary ? (
            <>
              <MetricCardSkeleton />
              <MetricCardSkeleton />
              <MetricCardSkeleton />
              <MetricCardSkeleton />
              <MetricCardSkeleton />
            </>
          ) : (
            marketSummary.assets.map((asset) => {
              const isPositive = asset.daily_return >= 0;
              const color = ASSET_COLORS[asset.ticker] || '#6366f1';
              return (
                <div
                  key={asset.ticker}
                  onClick={() => navigate(`/market?ticker=${asset.ticker}`)}
                  className="bg-white rounded-xl border border-slate-200/80 p-4 shadow-card hover:shadow-card-hover hover:border-slate-300 transition-all cursor-pointer group"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-1.5 font-semibold text-sm text-slate-900">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                      <span>{asset.ticker}</span>
                    </div>
                    <span
                      className={`text-xs font-medium flex items-center gap-0.5 ${
                        isPositive ? 'text-emerald-600' : 'text-rose-600'
                      }`}
                    >
                      {isPositive ? (
                        <ArrowUpRight className="w-3.5 h-3.5" />
                      ) : (
                        <ArrowDownRight className="w-3.5 h-3.5" />
                      )}
                      {formatPercent(asset.daily_return, false)}
                    </span>
                  </div>
                  <div className="text-lg font-bold text-slate-800 tracking-tight">
                    {formatCurrency(asset.close)}
                  </div>
                  <div className="mt-2 pt-2 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-500">
                    <span>RSI: {asset.rsi.toFixed(1)}</span>
                    <span>SMA20: {formatPercent(asset.sma20_ratio, false)}</span>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* AI Model Insight Panel (Scientifically Honest) */}
      <Card className="bg-gradient-to-r from-indigo-50/40 via-white to-white border-indigo-100">
        <CardContent className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 p-5">
          <div className="flex items-start gap-3.5">
            <div className="w-9 h-9 rounded-xl bg-indigo-600 text-white flex items-center justify-center shrink-0 shadow-sm shadow-indigo-200">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <h4 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                DDPG Model Allocation Insight
                <Badge variant="brand" size="sm">
                  Autonomous Policy
                </Badge>
              </h4>
              <p className="text-xs text-slate-600 mt-1 max-w-3xl leading-relaxed">
                {highestAllocAsset && lowestAllocAsset ? (
                  <>
                    Based on the latest available historical market state ({prediction?.date}), the trained
                    DDPG actor assigns the largest portfolio allocation to{' '}
                    <strong className="text-slate-900 font-semibold">{highestAllocAsset[0]}</strong> (
                    {formatPercent(highestAllocAsset[1], false)}) and lowest to{' '}
                    <strong className="text-slate-900 font-semibold">{lowestAllocAsset[0]}</strong> (
                    {formatPercent(lowestAllocAsset[1], false)}). Rebalancing friction is constrained by the
                    0.1% transaction cost penalty.
                  </>
                ) : (
                  'Model inference outputs deterministic portfolio allocations from the 50-dimensional observation state.'
                )}
              </p>
            </div>
          </div>
          <button
            onClick={() => navigate('/portfolio')}
            className="shrink-0 text-xs font-semibold text-indigo-600 hover:text-indigo-700 flex items-center gap-1 cursor-pointer bg-white px-3.5 py-2 rounded-lg border border-indigo-200 shadow-xs"
          >
            <span>Run Custom Inference</span>
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </CardContent>
      </Card>
    </div>
  );
};
