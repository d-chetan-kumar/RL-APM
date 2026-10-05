import React, { useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { ChartSkeleton, TableSkeleton } from '../components/ui/Skeleton';
import { ErrorCard } from '../components/ui/ErrorCard';
import { EquityCurveChart } from '../components/charts/EquityCurveChart';
import { DrawdownChart } from '../components/charts/DrawdownChart';
import { api } from '../services/api';
import type { BacktestMetricsResponse, EquityCurveResponse } from '../types';
import {
  formatCurrency,
  formatPercent,
  formatRatio,
} from '../lib/formatters';
import { Info, BarChart3, TrendingDown } from 'lucide-react';

export const Performance: React.FC = () => {
  const [backtest, setBacktest] = useState<BacktestMetricsResponse | null>(null);
  const [equityCurve, setEquityCurve] = useState<EquityCurveResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadPerformanceData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [bt, eq] = await Promise.all([api.getBacktest(), api.getEquityCurve()]);
      setBacktest(bt);
      setEquityCurve(eq);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load performance evaluation');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadPerformanceData();
  }, []);

  if (error) {
    return (
      <div className="py-8">
        <ErrorCard title="Performance Data Unavailable" message={error} onRetry={loadPerformanceData} />
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-12">
      {/* Header Context Card */}
      <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-card flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <Badge variant="brand" size="sm">
              Out-of-Sample Evaluation
            </Badge>
            <Badge variant="slate" size="sm">
              Horizon: 2024-01-01 to 2024-12-31
            </Badge>
          </div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-indigo-600" />
            <span>Held-Out Performance Analysis</span>
          </h2>
          <p className="text-xs text-slate-500 mt-1 max-w-2xl">
            Quantitative out-of-sample backtest comparing the trained DDPG agent against an
            equal-weight (1/N) benchmark across risk-adjusted financial metrics.
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs text-slate-500 bg-slate-50 p-3 rounded-xl border border-slate-200/70">
          <Info className="w-4 h-4 text-slate-400 shrink-0" />
          <span>Capital: $100k • Cost: 0.1% • Friction Enforced</span>
        </div>
      </div>

      {/* Metrics Comparison Table */}
      <Card>
        <CardHeader>
          <div>
            <CardTitle>Comparative Performance Summary</CardTitle>
            <p className="text-xs text-slate-500 mt-0.5">
              Strictly measured evaluation values from the 2024 held-out dataset
            </p>
          </div>
          <Badge variant="outline" size="sm">
            252 Trading Days
          </Badge>
        </CardHeader>
        <CardContent>
          {isLoading || !backtest ? (
            <TableSkeleton rows={7} />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500 font-semibold uppercase tracking-wider text-[11px]">
                    <th className="pb-3 pt-1">Evaluation Metric</th>
                    <th className="pb-3 pt-1 text-right text-indigo-600">DDPG Agent Policy</th>
                    <th className="pb-3 pt-1 text-right text-amber-600">Equal-Weight Benchmark</th>
                    <th className="pb-3 pt-1 text-right text-slate-400">Difference</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-mono">
                  <tr className="text-slate-800">
                    <td className="py-3 font-sans font-medium text-slate-700">Final Portfolio Value</td>
                    <td className="py-3 text-right font-bold text-slate-900">
                      {formatCurrency(backtest.ddpg.final_portfolio_value)}
                    </td>
                    <td className="py-3 text-right font-bold text-slate-900">
                      {formatCurrency(backtest.equal_weight.final_portfolio_value)}
                    </td>
                    <td className="py-3 text-right text-slate-500 font-sans">
                      {formatCurrency(backtest.ddpg.final_portfolio_value - backtest.equal_weight.final_portfolio_value)}
                    </td>
                  </tr>

                  <tr className="text-slate-800">
                    <td className="py-3 font-sans font-medium text-slate-700">Cumulative Return</td>
                    <td className="py-3 text-right font-semibold text-emerald-600">
                      {formatPercent(backtest.ddpg.cumulative_return)}
                    </td>
                    <td className="py-3 text-right font-semibold text-emerald-600">
                      {formatPercent(backtest.equal_weight.cumulative_return)}
                    </td>
                    <td className="py-3 text-right text-slate-500 font-sans">
                      {formatPercent(backtest.ddpg.cumulative_return - backtest.equal_weight.cumulative_return)}
                    </td>
                  </tr>

                  <tr className="text-slate-800">
                    <td className="py-3 font-sans font-medium text-slate-700">Annualized Return (CAGR)</td>
                    <td className="py-3 text-right font-medium">
                      {formatPercent(backtest.ddpg.annualized_return)}
                    </td>
                    <td className="py-3 text-right font-medium">
                      {formatPercent(backtest.equal_weight.annualized_return)}
                    </td>
                    <td className="py-3 text-right text-slate-500 font-sans">
                      {formatPercent(backtest.ddpg.annualized_return - backtest.equal_weight.annualized_return)}
                    </td>
                  </tr>

                  <tr className="text-slate-800">
                    <td className="py-3 font-sans font-medium text-slate-700">Annualized Volatility ($\sigma$)</td>
                    <td className="py-3 text-right font-medium text-slate-700">
                      {formatPercent(backtest.ddpg.annualized_volatility, false)}
                    </td>
                    <td className="py-3 text-right font-medium text-slate-700">
                      {formatPercent(backtest.equal_weight.annualized_volatility, false)}
                    </td>
                    <td className="py-3 text-right text-slate-500 font-sans">
                      {formatPercent(backtest.ddpg.annualized_volatility - backtest.equal_weight.annualized_volatility, false)}
                    </td>
                  </tr>

                  <tr className="text-slate-800">
                    <td className="py-3 font-sans font-medium text-slate-700">Sharpe Ratio ($r_f=0$)</td>
                    <td className="py-3 text-right font-bold text-indigo-700">
                      {formatRatio(backtest.ddpg.sharpe_ratio)}
                    </td>
                    <td className="py-3 text-right font-bold text-amber-700">
                      {formatRatio(backtest.equal_weight.sharpe_ratio)}
                    </td>
                    <td className="py-3 text-right text-slate-500 font-sans">
                      {(backtest.ddpg.sharpe_ratio - backtest.equal_weight.sharpe_ratio).toFixed(2)}
                    </td>
                  </tr>

                  <tr className="text-slate-800">
                    <td className="py-3 font-sans font-medium text-slate-700">Maximum Drawdown (MDD)</td>
                    <td className="py-3 text-right font-medium text-rose-600">
                      {formatPercent(backtest.ddpg.max_drawdown, false)}
                    </td>
                    <td className="py-3 text-right font-medium text-amber-600">
                      {formatPercent(backtest.equal_weight.max_drawdown, false)}
                    </td>
                    <td className="py-3 text-right text-slate-500 font-sans">
                      {formatPercent(backtest.ddpg.max_drawdown - backtest.equal_weight.max_drawdown, false)}
                    </td>
                  </tr>

                  <tr className="text-slate-800">
                    <td className="py-3 font-sans font-medium text-slate-700">Cumulative Transaction Costs</td>
                    <td className="py-3 text-right font-medium text-slate-800">
                      {formatCurrency(backtest.ddpg.total_transaction_costs)}
                    </td>
                    <td className="py-3 text-right font-medium text-slate-800">
                      {formatCurrency(backtest.equal_weight.total_transaction_costs)}
                    </td>
                    <td className="py-3 text-right text-slate-500 font-sans">
                      {formatCurrency(backtest.ddpg.total_transaction_costs - backtest.equal_weight.total_transaction_costs)}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Primary Equity Curves Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Daily Out-of-Sample Trajectory (2024)</CardTitle>
          <Badge variant="brand" size="sm">DDPG vs. Benchmark</Badge>
        </CardHeader>
        <CardContent>
          {isLoading || !equityCurve ? (
            <ChartSkeleton height="h-80" />
          ) : (
            <EquityCurveChart data={equityCurve.points} height={360} />
          )}
        </CardContent>
      </Card>

      {/* Underwater Drawdown Chart */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <TrendingDown className="w-4 h-4 text-rose-600" />
            <CardTitle>Underwater Drawdown Profile</CardTitle>
          </div>
          <Badge variant="outline" size="sm">Peak-to-Trough</Badge>
        </CardHeader>
        <CardContent>
          {isLoading || !equityCurve ? (
            <ChartSkeleton height="h-60" />
          ) : (
            <DrawdownChart data={equityCurve.points} height={240} />
          )}
        </CardContent>
      </Card>

      {/* Research Note */}
      <Card className="bg-slate-50 border-slate-200">
        <CardContent className="p-4 flex items-start gap-3 text-xs text-slate-500">
          <Info className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold text-slate-700 block mb-0.5">Scientific Evaluation Notes</span>
            <p>
              Both strategies were initialized with $100,000.00 on January 2, 2024 and traded daily until
              December 31, 2024 under identical market frictions (0.10% transaction cost per dollar traded).
              The measured results represent strict backtest execution on unseen data. In accordance with
              academic rigor, these results are presented as measured without declarative rankings or marketing claims.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
