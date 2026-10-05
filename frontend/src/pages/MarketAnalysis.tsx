import React, { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { ChartSkeleton } from '../components/ui/Skeleton';
import { ErrorCard } from '../components/ui/ErrorCard';
import { AssetSelector } from '../components/market/AssetSelector';
import {
  PriceIndicatorChart,
  RSIChart,
  MACDChart,
} from '../components/charts/TechnicalCharts';
import { api } from '../services/api';
import type { IndicatorRow } from '../types';
import { formatCurrency, formatPercent } from '../lib/formatters';

const DATE_PRESETS = [
  { label: '3M', days: 90 },
  { label: '6M', days: 180 },
  { label: '1Y', days: 365 },
  { label: 'All', days: 0 },
];

export const MarketAnalysis: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const initialTicker = searchParams.get('ticker') || 'AAPL';

  const [selectedTicker, setSelectedTicker] = useState<string>(initialTicker);
  const [selectedPreset, setSelectedPreset] = useState<string>('1Y');
  const [indicators, setIndicators] = useState<IndicatorRow[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchIndicatorData = async (ticker: string, preset: string) => {
    setIsLoading(true);
    setError(null);
    try {
      let startDate: string | undefined = undefined;
      if (preset !== 'All') {
        const pObj = DATE_PRESETS.find((p) => p.label === preset);
        if (pObj && pObj.days > 0) {
          const d = new Date('2024-12-31');
          d.setDate(d.getDate() - pObj.days);
          startDate = d.toISOString().split('T')[0];
        }
      }

      const res = await api.getIndicators(ticker, startDate, undefined, 1000);
      setIndicators(res.indicators);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to fetch technical indicators');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchIndicatorData(selectedTicker, selectedPreset);
    setSearchParams({ ticker: selectedTicker });
  }, [selectedTicker, selectedPreset]);

  const latest = indicators.length > 0 ? indicators[indicators.length - 1] : null;

  return (
    <div className="space-y-6 pb-12">
      {/* Controls Bar: Asset Selector + Date Presets */}
      <div className="bg-white rounded-2xl border border-slate-200/80 p-4 shadow-card flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <AssetSelector
            selectedTicker={selectedTicker}
            onSelectTicker={setSelectedTicker}
          />
        </div>

        <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-xl border border-slate-200/70 w-fit">
          {DATE_PRESETS.map((p) => (
            <button
              key={p.label}
              onClick={() => setSelectedPreset(p.label)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                selectedPreset === p.label
                  ? 'bg-white text-slate-900 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {error ? (
        <ErrorCard
          title="Market Data Fetch Failed"
          message={error}
          onRetry={() => fetchIndicatorData(selectedTicker, selectedPreset)}
        />
      ) : (
        <>
          {/* Latest Metric Summary Pills */}
          {latest && (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
              <div className="bg-white p-3.5 rounded-xl border border-slate-200/80 shadow-card">
                <span className="text-[11px] text-slate-500 font-medium block">Latest Close</span>
                <span className="text-base font-bold text-slate-900 block mt-0.5">
                  {formatCurrency(latest.close)}
                </span>
              </div>
              <div className="bg-white p-3.5 rounded-xl border border-slate-200/80 shadow-card">
                <span className="text-[11px] text-slate-500 font-medium block">Daily Return</span>
                <span
                  className={`text-base font-bold block mt-0.5 ${
                    latest.daily_return >= 0 ? 'text-emerald-600' : 'text-rose-600'
                  }`}
                >
                  {formatPercent(latest.daily_return)}
                </span>
              </div>
              <div className="bg-white p-3.5 rounded-xl border border-slate-200/80 shadow-card">
                <span className="text-[11px] text-slate-500 font-medium block">RSI (14)</span>
                <span className="text-base font-bold text-indigo-600 block mt-0.5">
                  {latest.rsi_14.toFixed(2)}
                </span>
              </div>
              <div className="bg-white p-3.5 rounded-xl border border-slate-200/80 shadow-card">
                <span className="text-[11px] text-slate-500 font-medium block">MACD Hist</span>
                <span
                  className={`text-base font-bold block mt-0.5 ${
                    latest.macd_hist >= 0 ? 'text-emerald-600' : 'text-rose-600'
                  }`}
                >
                  {latest.macd_hist.toFixed(3)}
                </span>
              </div>
              <div className="bg-white p-3.5 rounded-xl border border-slate-200/80 shadow-card">
                <span className="text-[11px] text-slate-500 font-medium block">SMA 20 Ratio</span>
                <span className="text-base font-bold text-slate-800 block mt-0.5">
                  {formatPercent((latest.close / latest.sma_20) - 1.0)}
                </span>
              </div>
              <div className="bg-white p-3.5 rounded-xl border border-slate-200/80 shadow-card">
                <span className="text-[11px] text-slate-500 font-medium block">Bollinger Width</span>
                <span className="text-base font-bold text-slate-800 block mt-0.5">
                  {latest.bb_width.toFixed(2)}%
                </span>
              </div>
            </div>
          )}

          {/* 1. Price + Moving Averages + Bollinger Bands */}
          <Card>
            <CardHeader>
              <div>
                <CardTitle>Price Trajectory & Trend Bands</CardTitle>
                <p className="text-xs text-slate-500 mt-0.5">
                  {selectedTicker} Close with 20-day SMA, 50-day SMA, and Bollinger Band envelope
                </p>
              </div>
              <Badge variant="brand" size="sm">
                {selectedTicker}
              </Badge>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <ChartSkeleton height="h-80" />
              ) : (
                <PriceIndicatorChart data={indicators} height={340} />
              )}
            </CardContent>
          </Card>

          {/* 2. RSI & MACD Dual Cards */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* RSI */}
            <Card>
              <CardHeader>
                <div>
                  <CardTitle>Relative Strength Index (RSI 14)</CardTitle>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Momentum oscillator bounded [0, 100] with 30/70 overbought-oversold zones
                  </p>
                </div>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <ChartSkeleton height="h-48" />
                ) : (
                  <RSIChart data={indicators} height={200} />
                )}
              </CardContent>
            </Card>

            {/* MACD */}
            <Card>
              <CardHeader>
                <div>
                  <CardTitle>MACD (12, 26, 9)</CardTitle>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Trend-following momentum showing MACD line, signal line, and divergence histogram
                  </p>
                </div>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <ChartSkeleton height="h-48" />
                ) : (
                  <MACDChart data={indicators} height={200} />
                )}
              </CardContent>
            </Card>
          </div>
        </>
      )}
    </div>
  );
};
