import React, { useState } from 'react';
import { Layers, ChevronDown, ChevronUp } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import { Badge } from '../ui/Badge';
import { ASSET_COLORS } from '../../lib/utils';

interface StateVisualizerProps {
  stateVector?: number[];
  date: string;
}

const TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'];
const FEATURE_NAMES = [
  'Daily Return',
  'SMA-20 Ratio',
  'SMA-50 Ratio',
  'RSI-14 Norm',
  'MACD / Close',
  'Signal / Close',
  'Hist / Close',
  'BB Position',
  'BB Width Norm',
];

export const StateVisualizer: React.FC<StateVisualizerProps> = ({ stateVector, date }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!stateVector || stateVector.length !== 50) {
    return (
      <Card className="p-4 text-center text-xs text-slate-500">
        Run inference to view the 50-dimensional observation state vector.
      </Card>
    );
  }

  // Parse state: first 45 features = 5 assets x 9 features; last 5 = weights
  const assetFeatures: Record<string, number[]> = {};
  TICKERS.forEach((t, i) => {
    assetFeatures[t] = stateVector.slice(i * 9, (i + 1) * 9);
  });
  const currentWeights = stateVector.slice(45, 50);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between w-full">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-indigo-600" />
            <CardTitle>50-Dimensional Observation State Vector</CardTitle>
            <Badge variant="brand" size="sm">
              Date: {date}
            </Badge>
          </div>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-xs text-indigo-600 font-medium hover:text-indigo-700 flex items-center gap-1 cursor-pointer"
          >
            <span>{isExpanded ? 'Collapse' : 'Inspect All Dimensions'}</span>
            {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </CardHeader>
      <CardContent>
        {/* Compact summary pill row */}
        <div className="grid grid-cols-2 sm:grid-cols-6 gap-2 mb-3 text-xs">
          {TICKERS.map((ticker) => (
            <div
              key={ticker}
              className="bg-slate-50 p-2.5 rounded-lg border border-slate-200/70"
            >
              <div className="flex items-center gap-1.5 mb-1">
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: ASSET_COLORS[ticker] }} />
                <span className="font-semibold text-slate-800">{ticker}</span>
              </div>
              <p className="text-[11px] text-slate-500">9 Market Features</p>
            </div>
          ))}
          <div className="bg-indigo-50/50 p-2.5 rounded-lg border border-indigo-200/60">
            <span className="font-semibold text-indigo-900 block">Portfolio</span>
            <p className="text-[11px] text-indigo-700">5 Asset Weights</p>
          </div>
        </div>

        {/* Detailed feature grid when expanded */}
        {isExpanded && (
          <div className="space-y-4 pt-3 border-t border-slate-100">
            {TICKERS.map((ticker) => (
              <div key={ticker} className="space-y-1.5">
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: ASSET_COLORS[ticker] }} />
                  <span className="text-xs font-semibold text-slate-800">{ticker} Features</span>
                </div>
                <div className="grid grid-cols-3 sm:grid-cols-5 md:grid-cols-9 gap-1.5">
                  {FEATURE_NAMES.map((name, idx) => {
                    const val = assetFeatures[ticker][idx];
                    return (
                      <div
                        key={name}
                        className="bg-slate-50/80 p-1.5 rounded border border-slate-200/60 text-[10px]"
                      >
                        <span className="text-slate-400 block truncate" title={name}>
                          {name}
                        </span>
                        <span className="font-mono font-medium text-slate-700 block truncate">
                          {val !== undefined ? val.toFixed(4) : '-'}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}

            {/* Current weights vector */}
            <div className="space-y-1.5 pt-2">
              <span className="text-xs font-semibold text-slate-800">Portfolio Weights Feature Subset (Last 5 Dimensions)</span>
              <div className="grid grid-cols-5 gap-2">
                {TICKERS.map((t, idx) => (
                  <div key={t} className="bg-indigo-50/40 p-2 rounded border border-indigo-100 text-xs">
                    <span className="text-slate-500 text-[10px] block">{t} Weight</span>
                    <span className="font-mono font-semibold text-indigo-900">
                      {currentWeights[idx] !== undefined ? currentWeights[idx].toFixed(4) : '-'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
