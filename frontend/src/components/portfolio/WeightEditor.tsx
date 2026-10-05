import React from 'react';
import { ASSET_COLORS } from '../../lib/utils';
import { Button } from '../ui/Button';
import { RotateCcw, Sliders } from 'lucide-react';

interface WeightEditorProps {
  weights: Record<string, number>;
  onChange: (weights: Record<string, number>) => void;
  disabled?: boolean;
}

const TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'];

export const WeightEditor: React.FC<WeightEditorProps> = ({
  weights,
  onChange,
  disabled = false,
}) => {
  const total = Object.values(weights).reduce((sum, w) => sum + w, 0);
  const isNormalized = Math.abs(total - 1.0) < 0.005;

  const handleSliderChange = (ticker: string, val: number) => {
    onChange({
      ...weights,
      [ticker]: val / 100,
    });
  };

  const handleResetEqual = () => {
    const equalW = 1.0 / TICKERS.length;
    const newWeights: Record<string, number> = {};
    TICKERS.forEach((t) => {
      newWeights[t] = equalW;
    });
    onChange(newWeights);
  };

  const handleAutoNormalize = () => {
    if (total <= 0) return handleResetEqual();
    const newWeights: Record<string, number> = {};
    TICKERS.forEach((t) => {
      newWeights[t] = (weights[t] || 0) / total;
    });
    onChange(newWeights);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-slate-700">Initial Asset Weights</span>
          <span
            className={`text-[11px] px-2 py-0.5 rounded-full font-medium ${
              isNormalized
                ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                : 'bg-amber-50 text-amber-700 border border-amber-200'
            }`}
          >
            Sum: {(total * 100).toFixed(1)}%
          </span>
        </div>
        <div className="flex items-center gap-2">
          {!isNormalized && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleAutoNormalize}
              disabled={disabled}
              className="h-7 text-[11px] px-2.5 py-0"
              icon={<Sliders className="w-3 h-3" />}
            >
              Normalize
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={handleResetEqual}
            disabled={disabled}
            className="h-7 text-[11px] px-2.5 py-0 text-slate-500"
            icon={<RotateCcw className="w-3 h-3" />}
          >
            Equal
          </Button>
        </div>
      </div>

      <div className="space-y-3">
        {TICKERS.map((ticker) => {
          const w = weights[ticker] !== undefined ? weights[ticker] : 0.2;
          const pct = Math.round(w * 100);
          const color = ASSET_COLORS[ticker] || '#6366f1';

          return (
            <div key={ticker} className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2 font-medium text-slate-800">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                  <span>{ticker}</span>
                </div>
                <span className="font-semibold text-slate-700">{pct}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                step="1"
                value={pct}
                disabled={disabled}
                onChange={(e) => handleSliderChange(ticker, parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed"
              />
            </div>
          );
        })}
      </div>
    </div>
  );
};
