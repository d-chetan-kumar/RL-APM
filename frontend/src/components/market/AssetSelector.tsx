import React from 'react';
import { ASSET_COLORS, cn } from '../../lib/utils';

interface AssetSelectorProps {
  selectedTicker: string;
  onSelectTicker: (ticker: string) => void;
  tickers?: string[];
}

const DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'];

export const AssetSelector: React.FC<AssetSelectorProps> = ({
  selectedTicker,
  onSelectTicker,
  tickers = DEFAULT_TICKERS,
}) => {
  return (
    <div className="flex items-center gap-1.5 p-1 bg-slate-100 rounded-xl border border-slate-200/80 w-fit">
      {tickers.map((ticker) => {
        const isSelected = selectedTicker === ticker;
        const color = ASSET_COLORS[ticker] || '#6366f1';
        return (
          <button
            key={ticker}
            onClick={() => onSelectTicker(ticker)}
            className={cn(
              'flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all duration-150 cursor-pointer',
              isSelected
                ? 'bg-white text-slate-900 shadow-xs font-semibold'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/50'
            )}
          >
            <span
              className="w-2 h-2 rounded-full transition-transform"
              style={{ backgroundColor: color }}
            />
            <span>{ticker}</span>
          </button>
        );
      })}
    </div>
  );
};
