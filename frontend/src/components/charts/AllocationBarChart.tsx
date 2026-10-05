import React from 'react';
import { ASSET_COLORS } from '../../lib/utils';
import { formatPercent } from '../../lib/formatters';

interface AllocationBarChartProps {
  allocations: Record<string, number>;
}

export const AllocationBarChart: React.FC<AllocationBarChartProps> = ({ allocations }) => {
  const entries = Object.entries(allocations).sort((a, b) => b[1] - a[1]);

  return (
    <div className="space-y-3">
      {entries.map(([ticker, weight]) => {
        const pct = Math.max(0, weight * 100);
        const color = ASSET_COLORS[ticker] || '#6366f1';
        return (
          <div key={ticker} className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                <span className="font-semibold text-slate-800">{ticker}</span>
              </div>
              <span className="font-semibold text-slate-700">{formatPercent(weight, false)}</span>
            </div>
            <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500 ease-out"
                style={{
                  width: `${Math.min(100, Math.max(0.5, pct))}%`,
                  backgroundColor: color,
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
};
