import React from 'react';
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
} from 'recharts';
import { ASSET_COLORS } from '../../lib/utils';
import { formatPercent } from '../../lib/formatters';

interface AllocationDonutChartProps {
  allocations: Record<string, number>;
  height?: number;
}

export const AllocationDonutChart: React.FC<AllocationDonutChartProps> = ({
  allocations,
  height = 240,
}) => {
  const data = Object.entries(allocations).map(([ticker, weight]) => ({
    name: ticker,
    value: Math.max(0, weight),
    color: ASSET_COLORS[ticker] || '#64748b',
  }));

  return (
    <div className="w-full flex items-center justify-center relative" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={62}
            outerRadius={88}
            paddingAngle={2}
            dataKey="value"
          >
            {data.map((entry) => (
              <Cell key={entry.name} fill={entry.color} stroke="#ffffff" strokeWidth={2} />
            ))}
          </Pie>
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const item = payload[0];
                return (
                  <div className="bg-white p-2.5 rounded-lg border border-slate-200/90 shadow-md text-xs">
                    <div className="flex items-center gap-1.5 font-medium text-slate-800">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: item.payload.color }} />
                      <span>{item.name}:</span>
                      <span className="font-semibold">{formatPercent(item.value as number, false)}</span>
                    </div>
                  </div>
                );
              }
              return null;
            }}
          />
        </PieChart>
      </ResponsiveContainer>
      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
        <span className="text-[11px] font-medium text-slate-400">Total Assets</span>
        <span className="text-sm font-bold text-slate-800">5 Equities</span>
      </div>
    </div>
  );
};
