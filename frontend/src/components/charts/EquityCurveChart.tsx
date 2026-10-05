import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from 'recharts';
import type { EquityCurvePoint } from '../../types';
import { formatCurrency, formatCompactCurrency, formatDate } from '../../lib/formatters';
import { MODEL_COLORS } from '../../lib/utils';

interface EquityCurveChartProps {
  data: EquityCurvePoint[];
  height?: number;
}

export const EquityCurveChart: React.FC<EquityCurveChartProps> = ({
  data,
  height = 360,
}) => {
  // Sample dates to prevent XAxis overcrowding
  const tickInterval = Math.max(1, Math.floor(data.length / 8));

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
          <XAxis
            dataKey="date"
            interval={tickInterval}
            tickFormatter={(d) => {
              const parts = d.split('-');
              return parts.length >= 3 ? `${parts[1]}/${parts[0].slice(2)}` : d;
            }}
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={{ stroke: '#e2e8f0' }}
            tickLine={false}
          />
          <YAxis
            domain={['dataMin - 5000', 'dataMax + 5000']}
            tickFormatter={(v) => formatCompactCurrency(v)}
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={60}
          />
          <Tooltip
            content={({ active, payload, label }) => {
              if (active && payload && payload.length) {
                const ddpgVal = payload.find((p) => p.dataKey === 'ddpg_value')?.value as number;
                const benchVal = payload.find((p) => p.dataKey === 'equal_weight_value')?.value as number;
                const diff = ddpgVal !== undefined && benchVal !== undefined ? ddpgVal - benchVal : 0;
                return (
                  <div className="bg-white p-3.5 rounded-xl border border-slate-200/90 shadow-lg text-xs space-y-2 min-w-[200px]">
                    <p className="font-semibold text-slate-800 border-b border-slate-100 pb-1.5">
                      {formatDate(label)}
                    </p>
                    <div className="flex items-center justify-between text-slate-700">
                      <div className="flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: MODEL_COLORS.ddpg }} />
                        <span className="font-medium">DDPG Policy:</span>
                      </div>
                      <span className="font-semibold">{formatCurrency(ddpgVal)}</span>
                    </div>
                    <div className="flex items-center justify-between text-slate-700">
                      <div className="flex items-center gap-1.5">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: MODEL_COLORS.benchmark }} />
                        <span className="font-medium">Equal-Weight:</span>
                      </div>
                      <span className="font-semibold">{formatCurrency(benchVal)}</span>
                    </div>
                    <div className="pt-1.5 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-500">
                      <span>Spread (DDPG - EW):</span>
                      <span className={diff >= 0 ? 'text-emerald-600 font-medium' : 'text-slate-600 font-medium'}>
                        {diff >= 0 ? '+' : ''}{formatCurrency(diff)}
                      </span>
                    </div>
                  </div>
                );
              }
              return null;
            }}
          />
          <Legend
            verticalAlign="top"
            align="right"
            iconType="circle"
            wrapperStyle={{ paddingBottom: 15, fontSize: 12 }}
            formatter={(value) => (
              <span className="text-slate-700 font-medium">
                {value === 'ddpg_value' ? 'DDPG Agent Policy' : 'Equal-Weight Benchmark'}
              </span>
            )}
          />
          <Line
            type="monotone"
            dataKey="ddpg_value"
            stroke={MODEL_COLORS.ddpg}
            strokeWidth={2.2}
            dot={false}
            activeDot={{ r: 4, strokeWidth: 1, stroke: '#fff' }}
          />
          <Line
            type="monotone"
            dataKey="equal_weight_value"
            stroke={MODEL_COLORS.benchmark}
            strokeWidth={2.0}
            strokeDasharray="4 4"
            dot={false}
            activeDot={{ r: 4, strokeWidth: 1, stroke: '#fff' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
