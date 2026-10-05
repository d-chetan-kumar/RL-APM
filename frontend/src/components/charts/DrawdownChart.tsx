import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from 'recharts';
import type { EquityCurvePoint } from '../../types';
import { formatPercent, formatDate } from '../../lib/formatters';
import { MODEL_COLORS } from '../../lib/utils';

interface DrawdownChartProps {
  data: EquityCurvePoint[];
  height?: number;
}

export const DrawdownChart: React.FC<DrawdownChartProps> = ({ data, height = 240 }) => {
  // Compute cumulative peak drawdown strictly from actual equity series
  const drawdownData = useMemo(() => {
    let ddpgPeak = -Infinity;
    let ewPeak = -Infinity;

    return data.map((pt) => {
      if (pt.ddpg_value > ddpgPeak) ddpgPeak = pt.ddpg_value;
      if (pt.equal_weight_value > ewPeak) ewPeak = pt.equal_weight_value;

      const ddpgDD = ddpgPeak > 0 ? (ddpgPeak - pt.ddpg_value) / ddpgPeak : 0;
      const ewDD = ewPeak > 0 ? (ewPeak - pt.equal_weight_value) / ewPeak : 0;

      return {
        date: pt.date,
        ddpg_drawdown: -ddpgDD, // Negative for standard underwater plot
        ew_drawdown: -ewDD,
      };
    });
  }, [data]);

  const tickInterval = Math.max(1, Math.floor(data.length / 8));

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={drawdownData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
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
            tickFormatter={(v) => formatPercent(v, false, 0)}
            domain={[-0.30, 0]}
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={45}
          />
          <Tooltip
            content={({ active, payload, label }) => {
              if (active && payload && payload.length) {
                const ddpgDD = payload.find((p) => p.dataKey === 'ddpg_drawdown')?.value as number;
                const ewDD = payload.find((p) => p.dataKey === 'ew_drawdown')?.value as number;
                return (
                  <div className="bg-white p-3 rounded-xl border border-slate-200 shadow-md text-xs space-y-1.5 min-w-[190px]">
                    <p className="font-semibold text-slate-800 border-b border-slate-100 pb-1">
                      {formatDate(label)}
                    </p>
                    <div className="flex justify-between items-center text-slate-700">
                      <span className="flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: MODEL_COLORS.ddpg }} />
                        DDPG Drawdown:
                      </span>
                      <span className="font-semibold text-rose-600">{formatPercent(ddpgDD, false)}</span>
                    </div>
                    <div className="flex justify-between items-center text-slate-700">
                      <span className="flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: MODEL_COLORS.benchmark }} />
                        Equal-Weight Drawdown:
                      </span>
                      <span className="font-semibold text-amber-600">{formatPercent(ewDD, false)}</span>
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
            wrapperStyle={{ paddingBottom: 10, fontSize: 11 }}
            formatter={(value) => (
              <span className="text-slate-700 font-medium">
                {value === 'ddpg_drawdown' ? 'DDPG Underwater' : 'Equal-Weight Underwater'}
              </span>
            )}
          />
          <Area
            type="monotone"
            dataKey="ddpg_drawdown"
            stroke={MODEL_COLORS.ddpg}
            fill={MODEL_COLORS.ddpg}
            fillOpacity={0.12}
            strokeWidth={1.8}
          />
          <Area
            type="monotone"
            dataKey="ew_drawdown"
            stroke={MODEL_COLORS.benchmark}
            fill={MODEL_COLORS.benchmark}
            fillOpacity={0.08}
            strokeWidth={1.5}
            strokeDasharray="4 4"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};
