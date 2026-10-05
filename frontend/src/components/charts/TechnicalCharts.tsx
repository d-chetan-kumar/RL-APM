import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
  Cell,
  Legend,
} from 'recharts';
import type { IndicatorRow } from '../../types';
import { formatCurrency, formatDate } from '../../lib/formatters';

interface TechnicalChartProps {
  data: IndicatorRow[];
  height?: number;
}

export const PriceIndicatorChart: React.FC<TechnicalChartProps> = ({ data, height = 340 }) => {
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
              return parts.length >= 3 ? `${parts[1]}/${parts[2]}` : d;
            }}
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={{ stroke: '#e2e8f0' }}
            tickLine={false}
          />
          <YAxis
            domain={['dataMin * 0.96', 'dataMax * 1.04']}
            tickFormatter={(v) => `$${v.toFixed(0)}`}
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={55}
          />
          <Tooltip
            content={({ active, payload, label }) => {
              if (active && payload && payload.length) {
                return (
                  <div className="bg-white p-3 rounded-xl border border-slate-200 shadow-md text-xs space-y-1.5 min-w-[170px]">
                    <p className="font-semibold text-slate-800 border-b border-slate-100 pb-1">
                      {formatDate(label)}
                    </p>
                    {payload.map((p) => (
                      <div key={p.name} className="flex justify-between items-center text-slate-600">
                        <div className="flex items-center gap-1.5">
                          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: p.color }} />
                          <span>{p.name}:</span>
                        </div>
                        <span className="font-semibold">{formatCurrency(p.value as number)}</span>
                      </div>
                    ))}
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
          />
          <Line
            type="monotone"
            dataKey="close"
            name="Close Price"
            stroke="#0f172a"
            strokeWidth={2}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="sma_20"
            name="SMA 20"
            stroke="#3b82f6"
            strokeWidth={1.5}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="sma_50"
            name="SMA 50"
            stroke="#f59e0b"
            strokeWidth={1.5}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="bb_high"
            name="BB High"
            stroke="#94a3b8"
            strokeDasharray="3 3"
            strokeWidth={1}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="bb_low"
            name="BB Low"
            stroke="#94a3b8"
            strokeDasharray="3 3"
            strokeWidth={1}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export const RSIChart: React.FC<TechnicalChartProps> = ({ data, height = 180 }) => {
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
              return parts.length >= 3 ? `${parts[1]}/${parts[2]}` : d;
            }}
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={{ stroke: '#e2e8f0' }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 100]}
            ticks={[30, 50, 70]}
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={35}
          />
          <ReferenceLine y={70} stroke="#f43f5e" strokeDasharray="3 3" strokeWidth={1} label={{ value: 'Overbought (70)', fill: '#f43f5e', fontSize: 10, position: 'insideTopRight' }} />
          <ReferenceLine y={30} stroke="#10b981" strokeDasharray="3 3" strokeWidth={1} label={{ value: 'Oversold (30)', fill: '#10b981', fontSize: 10, position: 'insideBottomRight' }} />
          <Tooltip
            content={({ active, payload, label }) => {
              if (active && payload && payload.length) {
                const val = payload[0].value as number;
                return (
                  <div className="bg-white p-2.5 rounded-lg border border-slate-200 shadow-md text-xs">
                    <p className="font-semibold text-slate-800">{formatDate(label)}</p>
                    <p className="text-indigo-600 font-medium mt-1">RSI (14): {val.toFixed(2)}</p>
                  </div>
                );
              }
              return null;
            }}
          />
          <Line
            type="monotone"
            dataKey="rsi_14"
            name="RSI (14)"
            stroke="#6366f1"
            strokeWidth={1.8}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export const MACDChart: React.FC<TechnicalChartProps> = ({ data, height = 200 }) => {
  const tickInterval = Math.max(1, Math.floor(data.length / 8));

  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
          <XAxis
            dataKey="date"
            interval={tickInterval}
            tickFormatter={(d) => {
              const parts = d.split('-');
              return parts.length >= 3 ? `${parts[1]}/${parts[2]}` : d;
            }}
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={{ stroke: '#e2e8f0' }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={40}
          />
          <ReferenceLine y={0} stroke="#cbd5e1" strokeWidth={1} />
          <Tooltip
            content={({ active, payload, label }) => {
              if (active && payload && payload.length) {
                const macd = payload.find((p) => p.dataKey === 'macd')?.value as number;
                const sig = payload.find((p) => p.dataKey === 'macd_signal')?.value as number;
                const hist = payload.find((p) => p.dataKey === 'macd_hist')?.value as number;
                return (
                  <div className="bg-white p-2.5 rounded-lg border border-slate-200 shadow-md text-xs space-y-1">
                    <p className="font-semibold text-slate-800">{formatDate(label)}</p>
                    <div className="flex justify-between gap-4">
                      <span className="text-indigo-600 font-medium">MACD:</span>
                      <span className="font-semibold">{macd?.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span className="text-amber-600 font-medium">Signal:</span>
                      <span className="font-semibold">{sig?.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span className="text-slate-600 font-medium">Histogram:</span>
                      <span className={`font-semibold ${hist >= 0 ? 'text-emerald-600' : 'text-rose-600'}`}>
                        {hist?.toFixed(3)}
                      </span>
                    </div>
                  </div>
                );
              }
              return null;
            }}
          />
          <Bar dataKey="macd_hist" name="Histogram" maxBarSize={6}>
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.macd_hist >= 0 ? '#10b981' : '#f43f5e'}
              />
            ))}
          </Bar>
          <Line
            type="monotone"
            dataKey="macd"
            name="MACD"
            stroke="#4f46e5"
            strokeWidth={1.5}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="macd_signal"
            name="Signal"
            stroke="#f59e0b"
            strokeWidth={1.5}
            dot={false}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
