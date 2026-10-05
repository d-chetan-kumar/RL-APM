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
  ReferenceDot,
} from 'recharts';
import type { TrainingEpisodeRecord, ValidationCheckpointRecord } from '../../types';
import { formatCurrency, formatPercent } from '../../lib/formatters';

interface TrainingChartProps {
  data: TrainingEpisodeRecord[];
  height?: number;
}

export const RewardCurveChart: React.FC<TrainingChartProps> = ({ data, height = 240 }) => {
  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
          <XAxis
            dataKey="episode"
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={{ stroke: '#e2e8f0' }}
            tickLine={false}
            tickFormatter={(ep) => `Ep ${ep}`}
          />
          <YAxis
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={40}
          />
          <Tooltip
            content={({ active, payload, label }) => {
              if (active && payload && payload.length) {
                const epData = payload[0].payload as TrainingEpisodeRecord;
                return (
                  <div className="bg-white p-3 rounded-xl border border-slate-200 shadow-md text-xs space-y-1">
                    <p className="font-semibold text-slate-800 border-b border-slate-100 pb-1">
                      Episode {label}
                    </p>
                    <div className="flex justify-between gap-4">
                      <span className="text-slate-500">Cumulative Return:</span>
                      <span className="font-semibold text-indigo-600">{formatPercent(epData.cumulative_return)}</span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span className="text-slate-500">Total Reward:</span>
                      <span className="font-semibold">{epData.total_reward.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span className="text-slate-500">Duration:</span>
                      <span className="text-slate-700">{epData.duration_sec.toFixed(1)}s</span>
                    </div>
                  </div>
                );
              }
              return null;
            }}
          />
          <Line
            type="monotone"
            dataKey="total_reward"
            name="Episode Total Reward"
            stroke="#4f46e5"
            strokeWidth={2}
            dot={{ r: 3, fill: '#4f46e5' }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export const LossCurvesChart: React.FC<TrainingChartProps> = ({ data, height = 240 }) => {
  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
          <XAxis
            dataKey="episode"
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={{ stroke: '#e2e8f0' }}
            tickLine={false}
            tickFormatter={(ep) => `Ep ${ep}`}
          />
          <YAxis
            yAxisId="actor"
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={45}
          />
          <YAxis
            yAxisId="critic"
            orientation="right"
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={50}
            tickFormatter={(v) => v.toExponential(1)}
          />
          <Tooltip
            content={({ active, payload, label }) => {
              if (active && payload && payload.length) {
                const epData = payload[0].payload as TrainingEpisodeRecord;
                return (
                  <div className="bg-white p-3 rounded-xl border border-slate-200 shadow-md text-xs space-y-1">
                    <p className="font-semibold text-slate-800 border-b border-slate-100 pb-1">
                      Episode {label}
                    </p>
                    <div className="flex justify-between gap-4 text-indigo-600 font-medium">
                      <span>Mean Actor Loss:</span>
                      <span>{epData.mean_actor_loss.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between gap-4 text-amber-600 font-medium">
                      <span>Mean Critic Loss:</span>
                      <span>{epData.mean_critic_loss.toExponential(3)}</span>
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
          />
          <Line
            yAxisId="actor"
            type="monotone"
            dataKey="mean_actor_loss"
            name="Actor Loss"
            stroke="#6366f1"
            strokeWidth={1.8}
            dot={{ r: 2.5 }}
          />
          <Line
            yAxisId="critic"
            type="monotone"
            dataKey="mean_critic_loss"
            name="Critic Loss"
            stroke="#f59e0b"
            strokeWidth={1.8}
            dot={{ r: 2.5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export const ValidationCurveChart: React.FC<{
  data: ValidationCheckpointRecord[];
  height?: number;
}> = ({ data, height = 240 }) => {
  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
          <XAxis
            dataKey="episode"
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={{ stroke: '#e2e8f0' }}
            tickLine={false}
            tickFormatter={(ep) => `Ep ${ep}`}
          />
          <YAxis
            domain={['dataMin - 10000', 'dataMax + 10000']}
            tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={45}
          />
          <Tooltip
            content={({ active, payload, label }) => {
              if (active && payload && payload.length) {
                const item = payload[0].payload as ValidationCheckpointRecord;
                return (
                  <div className="bg-white p-3 rounded-xl border border-slate-200 shadow-md text-xs space-y-1 min-w-[190px]">
                    <div className="flex items-center justify-between border-b border-slate-100 pb-1">
                      <span className="font-semibold text-slate-800">Episode {label}</span>
                      {item.episode === 9 && (
                        <span className="text-[10px] bg-emerald-50 text-emerald-700 px-1.5 py-0.5 rounded font-medium border border-emerald-200/60">
                          Best Checkpoint
                        </span>
                      )}
                    </div>
                    <div className="flex justify-between gap-4">
                      <span className="text-slate-500">Val Portfolio:</span>
                      <span className="font-semibold">{formatCurrency(item.val_final_portfolio_value)}</span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span className="text-slate-500">Val Return:</span>
                      <span className="font-semibold text-emerald-600">{formatPercent(item.val_cumulative_return)}</span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span className="text-slate-500">Val Sharpe:</span>
                      <span className="font-medium">{item.val_sharpe_ratio.toFixed(2)}</span>
                    </div>
                  </div>
                );
              }
              return null;
            }}
          />
          <Line
            type="monotone"
            dataKey="val_final_portfolio_value"
            name="Validation Value"
            stroke="#10b981"
            strokeWidth={2}
            dot={{ r: 3, fill: '#10b981' }}
            activeDot={{ r: 5 }}
          />
          {/* Highlight Episode 9 (Best Checkpoint) */}
          <ReferenceDot
            x={9}
            y={167268.04}
            r={6}
            fill="#10b981"
            stroke="#ffffff"
            strokeWidth={2}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
