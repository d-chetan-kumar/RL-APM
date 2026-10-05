import React, { useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { ChartSkeleton } from '../components/ui/Skeleton';
import { ErrorCard } from '../components/ui/ErrorCard';
import {
  RewardCurveChart,
  LossCurvesChart,
  ValidationCurveChart,
} from '../components/charts/TrainingCharts';
import { api } from '../services/api';
import type {
  ModelInfoResponse,
  TrainingMetricsResponse,
  ValidationMetricsResponse,
} from '../types';
import { formatCurrency, formatPercent, formatRatio } from '../lib/formatters';
import { Activity, Award, Sliders, CheckCircle2 } from 'lucide-react';

export const TrainingLab: React.FC = () => {
  const [training, setTraining] = useState<TrainingMetricsResponse | null>(null);
  const [validation, setValidation] = useState<ValidationMetricsResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadTrainingData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [tr, val, info] = await Promise.all([
        api.getTrainingMetrics(),
        api.getValidationMetrics(),
        api.getModelInfo(),
      ]);
      setTraining(tr);
      setValidation(val);
      setModelInfo(info);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load training laboratory data');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadTrainingData();
  }, []);

  if (error) {
    return (
      <div className="py-8">
        <ErrorCard title="Training Data Unavailable" message={error} onRetry={loadTrainingData} />
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-12">
      {/* Overview Context Card */}
      <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-card flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <Badge variant="brand" size="sm">Training Environment</Badge>
            <Badge variant="outline" size="sm">2015-03-16 to 2021-12-31</Badge>
          </div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <Activity className="w-5 h-5 text-indigo-600" />
            <span>Training Dynamics & Validation Lab</span>
          </h2>
          <p className="text-xs text-slate-500 mt-1 max-w-2xl">
            Empirical convergence curves, actor-critic loss optimization, experience replay dynamics,
            and periodic validation checkpoints across 15 complete training episodes.
          </p>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center text-xs">
          <div className="bg-slate-50 p-2.5 rounded-xl border border-slate-200/60">
            <span className="text-[10px] text-slate-400 block font-medium">Episodes</span>
            <span className="text-sm font-bold text-slate-800">15 Full Runs</span>
          </div>
          <div className="bg-slate-50 p-2.5 rounded-xl border border-slate-200/60">
            <span className="text-[10px] text-slate-400 block font-medium">State Dim</span>
            <span className="text-sm font-bold text-slate-800">50 Inputs</span>
          </div>
          <div className="bg-slate-50 p-2.5 rounded-xl border border-slate-200/60">
            <span className="text-[10px] text-slate-400 block font-medium">Action Dim</span>
            <span className="text-sm font-bold text-slate-800">5 Continuous</span>
          </div>
          <div className="bg-indigo-50 p-2.5 rounded-xl border border-indigo-200/60">
            <span className="text-[10px] text-indigo-500 block font-medium">Replay Buffer</span>
            <span className="text-sm font-bold text-indigo-900">100,000 Trans</span>
          </div>
        </div>
      </div>

      {/* Episode Rewards & Actor/Critic Loss Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Episode Total Reward */}
        <Card>
          <CardHeader>
            <div>
              <CardTitle>Episode Cumulative Reward</CardTitle>
              <p className="text-xs text-slate-500 mt-0.5">
                Reward = Net portfolio return minus transaction cost friction
              </p>
            </div>
            <Badge variant="brand" size="sm">Training Curve</Badge>
          </CardHeader>
          <CardContent>
            {isLoading || !training ? (
              <ChartSkeleton height="h-60" />
            ) : (
              <RewardCurveChart data={training.episodes} height={240} />
            )}
          </CardContent>
        </Card>

        {/* Actor and Critic Loss */}
        <Card>
          <CardHeader>
            <div>
              <CardTitle>Actor & Critic Optimization Loss</CardTitle>
              <p className="text-xs text-slate-500 mt-0.5">
                Actor policy gradient vs. Critic Mean Squared Bellman Error (MSBE)
              </p>
            </div>
            <Badge variant="slate" size="sm">Dual Axis</Badge>
          </CardHeader>
          <CardContent>
            {isLoading || !training ? (
              <ChartSkeleton height="h-60" />
            ) : (
              <LossCurvesChart data={training.episodes} height={240} />
            )}
          </CardContent>
        </Card>
      </div>

      {/* Validation Progression & Checkpoints Table */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Award className="w-4 h-4 text-emerald-600" />
            <div>
              <CardTitle>Deterministic Validation Progression (2022–2023)</CardTitle>
              <p className="text-xs text-slate-500 mt-0.5">
                Periodic zero-noise evaluation checkpoints. Episode 9 achieved peak validation value.
              </p>
            </div>
          </div>
          <Badge variant="emerald" size="sm">Best Checkpoint: Ep 9</Badge>
        </CardHeader>
        <CardContent className="space-y-6">
          {isLoading || !validation ? (
            <ChartSkeleton height="h-60" />
          ) : (
            <>
              <ValidationCurveChart data={validation.checkpoints} height={240} />

              <div className="overflow-x-auto pt-2">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-500 font-medium uppercase tracking-wider text-[11px]">
                      <th className="pb-2.5">Checkpoint</th>
                      <th className="pb-2.5 text-right">Portfolio Value</th>
                      <th className="pb-2.5 text-right">Cumulative Return</th>
                      <th className="pb-2.5 text-right">Sharpe Ratio</th>
                      <th className="pb-2.5 text-right">Max Drawdown</th>
                      <th className="pb-2.5 text-right">Transaction Costs</th>
                      <th className="pb-2.5 text-center">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-mono">
                    {validation.checkpoints.map((cp) => {
                      const isBest = cp.episode === 9;
                      return (
                        <tr
                          key={cp.episode}
                          className={`text-slate-700 ${isBest ? 'bg-emerald-50/40 font-semibold' : ''}`}
                        >
                          <td className="py-2.5 font-sans font-medium text-slate-900">
                            Episode {cp.episode}
                          </td>
                          <td className="py-2.5 text-right">
                            {formatCurrency(cp.val_final_portfolio_value)}
                          </td>
                          <td className="py-2.5 text-right text-emerald-600">
                            {formatPercent(cp.val_cumulative_return)}
                          </td>
                          <td className="py-2.5 text-right font-medium text-indigo-700">
                            {formatRatio(cp.val_sharpe_ratio)}
                          </td>
                          <td className="py-2.5 text-right text-rose-600">
                            {formatPercent(cp.val_max_drawdown, false)}
                          </td>
                          <td className="py-2.5 text-right text-slate-500">
                            {formatCurrency(cp.val_transaction_costs)}
                          </td>
                          <td className="py-2.5 text-center font-sans">
                            {isBest ? (
                              <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-700 bg-emerald-100/70 px-2 py-0.5 rounded-full">
                                <CheckCircle2 className="w-3 h-3" /> Selected Policy
                              </span>
                            ) : (
                              <span className="text-slate-400 text-[11px]">Evaluated</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* Hyperparameter Specifications */}
      {modelInfo && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-indigo-600" />
              <CardTitle>Hyperparameter Architecture & Training Configuration</CardTitle>
            </div>
            <Badge variant="outline" size="sm">Confirmed by Backend</Badge>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-7 gap-3 text-xs">
              <div className="bg-slate-50 p-3 rounded-xl border border-slate-200/60">
                <span className="text-slate-500 text-[11px] block">Discount Factor ($\gamma$)</span>
                <span className="text-sm font-bold text-slate-900 block mt-0.5">{modelInfo.gamma}</span>
              </div>
              <div className="bg-slate-50 p-3 rounded-xl border border-slate-200/60">
                <span className="text-slate-500 text-[11px] block">Target Polyak ($\tau$)</span>
                <span className="text-sm font-bold text-slate-900 block mt-0.5">{modelInfo.tau}</span>
              </div>
              <div className="bg-slate-50 p-3 rounded-xl border border-slate-200/60">
                <span className="text-slate-500 text-[11px] block">Actor Learning Rate</span>
                <span className="text-sm font-bold text-slate-900 block mt-0.5">{modelInfo.actor_learning_rate}</span>
              </div>
              <div className="bg-slate-50 p-3 rounded-xl border border-slate-200/60">
                <span className="text-slate-500 text-[11px] block">Critic Learning Rate</span>
                <span className="text-sm font-bold text-slate-900 block mt-0.5">{modelInfo.critic_learning_rate}</span>
              </div>
              <div className="bg-slate-50 p-3 rounded-xl border border-slate-200/60">
                <span className="text-slate-500 text-[11px] block">Batch Size</span>
                <span className="text-sm font-bold text-slate-900 block mt-0.5">{modelInfo.batch_size}</span>
              </div>
              <div className="bg-slate-50 p-3 rounded-xl border border-slate-200/60">
                <span className="text-slate-500 text-[11px] block">Replay Capacity</span>
                <span className="text-sm font-bold text-slate-900 block mt-0.5">{modelInfo.replay_buffer_size.toLocaleString()}</span>
              </div>
              <div className="bg-slate-50 p-3 rounded-xl border border-slate-200/60">
                <span className="text-slate-500 text-[11px] block">Transaction Cost</span>
                <span className="text-sm font-bold text-slate-900 block mt-0.5">{formatPercent(modelInfo.transaction_cost, false)}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
