import React, { useEffect, useState } from 'react';
import {
  Brain,
  Sliders,
  Cpu,
  Calendar,
  CheckCircle2,
  AlertCircle,
  Shield,
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { AllocationDonutChart } from '../components/charts/AllocationDonutChart';
import { AllocationBarChart } from '../components/charts/AllocationBarChart';
import { WeightEditor } from '../components/portfolio/WeightEditor';
import { StateVisualizer } from '../components/portfolio/StateVisualizer';
import { api } from '../services/api';
import type { ModelInfoResponse, PredictionResponse } from '../types';
import { formatPercent } from '../lib/formatters';
import { ASSET_COLORS } from '../lib/utils';

const DEFAULT_WEIGHTS: Record<string, number> = {
  AAPL: 0.2,
  MSFT: 0.2,
  GOOGL: 0.2,
  AMZN: 0.2,
  NVDA: 0.2,
};

export const AIPortfolio: React.FC = () => {
  const [currentWeights, setCurrentWeights] = useState<Record<string, number>>(DEFAULT_WEIGHTS);
  const [selectedDate, setSelectedDate] = useState<string>('');
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);

  const [isLoading, setIsLoading] = useState(false);
  const [isInitialLoading, setIsInitialLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const loadInitialData = async () => {
    setIsInitialLoading(true);
    try {
      const [info, pred] = await Promise.all([
        api.getModelInfo(),
        api.predictPortfolio(),
      ]);
      setModelInfo(info);
      setPrediction(pred);
      if (pred.date) setSelectedDate(pred.date);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to initialize model inference service');
    } finally {
      setIsInitialLoading(false);
    }
  };

  useEffect(() => {
    loadInitialData();
  }, []);

  const handleRunInference = async () => {
    setIsLoading(true);
    setError(null);
    setSuccessMessage(null);
    try {
      const res = await api.predictPortfolio({
        date: selectedDate ? selectedDate : undefined,
        current_weights: currentWeights,
      });
      setPrediction(res);
      setSuccessMessage(`DDPG inference completed successfully for ${res.date}`);
      setTimeout(() => setSuccessMessage(null), 4000);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Model inference failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6 pb-12">
      {/* Top Banner & Control Card */}
      <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-card space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Badge variant="brand" size="sm">
                Neural Actor Policy
              </Badge>
              <Badge variant="outline" size="sm">
                Deterministic Inference
              </Badge>
            </div>
            <h2 className="text-xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
              <Brain className="w-5 h-5 text-indigo-600" />
              <span>AI Portfolio Rebalancing Lab</span>
            </h2>
            <p className="text-xs text-slate-500 max-w-2xl">
              Simulate the trained DDPG Actor's allocation decision given arbitrary initial portfolio
              weights and a historical market date.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <Button
              variant="primary"
              size="md"
              onClick={handleRunInference}
              isLoading={isLoading}
              disabled={isInitialLoading}
              icon={<Cpu className="w-4 h-4" />}
            >
              Run Policy Inference
            </Button>
          </div>
        </div>

        {/* Feedback Messages */}
        {error && (
          <div className="p-3 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-700 flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}
        {successMessage && (
          <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-xl text-xs text-emerald-700 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 shrink-0" />
            <span>{successMessage}</span>
          </div>
        )}
      </div>

      {/* Main Grid: Controls & Weights (Left 1/3) + Allocation Results (Right 2/3) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Inference Parameters */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-xs">
                <Sliders className="w-4 h-4 text-indigo-600" />
                <span>Inference Parameters</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-5">
              {/* Historical Date Input */}
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-slate-700 flex items-center gap-1.5">
                  <Calendar className="w-3.5 h-3.5 text-slate-400" />
                  Historical State Date
                </label>
                <input
                  type="date"
                  value={selectedDate}
                  min="2015-03-16"
                  max="2024-12-31"
                  onChange={(e) => setSelectedDate(e.target.value)}
                  className="w-full text-xs px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:bg-white"
                />
                <span className="text-[10px] text-slate-400 block">
                  Available range: 2015-03-16 to 2024-12-31
                </span>
              </div>

              {/* Weight Editor with sliders */}
              <WeightEditor
                weights={currentWeights}
                onChange={setCurrentWeights}
                disabled={isLoading}
              />
            </CardContent>
          </Card>

          {/* Model Specification Card */}
          {modelInfo && (
            <Card className="bg-slate-50/50">
              <CardHeader>
                <CardTitle className="text-xs">Model Metadata</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-xs">
                <div className="flex justify-between py-1 border-b border-slate-200/50">
                  <span className="text-slate-500">Algorithm:</span>
                  <span className="font-semibold text-slate-800">{modelInfo.model}</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-200/50">
                  <span className="text-slate-500">Observation Space:</span>
                  <span className="font-semibold text-slate-800">{modelInfo.state_dimension} dims</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-200/50">
                  <span className="text-slate-500">Action Space:</span>
                  <span className="font-semibold text-slate-800">{modelInfo.action_dimension} continuous logits</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-200/50">
                  <span className="text-slate-500">Target Softmax:</span>
                  <span className="font-semibold text-slate-800">Softmax ($\tau=0.005$)</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Checkpoint:</span>
                  <span className="font-mono text-[11px] text-indigo-600 font-medium">
                    {modelInfo.checkpoint_name}
                  </span>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right Column: Allocation Output Visualizations */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <div>
                <CardTitle>Recommended Target Allocations</CardTitle>
                <p className="text-xs text-slate-500 mt-0.5">
                  Output of the trained Actor network on observation state for {prediction?.date || 'latest'}
                </p>
              </div>
              <Badge variant="emerald" size="sm">
                Total Weight: {prediction ? `${(prediction.total_weight * 100).toFixed(1)}%` : '100%'}
              </Badge>
            </CardHeader>
            <CardContent>
              {prediction ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                  <AllocationDonutChart allocations={prediction.allocations} height={230} />
                  <div className="space-y-2">
                    <h4 className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
                      Allocation Breakdown
                    </h4>
                    <AllocationBarChart allocations={prediction.allocations} />
                  </div>
                </div>
              ) : (
                <div className="h-48 flex items-center justify-center text-slate-400 text-xs">
                  Run inference to view allocations
                </div>
              )}

              {/* Detailed asset weight table */}
              {prediction && (
                <div className="mt-6 pt-4 border-t border-slate-100 overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="text-slate-400 border-b border-slate-100">
                        <th className="pb-2 font-medium">Asset</th>
                        <th className="pb-2 font-medium">Target Weight</th>
                        <th className="pb-2 font-medium">Initial Weight</th>
                        <th className="pb-2 font-medium">Net Rebalance</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {Object.entries(prediction.allocations).map(([ticker, targetW]) => {
                        const initW = currentWeights[ticker] || 0.2;
                        const delta = targetW - initW;
                        const color = ASSET_COLORS[ticker] || '#6366f1';
                        return (
                          <tr key={ticker} className="text-slate-700">
                            <td className="py-2.5 font-semibold flex items-center gap-2">
                              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                              {ticker}
                            </td>
                            <td className="py-2.5 font-bold text-slate-900">
                              {formatPercent(targetW, false)}
                            </td>
                            <td className="py-2.5 text-slate-500">
                              {formatPercent(initW, false)}
                            </td>
                            <td
                              className={`py-2.5 font-medium ${
                                delta >= 0 ? 'text-emerald-600' : 'text-rose-600'
                              }`}
                            >
                              {formatPercent(delta)}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>

          {/* 50-Dimensional State Inspection Card */}
          <StateVisualizer
            stateVector={prediction?.state_vector}
            date={prediction?.date || selectedDate}
          />
        </div>
      </div>

      {/* Research Disclaimer Card */}
      <Card className="bg-slate-50 border-slate-200">
        <CardContent className="p-4 flex items-start gap-3 text-xs text-slate-500">
          <Shield className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
          <p>
            <strong className="text-slate-700 font-semibold">Research Inference Notice:</strong>{' '}
            Allocations generated by the DDPG agent are research outputs evaluated on historical equity data.
            The system does not connect to brokerage APIs or execute live financial trades. Rebalancing calculations
            assume a constant 0.1% proportional transaction cost.
          </p>
        </CardContent>
      </Card>
    </div>
  );
};
