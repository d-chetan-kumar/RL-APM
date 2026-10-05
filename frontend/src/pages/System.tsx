import React, { useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { ErrorCard } from '../components/ui/ErrorCard';
import { api } from '../services/api';
import type { HealthResponse, ModelInfoResponse } from '../types';
import {
  ShieldCheck,
  Server,
  Database,
  Cpu,
  ExternalLink,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
} from 'lucide-react';

const ENDPOINTS = [
  { method: 'GET', path: '/api/health', desc: 'System status & subsystem availability flags' },
  { method: 'GET', path: '/api/market-data', desc: 'Filtered & paginated historical market records' },
  { method: 'GET', path: '/api/market-summary', desc: 'Latest financial indicator snapshot across all 5 assets' },
  { method: 'GET', path: '/api/indicators/{ticker}', desc: 'Technical indicators (SMA, RSI, MACD, Bollinger)' },
  { method: 'POST', path: '/api/predict', desc: 'DDPG model inference on 50-dimensional observation state' },
  { method: 'GET', path: '/api/portfolio/summary', desc: 'Combined backtest overview and latest inference' },
  { method: 'GET', path: '/api/model-info', desc: 'Verified DDPG hyperparameters and architectures' },
  { method: 'GET', path: '/api/backtest', desc: 'Held-out 2024 test evaluation metrics (DDPG vs. EW)' },
  { method: 'GET', path: '/api/backtest/equity-curve', desc: 'Daily equity trajectory series (252 days)' },
  { method: 'GET', path: '/api/training-metrics', desc: '15 training episode rewards, actor/critic losses' },
  { method: 'GET', path: '/api/validation-metrics', desc: 'Periodic deterministic validation checkpoints' },
];

export const System: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSystemData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [h, m] = await Promise.all([api.getHealth(), api.getModelInfo()]);
      setHealth(h);
      setModelInfo(m);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to query system status');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchSystemData();
  }, []);

  return (
    <div className="space-y-6 pb-12">
      {/* Overview Context Card */}
      <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-card flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <Badge variant={health?.status === 'ok' ? 'emerald' : 'rose'} size="sm">
              {health?.status === 'ok' ? 'Operational' : 'Degraded / Offline'}
            </Badge>
            <Badge variant="outline" size="sm">FastAPI + Uvicorn</Badge>
          </div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-indigo-600" />
            <span>System Health & Diagnostics</span>
          </h2>
          <p className="text-xs text-slate-500 mt-1 max-w-xl">
            Real-time status of backend services, neural model weight checkpoints, processed data
            integrity, and available REST API endpoints.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={fetchSystemData}
            isLoading={isLoading}
            icon={<RefreshCw className="w-3.5 h-3.5" />}
          >
            Refresh Status
          </Button>
          <a
            href="http://127.0.0.1:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-xs font-semibold shadow-xs transition-colors"
          >
            <span>Swagger UI</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>
      </div>

      {error ? (
        <ErrorCard title="System Diagnostics Unavailable" message={error} onRetry={fetchSystemData} />
      ) : (
        <>
          {/* Subsystem Health Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-slate-500">FastAPI Service</span>
                <Server className="w-4 h-4 text-slate-400" />
              </div>
              <div className="flex items-center gap-2">
                {health?.status === 'ok' ? (
                  <CheckCircle2 className="w-5 h-5 text-emerald-600" />
                ) : (
                  <AlertTriangle className="w-5 h-5 text-rose-600" />
                )}
                <span className="text-sm font-bold text-slate-900">
                  {health?.status === 'ok' ? 'Online' : 'Offline'}
                </span>
              </div>
              <span className="text-[10px] text-slate-400 block mt-2">Port: 8000 • Host: 127.0.0.1</span>
            </Card>

            <Card className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-slate-500">DDPG Actor Model</span>
                <Cpu className="w-4 h-4 text-slate-400" />
              </div>
              <div className="flex items-center gap-2">
                {health?.model_available ? (
                  <CheckCircle2 className="w-5 h-5 text-emerald-600" />
                ) : (
                  <AlertTriangle className="w-5 h-5 text-rose-600" />
                )}
                <span className="text-sm font-bold text-slate-900">
                  {health?.model_available ? 'Checkpoint Loaded' : 'Checkpoint Missing'}
                </span>
              </div>
              <span className="text-[10px] text-slate-400 block mt-2">models/ddpg_actor_best.pth</span>
            </Card>

            <Card className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-slate-500">Processed Market Data</span>
                <Database className="w-4 h-4 text-slate-400" />
              </div>
              <div className="flex items-center gap-2">
                {health?.data_available ? (
                  <CheckCircle2 className="w-5 h-5 text-emerald-600" />
                ) : (
                  <AlertTriangle className="w-5 h-5 text-rose-600" />
                )}
                <span className="text-sm font-bold text-slate-900">
                  {health?.data_available ? 'All 5 Datasets Ready' : 'Data Missing'}
                </span>
              </div>
              <span className="text-[10px] text-slate-400 block mt-2">data/processed/*.csv (10 Years)</span>
            </Card>

            <Card className="p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-slate-500">Research Metrics</span>
                <ShieldCheck className="w-4 h-4 text-slate-400" />
              </div>
              <div className="flex items-center gap-2">
                {health?.metrics_available ? (
                  <CheckCircle2 className="w-5 h-5 text-emerald-600" />
                ) : (
                  <AlertTriangle className="w-5 h-5 text-rose-600" />
                )}
                <span className="text-sm font-bold text-slate-900">
                  {health?.metrics_available ? 'Immutable & Valid' : 'Metrics Missing'}
                </span>
              </div>
              <span className="text-[10px] text-slate-400 block mt-2">results/metrics/*.csv</span>
            </Card>
          </div>

          {/* Architecture & Model Specifications */}
          {modelInfo && (
            <Card>
              <CardHeader>
                <CardTitle>Trained Neural Model Specifications</CardTitle>
                <Badge variant="brand" size="sm">{modelInfo.model} Policy</Badge>
              </CardHeader>
              <CardContent className="space-y-3 text-xs">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-slate-50 p-3 rounded-xl border border-slate-200/70 space-y-1">
                    <span className="text-slate-500 font-medium block">Actor Network Architecture</span>
                    <p className="font-mono text-[11px] text-slate-800 leading-relaxed">
                      {modelInfo.actor_architecture}
                    </p>
                  </div>
                  <div className="bg-slate-50 p-3 rounded-xl border border-slate-200/70 space-y-1">
                    <span className="text-slate-500 font-medium block">Critic Q-Network Architecture</span>
                    <p className="font-mono text-[11px] text-slate-800 leading-relaxed">
                      {modelInfo.critic_architecture}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
                  <div className="border border-slate-100 p-2.5 rounded-lg">
                    <span className="text-slate-400 text-[10px] block">Framework</span>
                    <span className="font-semibold text-slate-800">{modelInfo.framework} (PyTorch)</span>
                  </div>
                  <div className="border border-slate-100 p-2.5 rounded-lg">
                    <span className="text-slate-400 text-[10px] block">State Dimension</span>
                    <span className="font-semibold text-slate-800">{modelInfo.state_dimension} Inputs</span>
                  </div>
                  <div className="border border-slate-100 p-2.5 rounded-lg">
                    <span className="text-slate-400 text-[10px] block">Action Dimension</span>
                    <span className="font-semibold text-slate-800">{modelInfo.action_dimension} Logits</span>
                  </div>
                  <div className="border border-slate-100 p-2.5 rounded-lg">
                    <span className="text-slate-400 text-[10px] block">Asset Universe</span>
                    <span className="font-semibold text-slate-800">{modelInfo.assets.join(', ')}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Endpoint Catalogue */}
          <Card>
            <CardHeader>
              <div>
                <CardTitle>FastAPI Endpoint Catalogue</CardTitle>
                <p className="text-xs text-slate-500 mt-0.5">
                  Available REST routes consumed by this frontend
                </p>
              </div>
              <div className="flex items-center gap-2">
                <a
                  href="http://127.0.0.1:8000/docs"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs text-indigo-600 hover:text-indigo-700 font-medium flex items-center gap-1"
                >
                  <span>Interactive OpenAPI Docs</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-500 font-medium uppercase tracking-wider text-[11px]">
                      <th className="pb-2.5">Method</th>
                      <th className="pb-2.5">Endpoint Path</th>
                      <th className="pb-2.5">Description</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-mono">
                    {ENDPOINTS.map((ep) => (
                      <tr key={ep.path} className="text-slate-700 hover:bg-slate-50/50">
                        <td className="py-2.5">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                              ep.method === 'GET'
                                ? 'bg-indigo-50 text-indigo-700 border border-indigo-200/60'
                                : 'bg-emerald-50 text-emerald-700 border border-emerald-200/60'
                            }`}
                          >
                            {ep.method}
                          </span>
                        </td>
                        <td className="py-2.5 font-bold text-slate-800">{ep.path}</td>
                        <td className="py-2.5 font-sans text-slate-600">{ep.desc}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
};
