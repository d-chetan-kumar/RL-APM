import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import {
  Network,
  Database,
  Sliders,
  Brain,
  Repeat,
  DollarSign,
  ArrowRight,
  Zap,
} from 'lucide-react';

const PIPELINE_STEPS = [
  {
    step: '01',
    title: 'Market Data Ingestion',
    desc: 'Historical daily OHLCV equity prices collected via yfinance for AAPL, MSFT, GOOGL, AMZN, and NVDA.',
    icon: Database,
  },
  {
    step: '02',
    title: 'Feature Engineering',
    desc: 'Nine indicators per asset computed chronologically: SMA-20, SMA-50, RSI-14, MACD, Signal, Hist, and Bollinger Bands.',
    icon: Sliders,
  },
  {
    step: '03',
    title: '50-Dim State Construction',
    desc: '45 normalized market features (5 assets x 9 indicators) concatenated with 5 current portfolio weights without look-ahead bias.',
    icon: Zap,
  },
  {
    step: '04',
    title: 'DDPG Actor Policy',
    desc: 'Multi-layer perceptron (50 -> 256 -> 256 -> 5) with LayerNorm and ReLU generating unconstrained action logits.',
    icon: Brain,
  },
  {
    step: '05',
    title: 'Softmax Action Normalization',
    desc: 'Numerically stable exponential Softmax converting action logits into a non-negative probability simplex summing to 1.0.',
    icon: Repeat,
  },
  {
    step: '06',
    title: 'Portfolio MDP Environment',
    desc: 'Executes daily asset reallocation, charges 0.1% proportional transaction costs, and computes net portfolio return.',
    icon: DollarSign,
  },
  {
    step: '07',
    title: 'Critic Evaluation & Polyak Sync',
    desc: 'Critic evaluates Q(s, a) to calculate Bellman gradients. Target networks update via soft Polyak averaging (tau = 0.005).',
    icon: Network,
  },
];

const NINE_FEATURES = [
  { name: 'Daily Return', formula: '(Close_t - Close_{t-1}) / Close_{t-1}', desc: '1-day asset percentage price change' },
  { name: 'SMA-20 Ratio', formula: '(Close / SMA_{20}) - 1.0', desc: 'Short-term trend divergence ratio' },
  { name: 'SMA-50 Ratio', formula: '(Close / SMA_{50}) - 1.0', desc: 'Medium-term baseline trend divergence' },
  { name: 'RSI-14 Norm', formula: 'RSI_{14} / 100.0', desc: 'Normalized momentum oscillator bounded [0, 1]' },
  { name: 'MACD / Close', formula: 'MACD / Close', desc: 'Normalized moving average convergence divergence' },
  { name: 'Signal / Close', formula: 'MACD_Signal / Close', desc: 'Normalized 9-day exponential moving average of MACD' },
  { name: 'Hist / Close', formula: 'MACD_Hist / Close', desc: 'Normalized momentum acceleration spread' },
  { name: 'BB Position', formula: '(Close - BB_Low) / (BB_High - BB_Low)', desc: 'Relative price position inside Bollinger volatility band' },
  { name: 'BB Width Norm', formula: 'BB_Width / 100.0', desc: 'Normalized relative volatility band width' },
];

export const Methodology: React.FC = () => {
  return (
    <div className="space-y-6 pb-12">
      {/* Overview Header */}
      <div className="bg-white rounded-2xl border border-slate-200/80 p-6 shadow-card space-y-2">
        <div className="flex items-center gap-2">
          <Badge variant="brand" size="sm">System Architecture</Badge>
          <Badge variant="outline" size="sm">Mathematical Formulation</Badge>
        </div>
        <h2 className="text-xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
          <Network className="w-5 h-5 text-indigo-600" />
          <span>Research Methodology & Mathematical Formulation</span>
        </h2>
        <p className="text-xs text-slate-500 max-w-3xl leading-relaxed">
          How RL-APM models quantitative multi-asset equity management as a Continuous-Action
          Markov Decision Process (MDP) and optimizes policy weights via Deep Deterministic Policy Gradients.
        </p>
      </div>

      {/* 1. End-to-End Pipeline Flow */}
      <Card>
        <CardHeader>
          <CardTitle>Continuous-Action Policy Pipeline</CardTitle>
          <Badge variant="brand" size="sm">7 Execution Phases</Badge>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 relative">
            {PIPELINE_STEPS.map((step, idx) => {
              const Icon = step.icon;
              return (
                <div
                  key={step.step}
                  className="bg-slate-50 rounded-xl p-4 border border-slate-200/70 relative flex flex-col justify-between"
                >
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-[11px] font-mono font-bold text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded border border-indigo-200/60">
                        {step.step}
                      </span>
                      <Icon className="w-4 h-4 text-slate-500" />
                    </div>
                    <h4 className="text-xs font-semibold text-slate-900 mb-1.5">{step.title}</h4>
                    <p className="text-[11px] text-slate-500 leading-relaxed">{step.desc}</p>
                  </div>
                  {idx < PIPELINE_STEPS.length - 1 && (
                    <div className="hidden lg:block absolute -right-2.5 top-1/2 -translate-y-1/2 z-10">
                      <ArrowRight className="w-4 h-4 text-slate-300" />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* 2. Chronological Split Timeline */}
      <Card>
        <CardHeader>
          <CardTitle>Strict Chronological Dataset Partitioning</CardTitle>
          <Badge variant="outline" size="sm">Zero Look-Ahead Guarantee</Badge>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-xs text-slate-500 max-w-2xl">
            To prevent statistical data leakage and forward look-ahead bias, all feature scaling,
            environment resets, and evaluations are partitioned along strict chronological boundaries.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
            {/* Train Split */}
            <div className="bg-indigo-50/50 rounded-xl p-4 border border-indigo-200/80">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-indigo-900">Training Horizon</span>
                <span className="text-[11px] font-medium text-indigo-700 bg-indigo-100/70 px-2 py-0.5 rounded">
                  1,714 Days
                </span>
              </div>
              <span className="text-sm font-bold text-indigo-950 font-mono block">
                2015-03-16 → 2021-12-31
              </span>
              <p className="text-[11px] text-slate-600 mt-2">
                DDPG agent trains on historical price steps, updating replay buffer and Actor-Critic weights with Gaussian exploration noise ($\sigma=0.10$).
              </p>
            </div>

            {/* Validation Split */}
            <div className="bg-emerald-50/50 rounded-xl p-4 border border-emerald-200/80">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-emerald-900">Validation Horizon</span>
                <span className="text-[11px] font-medium text-emerald-700 bg-emerald-100/70 px-2 py-0.5 rounded">
                  501 Days
                </span>
              </div>
              <span className="text-sm font-bold text-emerald-950 font-mono block">
                2022-01-01 → 2023-12-31
              </span>
              <p className="text-[11px] text-slate-600 mt-2">
                Periodic zero-noise checkpoints evaluated without gradient backpropagation. Model checkpoint at Episode 9 selected based on peak validation return.
              </p>
            </div>

            {/* Test Split */}
            <div className="bg-slate-100/70 rounded-xl p-4 border border-slate-300">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-slate-900">Held-Out Test Horizon</span>
                <span className="text-[11px] font-medium text-slate-700 bg-slate-200 px-2 py-0.5 rounded">
                  252 Days
                </span>
              </div>
              <span className="text-sm font-bold text-slate-950 font-mono block">
                2024-01-01 → 2024-12-31
              </span>
              <p className="text-[11px] text-slate-600 mt-2">
                Completely unseen out-of-sample data. Final DDPG checkpoint tested against the Equal-Weight benchmark under identical 0.1% transaction frictions.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 3. State Vector Breakdown */}
      <Card>
        <CardHeader>
          <CardTitle>50-Dimensional State Representation Details</CardTitle>
          <Badge variant="brand" size="sm">5 Assets x 9 Features + 5 Weights</Badge>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-xs text-slate-500">
            Every observation step vector <span className="font-mono font-medium text-slate-700">s_t ∈ ℝ⁵⁰</span> consists of nine normalized technical indicators
            computed per asset across canonical order (<span className="font-semibold text-slate-700">AAPL, MSFT, GOOGL, AMZN, NVDA</span>)
            plus the current portfolio weights <span className="font-mono font-medium text-slate-700">w_t ∈ Δ⁵</span>.
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500 font-medium uppercase tracking-wider text-[11px]">
                  <th className="pb-2.5">Indicator Feature</th>
                  <th className="pb-2.5">Normalization / Scaling Formula</th>
                  <th className="pb-2.5">Financial Interpretation</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono">
                {NINE_FEATURES.map((feat) => (
                  <tr key={feat.name} className="text-slate-700">
                    <td className="py-2.5 font-sans font-semibold text-slate-900">{feat.name}</td>
                    <td className="py-2.5 text-indigo-700 bg-slate-50/50 px-2 rounded font-medium">
                      {feat.formula}
                    </td>
                    <td className="py-2.5 font-sans text-slate-600">{feat.desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* 4. Mathematical Formulations Grid (Reward & Softmax) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Reward Formulation</CardTitle>
            <Badge variant="slate" size="sm">Friction-Adjusted</Badge>
          </CardHeader>
          <CardContent className="space-y-3 text-xs text-slate-600">
            <p>
              The baseline reward signal $R_t$ received by the agent at step $t$ directly reflects
              the portfolio net return after accounting for reallocation transaction fees:
            </p>
            <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-200/80 font-mono text-xs text-slate-900 text-center font-bold">
              {"R_t = r_{net, t} - c_{trans} · ∑ |w_{i, t} - w_{i, t}^{drift}|"}
            </div>
            <p className="text-[11px] text-slate-500">
              {"Where c_{trans} = 0.001 (0.1%) penalizes excessive portfolio churn and turnover."}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Softmax Action Normalization</CardTitle>
            <Badge variant="slate" size="sm">Continuous Simplex</Badge>
          </CardHeader>
          <CardContent className="space-y-3 text-xs text-slate-600">
            <p>
              The Actor network produces unconstrained continuous action logits a_t in R^5.
              The environment transforms these into valid non-negative weights on the unit simplex:
            </p>
            <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-200/80 font-mono text-xs text-slate-900 text-center font-bold">
              {"w_{i, t} = exp(a_{i, t} - max(a_t)) / ∑ exp(a_{j, t} - max(a_t))"}
            </div>
            <p className="text-[11px] text-slate-500">
              Subtracting max(a_t) prevents numerical overflow while ensuring sum w_i = 1 and w_i &gt;= 0.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
