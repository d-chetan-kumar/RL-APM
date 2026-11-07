import type {
  BacktestMetricsResponse,
  EquityCurveResponse,
  HealthResponse,
  IndicatorResponse,
  MarketDataResponse,
  MarketSummaryResponse,
  ModelInfoResponse,
  PortfolioSummaryResponse,
  PredictionRequest,
  PredictionResponse,
  TrainingMetricsResponse,
  ValidationMetricsResponse,
} from '../types';

import {
  FALLBACK_BACKTEST,
  FALLBACK_DAILY_ALLOCATIONS,
  FALLBACK_EQUITY_CURVE,
  FALLBACK_HEALTH,
  FALLBACK_INDICATORS,
  FALLBACK_MARKET_SUMMARY,
  FALLBACK_MODEL_INFO,
  FALLBACK_PREDICTION,
  FALLBACK_TRAINING_METRICS,
  FALLBACK_VALIDATION_METRICS,
} from '../data/researchFallbackData';

const RAW_API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
export const API_BASE_URL = RAW_API_BASE.replace(/\/+$/, '');

let _isLiveApi = false;

export function isLiveApiConnected(): boolean {
  return _isLiveApi;
}

// Timeout fetch wrapper (default 3 seconds timeout to quickly fall back on dead/cold servers)
async function fetchWithTimeout(url: string, options?: RequestInit, timeoutMs = 3000): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(id);
    return response;
  } catch (err) {
    clearTimeout(id);
    throw err;
  }
}

async function fetchJson<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  const res = await fetchWithTimeout(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options?.headers || {}),
    },
  });

  if (!res.ok) {
    let errMsg = `HTTP Error ${res.status}: ${res.statusText}`;
    try {
      const errorJson = await res.json();
      errMsg = errorJson.detail || errorJson.message || errMsg;
    } catch {
      // Ignore json parse error
    }
    throw new Error(errMsg);
  }

  return (await res.json()) as T;
}

export const api = {
  getHealth: async (): Promise<HealthResponse> => {
    try {
      const res = await fetchJson<HealthResponse>('/api/health');
      _isLiveApi = res.status === 'ok';
      return res;
    } catch {
      _isLiveApi = false;
      return FALLBACK_HEALTH;
    }
  },

  getMarketData: async (
    ticker: string = 'AAPL',
    startDate?: string,
    endDate?: string,
    limit: number = 500,
    offset: number = 0
  ): Promise<MarketDataResponse> => {
    const params = new URLSearchParams({
      ticker,
      limit: limit.toString(),
      offset: offset.toString(),
    });
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);

    try {
      const res = await fetchJson<MarketDataResponse>(`/api/market-data?${params.toString()}`);
      _isLiveApi = true;
      return res;
    } catch {
      const fallback = FALLBACK_INDICATORS[ticker] || FALLBACK_INDICATORS['AAPL'];
      let rows = fallback.indicators;
      if (startDate) rows = rows.filter((r) => r.date >= startDate);
      if (endDate) rows = rows.filter((r) => r.date <= endDate);
      const paged = rows.slice(offset, offset + limit);
      return {
        ticker,
        count: paged.length,
        total_available: rows.length,
        data: paged.map((r) => ({
          date: r.date,
          open: r.close,
          high: r.bb_high,
          low: r.bb_low,
          close: r.close,
          adj_close: r.close,
          volume: 50000000,
          daily_return: r.daily_return,
          sma_20: r.sma_20,
          sma_50: r.sma_50,
          rsi_14: r.rsi_14,
          macd: r.macd,
          macd_signal: r.macd_signal,
          macd_hist: r.macd_hist,
          bb_high: r.bb_high,
          bb_mid: r.bb_mid,
          bb_low: r.bb_low,
          bb_width: r.bb_width,
        })),
      };
    }
  },

  getMarketSummary: async (): Promise<MarketSummaryResponse> => {
    try {
      const res = await fetchJson<MarketSummaryResponse>('/api/market-summary');
      _isLiveApi = true;
      return res;
    } catch {
      return FALLBACK_MARKET_SUMMARY;
    }
  },

  getIndicators: async (
    ticker: string,
    startDate?: string,
    endDate?: string,
    limit: number = 500,
    offset: number = 0
  ): Promise<IndicatorResponse> => {
    const params = new URLSearchParams({
      limit: limit.toString(),
      offset: offset.toString(),
    });
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);

    try {
      const res = await fetchJson<IndicatorResponse>(`/api/indicators/${ticker}?${params.toString()}`);
      _isLiveApi = true;
      return res;
    } catch {
      const fallback = FALLBACK_INDICATORS[ticker] || FALLBACK_INDICATORS['AAPL'];
      let rows = fallback.indicators;
      if (startDate) rows = rows.filter((r) => r.date >= startDate);
      if (endDate) rows = rows.filter((r) => r.date <= endDate);
      const paged = rows.slice(offset, offset + limit);
      return {
        ticker: fallback.ticker,
        count: paged.length,
        indicators: paged,
      };
    }
  },

  predictPortfolio: async (request: PredictionRequest = {}): Promise<PredictionResponse> => {
    try {
      const res = await fetchJson<PredictionResponse>('/api/predict', {
        method: 'POST',
        body: JSON.stringify(request),
      });
      _isLiveApi = true;
      return res;
    } catch {
      if (request.date && FALLBACK_DAILY_ALLOCATIONS[request.date]) {
        return {
          date: request.date,
          allocations: FALLBACK_DAILY_ALLOCATIONS[request.date],
          total_weight: 1.0,
          model: 'DDPG',
          state_dimension: 50,
          inference_type: 'historical out-of-sample validated policy (research cache)',
          state_vector: Array(50).fill(0),
        };
      }
      return FALLBACK_PREDICTION;
    }
  },

  getPortfolioSummary: async (): Promise<PortfolioSummaryResponse> => {
    try {
      const res = await fetchJson<PortfolioSummaryResponse>('/api/portfolio/summary');
      _isLiveApi = true;
      return res;
    } catch {
      return {
        model_name: 'DDPG Agent',
        evaluation_period: FALLBACK_BACKTEST.evaluation_period,
        historical_backtest: {
          period: FALLBACK_BACKTEST.evaluation_period,
          initial_capital: FALLBACK_BACKTEST.initial_capital,
          ddpg: {
            final_portfolio_value: FALLBACK_BACKTEST.ddpg.final_portfolio_value,
            cumulative_return: FALLBACK_BACKTEST.ddpg.cumulative_return,
            sharpe_ratio: FALLBACK_BACKTEST.ddpg.sharpe_ratio,
            annualized_volatility: FALLBACK_BACKTEST.ddpg.annualized_volatility,
            max_drawdown: FALLBACK_BACKTEST.ddpg.max_drawdown,
            total_transaction_costs: FALLBACK_BACKTEST.ddpg.total_transaction_costs,
          },
          equal_weight_benchmark: {
            final_portfolio_value: FALLBACK_BACKTEST.equal_weight.final_portfolio_value,
            cumulative_return: FALLBACK_BACKTEST.equal_weight.cumulative_return,
            sharpe_ratio: FALLBACK_BACKTEST.equal_weight.sharpe_ratio,
            annualized_volatility: FALLBACK_BACKTEST.equal_weight.annualized_volatility,
            max_drawdown: FALLBACK_BACKTEST.equal_weight.max_drawdown,
            total_transaction_costs: FALLBACK_BACKTEST.equal_weight.total_transaction_costs,
          },
          nature: 'HISTORICAL OUT-OF-SAMPLE TEST (2024). Standalone Research Cache Fallback.',
        },
        latest_inference: {
          date: FALLBACK_PREDICTION.date,
          allocations: FALLBACK_PREDICTION.allocations,
          total_weight: FALLBACK_PREDICTION.total_weight,
          nature: 'LATEST HISTORICAL-DATA INFERENCE (Standalone Research Cache Fallback)',
        },
      };
    }
  },

  getModelInfo: async (): Promise<ModelInfoResponse> => {
    try {
      const res = await fetchJson<ModelInfoResponse>('/api/model-info');
      _isLiveApi = true;
      return res;
    } catch {
      return FALLBACK_MODEL_INFO;
    }
  },

  getBacktest: async (): Promise<BacktestMetricsResponse> => {
    try {
      const res = await fetchJson<BacktestMetricsResponse>('/api/backtest');
      _isLiveApi = true;
      return res;
    } catch {
      return FALLBACK_BACKTEST;
    }
  },

  getEquityCurve: async (): Promise<EquityCurveResponse> => {
    try {
      const res = await fetchJson<EquityCurveResponse>('/api/backtest/equity-curve');
      _isLiveApi = true;
      return res;
    } catch {
      return FALLBACK_EQUITY_CURVE;
    }
  },

  getTrainingMetrics: async (): Promise<TrainingMetricsResponse> => {
    try {
      const res = await fetchJson<TrainingMetricsResponse>('/api/training-metrics');
      _isLiveApi = true;
      return res;
    } catch {
      return FALLBACK_TRAINING_METRICS;
    }
  },

  getValidationMetrics: async (): Promise<ValidationMetricsResponse> => {
    try {
      const res = await fetchJson<ValidationMetricsResponse>('/api/validation-metrics');
      _isLiveApi = true;
      return res;
    } catch {
      return FALLBACK_VALIDATION_METRICS;
    }
  },
};
