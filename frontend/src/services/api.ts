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

const RAW_API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
export const API_BASE_URL = RAW_API_BASE.replace(/\/+$/, '');

async function fetchJson<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, {
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
  } catch (err: unknown) {
    if (err instanceof Error) {
      throw err;
    }
    throw new Error('Unknown network error occurred');
  }
}

export const api = {
  getHealth: () => fetchJson<HealthResponse>('/api/health'),

  getMarketData: (
    ticker: string = 'AAPL',
    startDate?: string,
    endDate?: string,
    limit: number = 500,
    offset: number = 0
  ) => {
    const params = new URLSearchParams({
      ticker,
      limit: limit.toString(),
      offset: offset.toString(),
    });
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    return fetchJson<MarketDataResponse>(`/api/market-data?${params.toString()}`);
  },

  getMarketSummary: () => fetchJson<MarketSummaryResponse>('/api/market-summary'),

  getIndicators: (
    ticker: string,
    startDate?: string,
    endDate?: string,
    limit: number = 500,
    offset: number = 0
  ) => {
    const params = new URLSearchParams({
      limit: limit.toString(),
      offset: offset.toString(),
    });
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    return fetchJson<IndicatorResponse>(`/api/indicators/${ticker}?${params.toString()}`);
  },

  predictPortfolio: (request: PredictionRequest = {}) =>
    fetchJson<PredictionResponse>('/api/predict', {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  getPortfolioSummary: () => fetchJson<PortfolioSummaryResponse>('/api/portfolio/summary'),

  getModelInfo: () => fetchJson<ModelInfoResponse>('/api/model-info'),

  getBacktest: () => fetchJson<BacktestMetricsResponse>('/api/backtest'),

  getEquityCurve: () => fetchJson<EquityCurveResponse>('/api/backtest/equity-curve'),

  getTrainingMetrics: () => fetchJson<TrainingMetricsResponse>('/api/training-metrics'),

  getValidationMetrics: () => fetchJson<ValidationMetricsResponse>('/api/validation-metrics'),
};
