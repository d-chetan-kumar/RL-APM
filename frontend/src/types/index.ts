/**
 * TypeScript definitions matching backend FastAPI schemas for RL-APM.
 */

export interface HealthResponse {
  status: string;
  service: string;
  model_available: boolean;
  data_available: boolean;
  metrics_available: boolean;
}

export interface MarketDataRow {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  adj_close: number;
  volume: number;
  daily_return: number;
  sma_20: number;
  sma_50: number;
  rsi_14: number;
  macd: number;
  macd_signal: number;
  macd_hist: number;
  bb_high: number;
  bb_mid: number;
  bb_low: number;
  bb_width: number;
}

export interface MarketDataResponse {
  ticker: string;
  count: number;
  total_available: number;
  data: MarketDataRow[];
}

export interface AssetSummary {
  ticker: string;
  date: string;
  close: number;
  daily_return: number;
  sma_20: number;
  sma_50: number;
  sma20_ratio: number;
  sma50_ratio: number;
  rsi: number;
  macd: number;
  macd_signal: number;
  macd_histogram: number;
  bollinger_position: number;
  bollinger_width: number;
}

export interface MarketSummaryResponse {
  date: string;
  assets: AssetSummary[];
}

export interface IndicatorRow {
  date: string;
  close: number;
  daily_return: number;
  sma_20: number;
  sma_50: number;
  rsi_14: number;
  macd: number;
  macd_signal: number;
  macd_hist: number;
  bb_high: number;
  bb_mid: number;
  bb_low: number;
  bb_width: number;
}

export interface IndicatorResponse {
  ticker: string;
  count: number;
  indicators: IndicatorRow[];
}

export interface PredictionRequest {
  date?: string;
  current_weights?: Record<string, number>;
}

export interface PredictionResponse {
  date: string;
  allocations: Record<string, number>;
  total_weight: number;
  model: string;
  state_dimension: number;
  inference_type: string;
  state_vector?: number[];
}

export interface ModelMetricRecord {
  model: string;
  final_portfolio_value: number;
  cumulative_return: number;
  annualized_return: number;
  annualized_volatility: number;
  sharpe_ratio: number;
  max_drawdown: number;
  total_transaction_costs: number;
}

export interface BacktestMetricsResponse {
  evaluation_period: string;
  initial_capital: number;
  ddpg: ModelMetricRecord;
  equal_weight: ModelMetricRecord;
}

export interface EquityCurvePoint {
  date: string;
  ddpg_value: number;
  equal_weight_value: number;
}

export interface EquityCurveResponse {
  initial_capital: number;
  start_date: string;
  end_date: string;
  points_count: number;
  points: EquityCurvePoint[];
}

export interface TrainingEpisodeRecord {
  episode: number;
  total_reward: number;
  portfolio_value: number;
  cumulative_return: number;
  transaction_costs: number;
  mean_actor_loss: number;
  mean_critic_loss: number;
  duration_sec: number;
}

export interface TrainingMetricsResponse {
  total_episodes: number;
  episodes: TrainingEpisodeRecord[];
}

export interface ValidationCheckpointRecord {
  episode: number;
  val_final_portfolio_value: number;
  val_cumulative_return: number;
  val_annualized_return: number;
  val_annualized_volatility: number;
  val_sharpe_ratio: number;
  val_max_drawdown: number;
  val_transaction_costs: number;
}

export interface ValidationMetricsResponse {
  checkpoints: ValidationCheckpointRecord[];
}

export interface PortfolioSummaryResponse {
  model_name: string;
  evaluation_period: string;
  historical_backtest: {
    period: string;
    initial_capital: number;
    ddpg: {
      final_portfolio_value: number;
      cumulative_return: number;
      sharpe_ratio: number;
      annualized_volatility: number;
      max_drawdown: number;
      total_transaction_costs: number;
    };
    equal_weight_benchmark: {
      final_portfolio_value: number;
      cumulative_return: number;
      sharpe_ratio: number;
      annualized_volatility: number;
      max_drawdown: number;
      total_transaction_costs: number;
    };
    nature: string;
  };
  latest_inference: {
    date: string;
    allocations: Record<string, number>;
    total_weight: number;
    nature: string;
  };
}

export interface ModelInfoResponse {
  model: string;
  framework: string;
  state_dimension: number;
  action_dimension: number;
  assets: string[];
  actor_architecture: string;
  critic_architecture: string;
  gamma: number;
  tau: number;
  actor_learning_rate: number;
  critic_learning_rate: number;
  replay_buffer_size: number;
  batch_size: number;
  initial_capital: number;
  transaction_cost: number;
  train_period: string;
  validation_period: string;
  test_period: string;
  checkpoint_name: string;
}
