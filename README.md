# Reinforcement Learning for Portfolio Optimization in Equity Markets (RL-APM)

A research-oriented capstone system designed to optimize multi-asset equity portfolios using Deep Reinforcement Learning—specifically the Deep Deterministic Policy Gradient (DDPG) algorithm—with financial feature engineering and an interactive modern dashboard.

---

## Tech Stack Overview

- **Core & DL**: Python, PyTorch
- **Algorithm**: DDPG (Deep Deterministic Policy Gradient)
- **Financial Data & Indicators**: Yahoo Finance (`yfinance`), Pandas, NumPy, SMA, RSI, MACD, Bollinger Bands
- **Backend API**: FastAPI, Uvicorn
- **Frontend**: React + Vite, Tailwind CSS, Lucide React, Recharts, Framer Motion

---

## Project Directory Structure

```text
RL-APM/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI routers, endpoint controllers, and request/response schemas
│   │   ├── core/         # Core system configurations, settings, constants, and logging setup
│   │   ├── data/         # Ingestion pipelines (yfinance) and technical indicator feature engineering
│   │   ├── environment/  # Custom Markov Decision Process (MDP) portfolio trading environment
│   │   ├── models/       # PyTorch neural network architectures (Actor, Critic, Replay Buffer)
│   │   ├── services/     # Training orchestrator, inference service, and backtesting engine
│   │   └── utils/        # Financial performance metrics (Sharpe, Sortino, Drawdown) and helpers
│   ├── requirements.txt  # Python backend dependencies
│   └── venv/             # Dedicated Python virtual environment
├── frontend/             # Interactive UI dashboard (React + Vite, Tailwind CSS, Recharts)
├── data/
│   ├── raw/              # Historical OHLCV market data collected from Yahoo Finance
│   └── processed/        # Cleaned datasets enriched with technical indicators and normalized features
├── models/               # Checkpoints, trained Actor-Critic weights, and serialized policies
├── notebooks/            # Jupyter notebooks for exploratory data analysis (EDA) and experimental research
├── results/
│   ├── figures/          # Plots, cumulative return curves, and comparative benchmark visualizations
│   └── metrics/          # Quantitative performance logs (CSV/JSON) and risk-adjusted metrics
└── README.md             # Project documentation and architecture guide
```

---

## Backend Virtual Environment Setup

A dedicated Python virtual environment is provisioned at `backend/venv`.

### Activation

- **Windows (PowerShell):**
  ```powershell
  backend\venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt):**
  ```cmd
  backend\venv\Scripts\activate.bat
  ```
- **macOS / Linux:**
  ```bash
  source backend/venv/bin/activate
  ```

### Package Installation (When Ready)

Once ready to install backend packages:
```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

---

## Reinforcement Learning Environment (`PortfolioEnvironment`)

The portfolio management MDP is implemented in [`backend/app/environment/portfolio_environment.py`](file:///c:/Users/HP/OneDrive/Desktop/RL-APM/backend/app/environment/portfolio_environment.py) and is specifically tailored for continuous action RL algorithms (such as DDPG).

### 1. State Representation (Dimension: 50)
At each time step $t$, the state vector $S_t \in \mathbb{R}^{50}$ consists of two concatenated components:
1. **Market Technical Indicators (45 dimensions):** 5 assets $\times$ 9 technical indicators per asset. The features are transformed/scaled to provide numerically stable inputs to the RL agent:
   - `Daily_Return`: Percentage price return over day $t$
   - `SMA_Ratio_20`: $(Close / SMA_{20}) - 1.0$ (short-term trend deviation)
   - `SMA_Ratio_50`: $(Close / SMA_{50}) - 1.0$ (medium-term trend deviation)
   - `RSI_14`: $RSI / 100.0$ (momentum oscillator scaled to $[0, 1]$)
   - `MACD_Scaled`: $MACD / Close$
   - `MACD_Signal_Scaled`: $MACD_{Signal} / Close$
   - `MACD_Hist_Scaled`: $MACD_{Hist} / Close$
   - `BB_Position`: $(Close - BB_{Low}) / (BB_{High} - BB_{Low} + 10^{-8})$ (%B position within bands)
   - `BB_Width`: $BB_{Width} / 100.0$ (volatility band width)
   - *Asset Ordering:* `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `NVDA`.
2. **Current Portfolio Allocation (5 dimensions):** Current asset weights $[w_{AAPL}, w_{MSFT}, w_{GOOGL}, w_{AMZN}, w_{NVDA}]$ before rebalancing.

### 2. Action Representation & Normalization (Dimension: 5)
- Continuous action vector $a_t \in \mathbb{R}^5$ representing raw target logits.
- **Normalization:** Softmax projection ensures valid allocations:
  $$w_{t+1, i}^* = \frac{e^{a_{t, i} - \max(a_t)}}{\sum_{j=1}^5 e^{a_{t, j} - \max(a_t)}}$$
  Guarantees: $w_{t+1, i}^* \ge 0$ and $\sum_{i=1}^5 w_{t+1, i}^* = 1.0$.

### 3. Portfolio Initialization
- **Initial Capital:** $\$100,000.00$ (configurable).
- **Initial Allocation:** Equal weighting (20% per asset: $[0.2, 0.2, 0.2, 0.2, 0.2]$).

### 4. Transaction Cost Model
Rebalancing incurs proportional transaction fees based on turnover:
$$\text{Turnover}_t = \sum_{i=1}^5 |w_{t+1, i}^* - w_{t, i}|$$
$$\text{Cost}_t = \text{Turnover}_t \times \text{transaction\_cost} \times V_t$$
- Default rate: `transaction_cost = 0.001` ($0.1\%$ per turnover unit).

### 5. Time-Step Transition & Look-Ahead Bias Prevention
1. **Decision Time ($D_t$):** State $S_t$ is constructed strictly from observations up to and including date $D_t$.
2. **Rebalancing:** Target weights $w_{t+1}^*$ are committed and transaction costs $\text{Cost}_t$ deducted.
3. **Price Realization ($D_t \rightarrow D_{t+1}$):** Asset price movement occurs over the subsequent market interval:
   $$r_{t+1, i} = \frac{P_{t+1, i}}{P_{t, i}} - 1$$
4. **Portfolio Update:**
   $$V_{t+1} = (V_t - \text{Cost}_t) \times \left(1 + \sum_{i=1}^5 w_{t+1, i}^* \cdot r_{t+1, i}\right)$$
5. **Effective Drift:** Post-movement weights drift naturally before next decision step.
6. **Time Advance:** Step advances to $t+1$, where new state $S_{t+1}$ reflects information at $D_{t+1}$. No future prices or indicators can leak into decision $a_t$.

### 6. Reward Formulation
$$\text{Reward}_t = \text{Net\_Return}_t - \text{Risk\_Penalty} - \text{Cost\_Penalty}$$
- $\text{Net\_Return}_t = (V_{t+1} - V_t) / V_t$
- Configurable risk terms: `risk_penalty` (default: 0.0) and `transaction_cost_penalty` (default: 0.0).

### 7. Running Environment Validation Tests
```bash
python backend/app/environment/test_environment.py
```

