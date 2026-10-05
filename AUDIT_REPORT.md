# IMPLEMENTATION-TO-PAPER AUDIT REPORT

**Paper Title:** *DDPG-Based Adaptive Portfolio Allocation for a Five-Asset Equity Portfolio*  
**Internal Project:** RL-APM (Reinforcement Learning-Based Adaptive Portfolio Management)  
**Document Audited:** `RL-APM PAPER.pdf` (8 pages, finalized preprint layout)  
**Primary Code Evidence:** `backend/app/`, `models/`, `data/`, and `results/metrics/`  
**Audit Date:** October 2, 2026  

---

## 1. Complete Implementation Architecture & Component Trace

Below is the verified end-to-end execution path traced through the codebase:

```
[Yahoo Finance] 
      │
      ▼
download_market_data.py ──► data/raw/{TICKER}.csv (Open, High, Low, Close, Adj Close, Volume)
      │
      ▼
feature_engineering.py ───► 9 raw features per asset (Daily_Return via Adj Close, SMA20, SMA50, 
      │                     RSI14, MACD, MACD_Signal, MACD_Hist, BB_High, BB_Low, BB_Width)
      │                     Warmup NaNs dropped ──► data/processed/{TICKER}_processed.csv
      ▼
portfolio_environment.py ─► Standardizes 45 indicators + appends 5 portfolio weights = State s_t ∈ R^50
      │
      ├───────────────────────┬───────────────────────┐
      ▼                       ▼                       ▼
Actor (networks.py)    Critic (networks.py)   PortfolioEnvironment (Step logic)
Linear(50,256)+LN+ReLU Linear(50,256)+LN+ReLU Action normalization via Softmax
Linear(256,256)+LN+ReLU Concat([s, a]) (261)  Turnover & Transaction Cost (c=0.001)
Linear(256, 5)          Linear(261,256)+LN    Gross return & capital compounding
Raw logits a_t ∈ R^5   Linear(256, 1) -> Q   Net percentage return reward r_t
      │                       │                       │
      └───────────────────────┼───────────────────────┘
                              ▼
ddpg_agent.py ─────────────► ReplayBuffer (100k capacity, batch size 64)
                             Gaussian noise (sigma=0.10, added to raw logits in training)
                             Bellman Target Q: y_t = r_t + γ(1 - d_t)Q_target(s', μ_target(s'))
                             Critic loss: MSE(Q(s, a), y_t), grad clipped to 1.0
                             Actor loss: -mean(Q(s, μ(s))), grad clipped to 1.0
                             Polyak soft updates: θ', φ' updated with τ=0.005
      │
      ▼
train_ddpg.py ─────────────► Chronological train (2015–2021, 1714 days) × 15 episodes
                             Periodic deterministic validation (2022–2023, 501 days) every 3 eps
                             Model selection: Episode 9 checkpoint saved as ddpg_actor_best.pth
      │
      ▼
evaluate_ddpg.py ──────────► Evaluates ddpg_actor_best.pth on held-out 2024 test split (252 days)
                             Compares with passive Equal-Weight benchmark
                             Outputs: test_metrics.csv, test_equity_curve.csv, figures
```

### Component Implementation Map

| Component | Source File | Class / Function | Actual Implementation / Operational Behavior |
| :--- | :--- | :--- | :--- |
| **Market Data Ingestion** | `backend/app/data/download_market_data.py` | `download_ticker_data()` | Fetches daily OHLCV from Yahoo Finance via `yfinance` for AAPL, MSFT, GOOGL, AMZN, NVDA (2015-01-01 to 2024-12-31); saves to `data/raw/`. |
| **Feature Extraction** | `backend/app/data/feature_engineering.py` | `compute_technical_indicators()` | Computes Daily Return (`Adj Close.pct_change()`), SMA20, SMA50, RSI14, MACD (12,26,9), and Bollinger Bands (20,2). Drops initial 49 warmup rows (starts 2015-03-16). |
| **State Construction** | `backend/app/environment/portfolio_environment.py` | `PortfolioEnvironment._get_state()` | Scales 9 indicators per asset (45 total) and appends the 5 current portfolio weights $[w_1, \dots, w_5]$ to form a 50-dimensional observation vector $s_t \in \mathbb{R}^{50}$. |
| **Portfolio Environment** | `backend/app/environment/portfolio_environment.py` | `PortfolioEnvironment.step()` | Applies rebalancing, computes turnover $T_t = \sum \|w_i - w_{i-1}\|$, assesses cost $C_t = c T_t V_t$, computes post-cost return, drifts weights, and returns $(s_{t+1}, r_t, d_t, \text{info})$. |
| **Action Normalization** | `backend/app/environment/portfolio_environment.py` | `PortfolioEnvironment.normalize_action()` | Applies numerically stable Softmax: $w_i = \frac{\exp(a_i - \max a)}{\sum \exp(a_j - \max a)}$. Performed inside the environment, NOT in the Actor network. |
| **Actor Network** | `backend/app/models/networks.py` | `Actor` | PyTorch MLP: Linear(50, 256) $\to$ LayerNorm $\to$ ReLU $\to$ Linear(256, 256) $\to$ LayerNorm $\to$ ReLU $\to$ Linear(256, 5). Outputs unbounded linear logits. |
| **Critic Network** | `backend/app/models/networks.py` | `Critic` | PyTorch two-stream architecture: State $\to$ Linear(50, 256) $\to$ LayerNorm $\to$ ReLU $\to$ Concat with Action (261) $\to$ Linear(261, 256) $\to$ LayerNorm $\to$ ReLU $\to$ Linear(256, 1). |
| **Exploration Noise** | `backend/app/models/noise.py` | `GaussianNoise` | Generates $\mathcal{N}(0, 0.10^2)$ noise. Added directly to raw action logits in `DDPGAgent.select_action()` during training only. Disabled during validation and testing. |
| **Experience Replay** | `backend/app/models/replay_buffer.py` | `ReplayBuffer` | Pre-allocated circular NumPy arrays of capacity 100,000 storing $(s, a, r, s', d)$. Uniform random mini-batch sampling ($B=64$) converted to PyTorch tensors. |
| **Optimization Loop** | `backend/app/models/ddpg_agent.py` | `DDPGAgent.update()` | Evaluates Bellman target with target networks; updates Critic with MSE loss; updates Actor with `-Q(s, μ(s)).mean()`; updates targets via Polyak averaging ($\tau = 0.005$). |
| **Training Pipeline** | `backend/app/training/train_ddpg.py` | `train_ddpg()` | Coordinates 15 training episodes on 2015–2021; executes deterministic validation evaluations every 3 episodes on 2022–2023; saves best checkpoint (Episode 9). |
| **Out-of-Sample Evaluation**| `backend/app/training/evaluate_ddpg.py` | `evaluate_test_period()` | Evaluates Episode 9 checkpoint on strictly held-out 2024 test split (252 days) against Equal-Weight; computes Sharpe, drawdown, returns, volatility; writes CSV results. |

---

## 2. Equation-by-Equation Audit Table (Equations 1–30)

| Eq. | Paper Formulation | Purpose in Paper | Actual Code Implementation | Source File & Function/Class | Status | Detailed Explanation & Audit Findings |
| :---: | :--- | :--- | :--- | :--- | :---: | :--- |
| **(1)** | $s_t = [x_{1,t}, \dots, x_{5,t}, w_t]$ | 50-D State vector definition | `np.concatenate([market_features, current_weights])` | `portfolio_environment.py`<br>`_get_state()` | **A** | **EXACTLY IMPLEMENTED.** 45 market features concatenated with 5 weights. Note: Exactly duplicated by Eq. (14). |
| **(2)** | $5 \times 9 + 5 = 50$ | Dimensionality arithmetic | `state_dim = 5 * 9 + 5` | `portfolio_environment.py`<br>`__init__()` | **A** | **EXACTLY IMPLEMENTED.** Dimension identity matches. Note: Redundant arithmetic identity. |
| **(3)** | $T_t = \sum_{i=1}^5 \|w_{i,t} - w_{i,t-1}\|$ | Portfolio turnover | `np.sum(np.abs(new_weights - prev_weights))` | `portfolio_environment.py`<br>`step()` | **A** | **EXACTLY IMPLEMENTED.** Exact $L_1$ turnover norm. Note: Exactly duplicated by Eq. (18). |
| **(4)** | $C_t = c T_t$ | Transaction cost rate | `turnover * self.transaction_cost * self.portfolio_value` | `portfolio_environment.py`<br>`step()` | **A** | **EXACTLY IMPLEMENTED.** Proportional transaction cost with $c=0.001$. Note: Exactly duplicated by Eq. (19). |
| **(5)** | $s_t \to a_t \to r_t \to s_{t+1}$ | MDP transition schematic | `env.step(action) -> next_s, r, done, info` | `portfolio_environment.py`<br>`step()` | **B** | **IMPLEMENTED BUT NOTATION DIFFERS.** A conceptual transition flowchart arrow, not an algebraic equation. |
| **(6)** | $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$ | Daily return feature | `df["Daily_Return"] = df["Adj Close"].pct_change()` | `feature_engineering.py`<br>`compute_technical_indicators()` | **B** | **IMPLEMENTED BUT NOTATION DIFFERS.** Paper defines $P_t$ as closing price, but code uses **Adjusted Close** (`Adj Close`) to adjust for splits and dividends. |
| **(7)** | $\text{SMARatio}_{n,t} = \frac{P_t}{\text{SMA}_{n,t}} - 1$ | 20 & 50-day SMA deviation | `(close / (sma_n + 1e-8)) - 1.0` | `portfolio_environment.py`<br>`_load_and_align_market_data()` | **A** | **EXACTLY IMPLEMENTED.** Matches code formula; code adds $+10^{-8}$ in denominator for numerical stability. |
| **(8)** | $\text{RSI}^*_t = \frac{\text{RSI}_{14,t}}{100}$ | Normalized 14-day RSI | `rsi_14 / 100.0` | `portfolio_environment.py`<br>`_load_and_align_market_data()` | **A** | **EXACTLY IMPLEMENTED.** 14-day Wilder RSI scaled to $[0, 1]$. |
| **(9)** | $\text{MACD}^* = \frac{\text{MACD}}{P}, \text{Sig}^* = \frac{\text{Sig}}{P}$ | Normalized MACD & Signal | `macd / (close + 1e-8)`, `macd_sig / (close + 1e-8)` | `portfolio_environment.py`<br>`_load_and_align_market_data()` | **A** | **EXACTLY IMPLEMENTED.** MACD (12, 26, 9) normalized by closing price. |
| **(10)** | $\text{Hist}^*_t = \frac{\text{MACD}_t - \text{Signal}_t}{P_t}$ | Normalized MACD Histogram | `macd_hist / (close + 1e-8)` | `portfolio_environment.py`<br>`_load_and_align_market_data()` | **A** | **EXACTLY IMPLEMENTED.** MACD difference normalized by closing price. |
| **(11)** | $\text{BBPos}_t = \frac{P_t - \text{BB}^{\text{low}}_t}{\text{BB}^{\text{high}}_t - \text{BB}^{\text{low}}_t + 10^{-8}}$ | Bollinger Band %B position | `(close - bb_low) / (bb_span + 1e-8)` | `portfolio_environment.py`<br>`_load_and_align_market_data()` | **A** | **EXACTLY IMPLEMENTED.** Identical down to the $10^{-8}$ machine epsilon. |
| **(12)** | $\text{BBWidth}_t = \frac{\text{BB}^{\text{high}}_t - \text{BB}^{\text{low}}_t}{100}$ | Normalized Bollinger Width | `((BB_High - BB_Low) / BB_Mid) * 100` then `/ 100.0` $\implies \frac{\text{BB}^{\text{high}} - \text{BB}^{\text{low}}}{\text{BB}^{\text{mid}}}$ | `feature_engineering.py`<br>`portfolio_environment.py` | **C** | **PARTIALLY IMPLEMENTED (FORMULA INCOMPLETE IN PAPER).** Code calculates standard Bollinger bandwidth normalized by the middle band $\text{BB}^{\text{mid}}$: $(\text{Span}/\text{Mid})\times 100$ and divides by 100. Paper Eq. (12) omitted $\text{BB}^{\text{mid}}_t$ from the denominator. |
| **(13)** | $x_t = [x_{1,t}, \dots, x_{5,t}]$ | 45-D Market feature block | `row_features` (45 indicators) | `portfolio_environment.py`<br>`_load_and_align_market_data()` | **A** | **EXACTLY IMPLEMENTED.** 9 features $\times$ 5 assets in canonical order (AAPL, MSFT, GOOGL, AMZN, NVDA). |
| **(14)** | $s_t = [x_t, w_t] \in \mathbb{R}^{50}$ | Complete 50-D State vector | `np.concatenate([market_features, current_weights])` | `portfolio_environment.py`<br>`_get_state()` | **A** | **EXACTLY IMPLEMENTED.** Appends 5 portfolio weights. Note: Duplicates Eq. (1) and (2). |
| **(15)** | $a_t = \pi_\theta(s_t)$ | Continuous action policy | `raw_action = self.actor(state_tensor)` | `ddpg_agent.py`<br>`select_action()` | **A** | **EXACTLY IMPLEMENTED.** Deterministic Actor mapping. Note: Duplicated verbatim by Eq. (22). |
| **(16)** | $w_{i,t} \ge 0, \sum_{i=1}^5 w_{i,t} = 1$ | Portfolio simplex constraint | `normalize_action()`: Numerically stable Softmax | `portfolio_environment.py`<br>`normalize_action()` | **A** | **EXACTLY IMPLEMENTED.** Code guarantees non-negativity and exact unity sum via Softmax in the environment. |
| **(17)** | $r_t = R^p_t - C_t$ | Reward formulation | `net_return = (V_{t+1} - V_t) / V_t`<br>`reward = net_return` | `portfolio_environment.py`<br>`step()` | **B** | **IMPLEMENTED BUT NOTATION DIFFERS.** Paper writes additive difference $R^p_t - C_t$. Code implements exact percentage compounded value delta: $\frac{V_{t+1}-V_t}{V_t} = (1 - C_t)(1 + R^p_t) - 1 = R^p_t - C_t - C_t R^p_t$. |
| **(18)** | $T_t = \sum_{i=1}^5 \|w_{i,t} - w_{i,t-1}\|$ | Turnover formulation | `np.sum(np.abs(new_weights - prev_weights))` | `portfolio_environment.py`<br>`step()` | **D** | **REDUNDANT/DUPLICATED.** Verbatim duplicate of Eq. (3). |
| **(19)** | $C_t = c T_t$ | Transaction cost rate | `turnover * self.transaction_cost` | `portfolio_environment.py`<br>`step()` | **D** | **REDUNDANT/DUPLICATED.** Verbatim duplicate of Eq. (4). |
| **(20)** | $(s_t, a_t) \to (R^p_t, C_t, r_t, s_{t+1})$ | Environment step mapping | `env.step()` step execution | `portfolio_environment.py`<br>`step()` | **B** | **IMPLEMENTED BUT NOTATION DIFFERS.** Conceptual tuple mapping schematic. |
| **(21)** | $(s_t, a_t, r_t, s_{t+1}, d_t)$ | Transition tuple | `replay_buffer.add(state, action, reward, next_state, done)` | `replay_buffer.py`<br>`add()` | **A** | **EXACTLY IMPLEMENTED.** Tuple structure matches replay buffer fields. Note: Duplicated by Eq. (24). |
| **(22)** | $a_t = \pi_\theta(s_t)$ | Actor forward mapping | `logits = self.fc3(x)` | `networks.py`<br>`Actor.forward()` | **D** | **REDUNDANT/DUPLICATED.** Verbatim duplicate of Eq. (15). |
| **(23)** | $Q_\phi(s_t, a_t)$ | Critic Q-function | `Critic.forward(state, action)` | `networks.py`<br>`Critic.forward()` | **A** | **EXACTLY IMPLEMENTED.** Evaluates continuous state-action pair via two-stream MLP. |
| **(24)** | $(s_t, a_t, r_t, s_{t+1}, d_t)$ | Transition tuple in replay | `replay_buffer.add(...)` | `replay_buffer.py`<br>`add()` | **D** | **REDUNDANT/DUPLICATED.** Verbatim duplicate of Eq. (21). |
| **(25)** | $y_t = r_t + \gamma (1 - d_t) Q_{\phi'}(s_{t+1}, \pi_{\theta'}(s_{t+1}))$ | Target Q Bellman equation | `target_q = rewards + (gamma * (1.0 - dones) * target_q_next)` | `ddpg_agent.py`<br>`update()` | **A** | **EXACTLY IMPLEMENTED.** Standard DDPG target calculation using target Actor and target Critic under `torch.no_grad()`. |
| **(26)** | $L(\phi) = \frac{1}{B} \sum_{i=1}^B (Q_\phi(s_i, a_i) - y_i)^2$ | Critic MSE Loss | `critic_loss = nn.functional.mse_loss(current_q, target_q)` | `ddpg_agent.py`<br>`update()` | **A** | **EXACTLY IMPLEMENTED.** PyTorch mean squared error over mini-batch $B=64$. |
| **(27)** | $J(\theta) = \frac{1}{B} \sum_{i=1}^B Q_\phi(s_i, \pi_\theta(s_i))$ | Actor policy objective | `actor_loss = -self.critic(states, predicted_actions).mean()` | `ddpg_agent.py`<br>`update()` | **A** | **EXACTLY IMPLEMENTED.** Minimizing `-Q.mean()` in PyTorch is mathematically identical to maximizing $J(\theta)$. |
| **(28)** | $\theta' \leftarrow \tau \theta + (1 - \tau)\theta'$ | Target Actor Polyak update | `target_param.data.copy_(tau * online + (1 - tau) * target)` | `ddpg_agent.py`<br>`_soft_update()` | **A** | **EXACTLY IMPLEMENTED.** Smooth exponential tracking with $\tau = 0.005$. |
| **(29)** | $\phi' \leftarrow \tau \phi + (1 - \tau)\phi'$ | Target Critic Polyak update | `target_param.data.copy_(tau * online + (1 - tau) * target)` | `ddpg_agent.py`<br>`_soft_update()` | **A** | **EXACTLY IMPLEMENTED.** Smooth exponential tracking with $\tau = 0.005$. |
| **(30)** | $s_t \to \pi_\theta(s_t) \to a_t \to \dots \to \text{Actor Update}$ | Training flow schematic | `train_ddpg.py` episode step & update loop | `train_ddpg.py`<br>`train_ddpg()` | **B** | **IMPLEMENTED BUT NOTATION DIFFERS.** A linear conceptual sequence string summarizing the loop. |

---

## 3. Detailed Audit Findings by Topic

### 3.1 State Construction (Equations 1, 2, 13, 14)
* **Ordering and Composition:** Verified in `portfolio_environment.py` (lines 181–214). The state tensor strictly follows a canonical asset order: **AAPL $\to$ MSFT $\to$ GOOGL $\to$ AMZN $\to$ NVDA**. For each asset, exactly 9 features appear in this sequence: Daily Return, SMA20 Ratio, SMA50 Ratio, RSI14, Scaled MACD, Scaled MACD Signal, Scaled MACD Hist, Bollinger Position, Bollinger Width.
* Following the 45 market features, the 5 current portfolio weights $[w_{\text{AAPL}}, w_{\text{MSFT}}, w_{\text{GOOGL}}, w_{\text{AMZN}}, w_{\text{NVDA}}]$ are appended.
* **Redundancy:** Equations (1) and (2) in Section III-B present the exact same mathematical content as Equations (13) and (14) in Section IV-C.

### 3.2 Transaction Cost & Portfolio Value (Equations 3, 4, 18, 19)
* **Turnover Calculation:** Traced to line 323 of `portfolio_environment.py`. Uses exact $L_1$ turnover: $T_t = \sum_{i=1}^5 |w_{i,t}^{\text{target}} - w_{i,t-1}^{\text{drifted}}|$.
* **Fee Deduction Sequence:**
  1. Transaction cost dollar amount is deducted from portfolio capital **before** market exposure:  
     $$\text{Capital After Fees} = V_t - C_t V_t = V_t (1 - c T_t)$$
  2. Asset returns $r_{t+1}$ over the period are realized on this net invested capital:  
     $$V_{t+1} = \text{Capital After Fees} \times (1 + \text{Gross Return}) = V_t (1 - c T_t)(1 + \sum_{i=1}^5 w_{i,t} r_{i,t+1})$$
  3. Portfolio value is directly reduced by transaction costs at each trading step.
* **Redundancy:** Equations (3) and (4) in Section III-D are identical duplicates of Equations (18) and (19) in Section IV-D.

### 3.3 Feature Engineering Analysis (Equations 6–12)
* **Equation (6) (Daily Return):** Paper writes closing price $P_t$, but code uses Adjusted Close (`df["Adj Close"].pct_change()`). This is superior financial practice (adjusts for splits and dividends), but the paper notation should explicitly specify adjusted close price $P_t^{\text{adj}}$.
* **Equation (11) (Bollinger Band Position):** Exactly matches code down to the $10^{-8}$ machine epsilon: $\frac{P_t - \text{BB}^{\text{low}}}{\text{BB}^{\text{high}} - \text{BB}^{\text{low}} + 10^{-8}}$.
* **Equation (12) (Bollinger Band Width) — Mismatch:**  
  * The paper states: $\text{BBWidth}_t = \frac{\text{BB}^{\text{high}}_t - \text{BB}^{\text{low}}_t}{100}$.  
  * The code implementation calculates standard technical bandwidth: $\text{BB\_Width} = \frac{\text{BB}^{\text{high}} - \text{BB}^{\text{low}}}{\text{BB}^{\text{mid}}} \times 100$ in `feature_engineering.py` (line 98) and then scales it in `portfolio_environment.py` (line 206) via `f_bbwidth = bb_width / 100.0`. The actual feature in the state vector is therefore $\frac{\text{BB}^{\text{high}} - \text{BB}^{\text{low}}}{\text{BB}^{\text{mid}}}$. The paper formula erroneously omitted $\text{BB}^{\text{mid}}$ from the denominator.

### 3.4 Actor & Critic Architecture (Equations 15, 16, 22, 23)
* **Location of Softmax Normalization:**  
  The PyTorch Actor network head (`self.fc3` in `networks.py`, line 53) outputs **unconstrained linear continuous logits** $a_t \in \mathbb{R}^5$. Softmax normalization is executed in the environment step (`PortfolioEnvironment.normalize_action()`). The paper's text should clarify that the Actor outputs continuous logits and the simplex projection is enforced in the execution environment.
* **Omission of LayerNorm in Paper Text:**  
  Both Actor and Critic utilize `nn.LayerNorm(256)` after each linear layer prior to ReLU activations (`networks.py`, lines 50, 52, 97, 101). The paper text in Sections V-A and V-B describes the networks as basic fully connected layers and omits mentioning LayerNorm.
* **Critic Input Merging:**  
  Matches code. State (50) is processed to 256 dimensions, concatenated with action (5) to form a 261-dimensional vector, and passed through Linear(261, 256) $\to$ LayerNorm $\to$ ReLU $\to$ Linear(256, 1).

### 3.5 Experience Replay, Exploration, & Hyperparameters
* **Noise Application:** Gaussian noise $\mathcal{N}(0, 0.10^2)$ is added directly to the Actor's **raw action logits**, not to normalized weights. Exploration occurs only when `add_noise=True` during training. Validation and test evaluation loops set `add_noise=False` (strictly deterministic).
* **Table I Parameter Verification:** Every hyperparameter in Table I matches the implementation exactly:
  * State dim: 50 | Action dim: 5 | Initial capital: \$100,000 | Initial weights: 0.20 each | Training episodes: 15 | Buffer capacity: 100,000 | Batch size: 64 | Actor LR: $1 \times 10^{-4}$ | Critic LR: $1 \times 10^{-3}$ | Discount factor $\gamma$: 0.99 | Soft update $\tau$: 0.005 | Exploration $\sigma$: 0.10 | Transaction fee rate $c$: 0.001.

### 3.6 Data Partitioning & Chronological Integrity
* **Partition Counts:** Verified against `data/processed/*.csv`:
  * Training: 2015-03-16 to 2021-12-31 $\implies$ **1,714 trading days**
  * Validation: 2022-01-01 to 2023-12-31 $\implies$ **501 trading days**
  * Held-Out Test: 2024-01-01 to 2024-12-31 $\implies$ **252 trading days**
  * Total: **2,467 trading days**
* **Strict Temporal Isolation:** The 2024 test data were strictly held out. The model checkpoint was selected based on Episode 9 validation performance (2022–2023) prior to running out-of-sample inference.

---

## 4. Empirical Result Reproduction Verification

All reported values in the paper were verified against the original experiment CSV files (`results/metrics/test_metrics.csv`, `results/metrics/test_equity_curve.csv`, `results/metrics/validation_metrics.csv`, and `results/metrics/training_metrics.csv`):

| Metric / Result | Paper Reported Value | Actual CSV File Value | Source File Reference | Audit Verification |
| :--- | :---: | :---: | :--- | :---: |
| **DDPG Final Portfolio Value** | \$144,625.75 | \$144,625.7518 | `test_metrics.csv` (Row 1) | **EXACT MATCH** |
| **DDPG Cumulative Return** | +44.63% | +44.6258% | `test_metrics.csv` (Row 1) | **EXACT MATCH** |
| **DDPG Annualized Return** | 44.84% | 44.8385% | `test_metrics.csv` (Row 1) | **EXACT MATCH** |
| **DDPG Annualized Volatility** | 38.41% | 38.4126% | `test_metrics.csv` (Row 1) | **EXACT MATCH** |
| **DDPG Sharpe Ratio ($r_f=0$)** | 1.17 | 1.1673 | `test_metrics.csv` (Row 1) | **EXACT MATCH** |
| **DDPG Maximum Drawdown** | 22.08% | 22.0754% | `test_metrics.csv` (Row 1) | **EXACT MATCH** |
| **DDPG Transaction Costs** | \$12,780.30 | \$12,780.3035 | `test_metrics.csv` (Row 1) | **EXACT MATCH** |
| **Equal-Weight Final Value** | \$157,948.88 | \$157,948.8845 | `test_metrics.csv` (Row 2) | **EXACT MATCH** |
| **Equal-Weight Cumulative Return** | +57.95% | +57.9489% | `test_metrics.csv` (Row 2) | **EXACT MATCH** |
| **Equal-Weight Annualized Return** | 58.24% | 58.2368% | `test_metrics.csv` (Row 2) | **EXACT MATCH** |
| **Equal-Weight Volatility** | 22.39% | 22.3866% | `test_metrics.csv` (Row 2) | **EXACT MATCH** |
| **Equal-Weight Sharpe Ratio** | 2.60 | 2.6014 | `test_metrics.csv` (Row 2) | **EXACT MATCH** |
| **Equal-Weight Max Drawdown** | 17.31% | 17.3086% | `test_metrics.csv` (Row 2) | **EXACT MATCH** |
| **Equal-Weight Transaction Costs**| \$344.62 | \$344.6212 | `test_metrics.csv` (Row 2) | **EXACT MATCH** |
| **Training Episode 1 Reward** | 1.861 | 1.86075 | `training_metrics.csv` (Row 1) | **EXACT MATCH** |
| **Training Episode 13 Peak Reward**| 5.886 | 5.88564 | `training_metrics.csv` (Row 13)| **EXACT MATCH** |
| **Training Episode 14 Reward** | 4.503 | 4.50262 | `training_metrics.csv` (Row 14)| **EXACT MATCH** |
| **Training Episode 15 Reward** | 3.688 | 3.68791 | `training_metrics.csv` (Row 15)| **EXACT MATCH** |
| **Validation Ep 3 Return (Value)** | +36.28% (\$136,280.89) | 0.362809 (\$136,280.89) | `validation_metrics.csv` (Row 1) | **EXACT MATCH** |
| **Validation Ep 6 Return (Value)** | +52.01% (\$152,006.39) | 0.520064 (\$152,006.39) | `validation_metrics.csv` (Row 2) | **EXACT MATCH** |
| **Validation Ep 9 Return (Value)** | +67.27% (\$167,268.04) | 0.672680 (\$167,268.04) | `validation_metrics.csv` (Row 3) | **EXACT MATCH** |
| **Validation Ep 9 Sharpe Ratio** | 0.76 | 0.7623 | `validation_metrics.csv` (Row 3) | **EXACT MATCH** |
| **Validation Ep 12 Return** | +38.14% | 0.381375 | `validation_metrics.csv` (Row 4) | **EXACT MATCH** |
| **Validation Ep 15 Return** | -3.20% | -0.031963 | `validation_metrics.csv` (Row 5) | **EXACT MATCH** |
| **NVDA Mean Weight (>50%, >90%)** | 59.69% (149 days, 129 days) | 59.6904% (149 days, 129 days) | `daily_allocation_2024.csv` | **EXACT MATCH** |
| **AAPL Mean Weight (>50%, >90%)** | 22.51% (58 days, 34 days) | 22.5143% (58 days, 34 days) | `daily_allocation_2024.csv` | **EXACT MATCH** |
| **GOOGL Mean Weight (>50%)** | 9.65% (24 days) | 9.6538% (24 days) | `daily_allocation_2024.csv` | **EXACT MATCH** |
| **AMZN Mean Weight (>50%)** | 6.22% (9 days) | 6.2225% (9 days) | `daily_allocation_2024.csv` | **EXACT MATCH** |
| **MSFT Mean Weight (Max Weight)** | 1.92% (39.86%) | 1.9190% (39.8640%) | `daily_allocation_2024.csv` | **EXACT MATCH** |
| **DDPG Mid-Year Peak (2024-07-10)**| \$177,569.49 | \$177,569.4929 | `test_equity_curve.csv` | **EXACT MATCH** |
| **DDPG Trough (2024-08-07)** | \$138,367.63 | \$138,367.6259 | `test_equity_curve.csv` | **EXACT MATCH** |
| **DDPG 2nd Peak (2024-11-11)** | \$169,139.79 | \$169,139.7918 | `test_equity_curve.csv` | **EXACT MATCH** |

---

## 5. Figures and Tables Audit

* **Figure 1 (Page 5):** Overall workflow schematic. Captioned properly, clean presentation.
* **Figure 2 (Page 6):** Actor-Critic architecture and training flow. Diagram reflects two-stream Critic and 50-D state.
* **Figure 3 (Page 7):** Training reward and validation performance. Corresponds to `training_validation_performance.png`. Both panels (a) and (b) accurately plot `training_metrics.csv` and `validation_metrics.csv`.
* **Figure 4 (Page 7):** Out-of-sample 2024 equity curve. Corresponds to `results_equity_curve.png`. Plotted curves follow `test_equity_curve.csv` daily points without smoothing.
* **Figure 5 (Page 8):** Daily portfolio allocation. Corresponds to `dynamic_allocation_2024.png`. Reflects the 252 daily allocations in `daily_allocation_2024.csv`.
* **Table I (Page 3):** DDPG Training Configuration. All 13 parameters verified against source code.
* **Table II (Page 8):** Held-Out 2024 Performance. All 7 metrics for both DDPG and Equal-Weight match `test_metrics.csv` to two decimal places.
* **Defect / Unresolved LaTeX Reference Found:**  
  * On **Page 3, Section IV-A, line 105**: the text contains an uncompiled cross-reference:  
    `"The overall workflow of the proposed framework is illustrated in Fig.??."`  
    This refers to **Fig. 1** on Page 5.

---

## 6. Paper Claim Audit (Claims vs. Empirical Evidence)

| Claim in Paper | Location | Why It May Be Too Strong / Imprecise | Empirical Evidence Available | Recommended Status |
| :--- | :--- | :--- | :--- | :--- |
| *"The DDPG Actor generates continuous portfolio allocations..."* | Abstract & Section I-A | The Actor outputs unconstrained linear logits $a_t \in \mathbb{R}^5$. Simplex allocation weights $[0, 1]$ are generated by environment-level Softmax. | `networks.py` and `portfolio_environment.py` | **SHOULD CLARIFY:** Specify that the Actor produces continuous logits which are mapped to portfolio weights via Softmax in the environment. |
| *"The network consists of two fully connected hidden layers with 256 neurons each, with ReLU activation applied after each hidden layer."* | Section V-A & Section V-B | Omits mentioning Layer Normalization (`nn.LayerNorm`), which is implemented between each linear layer and ReLU. | `networks.py`, lines 50, 52, 97, 101 | **SHOULD CLARIFY:** Add LayerNorm to the network description for exact architectural reproducibility. |
| *"Based on validation cumulative return, the checkpoint from Episode 9 was selected for evaluation on the held-out test period."* | Section VI-A | Omission of validation risk profile. Episode 9 had a high validation maximum drawdown (55.37%), which was not reported alongside the +67.27% return. | `validation_metrics.csv` (`val_max_drawdown = 0.5537`) | **NO CHANGE REQUIRED** (Model selection was strictly criterion-driven by cumulative return), but noting validation drawdown adds transparency. |
| *"The concentration coincided with periods of portfolio growth during increases in NVDA and AAPL prices..."* | Section VI-C | Uses descriptive phrasing ("coincided with") rather than claiming direct causal optimality. | Daily allocation history and asset price trajectories in 2024 | **FULLY SUPPORTED:** Well-calibrated, non-causal prose. |
| Benchmark comparison in Discussion & Conclusion | Section VI-D & Section VII | Fully objective. Openly reports that the passive Equal-Weight benchmark achieved higher return, higher Sharpe ratio, lower volatility, and lower drawdown. | `test_metrics.csv` | **FULLY SUPPORTED:** Exemplary academic objectivity. No overstated superiority claims. |

---

## 7. Mathematical Necessity Audit (Equations 1–30)

* **CORE (12 Equations):**  
  * **Eq. (7):** SMA Ratio feature
  * **Eq. (8):** RSI normalization
  * **Eq. (9):** MACD and Signal scaling
  * **Eq. (10):** MACD Histogram scaling
  * **Eq. (11):** Bollinger Band position (%B)
  * **Eq. (14):** 50-D state vector definition
  * **Eq. (16):** Portfolio simplex constraint
  * **Eq. (17):** Net return reward objective
  * **Eq. (23):** Critic Q-value function
  * **Eq. (25):** Bellman target Q update
  * **Eq. (26):** Critic MSE TD loss
  * **Eq. (27):** Actor policy gradient objective
  * **Eq. (28)–(29):** Polyak target network soft updates ($\tau = 0.005$)

* **SUPPORTING (5 Equations):**  
  * **Eq. (1):** State vector definition (Supporting, but redundant with Eq. 14)
  * **Eq. (3):** Turnover norm $T_t = \sum \|w_i - w_{i-1}\|$ (Supporting, but duplicated in Eq. 18)
  * **Eq. (4):** Transaction cost $C_t = c T_t$ (Supporting, but duplicated in Eq. 19)
  * **Eq. (6):** Daily return definition (Supporting; should state Adjusted Close)
  * **Eq. (13):** 45-D market feature block definition
  * **Eq. (21):** Experience replay tuple format (Supporting, but duplicated in Eq. 24)

* **REDUNDANT (6 Equations):**  
  * **Eq. (2):** Arithmetic identity $5 \times 9 + 5 = 50$ (duplicates text description)
  * **Eq. (18):** Exact verbatim duplicate of Eq. (3)
  * **Eq. (19):** Exact verbatim duplicate of Eq. (4)
  * **Eq. (22):** Exact verbatim duplicate of Eq. (15) ($a_t = \pi_\theta(s_t)$)
  * **Eq. (24):** Exact verbatim duplicate of Eq. (21) ($(s_t, a_t, r_t, s_{t+1}, d_t)$)

* **OPTIONAL / FLOWCHART ARROW STRINGS (3 Equations):**  
  * **Eq. (5):** $s_t \to a_t \to r_t \to s_{t+1}$
  * **Eq. (20):** $(s_t, a_t) \to (R^p_t, C_t, r_t, s_{t+1})$
  * **Eq. (30):** $s_t \to \pi_\theta(s_t) \to a_t \to (r_t, s_{t+1}) \to \mathcal{B} \to \text{Critic Update} \to \text{Actor Update}$  
  *(These are inline schematic text diagrams formatted as numbered equations).*

* **INCORRECT / INCOMPLETE FORMULATION (1 Equation):**  
  * **Eq. (12):** Bollinger Band Width. In the paper: $\text{BBWidth}_t = \frac{\text{BB}^{\text{high}}_t - \text{BB}^{\text{low}}_t}{100}$. In implementation: $\frac{\text{BB}^{\text{high}}_t - \text{BB}^{\text{low}}_t}{\text{BB}^{\text{mid}}_t}$. The denominator $\text{BB}^{\text{mid}}_t$ was omitted in the paper formula.

---

## 8. Final Audit Synthesis

### A. Overall Verdict
**MOSTLY CONSISTENT — MINOR CORRECTIONS**

**Justification:**  
The paper is thoroughly grounded in the actual codebase. Every empirical metric reported across the 15 training episodes, the 5 validation checkpoints, and the 252 held-out 2024 test days matches the generated CSV results bit-for-bit. The research methodology, split dates, trading day counts, hyperparameters, and benchmark comparisons are faithfully represented.  
The status is categorized as "Mostly Consistent — Minor Corrections" due to:
1. One mathematical feature formula mismatch (Eq. 12 omits $\text{BB}^{\text{mid}}_t$).
2. Six redundant/duplicated equations (Eq. 3 $\equiv$ 18, Eq. 4 $\equiv$ 19, Eq. 1 $\approx$ 14, Eq. 15 $\equiv$ 22, Eq. 21 $\equiv$ 24).
3. One unresolved LaTeX cross-reference glitch (`Fig.??` on Page 3).
4. Omission of `LayerNorm` in the Actor/Critic network descriptions.

### B. Implementation Mismatches
1. **Bollinger Band Width (Equation 12):** The paper formula divides the band span by 100 rather than the middle band $\text{BB}^{\text{mid}}_t$.
2. **Daily Return Price Basis (Equation 6):** The paper specifies closing price $P_t$, whereas the code uses Adjusted Close (`Adj Close`).
3. **Actor Simplex Normalization Location:** The paper discusses Softmax normalization as an Actor property, but the PyTorch Actor outputs unbounded linear logits, and Softmax is applied in the environment.
4. **Network Layer Normalization:** The paper describes Actor and Critic hidden layers as standard Linear + ReLU, omitting the `LayerNorm` layers present in `networks.py`.

### C. Redundant Equations
* **Equations (1) & (2) vs. (13) & (14):** Both define the 50-D state vector and 45-D market feature concatenation.
* **Equations (3) & (4) vs. (18) & (19):** Both define turnover $T_t = \sum \|w_{i,t} - w_{i,t-1}\|$ and cost $C_t = c T_t$ verbatim.
* **Equation (15) vs. Equation (22):** Both state $a_t = \pi_\theta(s_t)$ verbatim.
* **Equation (21) vs. Equation (24):** Both state $(s_t, a_t, r_t, s_{t+1}, d_t)$ verbatim.
* **Equations (5), (20), and (30):** Three separate conceptual arrow strings formatted as numbered display equations.

### D. Missing Formulations
* The exact discrete portfolio value compounding formula is not explicitly given:
  $$V_{t+1} = V_t (1 - c T_t)(1 + \sum_{i=1}^5 w_{i,t} R_{i,t+1})$$
  Adding this would clarify why the reward $r_t$ in code is the exact net percentage return rather than the additive approximation $R^p_t - C_t$.

### E. Result Verification
* **100% of reported test metrics match** the underlying data files (`test_metrics.csv`, `test_equity_curve.csv`).
* **100% of reported training and validation metrics match** (`training_metrics.csv`, `validation_metrics.csv`).
* **100% of reported allocation statistics match** (`daily_allocation_2024.csv`).

### F. Figure/Table Verification
* **Fig. 1–5 and Tables I–II** correspond directly to actual project outputs. No fabricated curves or approximated data were detected.
* One unresolved cross-reference defect exists on Page 3, line 105 (`Fig.??`).

### G. Final Action List for Manual Correction

#### MUST FIX
1. **Fix Unresolved Cross-Reference:** On Page 3, Section IV-A, change `Fig.??` to `Fig. 1`.
2. **Correct Equation (12):** Update the denominator to include $\text{BB}^{\text{mid}}_t$:
   $$\text{BBWidth}_t = \frac{\text{BB}^{\text{high}}_t - \text{BB}^{\text{low}}_t}{\text{BB}^{\text{mid}}_t}$$

#### SHOULD FIX
1. **Clarify Equation (6):** Specify that $P_t$ is the dividend/split-adjusted closing price ($P_t^{\text{adj}}$).
2. **Include LayerNorm in Architecture Description (Sections V-A, V-B):** State that `LayerNorm` is applied to hidden layers before ReLU activation in both Actor and Critic.
3. **Clarify Normalization Pipeline (Section IV-C, V-A):** Note that the Actor network outputs unconstrained continuous logits, and the long-only fully invested simplex constraint is enforced via Softmax inside the environment.

#### OPTIONAL (Page-Budget / Equation Streamlining)
1. **Consolidate Redundant Equations:** To free up space or streamline presentation, merge Eq. (1)–(2) with (13)–(14), eliminate the duplicate turnover/cost equations (18)–(19) in favor of (3)–(4), and merge duplicate policy/tuple definitions ((15)/(22) and (21)/(24)).
2. **Convert Arrow Equations (5), (20), (30) to Inline Text:** Changing these conceptual schematic sequences into inline text reduces formal equation clutter.

#### NO CHANGE REQUIRED
* The empirical results, out-of-sample discussion, tables, benchmark comparisons, and restrained academic tone are sound and should be retained as currently written.
