# RL-APM Production Deployment Guide

## 1. System Architecture

RL-APM ("Reinforcement Learning for Portfolio Optimization in Equity Markets") is structured into a decoupled, production-ready system:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React + Vite)                         │
│   • Clean white AI fintech design system (Tailwind CSS)               │
│   • Recharts analytical visualizations & Framer Motion transitions    │
│   • Dynamic API client configured via VITE_API_BASE_URL               │
│   • Static build output: frontend/dist (SPA with client routing)      │
│   • Deployment target: Vercel / Netlify / Render Static / Nginx       │
└───────────────────────────────────▲────────────────────────────────────┘
                                    │ HTTP / JSON REST
                                    │ (CORS configured)
┌───────────────────────────────────▼────────────────────────────────────┐
│                        BACKEND (FastAPI + PyTorch)                     │
│   • High-performance ASGI service (Uvicorn)                           │
│   • Pre-warmed PyTorch DDPG Actor checkpoint (CPU inference)          │
│   • Cached historical technical feature matrices (2015–2024)          │
│   • Verified 2024 held-out evaluation metrics & equity curves         │
│   • Dynamic PORT & CORS_ORIGINS handling via environment variables    │
│   • Deployment target: Render / Railway / Fly.io / Docker Container   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Production Environment Variables

### Backend Configuration (`backend/.env` or Platform Dashboard)
| Variable | Default (Local) | Production Example | Description |
|---|---|---|---|
| `PORT` | `8000` | `10000` (assigned by cloud host) | Web service listening port |
| `HOST` | `0.0.0.0` | `0.0.0.0` | Network binding interface |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | `https://rl-apm.vercel.app` | Comma-separated allowed frontend origins |
| `FRONTEND_URL` | *(None)* | `https://rl-apm.vercel.app` | Primary frontend origin added to CORS |

### Frontend Configuration (`frontend/.env` or Platform Dashboard)
| Variable | Default (Local) | Production Example | Description |
|---|---|---|---|
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | `https://rl-apm-backend.onrender.com` | Base URL of the deployed FastAPI backend (no trailing slash required) |

---

## 3. Production Deployment Options

### Option A: One-Click Render Blueprint (`render.yaml`)
Both frontend and backend can be deployed automatically via Render using the included `render.yaml` configuration:
1. Push the repository to GitHub.
2. In the Render Dashboard, click **New +** $\rightarrow$ **Blueprint**.
3. Connect your repository. Render automatically reads `render.yaml` and provisions:
   - `rl-apm-backend`: Python 3.11 web service with `pip install -r backend/requirements.txt` and `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`.
   - `rl-apm-frontend`: Static web app with `npm run build` pointing to `frontend/dist` with SPA rewrite rules and automatic backend connection.

### Option B: Vercel (Frontend) + Render / Railway (Backend)

#### 1. Backend on Render or Railway
- **Root Directory**: `.` (Repository root)
- **Build Command**: `pip install -r backend/requirements.txt`
- **Start Command**: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
- **Health Check Path**: `/api/health`
- **Environment Variables**:
  - `PYTHON_VERSION`: `3.11.9`
  - `CORS_ORIGINS`: `https://<your-vercel-domain>.vercel.app`

#### 2. Frontend on Vercel
- **Root Directory**: `frontend`
- **Framework Preset**: `Vite`
- **Build Command**: `npm run build`
- **Output Directory**: `dist`
- **Configuration**: Uses `frontend/vercel.json` for client-side routing rewrites (`/* -> /index.html`).
- **Environment Variables**:
  - `VITE_API_BASE_URL`: `https://<your-backend-domain>.onrender.com`

### Option C: Docker Container Deployment (`docker-compose.yml`)
The repository includes production Dockerfiles for both services:
```bash
# Build and run both backend and frontend locally or on any cloud VM
docker compose up --build -d
```
- Backend container listens on port `8000`.
- Frontend container runs Nginx on port `5173` with SPA fallback routing.

---

## 4. Local Execution

### Backend
```bash
# Activate Python environment
backend\venv\Scripts\activate  # Windows
# or source backend/venv/bin/activate  # Linux/macOS

# Run with uvicorn
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger Documentation: `http://127.0.0.1:8000/docs`

### Frontend
```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```
Frontend UI: `http://127.0.0.1:5173`

---

## 5. Critical Files & Checkpoints

The production deployment requires the following immutable assets:
1. **PyTorch Trained Actor Checkpoint**: `models/ddpg_actor_best.pth` (50-dimensional observation state, 5 continuous action logits normalized via softmax).
2. **Preprocessed Market Features**: `data/processed/market_features_2015_2024.csv` and `data/processed/clean_returns.csv`.
3. **Validated Research Results**:
   - `results/metrics/test_metrics.csv` (Frozen 2024 out-of-sample performance table).
   - `results/metrics/test_equity_curve.csv` (Daily backtest trajectory).
   - `results/metrics/training_metrics.csv` & `validation_metrics.csv`.
   - `results/figures/` (Generated research visualizations).

---

## 6. Health & Verification Endpoints

- **Health Check**: `GET /api/health`
  ```json
  {
    "status": "ok",
    "service": "RL-APM API",
    "version": "1.0.0",
    "model_available": true,
    "data_available": true,
    "metrics_available": true,
    "environment": "production"
  }
  ```
- **Model Metadata**: `GET /api/model-info`
- **Real DDPG Inference**: `POST /api/predict`
  ```json
  {
    "weights": [0.2, 0.2, 0.2, 0.2, 0.2]
  }
  ```

---

## 7. Known Limitations & Research Disclaimer

1. **Research Prototype**: RL-APM is an academic research prototype demonstrating continuous reinforcement learning for portfolio optimization. It is **not** a registered investment advisory service, algorithmic execution broker, or commercial trading software.
2. **No Live Brokerage Execution**: The application calculates theoretical daily rebalancing allocations based on historical market indicators. It does not place real orders with financial brokers or exchanges.
3. **Transaction Cost Sensitivity**: The 2024 held-out evaluation strictly enforces a 0.1% transaction cost ($c = 0.001$). Under active daily rebalancing, transaction frictions accumulated to $12,780.30, resulting in DDPG underperforming the static equal-weight benchmark (+44.63% vs. +57.95%). These results are reported transparently in adherence to quantitative research integrity.
