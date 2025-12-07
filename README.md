# 🚀 Hybrid Trading System for Taiwan Stock Index (^TWII)

A sophisticated algorithmic trading system that combines **LSTM-SSAM** (Long Short-Term Memory with Sequential Self-Attention) for price prediction with **Pro Trader RL** (Reinforcement Learning) for trading decisions.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **LSTM-SSAM Prediction** | T+1 and T+5 price prediction with MC Dropout uncertainty estimation |
| **Transfer Learning** | Pre-train on global indices → Fine-tune for ^TWII |
| **Feature Fusion** | 23 features including LSTM predictions and confidence scores |
| **PPO Agent** | Separate Buy and Sell agents with class balancing |
| **Backtesting** | Full simulation with stop-loss and performance metrics |

## 📊 Performance Results (2023-Present)

| Metric | Value |
|--------|-------|
| **Total Return (ROI)** | 85.49% |
| **Annualized Return** | 23.53% |
| **Sharpe Ratio** | 1.47 |
| **Max Drawdown** | -17.23% |
| **Win Rate** | 100% (5 trades) |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     HYBRID TRADING SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  LSTM T+1    │    │  LSTM T+5    │    │  Technical       │  │
│  │  Prediction  │    │  + MC Dropout│    │  Indicators      │  │
│  └──────┬───────┘    └──────┬───────┘    └────────┬─────────┘  │
│         │                   │                      │            │
│         └───────────────────┼──────────────────────┘            │
│                             │                                    │
│                    ┌────────▼────────┐                          │
│                    │  23 Features    │                          │
│                    │  (Feature Fusion)│                         │
│                    └────────┬────────┘                          │
│                             │                                    │
│         ┌───────────────────┴───────────────────┐               │
│         │                                       │               │
│  ┌──────▼──────┐                        ┌──────▼──────┐        │
│  │  Buy Agent  │                        │  Sell Agent │        │
│  │    (PPO)    │                        │    (PPO)    │        │
│  └──────┬──────┘                        └──────┬──────┘        │
│         │                                      │                │
│         └──────────────────┬───────────────────┘                │
│                            │                                     │
│                   ┌────────▼────────┐                           │
│                   │  Trading Signal │                           │
│                   └─────────────────┘                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
hybrid-trader-v01/
├── ptrl_hybrid_system.py        # Main hybrid system (all-in-one)
├── train_lstm_models.py         # LSTM model training script
├── twii_model_registry_5d.py    # T+5 LSTM model registry
├── twii_model_registry_multivariate.py  # T+1 LSTM model registry
├── trade_advisor.py             # Trading advice generator
├── ptrl_TW50_split_train.py     # Reference: Original RL training
├── ptrl_TW50_paper_version.py   # Reference: Paper implementation
│
├── models_hybrid/               # Trained RL models
│   ├── ppo_buy_base.zip         # Pre-trained Buy agent
│   ├── ppo_sell_base.zip        # Pre-trained Sell agent
│   ├── ppo_buy_twii_final.zip   # Fine-tuned Buy agent
│   └── ppo_sell_twii_final.zip  # Fine-tuned Sell agent
│
├── saved_models_multivariate/   # T+1 LSTM models
├── saved_models_5d/             # T+5 LSTM models
│
├── data/processed/              # Feature cache
│   └── *_features.pkl
│
└── results_hybrid/              # Backtest results
    └── final_performance.png
```

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/hybrid-trader-v01.git
cd hybrid-trader-v01

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
tensorflow>=2.10
stable-baselines3>=2.0
gymnasium
yfinance
pandas
numpy
ta
torch
tqdm
matplotlib
psutil
```

## 🚀 Quick Start

### 1. Train LSTM Models (Long-Period)

```bash
python train_lstm_models.py
```

This trains LSTM T+1 and T+5 models on 2000-2023 data.

### 2. Run Full Pipeline

```bash
python ptrl_hybrid_system.py
```

This will:
1. **Phase 1-3**: Pre-train RL agents on 5 global indices (if not already done)
2. **Phase 4**: Fine-tune for ^TWII and run backtesting

## 📈 Training Pipeline

### Phase 1: Data Expansion
- Download 5 global indices: ^TWII, ^GSPC, ^IXIC, ^SOX, ^DJI
- Date range: 2000-01-01 ~ Present

### Phase 2: Feature Engineering
- 23 features including:
  - Normalized OHLC prices
  - Donchian Channel, SuperTrend
  - Heikin-Ashi patterns
  - RSI, MFI, ATR
  - Relative Strength metrics
  - **LSTM_Pred_1d**: T+1 prediction
  - **LSTM_Pred_5d**: T+5 prediction
  - **LSTM_Conf_5d**: T+5 confidence (MC Dropout)

### Phase 3: Pre-training
- Buy Agent: 1,000,000 steps (class-balanced sampling)
- Sell Agent: 500,000 steps

### Phase 4: Fine-tuning & Backtesting
- Fine-tune on ^TWII (2000-2022) with LR=1e-5
- Backtest on (2023-Present)

## 📊 Output

After running `ptrl_hybrid_system.py`, you'll get:

- `models_hybrid/ppo_buy_twii_final.zip`: Fine-tuned Buy model
- `models_hybrid/ppo_sell_twii_final.zip`: Fine-tuned Sell model
- `results_hybrid/final_performance.png`: Performance chart

## 🔧 Configuration

Key parameters in `ptrl_hybrid_system.py`:

```python
SPLIT_DATE = '2023-01-01'  # Train/Test split

# Pre-training
TOTAL_TIMESTEPS_BUY = 1_000_000
TOTAL_TIMESTEPS_SELL = 500_000

# Fine-tuning (Transfer Learning)
FINETUNE_LR = 1e-5  # 1/10 of original
FINETUNE_BUY_STEPS = 200_000
FINETUNE_SELL_STEPS = 100_000
```

## 📚 References

- **Pro Trader RL**: [Paper Implementation](https://arxiv.org/abs/xxxx)
- **LSTM-SSAM**: Sequential Self-Attention for time series prediction
- **MC Dropout**: Uncertainty estimation via Monte Carlo Dropout

## 📄 License

MIT License

## 👤 Author

Phil Liang

---

*Built with Python, TensorFlow, Stable-Baselines3, and ❤️*
