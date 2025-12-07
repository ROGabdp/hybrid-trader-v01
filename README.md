# 🚀 Hybrid Trading System for Taiwan Stock Index (^TWII)

這是一個先進的演算法交易系統，結合了用於價格預測的 **LSTM-SSAM** (Long Short-Term Memory with Sequential Self-Attention) 以及用於交易決策的 **Pro Trader RL** (Reinforcement Learning)。

## ✨ 核心特色 (Key Features)

| 特色 | 說明 |
|---------|-------------|
| **LSTM-SSAM 預測** | T+1 與 T+5 價格預測，並使用 MC Dropout 進行不確定性估計 |
| **遷移學習 (Transfer Learning)** | 使用全球指數進行預訓練 (Pre-train) → 針對 ^TWII 進行微調 (Fine-tune) |
| **特徵融合 (Feature Fusion)** | 整合 23 種特徵，包含 LSTM 預測值與信心分數 |
| **PPO Agent** | 分離的買入 (Buy) 與賣出 (Sell) 代理人，並具備類別平衡機制 |
| **回測 (Backtesting)** | 完整的模擬回測，包含停損機制與績效指標計算 |

## 📊 績效結果 (2023-Present)

| 指標 (Metric) | 數值 (Value) |
|--------|-------|
| **總報酬率 (ROI)** | 85.49% |
| **年化報酬率 (Annualized Return)** | 23.53% |
| **夏普值 (Sharpe Ratio)** | 1.47 |
| **最大回撤 (Max Drawdown)** | -17.23% |
| **勝率 (Win Rate)** | 100% (5 次交易) |

## 🏗️ 系統架構 (Architecture)

```
┌─────────────────────────────────────────────────────────────────┐
│                     HYBRID TRADING SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  LSTM T+1    │    │  LSTM T+5    │    │    技術指標       │  │
│  │   預測模型    │    │  + MC Dropout│    │  (Indicators)    │  │
│  └──────┬───────┘    └──────┬───────┘    └────────┬─────────┘  │
│         │                   │                      │            │
│         └───────────────────┼──────────────────────┘            │
│                             │                                    │
│                    ┌────────▼────────┐                          │
│                    │    23 特徵融合   │                          │
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
│                   │    交易訊號      │                           │
│                   └─────────────────┘                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 專案結構 (Project Structure)

```
hybrid-trader-v01/
├── ptrl_hybrid_system.py        # 混合交易系統主程式 (All-in-one)
├── train_lstm_models.py         # LSTM 模型訓練腳本
├── twii_model_registry_5d.py    # T+5 LSTM 模型註冊管理
├── twii_model_registry_multivariate.py  # T+1 LSTM 模型註冊管理
├── trade_advisor.py             # 交易建議生成器
├── ptrl_TW50_split_train.py     # 參考：原始 RL 訓練程式
├── ptrl_TW50_paper_version.py   # 參考：論文實作版本
│
├── models_hybrid/               # 訓練好的 RL 模型
│   ├── ppo_buy_base.zip         # 預訓練 Buy Agent
│   ├── ppo_sell_base.zip        # 預訓練 Sell Agent
│   ├── ppo_buy_twii_final.zip   # 微調後 Buy Agent (^TWII)
│   └── ppo_sell_twii_final.zip  # 微調後 Sell Agent (^TWII)
│
├── saved_models_multivariate/   # T+1 LSTM 模型存檔
├── saved_models_5d/             # T+5 LSTM 模型存檔
│
├── data/processed/              # 特徵快取資料
│   └── *_features.pkl
│
└── results_hybrid/              # 回測結果
    └── final_performance.png
```

## 🛠️ 安裝說明 (Installation)

### 建議使用虛擬環境 (Virtual Environment)
在 Windows 上使用虛擬環境可以避免套件版本衝突，強烈建議使用。

**方法一：使用自動腳本 (推薦)**
```powershell
.\setup_env.ps1
```

**方法二：手動設定**
```powershell
# 1. 建立虛擬環境
python -m venv venv

# 2. 啟動虛擬環境
.\venv\Scripts\Activate.ps1

# 3. 安裝套件
pip install -r requirements.txt
```

### ⚡ GPU 加速設定 (重要)
本專案建議使用 NVIDIA 顯卡進行訓練加速。

**方法一：使用 setup_env.ps1 (自動)**
腳本會自動安裝支援 CUDA 11.8 的 PyTorch 版本。

**方法二：手動安裝**
若您手動執行 `pip install -r requirements.txt`，預設會安裝 CPU 版本。請執行以下指令將其替換為 GPU 版本：

```powershell
# 1. 移除 CPU 版本
pip uninstall torch torchvision torchaudio -y

# 2. 安裝 GPU 版本 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 系統需求 (Dependencies)

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

## 🚀 快速開始 (Quick Start)

### 1. 訓練 LSTM 模型 (長週期)

```bash
python train_lstm_models.py
```

此步驟將使用 2000-2023 年的數據訓練 LSTM T+1 與 T+5 模型。

### 2. 執行完整流程 (Full Pipeline)

```bash
python ptrl_hybrid_system.py
```

此指令將執行：
1. **Phase 1-3**: 使用 5 個全球指數預訓練 RL Agent (如果尚未完成)
2. **Phase 4**: 針對 ^TWII 進行微調 (Fine-tune) 並執行回測

## 📈 訓練流程 (Training Pipeline)

### Phase 1: 數據擴充 (Data Expansion)
- 下載 5 個全球指數：^TWII, ^GSPC, ^IXIC, ^SOX, ^DJI
- 數據範圍：2000-01-01 ~ Present

### Phase 2: 特徵工程 (Feature Engineering)
- 包含 23 種特徵：
  - 標準化 OHLC 價格
  - 唐奇安通道 (Donchian Channel)、超級趨勢 (SuperTrend)
  - 平均K線 (Heikin-Ashi) 型態
  - RSI, MFI, ATR 指標
  - 相對強度 (Relative Strength) 指標
  - **LSTM_Pred_1d**: T+1 預測漲幅
  - **LSTM_Pred_5d**: T+5 預測漲幅
  - **LSTM_Conf_5d**: T+5 信心度 (MC Dropout)

### Phase 3: 預訓練 (Pre-training)
- Buy Agent: 1,000,000 步 (類別平衡採樣)
- Sell Agent: 500,000 步

### Phase 4: 微調與回測 (Fine-tuning & Backtesting)
- 微調：針對 ^TWII (2000-2022) 進行訓練，Learning Rate = 1e-5
- 回測：驗證數據集 (2023-Present)

### Phase 5: 訓練監控 (Training Monitoring)
本系統整合了 **TensorBoard** 進行訓練過程的即時監控。

**自動記錄的指標：**
- `rollout/ep_rew_mean`: 平均獎勵
- `train/loss`: 總損失
- `train/policy_gradient_loss`: 策略梯度損失
- `train/value_loss`: 價值函數損失
- `train/entropy_loss`: 熵損失
- `eval/mean_reward`: 驗證集平均獎勵 (EvalCallback)

**如何使用 TensorBoard：**
```powershell
# 在專案目錄下執行
tensorboard --logdir ./tensorboard_logs/

# 然後開啟瀏覽器前往
# http://localhost:6006
```

**日誌存放位置：**
- `./tensorboard_logs/`: TensorBoard 日誌
- `./logs/`: EvalCallback 評估結果
- `models_hybrid/best_tuned/`: 驗證集最佳模型

---

## 📊 輸出結果 (Output)

執行 `ptrl_hybrid_system.py` 後，您將獲得：

- `models_hybrid/ppo_buy_twii_final.zip`: 微調後的 Buy Model
- `models_hybrid/ppo_sell_twii_final.zip`: 微調後的 Sell Model
- `results_hybrid/final_performance.png`: 績效圖表
- `tensorboard_logs/`: 訓練過程日誌 (可用 TensorBoard 查看)

## 🔧 參數設定 (Configuration)

可在 `ptrl_hybrid_system.py` 修改關鍵參數：

```python
SPLIT_DATE = '2023-01-01'  # 訓練/測試 切分點

# 預訓練參數
TOTAL_TIMESTEPS_BUY = 1_000_000
TOTAL_TIMESTEPS_SELL = 500_000

# 微調參數 (Transfer Learning)
FINETUNE_LR = 1e-5  # 原始學習率的 1/10
FINETUNE_BUY_STEPS = 200_000
FINETUNE_SELL_STEPS = 100_000
```

## 📚 參考文獻 (References)

- **Pro Trader RL**: [Paper Implementation](https://arxiv.org/abs/xxxx)
- **LSTM-SSAM**: Sequential Self-Attention for time series prediction
- **MC Dropout**: Uncertainty estimation via Monte Carlo Dropout

## 📄 授權 (License)

MIT License

## 👤 作者 (Author)

Phil Liang

---

*Built with Python, TensorFlow, Stable-Baselines3, and ❤️*
