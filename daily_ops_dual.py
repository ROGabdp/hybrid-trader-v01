# -*- coding: utf-8 -*-
"""
================================================================================
Daily Operations with Dual Strategy & Versioning
================================================================================
每日維運腳本 - 雙策略推論與版本控管

功能：
1. 建立當日專屬工作區 (daily_runs/{date}/)
2. LSTM 全量重訓與封存
3. 隔離式特徵工程 (使用當日模型)
4. 雙模型推論 (Aggressive vs Conservative)
5. 輸出戰情儀表板與日誌

作者：Phil Liang
日期：2025-12-07
================================================================================
"""

import os
import sys
import shutil
import pickle
from datetime import datetime, timedelta

# 設定 UTF-8 輸出
sys.stdout.reconfigure(encoding='utf-8')

# 抑制 TensorFlow 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# =============================================================================
# 設定路徑
# =============================================================================
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DAILY_RUNS_PATH = os.path.join(PROJECT_PATH, 'daily_runs')

# RL 模型路徑 (固定)
STRATEGY_A_PATH = os.path.join(PROJECT_PATH, 'models_hybrid')  # Aggressive
STRATEGY_B_PATH = os.path.join(PROJECT_PATH, 'models_hybrid_v2_conservative')  # Conservative

# LSTM 模型預設路徑
DEFAULT_LSTM_5D_PATH = os.path.join(PROJECT_PATH, 'saved_models_5d')
DEFAULT_LSTM_1D_PATH = os.path.join(PROJECT_PATH, 'saved_models_multivariate')


# =============================================================================
# Step 0: 建立當日專屬工作區
# =============================================================================
def create_daily_workspace(date_str: str) -> dict:
    """建立當日專屬工作區目錄結構"""
    
    daily_path = os.path.join(DAILY_RUNS_PATH, date_str)
    
    paths = {
        'root': daily_path,
        'lstm_models': os.path.join(daily_path, 'lstm_models'),
        'cache': os.path.join(daily_path, 'cache'),
        'reports': os.path.join(daily_path, 'reports'),
    }
    
    for key, path in paths.items():
        os.makedirs(path, exist_ok=True)
    
    print(f"[Workspace] 建立當日工作區: {daily_path}")
    return paths


# =============================================================================
# Step 1: LSTM 全量重訓與封存
# =============================================================================
def train_and_archive_lstm(workspace: dict, end_date: str):
    """
    訓練 LSTM 模型並封存到當日工作區
    
    Args:
        workspace: 當日工作區路徑字典
        end_date: 訓練結束日期 (YYYY-MM-DD)
    """
    print("\n" + "=" * 60)
    print("📚 Step 1: LSTM 全量重訓與封存")
    print("=" * 60)
    
    # 動態引入模型訓練模組
    try:
        import twii_model_registry_5d as registry_5d
        import twii_model_registry_multivariate as registry_1d
    except ImportError as e:
        print(f"[Error] 無法載入 LSTM 模組: {e}")
        return False
    
    start_date = "2000-01-01"
    
    # =========================================================================
    # 訓練 T+5 模型
    # =========================================================================
    print(f"\n[LSTM T+5] 訓練範圍: {start_date} ~ {end_date}")
    try:
        # 下載數據
        df_5d = yf.download("^TWII", start=start_date, end=end_date, auto_adjust=True, progress=False)
        if len(df_5d) < 100:
            print("[Error] 數據不足，跳過 T+5 訓練")
        else:
            # 訓練模型
            registry_5d.train_model(df_5d, start_date, end_date)
            print("[LSTM T+5] ✅ 訓練完成")
    except Exception as e:
        print(f"[LSTM T+5] 訓練失敗: {e}")
    
    # =========================================================================
    # 訓練 T+1 模型
    # =========================================================================
    print(f"\n[LSTM T+1] 訓練範圍: {start_date} ~ {end_date}")
    try:
        # 下載數據
        df_1d = yf.download("^TWII", start=start_date, end=end_date, auto_adjust=True, progress=False)
        if len(df_1d) < 100:
            print("[Error] 數據不足，跳過 T+1 訓練")
        else:
            # 訓練模型
            registry_1d.train_model(df_1d, start_date, end_date)
            print("[LSTM T+1] ✅ 訓練完成")
    except Exception as e:
        print(f"[LSTM T+1] 訓練失敗: {e}")
    
    # =========================================================================
    # 封存模型到當日工作區
    # =========================================================================
    print("\n[Archive] 封存模型到當日工作區...")
    
    archive_path = workspace['lstm_models']
    
    # 複製 T+5 模型
    for src_dir in [DEFAULT_LSTM_5D_PATH]:
        if os.path.exists(src_dir):
            dest_dir = os.path.join(archive_path, os.path.basename(src_dir))
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
            shutil.copytree(src_dir, dest_dir)
            print(f"  ✅ 已複製: {os.path.basename(src_dir)}")
    
    # 複製 T+1 模型
    for src_dir in [DEFAULT_LSTM_1D_PATH]:
        if os.path.exists(src_dir):
            dest_dir = os.path.join(archive_path, os.path.basename(src_dir))
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
            shutil.copytree(src_dir, dest_dir)
            print(f"  ✅ 已複製: {os.path.basename(src_dir)}")
    
    return True


# =============================================================================
# Step 2: 隔離式特徵工程
# =============================================================================
def isolated_feature_engineering(workspace: dict, end_date: str) -> pd.DataFrame:
    """
    使用當日封存的 LSTM 模型進行特徵工程
    
    Args:
        workspace: 當日工作區路徑字典
        end_date: 數據結束日期
    
    Returns:
        包含所有特徵的 DataFrame
    """
    print("\n" + "=" * 60)
    print("🔧 Step 2: 隔離式特徵工程")
    print("=" * 60)
    
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    import ta
    
    # 自訂 SelfAttention 層 (與原始模型相同)
    class SelfAttention(layers.Layer):
        def __init__(self, **kwargs):
            super(SelfAttention, self).__init__(**kwargs)
        
        def build(self, input_shape):
            self.units = input_shape[-1]
            self.W_q = self.add_weight(name='W_query', shape=(self.units, self.units),
                                       initializer='glorot_uniform', trainable=True)
            self.W_k = self.add_weight(name='W_key', shape=(self.units, self.units),
                                       initializer='glorot_uniform', trainable=True)
        
        def call(self, inputs, training=None):
            Q = tf.matmul(inputs, self.W_q)
            K = tf.matmul(inputs, self.W_k)
            attention = tf.nn.softmax(tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.units, tf.float32)))
            return tf.matmul(attention, inputs)
    
    # =========================================================================
    # 載入當日封存的 LSTM 模型
    # =========================================================================
    lstm_5d_path = os.path.join(workspace['lstm_models'], 'saved_models_5d')
    lstm_1d_path = os.path.join(workspace['lstm_models'], 'saved_models_multivariate')
    
    model_5d, scaler_5d, meta_5d = None, None, None
    model_1d, scaler_1d, meta_1d = None, None, None
    
    # 載入 T+5 模型
    if os.path.exists(lstm_5d_path):
        import glob
        import json
        
        keras_files = glob.glob(os.path.join(lstm_5d_path, "*.keras"))
        if keras_files:
            latest_keras = sorted(keras_files)[-1]
            model_5d = keras.models.load_model(latest_keras, custom_objects={'SelfAttention': SelfAttention})
            
            # 載入 scaler
            scaler_file = latest_keras.replace('model_', 'scaler_').replace('.keras', '.pkl')
            if os.path.exists(scaler_file):
                with open(scaler_file, 'rb') as f:
                    scaler_5d = pickle.load(f)
            
            # 載入 meta
            meta_file = latest_keras.replace('model_', 'meta_').replace('.keras', '.json')
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as f:
                    meta_5d = json.load(f)
            
            print(f"[LSTM T+5] ✅ 已載入: {os.path.basename(latest_keras)}")
    
    # 載入 T+1 模型
    if os.path.exists(lstm_1d_path):
        import glob
        import json
        
        keras_files = glob.glob(os.path.join(lstm_1d_path, "*.keras"))
        if keras_files:
            latest_keras = sorted(keras_files)[-1]
            model_1d = keras.models.load_model(latest_keras, custom_objects={'SelfAttention': SelfAttention})
            
            # 載入 scaler
            scaler_file = latest_keras.replace('model_', 'scaler_').replace('.keras', '.pkl')
            if os.path.exists(scaler_file):
                with open(scaler_file, 'rb') as f:
                    scaler_1d = pickle.load(f)
            
            # 載入 meta
            meta_file = latest_keras.replace('model_', 'meta_').replace('.keras', '.json')
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as f:
                    meta_1d = json.load(f)
            
            print(f"[LSTM T+1] ✅ 已載入: {os.path.basename(latest_keras)}")
    
    # =========================================================================
    # 下載最新數據
    # =========================================================================
    print("\n[Data] 下載 ^TWII 數據...")
    df = yf.download("^TWII", start="2020-01-01", end=end_date, auto_adjust=True, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print(f"[Data] 數據範圍: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"[Data] 總筆數: {len(df)}")
    
    # =========================================================================
    # 計算技術指標
    # =========================================================================
    print("\n[Features] 計算技術指標...")
    
    # 基礎價格指標
    df['Norm_Open'] = df['Open'] / df['Close'].rolling(20).mean()
    df['Norm_High'] = df['High'] / df['Close'].rolling(20).mean()
    df['Norm_Low'] = df['Low'] / df['Close'].rolling(20).mean()
    df['Norm_Close'] = df['Close'] / df['Close'].rolling(20).mean()
    
    # Donchian Channel
    df['DC_High'] = df['High'].rolling(20).max()
    df['DC_Low'] = df['Low'].rolling(20).min()
    df['DC_Position'] = (df['Close'] - df['DC_Low']) / (df['DC_High'] - df['DC_Low'] + 1e-8)
    
    # SuperTrend (簡化版)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=10)
    df['SuperTrend_Signal'] = np.where(df['Close'] > df['Close'].rolling(10).mean() + df['ATR'], 1,
                                        np.where(df['Close'] < df['Close'].rolling(10).mean() - df['ATR'], -1, 0))
    
    # Heikin-Ashi
    df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df['HA_Open'] = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    df['HA_Trend'] = np.where(df['HA_Close'] > df['HA_Open'], 1, -1)
    
    # RSI, MFI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14) / 100
    df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14) / 100
    
    # MA
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    df['MA_Ratio_10_60'] = df['MA10'] / df['MA60']
    
    # Relative Strength
    df['RS_5d'] = df['Close'].pct_change(5)
    df['RS_20d'] = df['Close'].pct_change(20)
    
    # Volume
    df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # =========================================================================
    # 計算 LSTM 預測特徵
    # =========================================================================
    print("\n[LSTM] 計算預測特徵...")
    
    df['LSTM_Pred_1d'] = 0.0
    df['LSTM_Pred_5d'] = 0.0
    df['LSTM_Conf_5d'] = 0.5
    
    # LSTM 特徵欄位
    lstm_features = ['Close', 'Volume', 'RSI', 'MFI']
    
    if model_5d is not None and scaler_5d is not None:
        LOOKBACK_5D = 30
        
        for i in range(LOOKBACK_5D, len(df)):
            try:
                window = df.iloc[i-LOOKBACK_5D:i][['Close', 'Volume']].copy()
                window['Volume'] = np.log1p(window['Volume'])
                
                # 添加 KD, MACD_Hist
                window['KD'] = df['RSI'].iloc[i-LOOKBACK_5D:i].values * 100
                window['MACD_Hist'] = df['Close'].iloc[i-LOOKBACK_5D:i].pct_change().fillna(0).values
                
                scaled = scaler_5d.transform(window.values)
                X = scaled.reshape(1, LOOKBACK_5D, -1)
                
                # MC Dropout 預測
                preds = []
                for _ in range(5):
                    pred = model_5d(X, training=True).numpy()[0, 0]
                    preds.append(pred)
                
                mean_pred = np.mean(preds)
                std_pred = np.std(preds)
                
                # 反正規化
                price_min = meta_5d.get('price_min', df['Close'].min())
                price_max = meta_5d.get('price_max', df['Close'].max())
                pred_price = mean_pred * (price_max - price_min) + price_min
                current_price = df['Close'].iloc[i]
                
                df.iloc[i, df.columns.get_loc('LSTM_Pred_5d')] = (pred_price - current_price) / current_price
                df.iloc[i, df.columns.get_loc('LSTM_Conf_5d')] = max(0, min(1, 1 - std_pred * 10))
                
            except Exception as e:
                pass
    
    if model_1d is not None and scaler_1d is not None:
        LOOKBACK_1D = 10
        
        for i in range(LOOKBACK_1D, len(df)):
            try:
                window = df.iloc[i-LOOKBACK_1D:i][['Close', 'Volume']].copy()
                window['Volume'] = np.log1p(window['Volume'])
                window['KD'] = df['RSI'].iloc[i-LOOKBACK_1D:i].values * 100
                window['MACD_Hist'] = df['Close'].iloc[i-LOOKBACK_1D:i].pct_change().fillna(0).values
                
                scaled = scaler_1d.transform(window.values)
                X = scaled.reshape(1, LOOKBACK_1D, -1)
                
                pred = model_1d.predict(X, verbose=0)[0, 0]
                
                price_min = meta_1d.get('price_min', df['Close'].min())
                price_max = meta_1d.get('price_max', df['Close'].max())
                pred_price = pred * (price_max - price_min) + price_min
                current_price = df['Close'].iloc[i]
                
                df.iloc[i, df.columns.get_loc('LSTM_Pred_1d')] = (pred_price - current_price) / current_price
                
            except Exception as e:
                pass
    
    # =========================================================================
    # 儲存快取
    # =========================================================================
    df = df.dropna()
    
    cache_path = os.path.join(workspace['cache'], 'twii_features.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump(df, f)
    
    print(f"\n[Cache] 特徵已儲存: {cache_path}")
    print(f"[Cache] 最終筆數: {len(df)}")
    
    return df


# =============================================================================
# Step 3: 雙模型推論
# =============================================================================
def dual_inference(workspace: dict, df: pd.DataFrame) -> dict:
    """
    使用兩套 RL 模型進行推論
    
    Args:
        workspace: 當日工作區路徑字典
        df: 特徵 DataFrame
    
    Returns:
        包含兩套策略建議的字典
    """
    print("\n" + "=" * 60)
    print("🎯 Step 3: 雙模型推論")
    print("=" * 60)
    
    from stable_baselines3 import PPO
    
    # 特徵欄位 (與訓練時相同)
    FEATURE_COLS = [
        'Norm_Open', 'Norm_High', 'Norm_Low', 'Norm_Close',
        'DC_Position', 'SuperTrend_Signal', 'HA_Trend',
        'RSI', 'MFI', 'ATR', 'MA_Ratio_10_60',
        'RS_5d', 'RS_20d', 'Vol_Ratio',
        'LSTM_Pred_1d', 'LSTM_Pred_5d', 'LSTM_Conf_5d'
    ]
    
    # 取得最新一筆數據
    latest = df.iloc[-1]
    
    # 準備特徵向量
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    features = latest[available_cols].values.astype(np.float32)
    
    # 補齊缺失的欄位
    if len(features) < 23:
        features = np.pad(features, (0, 23 - len(features)), mode='constant', constant_values=0)
    
    results = {}
    
    # =========================================================================
    # Strategy A: Aggressive (ROI 85%)
    # =========================================================================
    print("\n[Strategy A] Aggressive (ROI 85%)...")
    
    buy_a_path = os.path.join(STRATEGY_A_PATH, 'ppo_buy_twii_final.zip')
    sell_a_path = os.path.join(STRATEGY_A_PATH, 'ppo_sell_twii_final.zip')
    
    if os.path.exists(buy_a_path) and os.path.exists(sell_a_path):
        buy_model_a = PPO.load(buy_a_path)
        sell_model_a = PPO.load(sell_a_path)
        
        # Buy 推論
        buy_action_a, _ = buy_model_a.predict(features, deterministic=True)
        buy_probs_a = buy_model_a.policy.get_distribution(
            buy_model_a.policy.obs_to_tensor(features.reshape(1, -1))[0]
        ).distribution.probs.detach().numpy()[0]
        
        # Sell 推論 (需要加入持有報酬)
        sell_features = np.concatenate([features, [1.0]])  # 假設持有報酬 0%
        sell_action_a, _ = sell_model_a.predict(sell_features, deterministic=True)
        
        results['strategy_a'] = {
            'name': 'Aggressive (ROI 85%)',
            'buy_action': int(buy_action_a),
            'buy_signal': 'BUY' if buy_action_a == 1 else 'HOLD',
            'buy_confidence': float(buy_probs_a[1]) if buy_action_a == 1 else float(buy_probs_a[0]),
            'sell_action': int(sell_action_a),
            'sell_signal': 'SELL' if sell_action_a == 1 else 'HOLD',
        }
        print(f"  Buy: {results['strategy_a']['buy_signal']} (Conf: {results['strategy_a']['buy_confidence']:.2%})")
        print(f"  Sell: {results['strategy_a']['sell_signal']}")
    else:
        print(f"  [Warning] 找不到模型: {buy_a_path}")
        results['strategy_a'] = {'name': 'Aggressive', 'error': 'Model not found'}
    
    # =========================================================================
    # Strategy B: Conservative (MDD -6%)
    # =========================================================================
    print("\n[Strategy B] Conservative (MDD -6%)...")
    
    buy_b_path = os.path.join(STRATEGY_B_PATH, 'ppo_buy_twii_final.zip')
    sell_b_path = os.path.join(STRATEGY_B_PATH, 'ppo_sell_twii_final.zip')
    
    if os.path.exists(buy_b_path) and os.path.exists(sell_b_path):
        buy_model_b = PPO.load(buy_b_path)
        sell_model_b = PPO.load(sell_b_path)
        
        # Buy 推論
        buy_action_b, _ = buy_model_b.predict(features, deterministic=True)
        buy_probs_b = buy_model_b.policy.get_distribution(
            buy_model_b.policy.obs_to_tensor(features.reshape(1, -1))[0]
        ).distribution.probs.detach().numpy()[0]
        
        # Sell 推論
        sell_features = np.concatenate([features, [1.0]])
        sell_action_b, _ = sell_model_b.predict(sell_features, deterministic=True)
        
        results['strategy_b'] = {
            'name': 'Conservative (MDD -6%)',
            'buy_action': int(buy_action_b),
            'buy_signal': 'BUY' if buy_action_b == 1 else 'HOLD',
            'buy_confidence': float(buy_probs_b[1]) if buy_action_b == 1 else float(buy_probs_b[0]),
            'sell_action': int(sell_action_b),
            'sell_signal': 'SELL' if sell_action_b == 1 else 'HOLD',
        }
        print(f"  Buy: {results['strategy_b']['buy_signal']} (Conf: {results['strategy_b']['buy_confidence']:.2%})")
        print(f"  Sell: {results['strategy_b']['sell_signal']}")
    else:
        print(f"  [Warning] 找不到模型: {buy_b_path}")
        results['strategy_b'] = {'name': 'Conservative', 'error': 'Model not found'}
    
    return results


# =============================================================================
# Step 4: 輸出戰情儀表板與日誌
# =============================================================================
def generate_report(workspace: dict, df: pd.DataFrame, inference_results: dict, date_str: str):
    """
    輸出戰情儀表板並儲存日誌
    
    Args:
        workspace: 當日工作區路徑字典
        df: 特徵 DataFrame
        inference_results: 推論結果字典
        date_str: 日期字串
    """
    print("\n" + "=" * 60)
    print("📊 Step 4: 戰情儀表板")
    print("=" * 60)
    
    latest = df.iloc[-1]
    
    # 市場數據
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(f"  Hybrid Trading System - Daily Report")
    report_lines.append(f"  日期: {date_str}")
    report_lines.append(f"  報告生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 60)
    
    report_lines.append("\n📈 市場數據 (^TWII)")
    report_lines.append("-" * 40)
    report_lines.append(f"  收盤價:     {latest['Close']:.2f}")
    report_lines.append(f"  最高價:     {latest['High']:.2f}")
    report_lines.append(f"  最低價:     {latest['Low']:.2f}")
    report_lines.append(f"  成交量:     {latest['Volume']:,.0f}")
    
    report_lines.append("\n📊 技術指標")
    report_lines.append("-" * 40)
    report_lines.append(f"  RSI (14):   {latest.get('RSI', 0) * 100:.1f}")
    report_lines.append(f"  MFI (14):   {latest.get('MFI', 0) * 100:.1f}")
    report_lines.append(f"  ATR:        {latest.get('ATR', 0):.2f}")
    report_lines.append(f"  DC 位置:    {latest.get('DC_Position', 0):.2%}")
    
    report_lines.append("\n🤖 LSTM 預測")
    report_lines.append("-" * 40)
    report_lines.append(f"  T+1 預測漲幅:  {latest.get('LSTM_Pred_1d', 0) * 100:+.2f}%")
    report_lines.append(f"  T+5 預測漲幅:  {latest.get('LSTM_Pred_5d', 0) * 100:+.2f}%")
    report_lines.append(f"  T+5 信心度:    {latest.get('LSTM_Conf_5d', 0.5) * 100:.1f}%")
    
    report_lines.append("\n🎯 策略建議")
    report_lines.append("-" * 40)
    
    # Strategy A
    if 'strategy_a' in inference_results and 'error' not in inference_results['strategy_a']:
        sa = inference_results['strategy_a']
        report_lines.append(f"\n  【策略 A: {sa['name']}】")
        report_lines.append(f"    買入訊號: {sa['buy_signal']} (信心度: {sa['buy_confidence']:.1%})")
        report_lines.append(f"    賣出訊號: {sa['sell_signal']}")
    else:
        report_lines.append("\n  【策略 A: 無法載入】")
    
    # Strategy B
    if 'strategy_b' in inference_results and 'error' not in inference_results['strategy_b']:
        sb = inference_results['strategy_b']
        report_lines.append(f"\n  【策略 B: {sb['name']}】")
        report_lines.append(f"    買入訊號: {sb['buy_signal']} (信心度: {sb['buy_confidence']:.1%})")
        report_lines.append(f"    賣出訊號: {sb['sell_signal']}")
    else:
        report_lines.append("\n  【策略 B: 無法載入】")
    
    report_lines.append("\n" + "=" * 60)
    
    # 輸出到終端機
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # 儲存到檔案
    report_path = os.path.join(workspace['reports'], 'summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n[Report] 報告已儲存: {report_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    """主程式進入點"""
    
    print("\n" + "=" * 70)
    print("  🚀 Daily Operations with Dual Strategy & Versioning")
    print("=" * 70)
    
    # 取得今天日期
    today = datetime.now()
    date_str = today.strftime('%Y-%m-%d')
    
    # 如果是週末，使用上一個交易日
    if today.weekday() == 5:  # Saturday
        today = today - timedelta(days=1)
        date_str = today.strftime('%Y-%m-%d')
        print(f"[Info] 今天是週六，使用週五日期: {date_str}")
    elif today.weekday() == 6:  # Sunday
        today = today - timedelta(days=2)
        date_str = today.strftime('%Y-%m-%d')
        print(f"[Info] 今天是週日，使用週五日期: {date_str}")
    
    print(f"\n📅 執行日期: {date_str}")
    
    # Step 0: 建立當日工作區
    workspace = create_daily_workspace(date_str)
    
    # Step 1: LSTM 訓練與封存
    train_and_archive_lstm(workspace, date_str)
    
    # Step 2: 隔離式特徵工程
    df = isolated_feature_engineering(workspace, date_str)
    
    # Step 3: 雙模型推論
    inference_results = dual_inference(workspace, df)
    
    # Step 4: 輸出報告
    generate_report(workspace, df, inference_results, date_str)
    
    print("\n" + "=" * 70)
    print("  ✅ Daily Operations 完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
