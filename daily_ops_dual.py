# -*- coding: utf-8 -*-
"""
================================================================================
Daily Operations with Dual Strategy & Versioning (v2 - Fixed)
================================================================================
每日維運腳本 - 雙策略推論與版本控管

修正重點：
1. 引用主系統 (ptrl_hybrid_system) 確保特徵工程一致性
2. 透過模型注入 (Model Injection) 強制使用當日訓練的 LSTM 模型
3. 使用 subprocess 執行 LSTM 訓練以釋放 GPU 記憶體

功能：
1. 建立當日專屬工作區 (daily_runs/{date}/)
2. LSTM 全量重訓與封存 (subprocess)
3. 隔離式特徵工程 (模型注入 + 主系統計算)
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
import subprocess
import json
import glob
from datetime import datetime, timedelta

# 設定 UTF-8 輸出
sys.stdout.reconfigure(encoding='utf-8')

# 抑制 TensorFlow 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import yfinance as yf

# =============================================================================
# 引用主系統 (關鍵修正)
# =============================================================================
import ptrl_hybrid_system as core_system

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
        'lstm_5d': os.path.join(daily_path, 'lstm_models', 'saved_models_5d'),
        'lstm_1d': os.path.join(daily_path, 'lstm_models', 'saved_models_multivariate'),
        'cache': os.path.join(daily_path, 'cache'),
        'reports': os.path.join(daily_path, 'reports'),
    }
    
    for key, path in paths.items():
        os.makedirs(path, exist_ok=True)
    
    print(f"[Workspace] 建立當日工作區: {daily_path}")
    return paths


# =============================================================================
# Step 1: LSTM 全量重訓與封存 (使用 subprocess)
# =============================================================================
def train_and_archive_lstm(workspace: dict, end_date: str):
    """
    使用 subprocess 訓練 LSTM 模型並封存到當日工作區
    
    使用 subprocess 的好處：
    - 訓練結束後自動釋放 GPU 記憶體
    - 避免訓練過程中的記憶體洩漏影響後續推論
    
    Args:
        workspace: 當日工作區路徑字典
        end_date: 訓練結束日期 (YYYY-MM-DD)
    """
    print("\n" + "=" * 60)
    print("📚 Step 1: LSTM 全量重訓與封存 (subprocess)")
    print("=" * 60)
    
    # =========================================================================
    # 使用 subprocess 執行訓練腳本
    # =========================================================================
    train_script = os.path.join(PROJECT_PATH, 'train_lstm_models.py')
    
    if os.path.exists(train_script):
        print(f"\n[Training] 執行 LSTM 訓練腳本...")
        print(f"[Training] 結束日期: {end_date}")
        
        try:
            # 執行訓練腳本 (在獨立進程中)
            result = subprocess.run(
                [sys.executable, train_script],
                cwd=PROJECT_PATH,
                capture_output=True,
                text=True,
                timeout=600  # 10 分鐘超時
            )
            
            if result.returncode == 0:
                print("[Training] ✅ LSTM 訓練完成")
            else:
                print(f"[Training] ⚠️ 訓練腳本返回非零代碼: {result.returncode}")
                if result.stderr:
                    print(f"[Training] stderr: {result.stderr[:500]}")
                    
        except subprocess.TimeoutExpired:
            print("[Training] ⚠️ 訓練超時 (10 分鐘)")
        except Exception as e:
            print(f"[Training] ⚠️ 執行訓練腳本失敗: {e}")
    else:
        print(f"[Warning] 找不到訓練腳本: {train_script}")
        print("[Warning] 將使用現有模型...")
    
    # =========================================================================
    # 封存模型到當日工作區
    # =========================================================================
    print("\n[Archive] 封存模型到當日工作區...")
    
    # 複製 T+5 模型
    if os.path.exists(DEFAULT_LSTM_5D_PATH):
        dest_dir = workspace['lstm_5d']
        
        # 複製所有模型檔案
        for file_pattern in ['*.keras', '*.pkl', '*.json', '*.png']:
            for src_file in glob.glob(os.path.join(DEFAULT_LSTM_5D_PATH, file_pattern)):
                dest_file = os.path.join(dest_dir, os.path.basename(src_file))
                shutil.copy2(src_file, dest_file)
        
        print(f"  ✅ T+5 模型已封存: {dest_dir}")
    else:
        print(f"  ⚠️ 找不到 T+5 模型: {DEFAULT_LSTM_5D_PATH}")
    
    # 複製 T+1 模型
    if os.path.exists(DEFAULT_LSTM_1D_PATH):
        dest_dir = workspace['lstm_1d']
        
        for file_pattern in ['*.keras', '*.pkl', '*.json', '*.png']:
            for src_file in glob.glob(os.path.join(DEFAULT_LSTM_1D_PATH, file_pattern)):
                dest_file = os.path.join(dest_dir, os.path.basename(src_file))
                shutil.copy2(src_file, dest_file)
        
        print(f"  ✅ T+1 模型已封存: {dest_dir}")
    else:
        print(f"  ⚠️ 找不到 T+1 模型: {DEFAULT_LSTM_1D_PATH}")
    
    return True


# =============================================================================
# Step 2: 隔離式特徵工程 (模型注入 + 主系統計算)
# =============================================================================
def isolated_feature_engineering(workspace: dict, end_date: str) -> pd.DataFrame:
    """
    使用當日封存的 LSTM 模型進行特徵工程
    
    關鍵修正：
    1. 從當日工作區載入 LSTM 模型
    2. 透過模型注入 (Monkey Patching) 覆蓋 core_system._LSTM_MODELS
    3. 呼叫 core_system.calculate_features() 確保特徵計算一致
    
    Args:
        workspace: 當日工作區路徑字典
        end_date: 數據結束日期
    
    Returns:
        包含所有特徵的 DataFrame
    """
    print("\n" + "=" * 60)
    print("🔧 Step 2: 隔離式特徵工程 (模型注入)")
    print("=" * 60)
    
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    
    # =========================================================================
    # 定義 SelfAttention 層 (與訓練時相同)
    # =========================================================================
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
    # 從當日工作區載入 LSTM 模型
    # =========================================================================
    print("\n[Model Injection] 載入當日封存的 LSTM 模型...")
    
    model_5d, scaler_5d, meta_5d = None, None, None
    model_1d, scaler_1d, meta_1d = None, None, None
    
    # 載入 T+5 模型
    lstm_5d_path = workspace['lstm_5d']
    keras_files_5d = glob.glob(os.path.join(lstm_5d_path, "*.keras"))
    
    if keras_files_5d:
        latest_keras = sorted(keras_files_5d)[-1]
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
        
        print(f"  ✅ T+5 模型: {os.path.basename(latest_keras)}")
    else:
        print(f"  ⚠️ 找不到 T+5 模型檔案: {lstm_5d_path}")
    
    # 載入 T+1 模型
    lstm_1d_path = workspace['lstm_1d']
    keras_files_1d = glob.glob(os.path.join(lstm_1d_path, "*.keras"))
    
    if keras_files_1d:
        latest_keras = sorted(keras_files_1d)[-1]
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
        
        print(f"  ✅ T+1 模型: {os.path.basename(latest_keras)}")
    else:
        print(f"  ⚠️ 找不到 T+1 模型檔案: {lstm_1d_path}")
    
    # =========================================================================
    # 模型注入 (Monkey Patching core_system._LSTM_MODELS)
    # =========================================================================
    print("\n[Model Injection] 注入模型到主系統...")
    
    # 確保 _LSTM_MODELS 字典存在
    if not hasattr(core_system, '_LSTM_MODELS'):
        core_system._LSTM_MODELS = {}
    
    # 注入 T+5 模型
    core_system._LSTM_MODELS['model_5d'] = model_5d
    core_system._LSTM_MODELS['scaler_feat_5d'] = scaler_5d
    core_system._LSTM_MODELS['meta_5d'] = meta_5d
    
    # 注入 T+1 模型
    core_system._LSTM_MODELS['model_1d'] = model_1d
    core_system._LSTM_MODELS['scaler_feat_1d'] = scaler_1d
    core_system._LSTM_MODELS['meta_1d'] = meta_1d
    
    # 標記為已載入
    core_system._LSTM_MODELS['loaded'] = True
    
    print("  ✅ 模型注入完成")
    
    # =========================================================================
    # 下載最新數據
    # =========================================================================
    print("\n[Data] 下載 ^TWII 數據...")
    
    # 下載足夠長的歷史數據以計算所有指標
    raw_df = yf.download("^TWII", start="2020-01-01", end=end_date, auto_adjust=True, progress=False)
    
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
    
    print(f"[Data] 數據範圍: {raw_df.index[0].strftime('%Y-%m-%d')} ~ {raw_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"[Data] 總筆數: {len(raw_df)}")
    
    # =========================================================================
    # 使用主系統計算特徵 (關鍵修正)
    # =========================================================================
    print("\n[Features] 使用主系統計算特徵 (確保一致性)...")
    
    # 呼叫主系統的 calculate_features 函數
    # 這確保所有指標計算邏輯與訓練時 100% 一致
    try:
        df = core_system.calculate_features(
            df=raw_df.copy(),
            benchmark_df=raw_df.copy(),  # 使用自身作為 benchmark
            ticker="^TWII",
            use_cache=False  # 不使用快取，確保重新計算
        )
        print(f"[Features] ✅ 特徵計算完成，總欄位數: {len(df.columns)}")
    except Exception as e:
        print(f"[Features] ⚠️ 特徵計算失敗: {e}")
        print("[Features] 嘗試使用簡化特徵...")
        df = raw_df.copy()
    
    # =========================================================================
    # 儲存快取到當日工作區
    # =========================================================================
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
    
    # 使用主系統定義的特徵欄位 (確保一致性)
    FEATURE_COLS = core_system.FEATURE_COLS
    
    # 取得最新一筆數據
    latest = df.iloc[-1]
    
    # 準備特徵向量
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    
    if len(available_cols) < len(FEATURE_COLS):
        missing = set(FEATURE_COLS) - set(available_cols)
        print(f"[Warning] 缺少特徵欄位: {missing}")
    
    features = latest[available_cols].values.astype(np.float32)
    
    # 補齊缺失的欄位 (填充 0)
    if len(features) < len(FEATURE_COLS):
        features = np.pad(features, (0, len(FEATURE_COLS) - len(features)), mode='constant', constant_values=0)
    
    # 處理 NaN
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    results = {}
    
    # =========================================================================
    # Strategy A: Aggressive (ROI 85%)
    # =========================================================================
    print("\n[Strategy A] Aggressive (ROI 85%)...")
    
    buy_a_path = os.path.join(STRATEGY_A_PATH, 'ppo_buy_twii_final.zip')
    sell_a_path = os.path.join(STRATEGY_A_PATH, 'ppo_sell_twii_final.zip')
    
    if os.path.exists(buy_a_path) and os.path.exists(sell_a_path):
        try:
            buy_model_a = PPO.load(buy_a_path)
            sell_model_a = PPO.load(sell_a_path)
            
            # Buy 推論
            buy_action_a, _ = buy_model_a.predict(features, deterministic=True)
            
            # 計算信心度
            try:
                obs_tensor = buy_model_a.policy.obs_to_tensor(features.reshape(1, -1))[0]
                buy_probs_a = buy_model_a.policy.get_distribution(obs_tensor).distribution.probs.detach().numpy()[0]
                buy_confidence = float(buy_probs_a[1]) if buy_action_a == 1 else float(buy_probs_a[0])
            except:
                buy_confidence = 0.5
            
            # Sell 推論 (需要加入持有報酬)
            sell_features = np.concatenate([features, [1.0]])  # 假設持有報酬 0%
            sell_action_a, _ = sell_model_a.predict(sell_features, deterministic=True)
            
            results['strategy_a'] = {
                'name': 'Aggressive (ROI 85%)',
                'buy_action': int(buy_action_a),
                'buy_signal': 'BUY' if buy_action_a == 1 else 'HOLD',
                'buy_confidence': buy_confidence,
                'sell_action': int(sell_action_a),
                'sell_signal': 'SELL' if sell_action_a == 1 else 'HOLD',
            }
            print(f"  Buy: {results['strategy_a']['buy_signal']} (Conf: {results['strategy_a']['buy_confidence']:.2%})")
            print(f"  Sell: {results['strategy_a']['sell_signal']}")
            
        except Exception as e:
            print(f"  [Error] 推論失敗: {e}")
            results['strategy_a'] = {'name': 'Aggressive', 'error': str(e)}
    else:
        print(f"  [Warning] 找不到模型")
        results['strategy_a'] = {'name': 'Aggressive', 'error': 'Model not found'}
    
    # =========================================================================
    # Strategy B: Conservative (MDD -6%)
    # =========================================================================
    print("\n[Strategy B] Conservative (MDD -6%)...")
    
    buy_b_path = os.path.join(STRATEGY_B_PATH, 'ppo_buy_twii_final.zip')
    sell_b_path = os.path.join(STRATEGY_B_PATH, 'ppo_sell_twii_final.zip')
    
    if os.path.exists(buy_b_path) and os.path.exists(sell_b_path):
        try:
            buy_model_b = PPO.load(buy_b_path)
            sell_model_b = PPO.load(sell_b_path)
            
            # Buy 推論
            buy_action_b, _ = buy_model_b.predict(features, deterministic=True)
            
            try:
                obs_tensor = buy_model_b.policy.obs_to_tensor(features.reshape(1, -1))[0]
                buy_probs_b = buy_model_b.policy.get_distribution(obs_tensor).distribution.probs.detach().numpy()[0]
                buy_confidence = float(buy_probs_b[1]) if buy_action_b == 1 else float(buy_probs_b[0])
            except:
                buy_confidence = 0.5
            
            # Sell 推論
            sell_features = np.concatenate([features, [1.0]])
            sell_action_b, _ = sell_model_b.predict(sell_features, deterministic=True)
            
            results['strategy_b'] = {
                'name': 'Conservative (MDD -6%)',
                'buy_action': int(buy_action_b),
                'buy_signal': 'BUY' if buy_action_b == 1 else 'HOLD',
                'buy_confidence': buy_confidence,
                'sell_action': int(sell_action_b),
                'sell_signal': 'SELL' if sell_action_b == 1 else 'HOLD',
            }
            print(f"  Buy: {results['strategy_b']['buy_signal']} (Conf: {results['strategy_b']['buy_confidence']:.2%})")
            print(f"  Sell: {results['strategy_b']['sell_signal']}")
            
        except Exception as e:
            print(f"  [Error] 推論失敗: {e}")
            results['strategy_b'] = {'name': 'Conservative', 'error': str(e)}
    else:
        print(f"  [Warning] 找不到模型")
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
    report_lines.append(f"  收盤價:     {latest.get('Close', 0):.2f}")
    report_lines.append(f"  最高價:     {latest.get('High', 0):.2f}")
    report_lines.append(f"  最低價:     {latest.get('Low', 0):.2f}")
    report_lines.append(f"  成交量:     {latest.get('Volume', 0):,.0f}")
    
    report_lines.append("\n📊 技術指標")
    report_lines.append("-" * 40)
    rsi_val = latest.get('RSI', 0)
    mfi_val = latest.get('MFI', 0)
    atr_val = latest.get('ATR', 0)
    dc_val = latest.get('DC_Position', 0)
    
    # RSI/MFI 可能已經是 0-1 或 0-100，統一顯示
    rsi_display = rsi_val * 100 if rsi_val <= 1 else rsi_val
    mfi_display = mfi_val * 100 if mfi_val <= 1 else mfi_val
    
    report_lines.append(f"  RSI (14):   {rsi_display:.1f}")
    report_lines.append(f"  MFI (14):   {mfi_display:.1f}")
    report_lines.append(f"  ATR:        {atr_val:.2f}")
    report_lines.append(f"  DC 位置:    {dc_val:.2%}")
    
    report_lines.append("\n🤖 LSTM 預測")
    report_lines.append("-" * 40)
    lstm_1d = latest.get('LSTM_Pred_1d', 0)
    lstm_5d = latest.get('LSTM_Pred_5d', 0)
    lstm_conf = latest.get('LSTM_Conf_5d', 0.5)
    
    report_lines.append(f"  T+1 預測漲幅:  {lstm_1d * 100:+.2f}%")
    report_lines.append(f"  T+5 預測漲幅:  {lstm_5d * 100:+.2f}%")
    report_lines.append(f"  T+5 信心度:    {lstm_conf * 100:.1f}%")
    
    report_lines.append("\n🎯 策略建議")
    report_lines.append("-" * 40)
    
    # Strategy A
    if 'strategy_a' in inference_results and 'error' not in inference_results['strategy_a']:
        sa = inference_results['strategy_a']
        report_lines.append(f"\n  【策略 A: {sa['name']}】")
        report_lines.append(f"    買入訊號: {sa['buy_signal']} (信心度: {sa['buy_confidence']:.1%})")
        report_lines.append(f"    賣出訊號: {sa['sell_signal']}")
    else:
        error_msg = inference_results.get('strategy_a', {}).get('error', '未知錯誤')
        report_lines.append(f"\n  【策略 A: 無法載入 ({error_msg})】")
    
    # Strategy B
    if 'strategy_b' in inference_results and 'error' not in inference_results['strategy_b']:
        sb = inference_results['strategy_b']
        report_lines.append(f"\n  【策略 B: {sb['name']}】")
        report_lines.append(f"    買入訊號: {sb['buy_signal']} (信心度: {sb['buy_confidence']:.1%})")
        report_lines.append(f"    賣出訊號: {sb['sell_signal']}")
    else:
        error_msg = inference_results.get('strategy_b', {}).get('error', '未知錯誤')
        report_lines.append(f"\n  【策略 B: 無法載入 ({error_msg})】")
    
    report_lines.append("\n" + "=" * 60)
    report_lines.append("  工作區路徑: " + workspace['root'])
    report_lines.append("=" * 60)
    
    # 輸出到終端機
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # 儲存到檔案
    report_path = os.path.join(workspace['reports'], 'summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # 同時儲存 JSON 格式 (方便程式讀取)
    json_path = os.path.join(workspace['reports'], 'summary.json')
    json_data = {
        'date': date_str,
        'generated_at': datetime.now().isoformat(),
        'market_data': {
            'close': float(latest.get('Close', 0)),
            'high': float(latest.get('High', 0)),
            'low': float(latest.get('Low', 0)),
            'volume': float(latest.get('Volume', 0)),
        },
        'indicators': {
            'rsi': float(rsi_display),
            'mfi': float(mfi_display),
            'atr': float(atr_val),
            'dc_position': float(dc_val),
        },
        'lstm_predictions': {
            'pred_1d': float(lstm_1d),
            'pred_5d': float(lstm_5d),
            'conf_5d': float(lstm_conf),
        },
        'strategies': inference_results,
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Report] 報告已儲存:")
    print(f"  - TXT: {report_path}")
    print(f"  - JSON: {json_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    """主程式進入點"""
    
    print("\n" + "=" * 70)
    print("  🚀 Daily Operations with Dual Strategy & Versioning (v2)")
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
    
    # Step 2: 隔離式特徵工程 (模型注入)
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
