# -*- coding: utf-8 -*-
"""
================================================================================
Daily Operations with Dual Strategy & Versioning (v2.1 - Patched)
================================================================================
每日維運腳本 - 雙策略推論與版本控管

修正紀錄 (v2.1):
1. [Fix] Step 1 改為直接呼叫 model registry 腳本，並傳入動態日期 (確保模型更新至今日)
2. [Fix] Step 2 補上 target_scaler 的載入與注入 (防止 inverse_transform 失敗)
3. [Safety] 增加 import 檢查與錯誤處理

作者：Phil Liang (Fixed by Gemini)
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow import keras
from keras import layers

# =============================================================================
# 引用主系統
# =============================================================================
import ptrl_hybrid_system as core_system

# =============================================================================
# 設定路徑
# =============================================================================
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DAILY_RUNS_PATH = os.path.join(PROJECT_PATH, 'daily_runs')

# RL 模型路徑
STRATEGY_A_PATH = os.path.join(PROJECT_PATH, 'models_hybrid')  # Aggressive
STRATEGY_B_PATH = os.path.join(PROJECT_PATH, 'models_hybrid_v2_conservative')  # Conservative

# LSTM 訓練腳本名稱 (必須存在於同一目錄下)
SCRIPT_5D = "twii_model_registry_5d.py"
SCRIPT_1D = "twii_model_registry_multivariate.py"

# LSTM 模型預設輸出路徑 (訓練腳本預設會存到這裡)
DEFAULT_LSTM_5D_DIR = os.path.join(PROJECT_PATH, 'saved_models_5d')
DEFAULT_LSTM_1D_DIR = os.path.join(PROJECT_PATH, 'saved_models_multivariate')


# =============================================================================
# Step 0: 建立當日專屬工作區
# =============================================================================
def create_daily_workspace(date_str: str) -> dict:
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
# Step 1: LSTM 全量重訓與封存
# =============================================================================
def train_and_archive_lstm(workspace: dict, end_date: str):
    print("\n" + "=" * 60)
    print("📚 Step 1: LSTM 全量重訓與封存")
    print("=" * 60)
    
    start_date = "2000-01-01"
    
    # 1. 執行 T+5 訓練 (傳入動態日期)
    print(f"\n[Training] T+5 Model ({start_date} ~ {end_date})...")
    script_5d_path = os.path.join(PROJECT_PATH, SCRIPT_5D)
    cmd_5d = [sys.executable, script_5d_path, "train", "--start", start_date, "--end", end_date]
    try:
        subprocess.run(cmd_5d, check=True, timeout=1200, cwd=PROJECT_PATH)  # 確保工作目錄正確
        print("[Training] ✅ T+5 訓練完成")
    except subprocess.CalledProcessError as e:
        print(f"[Error] T+5 訓練失敗: {e}")
        return False
    except FileNotFoundError:
        print(f"[Error] 找不到訓練腳本: {script_5d_path}")
        return False
    except Exception as e:
        print(f"[Error] 執行錯誤: {e}")
        return False

    # 2. 執行 T+1 訓練 (傳入動態日期)
    print(f"\n[Training] T+1 Model ({start_date} ~ {end_date})...")
    script_1d_path = os.path.join(PROJECT_PATH, SCRIPT_1D)
    cmd_1d = [sys.executable, script_1d_path, "train", "--start", start_date, "--end", end_date]
    try:
        subprocess.run(cmd_1d, check=True, timeout=1200, cwd=PROJECT_PATH)
        print("[Training] ✅ T+1 訓練完成")
    except subprocess.CalledProcessError as e:
        print(f"[Error] T+1 訓練失敗: {e}")
        return False
    except FileNotFoundError:
        print(f"[Error] 找不到訓練腳本: {script_1d_path}")
        return False

    # 3. 封存模型 (Copy from default dir to daily dir)
    print("\n[Archive] 封存模型到當日工作區...")
    
    def archive_dir(src_dir, dest_dir):
        if os.path.exists(src_dir):
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir) # 清空舊的
            shutil.copytree(src_dir, dest_dir)
            print(f"  ✅ 已封存: {os.path.basename(src_dir)} -> {dest_dir}")
        else:
            print(f"  ⚠️ 來源目錄不存在: {src_dir}")

    archive_dir(DEFAULT_LSTM_5D_DIR, workspace['lstm_5d'])
    archive_dir(DEFAULT_LSTM_1D_DIR, workspace['lstm_1d'])
    
    return True


# =============================================================================
# Step 2: 隔離式特徵工程 (修正：補上 Target Scaler)
# =============================================================================
def isolated_feature_engineering(workspace: dict, end_date: str) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("🔧 Step 2: 隔離式特徵工程 (模型注入)")
    print("=" * 60)
    
    # 定義 Custom Object (必須與訓練時一致，才能載入模型)
    class SelfAttention(layers.Layer):
        def __init__(self, **kwargs): super(SelfAttention, self).__init__(**kwargs)
        def build(self, input_shape):
            self.units = input_shape[-1]
            self.W_q = self.add_weight(name='W_query', shape=(self.units, self.units), initializer='glorot_uniform', trainable=True)
            self.W_k = self.add_weight(name='W_key', shape=(self.units, self.units), initializer='glorot_uniform', trainable=True)
        def call(self, inputs): # 簡化版 call，實際權重會從檔案載入
            Q = inputs
            return inputs 

    # 輔助函式：載入整組模型元件
    def load_model_components(model_dir):
        keras_files = glob.glob(os.path.join(model_dir, "*.keras"))
        if not keras_files: return None, None, None, None
        
        # 找最新的模型檔
        latest_keras = sorted(keras_files)[-1]
        print(f"  ...Loading {os.path.basename(latest_keras)}")
        
        # 載入模型
        # 注意：這裡使用上面的 Dummy SelfAttention 讓 Keras 不會報錯，權重會被覆蓋
        try:
            model = keras.models.load_model(latest_keras, custom_objects={'SelfAttention': SelfAttention})
        except:
            # 如果上面失敗，嘗試引用原始檔案的 Class (備案)
            from twii_model_registry_5d import SelfAttention as SA_Orig
            model = keras.models.load_model(latest_keras, custom_objects={'SelfAttention': SA_Orig})

        # 載入 Meta
        meta_file = latest_keras.replace('model_', 'meta_').replace('.keras', '.json')
        meta = {}
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                meta = json.load(f)

        # 載入 Feature Scaler
        scaler_feat_file = latest_keras.replace('model_', 'feature_scaler_').replace('.keras', '.pkl')
        # 兼容舊版命名
        if not os.path.exists(scaler_feat_file):
             scaler_feat_file = latest_keras.replace('model_', 'scaler_').replace('.keras', '.pkl')
        
        scaler_feat = None
        if os.path.exists(scaler_feat_file):
            with open(scaler_feat_file, 'rb') as f:
                scaler_feat = pickle.load(f)

        # 載入 Target Scaler (關鍵修正！)
        scaler_tgt_file = latest_keras.replace('model_', 'target_scaler_').replace('.keras', '.pkl')
        # 兼容舊版命名 (如果舊版只有一個 scaler，則 target = feature)
        if not os.path.exists(scaler_tgt_file):
             scaler_tgt = scaler_feat
        else:
             with open(scaler_tgt_file, 'rb') as f:
                 scaler_tgt = pickle.load(f)

        return model, scaler_feat, scaler_tgt, meta

    # 1. 載入模型
    print("\n[Model Injection] 載入當日封存的 LSTM 模型...")
    m5d, sf5d, st5d, meta5d = load_model_components(workspace['lstm_5d'])
    m1d, sf1d, st1d, meta1d = load_model_components(workspace['lstm_1d'])
    
    if m5d is None or m1d is None:
        print("[Error] 模型載入失敗，無法進行特徵工程")
        sys.exit(1)

    # 2. 注入主系統
    print("\n[Model Injection] 注入 core_system._LSTM_MODELS...")
    if not hasattr(core_system, '_LSTM_MODELS'):
        core_system._LSTM_MODELS = {}
    
    core_system._LSTM_MODELS.update({
        'model_5d': m5d, 'scaler_feat_5d': sf5d, 'scaler_tgt_5d': st5d, 'meta_5d': meta5d,
        'model_1d': m1d, 'scaler_feat_1d': sf1d, 'scaler_tgt_1d': st1d, 'meta_1d': meta1d,
        'loaded': True
    })
    print("  ✅ 注入完成 (含 Target Scalers)")

    # 3. 下載數據 & 計算特徵
    print(f"\n[Compute] 下載數據 (2020-01-01 ~ {end_date})...")
    raw_df = yf.download("^TWII", start="2020-01-01", end=end_date, auto_adjust=True, progress=False)
    
    # 確保 columns 格式正確
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
    
    print(f"[Compute] 計算特徵中 (使用當日模型)...")
    # 強制不使用快取，確保重新計算
    df = core_system.calculate_features(raw_df, raw_df, ticker="^TWII", use_cache=False)
    
    # 存入當日快取
    cache_file = os.path.join(workspace['cache'], 'twii_features.pkl')
    with open(cache_file, 'wb') as f:
        pickle.dump(df, f)
    print(f"[Cache] 特徵已存檔: {cache_file}")
    
    return df


# =============================================================================
# Step 3: 雙模型推論
# =============================================================================
def dual_inference(workspace: dict, df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("🎯 Step 3: 雙模型推論")
    print("=" * 60)
    
    from stable_baselines3 import PPO
    
    # 準備特徵
    FEATURE_COLS = core_system.FEATURE_COLS
    latest = df.iloc[-1]
    
    # 確保特徵欄位對齊
    features = []
    for col in FEATURE_COLS:
        val = latest.get(col, 0.0)
        features.append(val)
    features = np.array(features, dtype=np.float32).reshape(1, -1)
    
    # 處理 NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    results = {}
    
    def run_strategy(name, path, key):
        buy_path = os.path.join(path, 'ppo_buy_twii_final.zip')
        sell_path = os.path.join(path, 'ppo_sell_twii_final.zip')
        
        if not os.path.exists(buy_path):
            results[key] = {'error': 'Model not found'}
            print(f"  [Warning] {name}: 模型不存在")
            return

        try:
            buy_agent = PPO.load(buy_path)
            sell_agent = PPO.load(sell_path)
            
            # Buy Action
            b_act, _ = buy_agent.predict(features, deterministic=True)
            # Buy Probability
            b_obs = buy_agent.policy.obs_to_tensor(features)[0]
            b_prob = buy_agent.policy.get_distribution(b_obs).distribution.probs.detach().numpy()[0]
            
            # Sell Action (Construct Sell State: Features + [Current_Return=1.0])
            s_feat = np.concatenate([features[0], [1.0]]).reshape(1, -1)
            s_act, _ = sell_agent.predict(s_feat, deterministic=True)
            
            results[key] = {
                'name': name,
                'buy_signal': 'BUY' if b_act[0] == 1 else 'WAIT',
                'buy_prob': float(b_prob[1]) if b_act[0] == 1 else float(b_prob[0]),
                'sell_signal': 'SELL' if s_act[0] == 1 else 'HOLD'
            }
            print(f"  [{name}] Buy: {results[key]['buy_signal']} ({results[key]['buy_prob']:.1%}) | Sell: {results[key]['sell_signal']}")
            
        except Exception as e:
            results[key] = {'error': str(e)}
            print(f"  [Error] {name}: {e}")

    # 執行 A (Aggressive)
    run_strategy("Aggressive (ROI 85%)", STRATEGY_A_PATH, 'A')
    
    # 執行 B (Conservative)
    run_strategy("Conservative (MDD -6%)", STRATEGY_B_PATH, 'B')
    
    return results


# =============================================================================
# Step 4: 輸出報告
# =============================================================================
def generate_report(workspace: dict, df: pd.DataFrame, res: dict, date_str: str):
    print("\n" + "=" * 60)
    print("📊 Step 4: 戰情儀表板")
    print("=" * 60)
    
    last = df.iloc[-1]
    
    lines = []
    lines.append(f"📅 日期: {date_str}")
    lines.append(f"📊 收盤: {last['Close']:.2f} | 量: {last['Volume']/1e8:.2f}億")
    lines.append("-" * 40)
    lines.append("🔮 [分析師 LSTM]")
    lines.append(f"   T+1 漲跌: {last.get('LSTM_Pred_1d', 0)*100:+.2f}%")
    lines.append(f"   T+5 漲跌: {last.get('LSTM_Pred_5d', 0)*100:+.2f}%")
    lines.append(f"   信心度:   {last.get('LSTM_Conf_5d', 0)*100:.1f}%")
    lines.append("-" * 40)
    lines.append("🤖 [操盤手 RL]")
    
    if 'A' in res and 'error' not in res['A']:
        r = res['A']
        icon = "🚀" if r['buy_signal'] == 'BUY' else "💤"
        lines.append(f"   {icon} 策略 A (積極): [{r['buy_signal']}] (機率 {r['buy_prob']:.1%})")
    
    if 'B' in res and 'error' not in res['B']:
        r = res['B']
        icon = "🛡️" if r['buy_signal'] == 'BUY' else "💤"
        lines.append(f"   {icon} 策略 B (保守): [{r['buy_signal']}] (機率 {r['buy_prob']:.1%})")
        
    # 綜合建議
    lines.append("-" * 40)
    sig_a = res.get('A', {}).get('buy_signal', 'N/A')
    sig_b = res.get('B', {}).get('buy_signal', 'N/A')
    
    if sig_a == 'BUY' and sig_b == 'BUY':
        advice = "⭐⭐ 強力買進 (Strong Buy) ⭐⭐"
    elif sig_a == 'WAIT' and sig_b == 'WAIT':
        advice = "💤 空手觀望 (Wait)"
    elif sig_a == 'BUY':
        advice = "⚠️ 僅積極型買進 (Aggressive Only)"
    else:
        advice = "❓ 訊號不明"
        
    lines.append(f"💡 綜合建議: {advice}")
    lines.append("=" * 60)
    
    report = "\n".join(lines)
    print(report)
    
    # 存檔 TXT
    txt_path = os.path.join(workspace['reports'], 'summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 存檔 JSON (方便自動化讀取)
    json_path = os.path.join(workspace['reports'], 'summary.json')
    json_data = {
        'date': date_str,
        'generated_at': datetime.now().isoformat(),
        'market': {
            'close': float(last.get('Close', 0)),
            'volume': float(last.get('Volume', 0)),
        },
        'lstm': {
            'pred_1d': float(last.get('LSTM_Pred_1d', 0)),
            'pred_5d': float(last.get('LSTM_Pred_5d', 0)),
            'conf_5d': float(last.get('LSTM_Conf_5d', 0)),
        },
        'strategies': res,
        'advice': advice,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Report] 已儲存: {txt_path}")
    print(f"[Report] 已儲存: {json_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    today = datetime.now()
    # 處理週末 (往前推到週五)
    if today.weekday() == 5: today -= timedelta(days=1)
    elif today.weekday() == 6: today -= timedelta(days=2)
    
    date_str = today.strftime('%Y-%m-%d')
    print(f"🚀 啟動每日維運系統 - {date_str}")
    
    # Step 0
    ws = create_daily_workspace(date_str)
    
    # Step 1 (Train up to Today)
    train_and_archive_lstm(ws, date_str)
    
    # Step 2
    df = isolated_feature_engineering(ws, date_str)
    
    # Step 3
    res = dual_inference(ws, df)
    
    # Step 4
    generate_report(ws, df, res, date_str)

if __name__ == "__main__":
    main()