"""
#9 多步功率预测.py  （第三版：层1补充无SCADA对比）
================================================
利用当前时刻及历史时序数据预测未来多个时刻风机功率

研究背景
--------
第二版（v2）采用三层递进分层实验设计。本版（v3）在层1中补充了
"仅 LiDAR 无 SCADA"对比组（N1/N2/N3/N4），专门回答：
  "层1中 LiDAR+SCADA 效果不如纯 SCADA 的原因，是 LiDAR 信息不足，
   还是 LiDAR 与 SCADA 风速相互干扰所致？"

三层实验设计（第三版）
----------------------
层1 ── 风速距离结构实验

  含 SCADA 组（M 组）：
    M0  : 仅 SCADA 机舱风速 + 历史功率（无激光雷达基准）
    M1  : 单距离 LiDAR（逐距离 40, 60, ..., 300 m）+ SCADA + 历史功率
    M2  : 近距组（40~120m LiDAR HWS）+ SCADA + 历史功率
    M3  : 中距组（150~210m LiDAR HWS）+ SCADA + 历史功率
    M4  : 远距组（240~300m LiDAR HWS）+ SCADA + 历史功率
    M5  : 全距离（40~300m LiDAR HWS）+ SCADA + 历史功率

  无 SCADA 组（N 组，新增）：
    N1  : 单距离 LiDAR（逐距离 40, 60, ..., 300 m）+ 历史功率（不含SCADA风速）
    N2  : 近距组（40~120m LiDAR HWS）+ 历史功率（不含SCADA风速）
    N3  : 中距组（150~210m LiDAR HWS）+ 历史功率（不含SCADA风速）
    N4  : 远距组（240~300m LiDAR HWS）+ 历史功率（不含SCADA风速）
    注：全距离无SCADA（N5 概念）= 层3的 L3a，已涵盖。

层2 ── 气象特征增益实验（在 M5 基础上，回答"VShear/HShear/TI 在多步预测中是否有增益？"）
    E1  : M5 + VShear（10 个距离）
    E2  : M5 + HShear（10 个距离）
    E3  : M5 + TI_avg（10 个距离）
    E4  : M5 + VShear + HShear + TI_avg（全气象特征）
    注：M5 本身作为 E0 基准，结果直接引用层1的 M5_Lall+SCADA。

层3 ── 去 SCADA 实验（工程替代性验证：激光雷达能否完全替代机舱测风？）
    L3a : 仅全距离 LiDAR HWS + 历史功率（无 SCADA 风速）
    L3b : 仅全距离 LiDAR HWS + 全气象特征 + 历史功率（无 SCADA 风速）

建模策略
--------
- LightGBM / XGBoost：所有 34 个场景均运行，直接多步策略
  （每预测步长单独训练一个模型，输入为展平的历史特征矩阵）
- LSTM（双层多输出）：仅在 4 个关键场景运行（M0/M5/L3a/L3b），节省运行时间
  （一个模型同时预测所有 FORECAST_STEPS 步）

时间连续性约束
--------------
LOOK_BACK = 6 步（60 min 历史），FORECAST_STEPS = [1, 2, 3, 6]（+10~+60 min）
窗口仅在连续时间段内构建，不跨越 140+ 处时间戳间隙。

运行方式
--------
    cd 利用激光雷达数据辅助风机功率预测/
    python "CODE/#9多步功率预测.py"

依赖
----
    pip install pandas numpy scikit-learn lightgbm xgboost matplotlib tensorflow
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ──────────────────────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.dirname(SCRIPT_DIR)
DATASET_PATH = os.path.join(BASE_DIR, "PROCESS_DATA", "#7构建好的训练数据集.csv")
OUTPUT_CSV   = os.path.join(BASE_DIR, "PROCESS_DATA", "#9多步预测结果.csv")
OUTPUT_DIR   = os.path.join(BASE_DIR, "PROCESS_DATA")

# ──────────────────────────────────────────────────────────────
# 超参数
# ──────────────────────────────────────────────────────────────
LIDAR_DISTANCES = [40, 60, 90, 120, 150, 180, 210, 240, 270, 300]
_NEAR_DISTS     = [40, 60, 90, 120]
_MID_DISTS      = [150, 180, 210]
_FAR_DISTS      = [240, 270, 300]

LOOK_BACK      = 6                     # 历史回望步数：6 × 10min = 60 min
FORECAST_STEPS = [1, 2, 3, 6]         # 预测步长：+10, +20, +30, +60 min
MAX_H          = max(FORECAST_STEPS)   # 最大预测步长
TEST_RATIO     = 0.20                  # 测试集比例
VAL_RATIO      = 0.10                  # 验证集比例（从训练集末尾取）
RANDOM_STATE   = 42
TIME_GAP       = pd.Timedelta("10min")

# ──────────────────────────────────────────────────────────────
# 特征列组
# ──────────────────────────────────────────────────────────────
_all_hws    = [f"HWS_{d}m"    for d in LIDAR_DISTANCES]
_near_hws   = [f"HWS_{d}m"    for d in _NEAR_DISTS]
_mid_hws    = [f"HWS_{d}m"    for d in _MID_DISTS]
_far_hws    = [f"HWS_{d}m"    for d in _FAR_DISTS]
_all_vshear = [f"VShear_{d}m" for d in LIDAR_DISTANCES]
_all_hshear = [f"HShear_{d}m" for d in LIDAR_DISTANCES]
_all_ti     = [f"TI_avg_{d}m" for d in LIDAR_DISTANCES]
_all_met    = _all_vshear + _all_hshear + _all_ti   # 30 个气象特征列

# ──────────────────────────────────────────────────────────────
# 三层场景定义
# 格式：key -> (feat_cols, display_name, layer_number)
# ──────────────────────────────────────────────────────────────

# 层1：风速距离结构（15 个含SCADA场景 + 13 个无SCADA场景 = 28 个场景）
L1_SCENARIOS = {
    # ── 含 SCADA 组（M 组）──────────────────────────────────────
    "M0_SCADA": (
        ["HWS_scada", "power"],
        "M0: 仅SCADA（无LiDAR）", 1),
    **{f"M1_{d}m": (
        [f"HWS_{d}m", "HWS_scada", "power"],
        f"M1: LiDAR {d}m + SCADA", 1)
       for d in LIDAR_DISTANCES},
    "M2_near": (
        _near_hws + ["HWS_scada", "power"],
        "M2: 近距(40-120m) + SCADA", 1),
    "M3_mid": (
        _mid_hws + ["HWS_scada", "power"],
        "M3: 中距(150-210m) + SCADA", 1),
    "M4_far": (
        _far_hws + ["HWS_scada", "power"],
        "M4: 远距(240-300m) + SCADA", 1),
    "M5_Lall+SCADA": (
        _all_hws + ["HWS_scada", "power"],
        "M5: 全距离LiDAR + SCADA", 1),
    # ── 无 SCADA 组（N 组）：回答"LiDAR 与 SCADA 是否相互干扰？"────
    **{f"N1_{d}m_only": (
        [f"HWS_{d}m", "power"],
        f"N1: 仅LiDAR {d}m（无SCADA）", 1)
       for d in LIDAR_DISTANCES},
    "N2_near_only": (
        _near_hws + ["power"],
        "N2: 仅近距(40-120m)（无SCADA）", 1),
    "N3_mid_only": (
        _mid_hws + ["power"],
        "N3: 仅中距(150-210m)（无SCADA）", 1),
    "N4_far_only": (
        _far_hws + ["power"],
        "N4: 仅远距(240-300m)（无SCADA）", 1),
}

# 层2：气象特征增益（4 个场景，M5 结果作为 E0 基准直接引用）
L2_SCENARIOS = {
    "E1_AllHWS+SCADA+VShear": (
        _all_hws + ["HWS_scada"] + _all_vshear + ["power"],
        "E1: 全HWS+SCADA+VShear", 2),
    "E2_AllHWS+SCADA+HShear": (
        _all_hws + ["HWS_scada"] + _all_hshear + ["power"],
        "E2: 全HWS+SCADA+HShear", 2),
    "E3_AllHWS+SCADA+TI": (
        _all_hws + ["HWS_scada"] + _all_ti + ["power"],
        "E3: 全HWS+SCADA+TI", 2),
    "E4_AllHWS+SCADA+AllMet": (
        _all_hws + ["HWS_scada"] + _all_met + ["power"],
        "E4: 全HWS+SCADA+全气象", 2),
}

# 层3：去 SCADA 验证（2 个场景）
L3_SCENARIOS = {
    "L3a_Lall_only": (
        _all_hws + ["power"],
        "L3a: 仅全距离LiDAR HWS", 3),
    "L3b_Lall+met_only": (
        _all_hws + _all_met + ["power"],
        "L3b: 仅LiDAR HWS+全气象", 3),
}

ALL_SCENARIOS = {**L1_SCENARIOS, **L2_SCENARIOS, **L3_SCENARIOS}

# LSTM 仅在以下 4 个关键场景运行（21 个全跑耗时过长）
LSTM_SCENARIOS = {"M0_SCADA", "M5_Lall+SCADA", "L3a_Lall_only", "L3b_Lall+met_only"}


# ════════════════════════════════════════════════════════════════
# 1. 数据加载与预处理
# ════════════════════════════════════════════════════════════════

def load_wide_dataset():
    """
    读取 #7 数据集，转为宽格式（每行一个时间戳）。

    相比 v1 新增：VShear_{d}m、HShear_{d}m、TI_avg_{d}m（各距离气象特征）

    列说明
    ------
    HWS_scada      : Distance=0 的机舱风速 HWS(hub)
    power          : ACTIVE_POWER_#56_对齐_前10分钟均值（目标序列）
    HWS_{d}m       : 各距离激光雷达水平风速（d = 40,60,...,300m）
    VShear_{d}m    : 各距离垂直风切变
    HShear_{d}m    : 各距离水平风切变
    TI_avg_{d}m    : 各距离湍流强度均值（TI1~TI4 平均）
    segment_id     : 连续时间段编号（间隔 ≠ 10 min 则新段开始）
    """
    print("=" * 60)
    print("【数据加载与预处理】")
    print("=" * 60)

    df_raw = pd.read_csv(DATASET_PATH, encoding="gbk")
    df_raw["DateAndTime"] = pd.to_datetime(df_raw["DateAndTime"])
    print(f"  原始数据：{len(df_raw):,} 行，{df_raw['DateAndTime'].nunique():,} 个时间戳")

    # ── SCADA 行 (Distance=0) ─────────────────────────────────
    d0 = df_raw[df_raw["Distance"] == 0][[
        "DateAndTime", "HWS(hub)", "ACTIVE_POWER_#56_对齐_前10分钟均值"
    ]].rename(columns={
        "HWS(hub)": "HWS_scada",
        "ACTIVE_POWER_#56_对齐_前10分钟均值": "power",
    }).copy()

    # ── 各距离 LiDAR 行：HWS + VShear + HShear + TI_avg ───────
    df_wide = d0
    for dist in LIDAR_DISTANCES:
        sub = df_raw[df_raw["Distance"] == dist][[
            "DateAndTime", "HWS(hub)", "VShear", "HShear", "TI1", "TI2", "TI3", "TI4"
        ]].copy()
        sub[f"TI_avg_{dist}m"] = sub[["TI1", "TI2", "TI3", "TI4"]].mean(axis=1)
        sub = sub.rename(columns={
            "HWS(hub)": f"HWS_{dist}m",
            "VShear":   f"VShear_{dist}m",
            "HShear":   f"HShear_{dist}m",
        }).drop(columns=["TI1", "TI2", "TI3", "TI4"])
        df_wide = df_wide.merge(sub, on="DateAndTime", how="left")

    df_wide = df_wide.sort_values("DateAndTime").reset_index(drop=True)
    df_wide = df_wide.dropna(subset=["power"]).reset_index(drop=True)

    # ── 连续时间段标注 ─────────────────────────────────────────
    dt_diff = df_wide["DateAndTime"].diff()
    df_wide["segment_id"] = (dt_diff != TIME_GAP).cumsum()

    seg_sizes = df_wide.groupby("segment_id").size()
    n_usable  = int((seg_sizes >= LOOK_BACK + MAX_H).sum())
    print(f"  有效时间戳：{len(df_wide):,}")
    print(f"  特征列总数：{len(df_wide.columns)} "
          f"（包含 HWS × {len(LIDAR_DISTANCES)} + VShear × {len(LIDAR_DISTANCES)} + "
          f"HShear × {len(LIDAR_DISTANCES)} + TI_avg × {len(LIDAR_DISTANCES)}）")
    print(f"  连续段总数：{df_wide['segment_id'].nunique()}，"
          f"其中长度 ≥ {LOOK_BACK + MAX_H} 的可用段：{n_usable} 个")
    print(f"  时间范围：{df_wide['DateAndTime'].min()} ~ {df_wide['DateAndTime'].max()}")
    return df_wide


# ════════════════════════════════════════════════════════════════
# 2. 滑动窗口构建（时间连续性严格保证）
# ════════════════════════════════════════════════════════════════

def build_windows(df, feat_cols, look_back=LOOK_BACK, max_h=MAX_H):
    """
    在每个连续时间段内构建滑动窗口。

    窗口定义（以当前时刻索引 i 为锚点）
    ------------------------------------
      输入 X：feat_vals[i - look_back : i]  — 最近 look_back 步特征序列
      目标 Y：power_vals[i : i + max_h]      — 未来 max_h 步功率

    保证窗口内所有时间步均在同一连续段内，不跨越间隙。

    Returns
    -------
    X_seq  : (N, look_back, n_feat)   float32，用于 LSTM
    Y      : (N, max_h)               float32，全部预测目标步
    T_curr : (N,)                     datetime，窗口最后一个输入时刻
    """
    Xs, Ys, Ts = [], [], []

    for _, seg_df in df.groupby("segment_id", sort=True):
        seg_df = seg_df.sort_values("DateAndTime").reset_index(drop=True)
        n = len(seg_df)
        if n < look_back + max_h:
            continue

        # 中位数填充缺失值（分段计算，避免跨段污染）
        feats = seg_df[feat_cols].copy()
        for c in feat_cols:
            med = feats[c].median()
            feats[c] = feats[c].fillna(med if not np.isnan(med) else 0.0)

        feat_vals  = feats.values
        power_vals = seg_df["power"].values
        times      = seg_df["DateAndTime"].values

        for i in range(look_back, n - max_h + 1):
            future = power_vals[i:i + max_h]
            if np.any(np.isnan(future)):
                continue
            Xs.append(feat_vals[i - look_back:i])
            Ys.append(future)
            Ts.append(times[i - 1])

    if not Xs:
        return None, None, None

    return (
        np.array(Xs, dtype=np.float32),
        np.array(Ys, dtype=np.float32),
        np.array(Ts),
    )


def time_split_windows(X, Y, T, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO):
    """按时间顺序划分训练 / 验证 / 测试集（不打乱，保持时序）。"""
    n     = len(X)
    n_te  = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_tr  = n - n_te - n_val
    return (
        (X[:n_tr],             Y[:n_tr],             T[:n_tr]),
        (X[n_tr:n_tr + n_val], Y[n_tr:n_tr + n_val], T[n_tr:n_tr + n_val]),
        (X[n_tr + n_val:],     Y[n_tr + n_val:],     T[n_tr + n_val:]),
    )


def flat_feat_names(feat_cols, look_back):
    """返回展平后的特征名列表，便于 LightGBM 特征重要性解释。"""
    return [f"{c}_t-{look_back-1-k}"
            for k in range(look_back) for c in feat_cols]


# ════════════════════════════════════════════════════════════════
# 3. 通用工具
# ════════════════════════════════════════════════════════════════

def evaluate_step(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t, y_p = y_true[mask], y_pred[mask]
    if len(y_t) < 2:
        return {"RMSE": float("nan"), "MAE": float("nan"), "R2": float("nan"), "N": 0}
    return {
        "RMSE": round(float(np.sqrt(mean_squared_error(y_t, y_p))), 2),
        "MAE":  round(float(mean_absolute_error(y_t, y_p)), 2),
        "R2":   round(float(r2_score(y_t, y_p)), 4),
        "N":    int(len(y_t)),
    }


def try_lgb():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        print("  ⚠️  lightgbm 未安装，跳过 LightGBM")
        return None


def try_xgb():
    try:
        import xgboost as xgb
        return xgb
    except ImportError:
        print("  ⚠️  xgboost 未安装，跳过 XGBoost")
        return None


def try_tf():
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        return tf
    except ImportError:
        print("  ⚠️  tensorflow 未安装，跳过 LSTM")
        return None


def try_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        import subprocess

        try:
            subprocess.run(["apt-get", "install", "-y", "-q", "fonts-noto-cjk"],
                           capture_output=True, timeout=60)
        except Exception:
            pass

        # 直接 addfont，不重建缓存（_load_fontmanager 会重置状态导致字体失效）
        for fp in [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        ]:
            if os.path.exists(fp):
                fm.fontManager.addfont(fp)
                _p = fm.FontProperties(fname=fp)
                plt.rcParams["font.sans-serif"] = [_p.get_name(), "DejaVu Sans"]
                break
        plt.rcParams["axes.unicode_minus"] = False
        return plt
    except ImportError:
        return None


# ════════════════════════════════════════════════════════════════
# 4. 树模型：直接多步策略（Direct Multi-Step）
# ════════════════════════════════════════════════════════════════

def run_tree_scenario(df, feat_cols, s_key, s_name, layer):
    """
    对每个预测步长 h 单独训练一个 LightGBM/XGBoost 模型。
    输入特征：X_flat = (N, look_back × n_feat) — 展平的历史特征矩阵。
    各步模型相互独立，预测误差不会在步长间累积（Direct 策略）。
    """
    X, Y, T = build_windows(df, feat_cols)
    if X is None:
        print(f"  ⚠️  {s_name} 没有有效窗口，跳过")
        return []

    N, L, F = X.shape
    X_flat     = X.reshape(N, L * F)
    feat_names = flat_feat_names(feat_cols, L)

    (X_tr, Y_tr, _), (X_val, Y_val, _), (X_te, Y_te, _) = time_split_windows(
        X_flat, Y, T)

    print(f"  窗口总数：{N}，训练：{len(X_tr)}，验证：{len(X_val)}，测试：{len(X_te)}")

    lgb = try_lgb()
    xgb = try_xgb()
    results = []

    for h in FORECAST_STEPS:
        y_tr_h  = Y_tr[:, h - 1]
        y_val_h = Y_val[:, h - 1]
        y_te_h  = Y_te[:, h - 1]

        for model_name, lib in [("LightGBM", lgb), ("XGBoost", xgb)]:
            if lib is None:
                continue

            if model_name == "LightGBM":
                m = lib.LGBMRegressor(
                    n_estimators=300, learning_rate=0.05, num_leaves=31,
                    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                    random_state=RANDOM_STATE, verbose=-1,
                    feature_name=feat_names,
                )
                m.fit(X_tr, y_tr_h,
                      eval_set=[(X_val, y_val_h)],
                      callbacks=[lib.early_stopping(20, verbose=False),
                                 lib.log_evaluation(-1)])
            else:
                m = lib.XGBRegressor(
                    n_estimators=300, learning_rate=0.05, max_depth=5,
                    subsample=0.8, colsample_bytree=0.8, eval_metric="rmse",
                    early_stopping_rounds=20, random_state=RANDOM_STATE, verbosity=0,
                )
                m.fit(X_tr, y_tr_h,
                      eval_set=[(X_val, y_val_h)], verbose=False)

            pred = m.predict(X_te)
            r    = evaluate_step(y_te_h, pred)
            r.update({
                "scenario_key":  s_key,
                "scenario_name": s_name,
                "layer":         layer,
                "model":         model_name,
                "step":          h,
                "step_min":      h * 10,
            })
            results.append(r)
            print(f"    {model_name:<12} +{h*10:>3}min  "
                  f"RMSE={r['RMSE']:>7.1f}  MAE={r['MAE']:>7.1f}  R²={r['R2']:.4f}")

    return results


# ════════════════════════════════════════════════════════════════
# 5. LSTM 多输出策略
# ════════════════════════════════════════════════════════════════

def run_lstm_scenario(df, feat_cols, s_key, s_name, layer):
    """
    训练双层 LSTM 多输出模型：一次预测所有 FORECAST_STEPS 步。
    仅在 LSTM_SCENARIOS 中的关键场景调用。

    架构：Input(look_back, n_feat)
           → LSTM(64, return_sequences=True) → LSTM(32)
           → Dropout(0.2) → Dense(32, relu) → Dense(max_h)
    """
    tf = try_tf()
    if tf is None:
        return []

    X, Y, T = build_windows(df, feat_cols)
    if X is None or len(X) < 30:
        print(f"  ⚠️  数据不足，跳过 {s_name}")
        return []

    N, L, F = X.shape
    (X_tr, Y_tr, _), (X_val, Y_val, _), (X_te, Y_te, _) = time_split_windows(X, Y, T)

    scaler_x = StandardScaler()
    X_tr_s   = scaler_x.fit_transform(X_tr.reshape(-1, F)).reshape(len(X_tr), L, F)
    X_val_s  = scaler_x.transform(X_val.reshape(-1, F)).reshape(len(X_val), L, F)
    X_te_s   = scaler_x.transform(X_te.reshape(-1, F)).reshape(len(X_te), L, F)

    scaler_y = StandardScaler()
    Y_tr_s   = scaler_y.fit_transform(Y_tr)
    Y_val_s  = scaler_y.transform(Y_val)

    inp = tf.keras.Input(shape=(L, F))
    x   = tf.keras.layers.LSTM(64, return_sequences=True)(inp)
    x   = tf.keras.layers.LSTM(32)(x)
    x   = tf.keras.layers.Dropout(0.2)(x)
    x   = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(MAX_H)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

    model.fit(
        X_tr_s, Y_tr_s,
        validation_data=(X_val_s, Y_val_s),
        epochs=100, batch_size=64,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, verbose=0)],
        verbose=0,
    )

    pred_s = model.predict(X_te_s, verbose=0)
    pred   = scaler_y.inverse_transform(pred_s)

    results = []
    for h in FORECAST_STEPS:
        r = evaluate_step(Y_te[:, h - 1], pred[:, h - 1])
        r.update({
            "scenario_key":  s_key,
            "scenario_name": s_name,
            "layer":         layer,
            "model":         "LSTM",
            "step":          h,
            "step_min":      h * 10,
        })
        results.append(r)
        print(f"    LSTM        +{h*10:>3}min  "
              f"RMSE={r['RMSE']:>7.1f}  MAE={r['MAE']:>7.1f}  R²={r['R2']:.4f}")

    return results


# ════════════════════════════════════════════════════════════════
# 6. 全部实验流程
# ════════════════════════════════════════════════════════════════

def run_all_experiments(df):
    """
    按层依次运行所有场景：
    - 树模型：所有 21 个场景
    - LSTM：仅 LSTM_SCENARIOS（4 个关键场景）
    """
    all_results = []

    layer_labels = {1: "层1：风速距离结构实验",
                    2: "层2：气象特征增益实验",
                    3: "层3：去SCADA验证实验"}
    current_layer = None

    for s_key, (feat_cols, s_name, layer) in ALL_SCENARIOS.items():
        if layer != current_layer:
            current_layer = layer
            print()
            print("╔" + "═" * 58 + "╗")
            print(f"║  {layer_labels[layer]:<54}  ║")
            print("╚" + "═" * 58 + "╝")

        print()
        print(f"  ── {s_name} ──")
        print(f"     特征列（{len(feat_cols)} 个）：{feat_cols[:4]}"
              f"{'...' if len(feat_cols) > 4 else ''}")

        print("  → 树模型（LightGBM / XGBoost，直接多步策略）")
        all_results.extend(run_tree_scenario(df, feat_cols, s_key, s_name, layer))

        if s_key in LSTM_SCENARIOS:
            print("  → LSTM（双层，多输出）")
            all_results.extend(run_lstm_scenario(df, feat_cols, s_key, s_name, layer))

    return all_results


# ════════════════════════════════════════════════════════════════
# 7. 结果汇总
# ════════════════════════════════════════════════════════════════

def print_summary(results):
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 结果已保存：{OUTPUT_CSV}")

    for layer_num, layer_label in [
        (1, "层1：风速距离结构"),
        (2, "层2：气象特征增益"),
        (3, "层3：去SCADA验证"),
    ]:
        sub_l = df_res[df_res["layer"] == layer_num]
        if sub_l.empty:
            continue
        print(f"\n{'='*60}\n【LightGBM RMSE — {layer_label}（行=场景，列=步长 min）】\n{'='*60}")
        sub_lgb = sub_l[sub_l["model"] == "LightGBM"]
        if not sub_lgb.empty:
            pivot = sub_lgb.pivot(
                index="scenario_key", columns="step_min", values="RMSE")
            print(pivot.to_string())

    return df_res


# ════════════════════════════════════════════════════════════════
# 8. 可视化
# ════════════════════════════════════════════════════════════════

_STEP_MINS = [h * 10 for h in FORECAST_STEPS]

# 步长颜色映射
_STEP_COLORS = {10: "#1f77b4", 20: "#ff7f0e", 30: "#2ca02c", 60: "#d62728"}

# 层3关键场景配置（用于折线图）
_L3_CFG = {
    "M0_SCADA":         ("M0: 仅SCADA",           "o", "solid",  "#1f77b4"),
    "M5_Lall+SCADA":    ("M5: 全LiDAR+SCADA",     "s", "dashed", "#2ca02c"),
    "L3a_Lall_only":    ("L3a: 仅全LiDAR HWS",    "D", "dashed", "#d62728"),
    "L3b_Lall+met_only":("L3b: 仅LiDAR HWS+气象", "^", "dotted", "#9467bd"),
}


def _setup_heatmap_ax(ax, pivot, title, plt):
    """在给定 ax 上绘制带数值标注的 RMSE 热力图。"""
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto",
                   vmin=pivot.values[~np.isnan(pivot.values)].min() * 0.98,
                   vmax=pivot.values[~np.isnan(pivot.values)].max() * 1.02)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"+{c}min" for c in pivot.columns], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title(title, fontsize=11, pad=6)
    for ri in range(len(pivot.index)):
        for ci in range(len(pivot.columns)):
            v = pivot.values[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f"{v:.0f}", ha="center", va="center",
                        fontsize=8, color="black")
    return im


def _plot_l1_distance_scan(df_res, plt):
    """
    层1 图A：逐距离扫描折线图（M1含SCADA vs N1无SCADA 双线对比）。
    x 轴 = LiDAR 测量距离（40~300m）
    y 轴 = RMSE（kW）
    实线 = M1（LiDAR + SCADA），虚线 = N1（仅 LiDAR，无 SCADA）
    线色 = 各预测步长（+10/+20/+30/+60 min）
    水平点线 = M0 SCADA 基准
    """
    m1_keys = [f"M1_{d}m"      for d in LIDAR_DISTANCES]
    n1_keys = [f"N1_{d}m_only" for d in LIDAR_DISTANCES]
    x_vals  = LIDAR_DISTANCES

    for model_name in ["LightGBM", "XGBoost"]:
        sub_m = df_res[df_res["model"] == model_name]
        m0    = sub_m[sub_m["scenario_key"] == "M0_SCADA"]

        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

        for ax_idx, (keys, linestyle, with_scada_label) in enumerate([
            (m1_keys, "solid",  "实线：LiDAR + SCADA"),
            (n1_keys, "dashed", "虚线：仅LiDAR（无SCADA）"),
        ]):
            ax = axes[ax_idx]
            for h in FORECAST_STEPS:
                step_min  = h * 10
                rmse_vals = []
                for dk in keys:
                    row = sub_m[(sub_m["scenario_key"] == dk) &
                                (sub_m["step_min"] == step_min)]
                    rmse_vals.append(row["RMSE"].values[0] if not row.empty else np.nan)

                ax.plot(x_vals, rmse_vals,
                        marker="o", color=_STEP_COLORS[step_min],
                        linewidth=2, markersize=6, label=f"+{step_min}min")

                # M0 baseline（点线）
                m0_rmse = m0[m0["step_min"] == step_min]["RMSE"].values
                if m0_rmse.size:
                    ax.axhline(m0_rmse[0], color=_STEP_COLORS[step_min],
                               linestyle=":", linewidth=1, alpha=0.6)

            ax.set_xlabel("LiDAR 测量距离 (m)", fontsize=12)
            ax.set_ylabel("RMSE (kW)", fontsize=12)
            ax.set_title(f"{model_name}：{with_scada_label}\n"
                         f"（点线 = SCADA 基准 M0）", fontsize=11)
            ax.set_xticks(x_vals)
            ax.legend(title="预测步长", fontsize=9)
            ax.grid(alpha=0.3)

        plt.suptitle(f"{model_name}：逐距离 LiDAR HWS — 含SCADA(左) vs 无SCADA(右)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        fpath = os.path.join(OUTPUT_DIR, f"#9_L1a_距离扫描_{model_name}.png")
        plt.savefig(fpath, dpi=150)
        plt.close(fig)
        print(f"  图表已保存：{fpath}")


def _plot_l1_scada_delta(df_res, plt):
    """
    层1 图C：SCADA 干扰量热力图（新增）。
    Δ RMSE = N1_RMSE（无SCADA）− M1_RMSE（含SCADA）
      Δ > 0：去掉 SCADA 后 RMSE 升高 → SCADA 对该距离有正向贡献
      Δ < 0：去掉 SCADA 后 RMSE 降低 → SCADA 对该距离造成干扰
    行 = 距离，列 = 预测步长，颜色 = Δ 值（绿=干扰，红=有益）
    """
    for model_name in ["LightGBM", "XGBoost"]:
        sub_m = df_res[df_res["model"] == model_name]
        deltas = []
        for d in LIDAR_DISTANCES:
            row_delta = []
            for sm in _STEP_MINS:
                r_m1 = sub_m[(sub_m["scenario_key"] == f"M1_{d}m") &
                             (sub_m["step_min"] == sm)]["RMSE"].values
                r_n1 = sub_m[(sub_m["scenario_key"] == f"N1_{d}m_only") &
                             (sub_m["step_min"] == sm)]["RMSE"].values
                row_delta.append(
                    float(r_n1[0] - r_m1[0]) if (r_m1.size and r_n1.size) else np.nan)
            deltas.append(row_delta)

        pivot = pd.DataFrame(deltas,
                             index=[f"{d}m" for d in LIDAR_DISTANCES],
                             columns=_STEP_MINS)

        fig, ax = plt.subplots(figsize=(8, 5))
        abs_max = np.nanmax(np.abs(pivot.values))
        im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto",
                       vmin=-abs_max, vmax=abs_max)
        plt.colorbar(im, ax=ax, label="Δ RMSE (kW)：N1 − M1（正 = SCADA有益，负 = SCADA干扰）")

        ax.set_xticks(range(len(_STEP_MINS)))
        ax.set_xticklabels([f"+{c}min" for c in _STEP_MINS], fontsize=10)
        ax.set_yticks(range(len(LIDAR_DISTANCES)))
        ax.set_yticklabels([f"{d}m" for d in LIDAR_DISTANCES], fontsize=9)
        ax.set_xlabel("预测步长 (min)", fontsize=11)
        ax.set_title(f"{model_name}：SCADA 干扰量 Δ RMSE = 无SCADA(N1) − 含SCADA(M1)\n"
                     f"（红 = SCADA 有益 / 绿 = SCADA 有害）", fontsize=11)

        for ri in range(len(LIDAR_DISTANCES)):
            for ci in range(len(_STEP_MINS)):
                v = pivot.values[ri, ci]
                if not np.isnan(v):
                    ax.text(ci, ri, f"{v:+.0f}", ha="center", va="center",
                            fontsize=8, color="black")

        plt.tight_layout()
        fpath = os.path.join(OUTPUT_DIR, f"#9_L1c_SCADA干扰量_{model_name}.png")
        plt.savefig(fpath, dpi=150)
        plt.close(fig)
        print(f"  图表已保存：{fpath}")


def _plot_l1_group_heatmap(df_res, plt):
    """
    层1 图B：距离组合对比热力图（含SCADA 与 无SCADA 两组并排）。
    上半部分 = M 组（含SCADA），下半部分 = N 组（无SCADA）
    行 = 场景，列 = 预测步长，颜色 = RMSE
    """
    group_keys   = ["M0_SCADA",
                    "M2_near", "M3_mid", "M4_far", "M5_Lall+SCADA",
                    "N2_near_only", "N3_mid_only", "N4_far_only", "L3a_Lall_only"]
    group_labels = ["M0:仅SCADA",
                    "M2:近距+SCADA", "M3:中距+SCADA", "M4:远距+SCADA", "M5:全距+SCADA",
                    "N2:仅近距", "N3:仅中距", "N4:仅远距", "L3a:仅全距(=N5)"]

    for model_name in ["LightGBM", "XGBoost"]:
        sub_m = df_res[df_res["model"] == model_name]
        rows  = []
        for gk in group_keys:
            row = sub_m[sub_m["scenario_key"] == gk].sort_values("step_min")
            rows.append(row["RMSE"].values if not row.empty else [np.nan] * len(_STEP_MINS))

        pivot = pd.DataFrame(rows, index=group_labels, columns=_STEP_MINS)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        im = _setup_heatmap_ax(ax, pivot,
                               f"{model_name}：距离组合 RMSE 热力图（kW，越绿越好）\n"
                               f"上半：含SCADA（M组）/ 下半：无SCADA（N组）", plt)
        plt.colorbar(im, ax=ax, label="RMSE (kW)")
        ax.set_xlabel("预测步长 (min)", fontsize=11)
        # 在 M5 和 N2 之间画分隔线
        ax.axhline(4.5, color="white", linewidth=2)
        plt.tight_layout()
        fpath = os.path.join(OUTPUT_DIR, f"#9_L1b_距离组合热力图_{model_name}.png")
        plt.savefig(fpath, dpi=150)
        plt.close(fig)
        print(f"  图表已保存：{fpath}")


def _plot_l2_met_extension(df_res, plt):
    """
    层2 图：气象特征增益柱状图。
    M5（E0 基准）+ E1（+VShear）+ E2（+HShear）+ E3（+TI）+ E4（+全气象）
    x 轴 = 预测步长，y 轴 = RMSE，每组 5 根柱
    """
    met_keys   = ["M5_Lall+SCADA", "E1_AllHWS+SCADA+VShear",
                  "E2_AllHWS+SCADA+HShear", "E3_AllHWS+SCADA+TI",
                  "E4_AllHWS+SCADA+AllMet"]
    met_labels = ["E0(M5):\n全HWS+SCADA",
                  "E1:\n+VShear", "E2:\n+HShear",
                  "E3:\n+TI", "E4:\n+全气象"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]

    for model_name in ["LightGBM", "XGBoost"]:
        sub_m = df_res[df_res["model"] == model_name]
        n_steps = len(FORECAST_STEPS)
        fig, axes = plt.subplots(1, n_steps, figsize=(3.5 * n_steps, 4.5), sharey=True)

        for ji, h in enumerate(FORECAST_STEPS):
            ax = axes[ji]
            for ki, (key, label, color) in enumerate(
                    zip(met_keys, met_labels, colors)):
                row = sub_m[(sub_m["scenario_key"] == key) &
                            (sub_m["step_min"] == h * 10)]
                if not row.empty:
                    v = row["RMSE"].values[0]
                    bar = ax.bar(label, v, color=color, alpha=0.85)
                    ax.text(bar[0].get_x() + bar[0].get_width() / 2,
                            v + 3, f"{v:.0f}", ha="center", va="bottom",
                            fontsize=7)
            ax.set_title(f"+{h*10} min", fontsize=10)
            ax.set_ylabel("RMSE (kW)" if ji == 0 else "", fontsize=10)
            ax.tick_params(axis="x", rotation=25, labelsize=7)
            ax.grid(axis="y", alpha=0.3)

        plt.suptitle(f"{model_name}：气象特征增益对比（层2）\n"
                     "在全距离LiDAR+SCADA基础上逐类添加气象特征",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        fpath = os.path.join(OUTPUT_DIR, f"#9_L2_气象特征增益_{model_name}.png")
        plt.savefig(fpath, dpi=150)
        plt.close(fig)
        print(f"  图表已保存：{fpath}")


def _plot_l3_scada_removal(df_res, plt):
    """
    层3 图：去SCADA对比折线图（包含树模型和 LSTM）。
    比较 M0、M5、L3a、L3b 在各预测步长下的 RMSE。
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for mi, model_name in enumerate(["LightGBM", "LSTM"]):
        ax  = axes[mi]
        sub = df_res[df_res["model"] == model_name]

        for s_key, (label, marker, ls, color) in _L3_CFG.items():
            s_sub = sub[sub["scenario_key"] == s_key].sort_values("step_min")
            if s_sub.empty:
                continue
            ax.plot(s_sub["step_min"], s_sub["RMSE"],
                    marker=marker, linestyle=ls, color=color,
                    linewidth=2, markersize=7, label=label)

        ax.set_xlabel("预测步长 (min)", fontsize=12)
        ax.set_ylabel("RMSE (kW)", fontsize=12)
        ax.set_title(f"{model_name}：去SCADA对比", fontsize=11)
        ax.set_xticks(_STEP_MINS)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("层3：去SCADA验证 — M0/M5 vs 仅LiDAR（L3a/L3b）RMSE 对比\n"
                 "（L3a/L3b 不含 SCADA 机舱风速，验证激光雷达工程替代可行性）",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fpath = os.path.join(OUTPUT_DIR, "#9_L3_去SCADA对比.png")
    plt.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"  图表已保存：{fpath}")


def _plot_lstm_key_scenarios(df_res, plt):
    """LSTM 关键场景 RMSE/R² 随步长变化曲线。"""
    sub = df_res[df_res["model"] == "LSTM"]
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for s_key, (label, marker, ls, color) in _L3_CFG.items():
        s_sub = sub[sub["scenario_key"] == s_key].sort_values("step_min")
        if s_sub.empty:
            continue
        axes[0].plot(s_sub["step_min"], s_sub["RMSE"],
                     marker=marker, linestyle=ls, color=color, linewidth=2, label=label)
        axes[1].plot(s_sub["step_min"], s_sub["R2"],
                     marker=marker, linestyle=ls, color=color, linewidth=2, label=label)

    for ax, ylabel, title in [
        (axes[0], "RMSE (kW)", "LSTM 关键场景 RMSE 随预测步长变化"),
        (axes[1], "R²",        "LSTM 关键场景 R² 随预测步长变化"),
    ]:
        ax.set_xlabel("预测步长 (min)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(_STEP_MINS)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(f"LSTM 多步预测：关键场景对比（历史回望 {LOOK_BACK * 10} min）",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fpath = os.path.join(OUTPUT_DIR, "#9_LSTM_关键场景.png")
    plt.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"  图表已保存：{fpath}")


def _plot_full_heatmap(df_res, plt):
    """全场景 RMSE 热力图（LightGBM），按三层分段显示。"""
    sub = df_res[df_res["model"] == "LightGBM"]
    if sub.empty:
        return

    # 按层分组，每层绘制一个子热力图
    for layer_num, layer_label in [(1, "层1"), (2, "层2"), (3, "层3")]:
        sub_l = sub[sub["layer"] == layer_num]
        if sub_l.empty:
            continue
        try:
            pivot = sub_l.pivot(
                index="scenario_key", columns="step_min", values="RMSE")
        except Exception:
            continue
        if pivot.empty:
            continue

        h = max(3, len(pivot) * 0.55)
        fig, ax = plt.subplots(figsize=(8, h))
        im = _setup_heatmap_ax(ax, pivot,
                               f"LightGBM RMSE 热力图（{layer_label}，越绿越好）", plt)
        plt.colorbar(im, ax=ax, label="RMSE (kW)")
        ax.set_xlabel("预测步长 (min)", fontsize=11)
        plt.tight_layout()
        fpath = os.path.join(OUTPUT_DIR, f"#9_全场景热力图_{layer_label}.png")
        plt.savefig(fpath, dpi=150)
        plt.close(fig)
        print(f"  图表已保存：{fpath}")


def plot_all(df_res, plt):
    if plt is None:
        print("  matplotlib 不可用，跳过图表生成")
        return

    print()
    print("=" * 60)
    print("【生成图表】")
    print("=" * 60)

    _plot_l1_distance_scan(df_res, plt)    # 层1A：逐距离扫描（含SCADA vs 无SCADA 双子图）
    _plot_l1_group_heatmap(df_res, plt)    # 层1B：距离组合热力图（M组+N组并排）
    _plot_l1_scada_delta(df_res, plt)      # 层1C：SCADA 干扰量热力图（新增）
    _plot_l2_met_extension(df_res, plt)    # 层2：气象特征增益柱状图
    _plot_l3_scada_removal(df_res, plt)    # 层3：去SCADA对比折线图
    _plot_lstm_key_scenarios(df_res, plt)  # LSTM 关键场景曲线
    _plot_full_heatmap(df_res, plt)        # 全场景热力图（LightGBM）


# ════════════════════════════════════════════════════════════════
# 9. 主入口
# ════════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 60)
    print("  多步功率预测对比实验（三层分层实验设计 v3）")
    print(f"  层1：{len(L1_SCENARIOS)} 场景（M0~M5含SCADA 15个 + N1~N4无SCADA 13个）")
    print(f"  层2：{len(L2_SCENARIOS)} 场景（气象特征增益 E1~E4）")
    print(f"  层3：{len(L3_SCENARIOS)} 场景（去SCADA，L3a~L3b）")
    print(f"  历史回望：{LOOK_BACK} 步（{LOOK_BACK*10} min）")
    print(f"  预测步长：{FORECAST_STEPS}（× 10 min）")
    print(f"  树模型：全部 {len(ALL_SCENARIOS)} 场景  "
          f"LSTM：{len(LSTM_SCENARIOS)} 个关键场景")
    print("=" * 60)

    df          = load_wide_dataset()
    all_results = run_all_experiments(df)
    df_res      = print_summary(all_results)
    plot_all(df_res, try_plt())

    print()
    print("🎉 全部实验完成！")


if __name__ == "__main__":
    main()

