"""
#9 多步功率预测.py
==================
利用当前时刻及历史时序数据预测未来多个时刻风机功率

研究背景
--------
在上一轮实验（#8）中，我们使用单个时刻的特征快照预测下一个 10 min 的功率。
本实验在此基础上拓展为：
  - 输入：历史 look_back 步的特征序列（多历史时刻输入）
  - 输出：未来 H 步的功率序列（多步预测输出）

两种场景
--------
场景1（无激光雷达）：仅使用 SCADA 机舱风速 + 历史功率
  → 代表激光雷达故障或不可用时的预测能力基线

场景2（有激光雷达）：
  2a. LiDAR 最优距离（90m）+ SCADA 风速 + 历史功率
  2b. 仅 LiDAR 最优距离（90m）+ 历史功率（不含 SCADA 风速）
      —— 用户假设：激光雷达前视风速可能比机舱测风更能代表来流，效果或更优
  2c. 全距离 LiDAR（40~300m）+ SCADA 风速 + 历史功率
  2d. 仅全距离 LiDAR（40~300m）+ 历史功率（不含 SCADA 风速）

时间连续性约束
--------------
数据中存在 140+ 处时间戳间隙，所有滑动窗口仅在连续时间段内构建，
不跨越间隙，避免窗口跨越非连续时间段导致时序混乱。

模型
----
- LightGBM：直接多步策略（每预测步长单独训练一个独立模型，输入为展平的历史特征）
- XGBoost ：同上
- LSTM    ：多输出策略（一个模型同时预测所有未来步长，架构为双层 LSTM）

不使用风向特征（DIR/Veer）：数据质量较差，暂不纳入

超参数
------
  LOOK_BACK      = 6     → 历史回望：6 步 × 10 min = 60 min 历史信息
  FORECAST_STEPS = [1, 2, 3, 6]  → 预测 +10, +20, +30, +60 min 四个步长

运行方式
--------
    cd 利用激光雷达数据辅助风机功率预测/
    python "CODE/#9多步功率预测.py"

依赖
----
    pip install pandas numpy scikit-learn lightgbm xgboost matplotlib
    pip install tensorflow   # 可选，用于 LSTM
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
BEST_DIST       = 90                    # #8 实验确定的最优单距离
LOOK_BACK       = 6                     # 历史回望步数：60 min
FORECAST_STEPS  = [1, 2, 3, 6]         # 预测步长：+10, +20, +30, +60 min
MAX_H           = max(FORECAST_STEPS)   # 最大预测步长
TEST_RATIO      = 0.20                  # 测试集比例
VAL_RATIO       = 0.10                  # 验证集比例（从训练集末尾取）
RANDOM_STATE    = 42
TIME_GAP        = pd.Timedelta("10min")

# ──────────────────────────────────────────────────────────────
# 场景定义
# ──────────────────────────────────────────────────────────────
_all_lidar  = [f"HWS_{d}m" for d in LIDAR_DISTANCES]
_best_lidar = [f"HWS_{BEST_DIST}m"]

SCENARIOS = {
    "S1_SCADA":      (["HWS_scada", "power"],
                      "场景1：SCADA 风速（无激光雷达）"),
    "S2a_L90+SCADA": (_best_lidar + ["HWS_scada", "power"],
                      f"场景2a：LiDAR {BEST_DIST}m + SCADA 风速"),
    "S2b_L90_only":  (_best_lidar + ["power"],
                      f"场景2b：仅 LiDAR {BEST_DIST}m（不含SCADA风速）"),
    "S2c_Lall+SCADA":(_all_lidar + ["HWS_scada", "power"],
                      "场景2c：全距离LiDAR + SCADA 风速"),
    "S2d_Lall_only": (_all_lidar + ["power"],
                      "场景2d：仅全距离LiDAR（不含SCADA风速）"),
}


# ════════════════════════════════════════════════════════════════
# 1. 数据加载与预处理
# ════════════════════════════════════════════════════════════════

def load_wide_dataset():
    """
    读取 #7 数据集，转为宽格式（每行一个时间戳）。

    列说明
    ------
    HWS_scada   : Distance=0 的机舱风速 HWS(hub)
    power       : ACTIVE_POWER_#56_对齐_前10分钟均值（目标序列）
    HWS_{d}m    : 各距离激光雷达反演水平风速（d = 40,60,...,300m）
    segment_id  : 连续时间段编号（间隔 ≠ 10 min 则新段开始）
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

    # ── 各距离 LiDAR HWS ──────────────────────────────────────
    df_wide = d0
    for dist in LIDAR_DISTANCES:
        sub = df_raw[df_raw["Distance"] == dist][
            ["DateAndTime", "HWS(hub)"]
        ].rename(columns={"HWS(hub)": f"HWS_{dist}m"})
        df_wide = df_wide.merge(sub, on="DateAndTime", how="left")

    df_wide = df_wide.sort_values("DateAndTime").reset_index(drop=True)
    df_wide = df_wide.dropna(subset=["power"]).reset_index(drop=True)

    # ── 连续时间段标注 ─────────────────────────────────────────
    dt_diff = df_wide["DateAndTime"].diff()
    df_wide["segment_id"] = (dt_diff != TIME_GAP).cumsum()

    seg_sizes = df_wide.groupby("segment_id").size()
    n_usable = int(((seg_sizes >= LOOK_BACK + MAX_H)).sum())
    print(f"  有效时间戳：{len(df_wide):,}")
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
    n_feat = len(feat_cols)
    Xs, Ys, Ts = [], [], []

    for _, seg_df in df.groupby("segment_id", sort=True):
        seg_df = seg_df.sort_values("DateAndTime").reset_index(drop=True)
        n = len(seg_df)
        if n < look_back + max_h:
            continue

        # 中位数填充每列缺失值（分段计算，避免跨段污染）
        feats = seg_df[feat_cols].copy()
        for c in feat_cols:
            med = feats[c].median()
            feats[c] = feats[c].fillna(med if not np.isnan(med) else 0.0)

        feat_vals  = feats.values              # (n, n_feat)
        power_vals = seg_df["power"].values    # (n,)
        times      = seg_df["DateAndTime"].values  # (n,)

        # i = index of first FUTURE step (= index right after input window)
        for i in range(look_back, n - max_h + 1):
            future = power_vals[i:i + max_h]
            if np.any(np.isnan(future)):
                continue
            Xs.append(feat_vals[i - look_back:i])   # (look_back, n_feat)
            Ys.append(future)                         # (max_h,)
            Ts.append(times[i - 1])                  # 最后一个输入时刻

    if not Xs:
        return None, None, None

    return (
        np.array(Xs, dtype=np.float32),   # (N, look_back, n_feat)
        np.array(Ys, dtype=np.float32),   # (N, max_h)
        np.array(Ts),                     # (N,)
    )


def time_split_windows(X, Y, T, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO):
    """按时间顺序划分训练 / 验证 / 测试集（不打乱，保持时序）。"""
    n = len(X)
    n_te  = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_tr  = n - n_te - n_val
    return (
        (X[:n_tr],          Y[:n_tr],          T[:n_tr]),
        (X[n_tr:n_tr+n_val], Y[n_tr:n_tr+n_val], T[n_tr:n_tr+n_val]),
        (X[n_tr+n_val:],    Y[n_tr+n_val:],    T[n_tr+n_val:]),
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

        # 确保 CJK 字体已安装（静默尝试）
        try:
            subprocess.run(["apt-get", "install", "-y", "-q", "fonts-noto-cjk"],
                           capture_output=True, timeout=60)
        except Exception:
            pass

        # 重建字体缓存，识别新安装的字体
        fm._load_fontmanager(try_read_cache=False)

        # 按优先级尝试加载中文字体
        _zh_candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
        ]
        for fp in _zh_candidates:
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

def run_tree_scenario(df, feat_cols, s_key, s_name):
    """
    对每个预测步长 h 单独训练一个 LightGBM/XGBoost 模型。

    输入特征：X_flat = (N, look_back × n_feat) — 展平的历史特征矩阵
    目标标签：y_h = power(t+h)，每步独立

    这是 Direct 多步策略，各步模型间无依赖，预测误差不会累积。
    """
    X, Y, T = build_windows(df, feat_cols)
    if X is None:
        print(f"  ⚠️  {s_name} 没有有效窗口，跳过")
        return []

    N, L, F = X.shape
    X_flat = X.reshape(N, L * F)   # (N, look_back × n_feat)
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
                    n_estimators=500, learning_rate=0.05, num_leaves=31,
                    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                    random_state=RANDOM_STATE, verbose=-1,
                    feature_name=feat_names,
                )
                m.fit(X_tr, y_tr_h,
                      eval_set=[(X_val, y_val_h)],
                      callbacks=[lib.early_stopping(30, verbose=False),
                                 lib.log_evaluation(-1)])
            else:
                m = lib.XGBRegressor(
                    n_estimators=500, learning_rate=0.05, max_depth=5,
                    subsample=0.8, colsample_bytree=0.8, eval_metric="rmse",
                    early_stopping_rounds=30, random_state=RANDOM_STATE, verbosity=0,
                )
                m.fit(X_tr, y_tr_h,
                      eval_set=[(X_val, y_val_h)], verbose=False)

            pred = m.predict(X_te)
            r = evaluate_step(y_te_h, pred)
            r.update({
                "scenario_key":  s_key,
                "scenario_name": s_name,
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

def run_lstm_scenario(df, feat_cols, s_key, s_name):
    """
    训练双层 LSTM 多输出模型：一次预测所有 FORECAST_STEPS 步。

    架构：Input(look_back, n_feat)
           → LSTM(64, return_sequences=True)
           → LSTM(32)
           → Dropout(0.2)
           → Dense(32, relu)
           → Dense(max_h)   # 同时输出所有预测步

    特征和目标均经 StandardScaler 标准化（仅在训练集拟合）。
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

    # 标准化输入特征（沿样本和时间步展开，逐特征计算均值方差）
    scaler_x = StandardScaler()
    X_tr_s  = scaler_x.fit_transform(X_tr.reshape(-1, F)).reshape(len(X_tr), L, F)
    X_val_s = scaler_x.transform(X_val.reshape(-1, F)).reshape(len(X_val), L, F)
    X_te_s  = scaler_x.transform(X_te.reshape(-1, F)).reshape(len(X_te), L, F)

    # 标准化输出功率
    scaler_y = StandardScaler()
    Y_tr_s   = scaler_y.fit_transform(Y_tr)
    Y_val_s  = scaler_y.transform(Y_val)

    # 构建模型
    inp = tf.keras.Input(shape=(L, F))
    x   = tf.keras.layers.LSTM(64, return_sequences=True)(inp)
    x   = tf.keras.layers.LSTM(32)(x)
    x   = tf.keras.layers.Dropout(0.2)(x)
    x   = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(MAX_H)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

    cb = [
        tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, verbose=0),
    ]
    model.fit(
        X_tr_s, Y_tr_s,
        validation_data=(X_val_s, Y_val_s),
        epochs=100, batch_size=64, callbacks=cb, verbose=0,
    )

    pred_s = model.predict(X_te_s, verbose=0)                # (N_te, MAX_H)
    pred   = scaler_y.inverse_transform(pred_s)               # (N_te, MAX_H)

    results = []
    for h in FORECAST_STEPS:
        r = evaluate_step(Y_te[:, h - 1], pred[:, h - 1])
        r.update({
            "scenario_key":  s_key,
            "scenario_name": s_name,
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
    all_results = []

    for s_key, (feat_cols, s_name) in SCENARIOS.items():
        print()
        print("=" * 60)
        print(f"【{s_name}】")
        print(f"  特征列：{feat_cols}")
        print("=" * 60)

        print("  → 树模型（LightGBM / XGBoost，直接多步策略）")
        all_results.extend(run_tree_scenario(df, feat_cols, s_key, s_name))

        print("  → LSTM（双层，多输出）")
        all_results.extend(run_lstm_scenario(df, feat_cols, s_key, s_name))

    return all_results


# ════════════════════════════════════════════════════════════════
# 7. 结果汇总
# ════════════════════════════════════════════════════════════════

def print_summary(results):
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 结果已保存：{OUTPUT_CSV}")

    for m in ["LightGBM", "XGBoost", "LSTM"]:
        sub = df_res[df_res["model"] == m]
        if sub.empty:
            continue
        pivot = sub.pivot(index="scenario_key", columns="step_min", values="RMSE")
        print(f"\n{'='*60}")
        print(f"【{m} RMSE 汇总（行=场景，列=预测步长 min）】")
        print(f"{'='*60}")
        print(pivot.to_string())

    return df_res


# ════════════════════════════════════════════════════════════════
# 8. 可视化
# ════════════════════════════════════════════════════════════════

# 场景显示配置
_SCENARIO_CFG = {
    "S1_SCADA":       ("S1: SCADA风速",     "o", "solid",  "#1f77b4"),
    "S2a_L90+SCADA":  ("S2a: LiDAR90+SCADA","s", "dashed", "#2ca02c"),
    "S2b_L90_only":   ("S2b: 仅LiDAR90",    "D", "dashed", "#d62728"),
    "S2c_Lall+SCADA": ("S2c: 全LiDAR+SCADA","^", "dotted", "#ff7f0e"),
    "S2d_Lall_only":  ("S2d: 仅全LiDAR",    "*", "dotted", "#9467bd"),
}

_STEP_MINS = [h * 10 for h in FORECAST_STEPS]


def _plot_rmse_r2_curves(df_res, plt, model_name):
    """为指定模型绘制 RMSE 和 R² 随预测步长变化的折线图。"""
    sub = df_res[df_res["model"] == model_name]
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for s_key, (label, marker, ls, color) in _SCENARIO_CFG.items():
        s_sub = sub[sub["scenario_key"] == s_key].sort_values("step_min")
        if s_sub.empty:
            continue
        axes[0].plot(s_sub["step_min"], s_sub["RMSE"],
                     marker=marker, linestyle=ls, color=color, label=label)
        axes[1].plot(s_sub["step_min"], s_sub["R2"],
                     marker=marker, linestyle=ls, color=color, label=label)

    for ax, ylabel, title in [
        (axes[0], "RMSE (kW)", f"{model_name}：各场景 RMSE 随预测步长变化"),
        (axes[1], "R²",        f"{model_name}：各场景 R² 随预测步长变化"),
    ]:
        ax.set_xlabel("预测步长 (min)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(_STEP_MINS)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(
        f"多步功率预测性能对比（{model_name}，历史回望={LOOK_BACK*10} min）",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    fpath = os.path.join(OUTPUT_DIR, f"#9_{model_name}_RMSE_vs_step.png")
    plt.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"  图表已保存：{fpath}")


def _plot_s1_vs_s2b(df_res, plt):
    """
    核心对比图：S1（SCADA）vs S2b（仅 LiDAR 90m）× 各模型 × 各预测步长。
    回答用户假设：激光雷达（仅前视风速）预测是否优于 SCADA 机舱风速？
    """
    models    = ["LightGBM", "XGBoost", "LSTM"]
    n_steps   = len(FORECAST_STEPS)
    fig, axes = plt.subplots(len(models), n_steps,
                             figsize=(4 * n_steps, 3.5 * len(models)),
                             sharey="row")

    for mi, model_name in enumerate(models):
        sub_m = df_res[df_res["model"] == model_name]
        for ji, h in enumerate(FORECAST_STEPS):
            ax = axes[mi][ji]
            sub_h = sub_m[sub_m["step"] == h]

            for s_key, label, color in [
                ("S1_SCADA",      "S1: SCADA", "#1f77b4"),
                ("S2a_L90+SCADA", "S2a: L90+SCADA", "#2ca02c"),
                ("S2b_L90_only",  "S2b: 仅L90", "#d62728"),
            ]:
                row = sub_h[sub_h["scenario_key"] == s_key]
                if not row.empty:
                    ax.bar(label, row["RMSE"].values[0], color=color, alpha=0.8)

            ax.set_title(f"{model_name}\n+{h*10} min", fontsize=9)
            ax.set_ylabel("RMSE (kW)" if ji == 0 else "", fontsize=9)
            ax.tick_params(axis="x", rotation=30, labelsize=7)
            ax.grid(axis="y", alpha=0.3)

    plt.suptitle("场景1（SCADA）vs 场景2a（LiDAR+SCADA）vs 场景2b（仅LiDAR）RMSE 对比",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fpath = os.path.join(OUTPUT_DIR, "#9_S1vsS2_对比.png")
    plt.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"  图表已保存：{fpath}")


def _plot_all_scenarios_heatmap(df_res, plt):
    """RMSE 热力图：行=场景，列=预测步长，颜色=RMSE 值（越深越好）。"""
    for model_name in ["LightGBM", "XGBoost", "LSTM"]:
        sub = df_res[df_res["model"] == model_name]
        if sub.empty:
            continue

        pivot = sub.pivot(index="scenario_key", columns="step_min", values="RMSE")
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
        plt.colorbar(im, ax=ax, label="RMSE (kW)")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"+{c}min" for c in pivot.columns], fontsize=11)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(
            [_SCENARIO_CFG.get(k, (k,))[0] for k in pivot.index], fontsize=10)
        ax.set_xlabel("预测步长", fontsize=12)
        ax.set_title(f"{model_name} RMSE 热力图（越绿越好）", fontsize=12)

        # 在格子中显示数值
        for ri in range(len(pivot.index)):
            for ci in range(len(pivot.columns)):
                v = pivot.values[ri, ci]
                if not np.isnan(v):
                    ax.text(ci, ri, f"{v:.0f}", ha="center", va="center",
                            fontsize=9, color="black")

        plt.tight_layout()
        fpath = os.path.join(OUTPUT_DIR, f"#9_{model_name}_RMSE_热力图.png")
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

    # 1. 各模型 RMSE/R² 曲线
    for m in ["LightGBM", "XGBoost", "LSTM"]:
        _plot_rmse_r2_curves(df_res, plt, m)

    # 2. S1 vs S2a vs S2b 核心对比
    _plot_s1_vs_s2b(df_res, plt)

    # 3. 全场景 RMSE 热力图
    _plot_all_scenarios_heatmap(df_res, plt)


# ════════════════════════════════════════════════════════════════
# 9. 主入口
# ════════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 60)
    print("  多步功率预测对比实验")
    print("  历史多步输入 → 未来多步输出")
    print(f"  历史回望：{LOOK_BACK} 步（{LOOK_BACK*10} min）")
    print(f"  预测步长：{FORECAST_STEPS}（单位：步，即 × 10 min）")
    print("=" * 60)

    df          = load_wide_dataset()
    all_results = run_all_experiments(df)
    df_res      = print_summary(all_results)
    plot_all(df_res, try_plt())

    print()
    print("🎉 全部实验完成！")


if __name__ == "__main__":
    main()
