"""
#8 功率预测模型对比.py
======================
利用激光雷达数据辅助风机功率预测 —— 模型训练与对比（第二版）

研究问题
--------
1. 激光雷达风速（HWS）是否能比 SCADA 机舱风速更好地预测功率？
2. 哪个距离的激光雷达风速对预测最有价值？
3. 多距离激光雷达风速的组合是否能进一步提升精度？
4. 垂直风切变（VShear）、水平风切变（HShear）、湍流强度（TI）等气象特征
   是否能在最优风速组合基础上进一步提升预测性能？

数据说明
--------
- 输入：PROCESS_DATA/#7构建好的训练数据集.csv（GBK 编码）
  ├── Distance=0  : SCADA 机舱风速 HWS(hub)、功率 ACTIVE_POWER_#56_对齐_前10分钟均值
  └── Distance>0  : 各距离激光雷达特征（HWS/RAWS/VShear/HShear/TI1~TI4）

- 目标变量：ACTIVE_POWER_#56_对齐_前10分钟均值（kW），预测下一个 10 分钟的功率
- 不使用风向特征（DIR/Veer）：现有风向数据质量较差，暂不纳入建模

实验方案
--------
Exp-0  Baseline      : SCADA HWS + 当前功率 → LightGBM / XGBoost
Exp-1  单距离 LiDAR  : 各距离 HWS + 当前功率（10 个距离分别测试）
Exp-2  多距离 LiDAR  : 近距（40-120m）/ 中距（150-210m）/ 远距（240-300m）/ 全距离 HWS 组合
Exp-3  气象特征扩展  : 全距离 HWS + 当前功率 + VShear / HShear / TI（逐类添加）
Exp-4  LSTM 时序     : 连续时间段内构建滑动窗口（look_back=6×10min）

关键约束
--------
- 以时间顺序划分训练集（前80%）和测试集（后20%），不随机打乱
- LSTM 滑动窗口必须在连续时间段内构建（不跨越时间戳缺失的间隔）

运行方式
--------
    cd 利用激光雷达数据辅助风机功率预测/
    python "CODE/#8功率预测模型对比.py"

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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

DATASET_PATH = os.path.join(BASE_DIR, "PROCESS_DATA", "#7构建好的训练数据集.csv")
OUTPUT_CSV   = os.path.join(BASE_DIR, "PROCESS_DATA", "#8模型对比结果.csv")
OUTPUT_DIR   = os.path.join(BASE_DIR, "PROCESS_DATA")

# ──────────────────────────────────────────────────────────────
# 全局超参数
# ──────────────────────────────────────────────────────────────
LIDAR_DISTANCES = [40, 60, 90, 120, 150, 180, 210, 240, 270, 300]
PREDICT_STEP    = 1       # 预测步长：1 步 = 10 min 超前
TEST_RATIO      = 0.20    # 按时序划分测试集比例
RANDOM_STATE    = 42
LSTM_LOOK_BACK  = 6       # LSTM 回望窗口：6×10min = 60 min
TIME_GAP        = pd.Timedelta("10min")   # 正常时间间隔


# ════════════════════════════════════════════════════════════════
# 1. 数据加载与预处理
# ════════════════════════════════════════════════════════════════

def load_wide_dataset():
    """
    读取 #7 数据集，转为宽格式（每行一个时间戳），返回 DataFrame。

    列包含：
      - DateAndTime
      - HWS_scada          : Distance=0 的 HWS(hub)（SCADA 机舱风速）
      - power_now          : 当前时刻 ACTIVE_POWER（用作输入特征）
      - power_target       : 下一时刻 ACTIVE_POWER（预测目标）
      - HWS_{d}m           : 各距离激光雷达反演风速
      - VShear_{d}m        : 各距离垂直风切变
      - HShear_{d}m        : 各距离水平风切变
      - TI_avg_{d}m        : 各距离湍流强度均值（TI1~TI4 平均）
      - segment_id         : 连续时间段编号（用于 LSTM 滑动窗口划分）
    """
    print("=" * 60)
    print("【数据加载与预处理】")
    print("=" * 60)

    df_raw = pd.read_csv(DATASET_PATH, encoding="gbk")
    df_raw["DateAndTime"] = pd.to_datetime(df_raw["DateAndTime"])
    print(f"  原始数据：{len(df_raw):,} 行，{df_raw['DateAndTime'].nunique():,} 个时间戳")

    # ── 提取 SCADA 行（Distance=0）─────────────────────────────
    d0 = df_raw[df_raw["Distance"] == 0][
        ["DateAndTime", "HWS(hub)", "ACTIVE_POWER_#56_对齐_前10分钟均值"]
    ].copy().rename(columns={
        "HWS(hub)": "HWS_scada",
        "ACTIVE_POWER_#56_对齐_前10分钟均值": "power_now",
    })

    # ── 提取各距离 LiDAR 特征 ──────────────────────────────────
    lidar_parts = []
    for dist in LIDAR_DISTANCES:
        sub = df_raw[df_raw["Distance"] == dist][
            ["DateAndTime", "HWS(hub)", "VShear", "HShear", "TI1", "TI2", "TI3", "TI4"]
        ].copy()
        sub["TI_avg"] = sub[["TI1", "TI2", "TI3", "TI4"]].mean(axis=1)
        sub = sub.rename(columns={
            "HWS(hub)": f"HWS_{dist}m",
            "VShear":   f"VShear_{dist}m",
            "HShear":   f"HShear_{dist}m",
            "TI_avg":   f"TI_avg_{dist}m",
        }).drop(columns=["TI1", "TI2", "TI3", "TI4"])
        lidar_parts.append(sub)

    lidar_wide = lidar_parts[0]
    for part in lidar_parts[1:]:
        lidar_wide = lidar_wide.merge(part, on="DateAndTime", how="outer")

    # ── 合并 SCADA + LiDAR ──────────────────────────────────────
    df_wide = d0.merge(lidar_wide, on="DateAndTime", how="inner")
    df_wide = df_wide.sort_values("DateAndTime").reset_index(drop=True)

    # ── 删除 power_now 为 NaN 的行 ──────────────────────────────
    df_wide = df_wide.dropna(subset=["power_now"]).reset_index(drop=True)
    print(f"  有效时间戳（power_now 非空）：{len(df_wide):,}")

    # ── 构建预测目标：下一步功率 ─────────────────────────────────
    df_wide["power_target"] = df_wide["power_now"].shift(-PREDICT_STEP)

    # ── 标注连续时间段（用于 LSTM 窗口划分）──────────────────────
    dt_diff = df_wide["DateAndTime"].diff()
    # 时间间隔不等于 10 min 时，视为新段的开始
    df_wide["segment_id"] = (dt_diff != TIME_GAP).cumsum()
    seg_sizes = df_wide["segment_id"].value_counts()
    print(f"  连续时间段数：{df_wide['segment_id'].nunique()}  "
          f"（最大段长 {seg_sizes.max()}，最小段长 {seg_sizes.min()}）")
    print(f"  时间范围：{df_wide['DateAndTime'].min()} ~ {df_wide['DateAndTime'].max()}")
    return df_wide


# ════════════════════════════════════════════════════════════════
# 2. 通用工具
# ════════════════════════════════════════════════════════════════

def evaluate(y_true, y_pred, label=""):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t, y_p = np.array(y_true)[mask], np.array(y_pred)[mask]
    return {
        "label":  label,
        "RMSE":   round(float(np.sqrt(mean_squared_error(y_t, y_p))), 2),
        "MAE":    round(float(mean_absolute_error(y_t, y_p)), 2),
        "R2":     round(float(r2_score(y_t, y_p)), 4),
        "N":      int(len(y_t)),
    }


def time_split(df, test_ratio=TEST_RATIO):
    """按时间顺序划分训练/验证/测试集，去除 power_target 为 NaN 的行。"""
    df_valid = df.dropna(subset=["power_target"]).reset_index(drop=True)
    n = len(df_valid)
    n_test = int(n * test_ratio)
    n_val  = int(n * 0.1)                  # 从训练集末尾取 10% 为验证集
    n_train_core = n - n_test - n_val
    return (
        df_valid.iloc[:n_train_core],       # train
        df_valid.iloc[n_train_core:n_train_core + n_val],   # val
        df_valid.iloc[n_train_core + n_val:],               # test
    )


def fill_features(df, feat_cols):
    """用列中位数填充缺失值。"""
    X = df[feat_cols].copy()
    for c in feat_cols:
        X[c] = X[c].fillna(X[c].median())
    return X.values


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

        # 重建字体缓存以识别新安装的字体（如 fonts-noto-cjk）
        fm._load_fontmanager(try_read_cache=False)

        # 依次尝试常见的中文字体路径
        _zh_font_candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
        ]
        for _fp in _zh_font_candidates:
            if os.path.exists(_fp):
                fm.fontManager.addfont(_fp)
                _prop = fm.FontProperties(fname=_fp)
                plt.rcParams["font.sans-serif"] = [_prop.get_name(), "DejaVu Sans"]
                break
        plt.rcParams["axes.unicode_minus"] = False
        return plt
    except ImportError:
        return None


# ════════════════════════════════════════════════════════════════
# 3. 树模型训练（LightGBM / XGBoost）
# ════════════════════════════════════════════════════════════════

def train_tree_models(X_tr, y_tr, X_val, y_val, X_te):
    """训练 LightGBM 和 XGBoost，返回 (lgb_model, xgb_model, pred_lgb, pred_xgb)。"""
    preds = {}

    lgb = try_lgb()
    if lgb:
        m = lgb.LGBMRegressor(
            n_estimators=1000, learning_rate=0.05, num_leaves=63,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbose=-1,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
        preds["LightGBM"] = (m, m.predict(X_te))

    xgb = try_xgb()
    if xgb:
        m = xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, eval_metric="rmse",
            early_stopping_rounds=50, random_state=RANDOM_STATE, verbosity=0,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds["XGBoost"] = (m, m.predict(X_te))

    return preds


def run_tree_experiment(df, feat_cols, exp_label, y_te_ref=None):
    """
    训练树模型并返回评估结果列表。
    feat_cols  : 特征列名列表
    exp_label  : 实验标签前缀
    y_te_ref   : 若提供则验证测试集对齐（用于跨实验比较）
    """
    train_df, val_df, test_df = time_split(df)
    X_tr  = fill_features(train_df, feat_cols)
    y_tr  = train_df["power_target"].values
    X_val = fill_features(val_df,   feat_cols)
    y_val = val_df["power_target"].values
    X_te  = fill_features(test_df,  feat_cols)
    y_te  = test_df["power_target"].values

    preds = train_tree_models(X_tr, y_tr, X_val, y_val, X_te)
    results = []
    for model_name, (model, pred) in preds.items():
        r = evaluate(y_te, pred, f"{model_name}|{exp_label}")
        r["model"]   = model_name
        r["exp"]     = exp_label
        r["features"] = "|".join(feat_cols)
        results.append(r)
        print(f"    {model_name:<12}  RMSE={r['RMSE']:>7.1f}  MAE={r['MAE']:>7.1f}  R²={r['R2']:.4f}  N={r['N']}")

    # 返回最后一个 LightGBM 模型（用于特征重要性）和 y_te
    lgb_model = preds.get("LightGBM", (None, None))[0]
    return results, lgb_model, y_te, preds.get("LightGBM", (None, None))[1]


# ════════════════════════════════════════════════════════════════
# 4. LSTM 模型（时间连续性校验）
# ════════════════════════════════════════════════════════════════

def build_sequences_continuous(df, feat_cols, look_back=LSTM_LOOK_BACK):
    """
    在每个连续时间段内构建滑动窗口序列。
    保证每个窗口内时间戳严格连续（间隔 = 10 min）。
    返回 X_seq (N, look_back, n_feat), y_seq (N,), timestamps (N,)
    """
    Xs, ys, ts = [], [], []
    for seg_id, seg_df in df.groupby("segment_id", sort=True):
        seg_df = seg_df.sort_values("DateAndTime").reset_index(drop=True)
        if len(seg_df) <= look_back:
            continue
        X_seg = fill_features(seg_df, feat_cols)   # (T, n_feat)
        y_seg = seg_df["power_target"].values       # (T,)
        t_seg = seg_df["DateAndTime"].values        # (T,)
        for i in range(look_back, len(seg_df)):
            if pd.isna(y_seg[i]):
                continue
            Xs.append(X_seg[i - look_back:i])
            ys.append(y_seg[i])
            ts.append(t_seg[i])
    if not Xs:
        return None, None, None
    return np.array(Xs), np.array(ys), np.array(ts)


def train_lstm_model(X_seq_tr, y_tr, X_seq_val, y_val, X_seq_te):
    tf = try_tf()
    if tf is None:
        return None, None

    look_back, n_feat = X_seq_tr.shape[1], X_seq_tr.shape[2]
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(look_back, n_feat)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=0)
    model.fit(X_seq_tr, y_tr, validation_data=(X_seq_val, y_val),
              epochs=100, batch_size=64, callbacks=[cb], verbose=0)
    return model, model.predict(X_seq_te, verbose=0).flatten()


def run_lstm_experiment(df, feat_cols, exp_label, look_back=LSTM_LOOK_BACK):
    """
    在连续时间段内构建序列，训练 LSTM，返回评估结果。
    """
    # 划分训练段、验证段、测试段（按时间顺序）
    train_df, val_df, test_df = time_split(df)

    scaler = StandardScaler()

    # fit scaler on train features
    X_train_all = fill_features(train_df, feat_cols)
    scaler.fit(X_train_all)

    def scale_df(sub_df):
        sub_scaled = sub_df.copy()
        vals = fill_features(sub_df, feat_cols)
        vals_sc = scaler.transform(vals)
        for i, c in enumerate(feat_cols):
            sub_scaled = sub_scaled.copy()
            # Replace values inline via temporary column
            sub_scaled = sub_scaled.assign(**{c: vals_sc[:, i]})
        return sub_scaled

    train_sc = scale_df(train_df)
    val_sc   = scale_df(val_df)
    test_sc  = scale_df(test_df)

    X_tr,  y_tr,  _ = build_sequences_continuous(train_sc, feat_cols, look_back)
    X_val, y_val, _ = build_sequences_continuous(val_sc,   feat_cols, look_back)
    X_te,  y_te,  _ = build_sequences_continuous(test_sc,  feat_cols, look_back)

    if X_tr is None or len(X_tr) < 10:
        print(f"    LSTM 数据不足，跳过 {exp_label}")
        return []

    _, pred = train_lstm_model(X_tr, y_tr, X_val, y_val, X_te)
    if pred is None:
        return []

    results = []
    r = evaluate(y_te, pred, f"LSTM|{exp_label}")
    r["model"]    = "LSTM"
    r["exp"]      = exp_label
    r["features"] = "|".join(feat_cols)
    results.append(r)
    print(f"    LSTM        RMSE={r['RMSE']:>7.1f}  MAE={r['MAE']:>7.1f}  R²={r['R2']:.4f}  N={r['N']}")
    return results


# ════════════════════════════════════════════════════════════════
# 5. 实验主流程
# ════════════════════════════════════════════════════════════════

def run_all_experiments(df):
    all_results = []

    # ── Exp-0: Baseline（SCADA HWS + 当前功率）─────────────────
    print()
    print("=" * 60)
    print("【Exp-0】Baseline：SCADA 风速 + 当前功率 → 预测下一步功率")
    print("=" * 60)
    feat_baseline = ["HWS_scada", "power_now"]
    res, lgb_baseline, y_te_base, pred_baseline = run_tree_experiment(
        df, feat_baseline, "Baseline(SCADA_HWS+power)")
    all_results.extend(res)

    # ── Exp-1: 逐距离 LiDAR HWS ──────────────────────────────
    print()
    print("=" * 60)
    print("【Exp-1】逐距离 LiDAR 风速：各距离 HWS + 当前功率")
    print("=" * 60)
    best_dist_rmse = {}
    for dist in LIDAR_DISTANCES:
        feat = ["HWS_scada", f"HWS_{dist}m", "power_now"]
        print(f"  距离 {dist:>3}m：")
        res, _, _, _ = run_tree_experiment(df, feat, f"LiDAR_HWS_{dist}m")
        all_results.extend(res)
        for r in res:
            if r["model"] == "LightGBM":
                best_dist_rmse[dist] = r["RMSE"]

    # 找最优单距离
    best_dist = min(best_dist_rmse, key=best_dist_rmse.get)
    print(f"\n  ✅ 最优单距离（LightGBM RMSE 最低）：{best_dist} m  "
          f"RMSE={best_dist_rmse[best_dist]:.1f}")

    # ── Exp-2: 多距离 LiDAR HWS 组合 ─────────────────────────
    print()
    print("=" * 60)
    print("【Exp-2】多距离 LiDAR HWS 组合 + 当前功率")
    print("=" * 60)
    dist_groups = {
        "近距(40-120m)":  [40, 60, 90, 120],
        "中距(150-210m)": [150, 180, 210],
        "远距(240-300m)": [240, 270, 300],
        "全距离(40-300m)": LIDAR_DISTANCES,
    }
    for grp_name, dists in dist_groups.items():
        feat = ["HWS_scada"] + [f"HWS_{d}m" for d in dists] + ["power_now"]
        print(f"  组合 {grp_name}：")
        res, lgb_allhws, _, pred_allhws = run_tree_experiment(
            df, feat, f"LiDAR_HWS_{grp_name}")
        all_results.extend(res)

    # ── Exp-3: 气象特征扩展（在全距离 HWS 基础上逐步添加）──────
    print()
    print("=" * 60)
    print("【Exp-3】气象特征扩展（全距离 HWS + 当前功率 + 各类气象特征）")
    print("=" * 60)
    feat_base_hws = ["HWS_scada"] + [f"HWS_{d}m" for d in LIDAR_DISTANCES] + ["power_now"]

    # 3a: + VShear（所有距离）
    feat_vshear = feat_base_hws + [f"VShear_{d}m" for d in LIDAR_DISTANCES]
    print("  + VShear（所有距离）：")
    res, _, _, _ = run_tree_experiment(df, feat_vshear, "AllHWS+VShear")
    all_results.extend(res)

    # 3b: + HShear（所有距离）
    feat_hshear = feat_base_hws + [f"HShear_{d}m" for d in LIDAR_DISTANCES]
    print("  + HShear（所有距离）：")
    res, _, _, _ = run_tree_experiment(df, feat_hshear, "AllHWS+HShear")
    all_results.extend(res)

    # 3c: + TI（所有距离均值）
    feat_ti = feat_base_hws + [f"TI_avg_{d}m" for d in LIDAR_DISTANCES]
    print("  + TI_avg（所有距离）：")
    res, lgb_allmet, _, pred_allmet = run_tree_experiment(df, feat_ti, "AllHWS+TI")
    all_results.extend(res)

    # 3d: + 全部气象特征（VShear + HShear + TI）
    feat_all_met = (feat_base_hws
                    + [f"VShear_{d}m" for d in LIDAR_DISTANCES]
                    + [f"HShear_{d}m" for d in LIDAR_DISTANCES]
                    + [f"TI_avg_{d}m" for d in LIDAR_DISTANCES])
    print("  + 全部气象特征（VShear+HShear+TI，所有距离）：")
    res, lgb_full, _, pred_full = run_tree_experiment(df, feat_all_met, "AllHWS+AllMet")
    all_results.extend(res)

    # ── Exp-4: LSTM（连续时间段滑动窗口）──────────────────────
    print()
    print("=" * 60)
    print("【Exp-4】LSTM 时序模型（仅在连续时间段内构建滑动窗口）")
    print("=" * 60)

    # 4a: Baseline（SCADA HWS + 当前功率）
    print("  LSTM Baseline（SCADA HWS + 当前功率）：")
    res = run_lstm_experiment(df, feat_baseline, "LSTM_Baseline")
    all_results.extend(res)

    # 4b: 全距离 HWS + 当前功率
    feat_lstm_hws = feat_base_hws
    print("  LSTM 全距离 HWS + 当前功率：")
    res = run_lstm_experiment(df, feat_lstm_hws, "LSTM_AllHWS")
    all_results.extend(res)

    # 4c: 全距离 HWS + 当前功率 + 全部气象特征
    print("  LSTM 全距离 HWS + 当前功率 + 全部气象特征：")
    res = run_lstm_experiment(df, feat_all_met, "LSTM_AllHWS+AllMet")
    all_results.extend(res)

    return all_results, lgb_full, feat_all_met, y_te_base, pred_baseline, best_dist


# ════════════════════════════════════════════════════════════════
# 6. 结果汇总与输出
# ════════════════════════════════════════════════════════════════

def print_summary(results):
    df_res = pd.DataFrame(results)

    print()
    print("=" * 60)
    print("【结果汇总：LightGBM 各实验 RMSE / R²】")
    print("=" * 60)
    lgb = df_res[df_res["model"] == "LightGBM"][["exp", "RMSE", "MAE", "R2", "N"]]
    print(lgb.to_string(index=False))

    print()
    print("=" * 60)
    print("【结果汇总：所有模型（含 XGBoost / LSTM）】")
    print("=" * 60)
    print(df_res[["model", "exp", "RMSE", "MAE", "R2", "N"]].to_string(index=False))

    df_res.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 全部结果已保存：{OUTPUT_CSV}")
    return df_res


# ════════════════════════════════════════════════════════════════
# 7. 可视化
# ════════════════════════════════════════════════════════════════

def plot_all(df_res, lgb_model, feat_all, y_te, pred_base, best_dist, plt):
    # ── Fig 1: Baseline vs 逐距离 LiDAR（LightGBM RMSE）──────
    dist_res = df_res[
        (df_res["model"] == "LightGBM") &
        (df_res["exp"].str.startswith("LiDAR_HWS_") & df_res["exp"].str.match(r"LiDAR_HWS_\d+m"))
    ].copy()
    if not dist_res.empty:
        dist_res["dist_num"] = dist_res["exp"].str.extract(r"(\d+)m$").astype(int)
        dist_res = dist_res.sort_values("dist_num")
        base_rmse = df_res[
            (df_res["model"] == "LightGBM") & (df_res["exp"] == "Baseline(SCADA_HWS+power)")
        ]["RMSE"].values[0]

        fig, ax = plt.subplots(figsize=(11, 5))
        bars = ax.bar(dist_res["dist_num"].astype(str), dist_res["RMSE"],
                      color="steelblue", label="SCADA HWS + 各距离 LiDAR HWS + 当前功率")
        ax.axhline(base_rmse, color="red", linestyle="--", linewidth=2,
                   label=f"基准（仅 SCADA HWS）RMSE={base_rmse:.1f} kW")
        for bar, val in zip(bars, dist_res["RMSE"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("激光雷达测量距离 (m)", fontsize=12)
        ax.set_ylabel("RMSE (kW)", fontsize=12)
        ax.set_title("逐距离 LiDAR 风速对功率预测精度的影响\n"
                     "（LightGBM，输入=SCADA HWS + 该距离 LiDAR HWS + 当前功率，预测步长 +10 min）",
                     fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(base_rmse, dist_res["RMSE"].max()) * 1.15)
        plt.tight_layout()
        fpath = os.path.join(OUTPUT_DIR, "#8_exp1_distance_vs_rmse.png")
        plt.savefig(fpath, dpi=150)
        plt.close(fig)
        print(f"  图表已保存：{fpath}")

    # ── Fig 2: 各实验 RMSE 综合对比（LightGBM）──────────────
    lgb_res = df_res[df_res["model"] == "LightGBM"].copy()
    lgb_res = lgb_res.sort_values("RMSE")
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(lgb_res) * 0.45)))
    axes[0].barh(lgb_res["exp"][::-1], lgb_res["RMSE"][::-1], color="steelblue")
    axes[0].set_xlabel("RMSE (kW)", fontsize=11)
    axes[0].set_title("LightGBM 各实验 RMSE（越低越好）", fontsize=12)
    axes[0].grid(axis="x", alpha=0.3)
    for i, (_, row) in enumerate(lgb_res[::-1].iterrows()):
        axes[0].text(row["RMSE"] + 1, i, f"{row['RMSE']:.1f}", va="center", fontsize=8)

    axes[1].barh(lgb_res["exp"][::-1], lgb_res["R2"][::-1], color="seagreen")
    axes[1].set_xlabel("R²", fontsize=11)
    axes[1].set_title("LightGBM 各实验 R²（越高越好）", fontsize=12)
    axes[1].grid(axis="x", alpha=0.3)
    for i, (_, row) in enumerate(lgb_res[::-1].iterrows()):
        axes[1].text(row["R2"] + 0.001, i, f"{row['R2']:.4f}", va="center", fontsize=8)

    plt.suptitle("LightGBM 各实验性能综合对比（预测步长 +10 min）",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fpath = os.path.join(OUTPUT_DIR, "#8_exp_comparison.png")
    plt.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"  图表已保存：{fpath}")

    # ── Fig 3: 各模型（LightGBM/XGBoost/LSTM）综合对比─────────
    key_exps = [
        "Baseline(SCADA_HWS+power)",
        f"LiDAR_HWS_{best_dist}m",
        f"LiDAR_HWS_全距离(40-300m)",
        "AllHWS+AllMet",
        "LSTM_Baseline",
        "LSTM_AllHWS",
        "LSTM_AllHWS+AllMet",
    ]
    rows = []
    for exp in key_exps:
        for model in ["LightGBM", "XGBoost", "LSTM"]:
            sub = df_res[(df_res["model"] == model) & (df_res["exp"] == exp)]
            if not sub.empty:
                r = sub.iloc[0]
                rows.append({"label": f"{model}\n{exp}", "RMSE": r["RMSE"], "R2": r["R2"]})
    if rows:
        dfp = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(dfp) * 0.45)))
        axes[0].barh(dfp["label"][::-1], dfp["RMSE"][::-1], color="steelblue")
        axes[0].set_xlabel("RMSE (kW)", fontsize=11)
        axes[0].set_title("关键实验 RMSE 对比", fontsize=12)
        axes[0].grid(axis="x", alpha=0.3)
        axes[1].barh(dfp["label"][::-1], dfp["R2"][::-1], color="darkorange")
        axes[1].set_xlabel("R²", fontsize=11)
        axes[1].set_title("关键实验 R² 对比", fontsize=12)
        axes[1].grid(axis="x", alpha=0.3)
        plt.suptitle("各模型×各特征组合性能对比（预测步长 +10 min）",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        fpath = os.path.join(OUTPUT_DIR, "#8_model_comparison.png")
        plt.savefig(fpath, dpi=150)
        plt.close(fig)
        print(f"  图表已保存：{fpath}")

    # ── Fig 4: LightGBM 最终模型特征重要性 ───────────────────
    if lgb_model is not None:
        df_imp = pd.DataFrame({
            "feature": feat_all,
            "importance": lgb_model.feature_importances_,
        }).sort_values("importance", ascending=False).head(30)
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.barh(df_imp["feature"][::-1], df_imp["importance"][::-1], color="darkorange")
        ax.set_xlabel("特征重要性（split count）", fontsize=12)
        ax.set_title("LightGBM（全特征）Top-30 特征重要性\n"
                     "（全距离 HWS + 当前功率 + 全气象特征）", fontsize=12)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        fpath = os.path.join(OUTPUT_DIR, "#8_feature_importance.png")
        plt.savefig(fpath, dpi=150)
        plt.close(fig)
        print(f"  图表已保存：{fpath}")

    # ── Fig 5: 预测值 vs 实际值（Baseline LightGBM）──────────
    if pred_base is not None and y_te is not None:
        vmin = min(float(np.nanmin(y_te)), float(np.nanmin(pred_base)))
        vmax = max(float(np.nanmax(y_te)), float(np.nanmax(pred_base)))
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_te, pred_base, s=5, alpha=0.3, color="steelblue")
        ax.plot([vmin, vmax], [vmin, vmax], "r--", label="完美预测线")
        mask = ~np.isnan(pred_base)
        r2 = r2_score(y_te[mask], pred_base[mask])
        ax.text(0.05, 0.92, f"R² = {r2:.4f}", transform=ax.transAxes,
                fontsize=11, color="darkred")
        ax.set_xlabel("实际功率 (kW)", fontsize=12)
        ax.set_ylabel("预测功率 (kW)", fontsize=12)
        ax.set_title("LightGBM Baseline 预测值 vs 实际值\n（SCADA HWS + 当前功率，步长 +10 min）",
                     fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fpath = os.path.join(OUTPUT_DIR, "#8_pred_vs_actual.png")
        plt.savefig(fpath, dpi=150)
        plt.close(fig)
        print(f"  图表已保存：{fpath}")


# ════════════════════════════════════════════════════════════════
# 8. 主入口
# ════════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 60)
    print("  激光雷达辅助风机功率预测 —— 模型训练与对比（第二版）")
    print("=" * 60)

    # 1. 数据加载
    df = load_wide_dataset()

    # 2. 实验
    results, lgb_full, feat_all_met, y_te_base, pred_base, best_dist = run_all_experiments(df)

    # 3. 汇总
    df_res = print_summary(results)

    # 4. 可视化
    plt_mod = try_plt()
    if plt_mod is not None:
        print()
        print("=" * 60)
        print("【生成图表】")
        print("=" * 60)
        plot_all(df_res, lgb_full, feat_all_met,
                 y_te_base, pred_base, best_dist, plt_mod)
    else:
        print("  matplotlib 未安装，跳过图表生成")

    print()
    print("🎉 全部实验完成！")


if __name__ == "__main__":
    main()
