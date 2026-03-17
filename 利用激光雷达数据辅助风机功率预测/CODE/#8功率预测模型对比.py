"""
#8 功率预测模型对比.py
======================
利用激光雷达数据辅助风机功率预测 —— 模型训练与对比

研究问题
--------
1. 激光雷达数据是否能提升风机功率预测精度？
2. 哪些距离下的激光雷达数据最有参考价值？
3. 不同预测时长下，激光雷达数据的辅助效果有何差异？
4. 哪些特征变量对功率预测贡献最大？

数据说明
--------
- 输入1：PROCESS_DATA/#7构建好的训练数据集.csv
  激光雷达 10 分钟平均数据，包含 11 个距离（0~300m）×各特征，
  其中 Distance=0 为 SCADA 机舱测量（无 RAWS/TI 等雷达特有特征）。

- 输入2：PROCESS_DATA/#3峡沙56号_时间戳对齐后数据_10分钟均值.csv
  包含目标变量：平均有功功率_风机导出_前10分钟均值（kW）

模型方案
--------
- Baseline：LightGBM / XGBoost，仅使用 SCADA 风速和风向作为输入
- LiDAR-per-distance：对每个距离（40m~300m）分别添加该距离激光雷达特征，
  对比与基准的精度提升
- LiDAR-all：使用全部距离激光雷达特征
- LSTM：利用历史时序序列（look_back 窗口），结合 LiDAR 特征

输出
----
- PROCESS_DATA/#8模型对比结果.csv：各模型在各预测步长下的 RMSE/MAE/R²
- 若环境支持绘图，生成以下 PNG 图片：
    #8_distance_vs_rmse.png          各距离激光雷达对 RMSE 的影响
    #8_model_comparison.png          各模型预测精度对比（柱状图）
    #8_feature_importance.png        LightGBM 全距离模型特征重要性
    #8_pred_vs_actual.png            最优模型预测值 vs 实际值散点图

运行方式
--------
    python "CODE/#8功率预测模型对比.py"

依赖
----
    pip install pandas numpy scikit-learn lightgbm xgboost matplotlib
    pip install tensorflow   # 可选，用于 LSTM
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

LIDAR_PATH = os.path.join(BASE_DIR, "PROCESS_DATA", "#7构建好的训练数据集.csv")
POWER_PATH = os.path.join(BASE_DIR, "PROCESS_DATA", "#3峡沙56号_时间戳对齐后数据_10分钟均值.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "PROCESS_DATA", "#8模型对比结果.csv")
FIG_DIST = os.path.join(BASE_DIR, "PROCESS_DATA", "#8_distance_vs_rmse.png")
FIG_MODEL = os.path.join(BASE_DIR, "PROCESS_DATA", "#8_model_comparison.png")
FIG_IMP = os.path.join(BASE_DIR, "PROCESS_DATA", "#8_feature_importance.png")
FIG_PRED = os.path.join(BASE_DIR, "PROCESS_DATA", "#8_pred_vs_actual.png")
FIG_STEP = os.path.join(BASE_DIR, "PROCESS_DATA", "#8_step_degradation.png")

# 距离列表（雷达测量距离，单位 m）
LIDAR_DISTANCES = [40, 60, 90, 120, 150, 180, 210, 240, 270, 300]

# LiDAR 特征列（每个距离均有）
LIDAR_FEATURE_COLS = ["RAWS", "HWS(hub)", "DIR(hub)", "Veer", "VShear", "HShear",
                       "TI1", "TI2", "TI3", "TI4"]

# SCADA 基础特征（Distance=0）
SCADA_FEATURE_COLS = ["HWS(hub)", "DIR(hub)"]

# 预测步长（单位：10 分钟间隔，1=10min, 2=20min, 3=30min）
PREDICT_STEPS = [0, 1, 2, 3]   # 0 表示当前时刻功率（无超前预测，上限性能）
LSTM_LOOK_BACK = 6              # LSTM 回望窗口：60 分钟
TEST_SIZE = 0.2                 # 训练/测试集划分比例
RANDOM_STATE = 42

# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate(y_true, y_pred, label=""):
    r = {
        "label": label,
        "RMSE": round(rmse(y_true, y_pred), 2),
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "R2": round(r2_score(y_true, y_pred), 4),
        "N": len(y_true),
    }
    return r


def try_import_lgb():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        print("  ⚠️  lightgbm 未安装，跳过 LightGBM 模型")
        return None


def try_import_xgb():
    try:
        import xgboost as xgb
        return xgb
    except ImportError:
        print("  ⚠️  xgboost 未安装，跳过 XGBoost 模型")
        return None


def try_import_tf():
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        return tf
    except ImportError:
        print("  ⚠️  tensorflow 未安装，跳过 LSTM 模型")
        return None


def try_import_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        # 尝试加载中文字体（Noto Sans CJK，Linux 常见位置）
        _zh_font_candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
        ]
        for _fp in _zh_font_candidates:
            if os.path.exists(_fp):
                fm.fontManager.addfont(_fp)
                _prop = fm.FontProperties(fname=_fp)
                plt.rcParams["font.family"] = _prop.get_name()
                break
        plt.rcParams["axes.unicode_minus"] = False
        return plt
    except ImportError:
        return None


# ──────────────────────────────────────────────────────────────
# 数据加载与预处理
# ──────────────────────────────────────────────────────────────

def load_and_prepare_data():
    """
    加载并整合 LiDAR 特征与功率目标变量，构建宽格式建模数据集。

    返回
    ----
    df_wide : pd.DataFrame
        每行对应一个 10 分钟时间戳，列包含：
        - DateAndTime
        - power_kw           : 目标变量（有功功率 kW）
        - HWS_scada          : SCADA 机舱风速（用于基准模型）
        - DIR_scada          : SCADA 机舱风向
        - HWS_{d}m ... TI4_{d}m : 各距离激光雷达特征
    """
    print("=" * 60)
    print("【数据加载】")
    print("=" * 60)

    # 1. 读取 LiDAR 数据（step 7）
    df7 = pd.read_csv(LIDAR_PATH)
    df7["DateAndTime"] = pd.to_datetime(df7["DateAndTime"])
    print(f"  LiDAR 数据：{len(df7):,} 行，{df7['DateAndTime'].nunique():,} 个时间戳")

    # 2. 读取功率数据（step 3）
    df3 = pd.read_csv(POWER_PATH, encoding="gbk")
    df3["时间"] = pd.to_datetime(df3["时间"])
    print(f"  功率数据：{len(df3):,} 行")

    # 3. 提取 SCADA 基础特征（Distance=0）
    d0 = df7[df7["Distance"] == 0][["DateAndTime", "HWS(hub)", "DIR(hub)"]].copy()
    d0 = d0.rename(columns={"HWS(hub)": "HWS_scada", "DIR(hub)": "DIR_scada"})

    # 4. 提取各距离 LiDAR 特征，转为宽格式
    d_lidar = df7[df7["Distance"] != 0].copy()
    pivot_parts = []
    for feat in LIDAR_FEATURE_COLS:
        pv = d_lidar.pivot_table(index="DateAndTime", columns="Distance", values=feat, aggfunc="first")
        pv.columns = [f"{feat}_{int(c)}m" for c in pv.columns]
        pivot_parts.append(pv)
    lidar_wide = pd.concat(pivot_parts, axis=1).reset_index()

    # 5. 合并所有数据
    df_wide = d0.merge(lidar_wide, on="DateAndTime", how="inner")
    df_wide = df_wide.merge(
        df3[["时间", "平均有功功率_风机导出_前10分钟均值"]].rename(
            columns={"时间": "DateAndTime", "平均有功功率_风机导出_前10分钟均值": "power_kw"}
        ),
        on="DateAndTime",
        how="left",
    )

    # 6. 过滤有效数据（功率非 NaN 且非负）
    before = len(df_wide)
    df_wide = df_wide.dropna(subset=["power_kw"])
    df_wide = df_wide[df_wide["power_kw"] >= -100]   # 允许轻微负值（停机/制动）
    df_wide = df_wide.sort_values("DateAndTime").reset_index(drop=True)
    print(f"  合并后有效时间戳：{len(df_wide):,}（过滤前：{before:,}）")
    print(f"  功率范围：{df_wide['power_kw'].min():.1f} ~ {df_wide['power_kw'].max():.1f} kW")
    print(f"  时间范围：{df_wide['DateAndTime'].min()} ~ {df_wide['DateAndTime'].max()}")
    return df_wide


# ──────────────────────────────────────────────────────────────
# 特征集构建
# ──────────────────────────────────────────────────────────────

def get_baseline_features():
    """基准特征：SCADA 机舱风速 + 风向（方向编码）。"""
    return ["HWS_scada", "DIR_sin_scada", "DIR_cos_scada"]


def add_dir_encoding(df):
    """对风向列做 sin/cos 编码，避免 0°/360° 不连续问题。"""
    df = df.copy()
    df["DIR_sin_scada"] = np.sin(np.deg2rad(df["DIR_scada"]))
    df["DIR_cos_scada"] = np.cos(np.deg2rad(df["DIR_scada"]))
    for d in LIDAR_DISTANCES:
        col = f"DIR(hub)_{d}m"
        if col in df.columns:
            df[f"DIR_sin_{d}m"] = np.sin(np.deg2rad(df[col]))
            df[f"DIR_cos_{d}m"] = np.cos(np.deg2rad(df[col]))
    return df


def get_lidar_features_for_dist(distance):
    """返回指定距离激光雷达的特征列名列表（含方向编码）。"""
    feats = []
    for feat in LIDAR_FEATURE_COLS:
        col = f"{feat}_{distance}m"
        if feat == "DIR(hub)":
            feats += [f"DIR_sin_{distance}m", f"DIR_cos_{distance}m"]
        else:
            feats.append(col)
    return feats


def get_all_lidar_features():
    """所有距离 LiDAR 特征。"""
    feats = []
    for d in LIDAR_DISTANCES:
        feats.extend(get_lidar_features_for_dist(d))
    return feats


# ──────────────────────────────────────────────────────────────
# 树模型训练（LightGBM / XGBoost）
# ──────────────────────────────────────────────────────────────def train_lgb_proper(X_train, y_train, X_val, y_val, X_test):
    lgb = try_import_lgb()
    if lgb is None:
        return None, None, None
    params = {
        "objective": "regression",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "verbose": -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    return model, model.predict(X_val), model.predict(X_test)


def train_xgb_proper(X_train, y_train, X_val, y_val, X_test):
    xgb = try_import_xgb()
    if xgb is None:
        return None, None, None
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="rmse",
        early_stopping_rounds=50,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, model.predict(X_val), model.predict(X_test)


# ──────────────────────────────────────────────────────────────
# LSTM 模型训练
# ──────────────────────────────────────────────────────────────

def train_lstm(X_seq_train, y_seq_train, X_seq_test, y_seq_test, look_back=LSTM_LOOK_BACK):
    """
    X_seq_*: shape (samples, look_back, n_features)
    y_seq_*: shape (samples,)
    """
    tf = try_import_tf()
    if tf is None:
        return None, None

    n_features = X_seq_train.shape[2]

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(look_back, n_features), return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

    cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=0)
    model.fit(
        X_seq_train, y_seq_train,
        validation_split=0.1,
        epochs=100,
        batch_size=64,
        callbacks=[cb],
        verbose=0,
    )
    pred = model.predict(X_seq_test, verbose=0).flatten()
    return model, pred


def make_sequences(X, y, look_back=LSTM_LOOK_BACK):
    """
    将时序数据转化为滑动窗口序列。
    返回 X_seq: (N, look_back, n_features), y_seq: (N,)
    """
    Xs, ys = [], []
    for i in range(look_back, len(X)):
        Xs.append(X[i - look_back:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


# ──────────────────────────────────────────────────────────────
# 实验主流程
# ──────────────────────────────────────────────────────────────

def run_experiments(df):
    """
    对各模型、各距离、各预测步长运行实验，返回结果 DataFrame。
    """
    df = add_dir_encoding(df)
    results = []

    # ── 1. 基准模型：SCADA only（预测步长 0 和 1）──────────────
    print()
    print("=" * 60)
    print("【实验 1：基准模型（仅 SCADA 特征）】")
    print("=" * 60)

    for step in [0, 1, 2, 3]:
        label = f"step+{step}" if step > 0 else "step+0(当前)"
        print(f"  预测步长：{step} × 10min ({step * 10} min 超前)")

        # 构建预测目标：step 步之后的功率
        df_exp = df.copy()
        if step > 0:
            df_exp["target"] = df_exp["power_kw"].shift(-step)
            df_exp = df_exp.dropna(subset=["target"])
        else:
            df_exp["target"] = df_exp["power_kw"]

        feat_cols = [c for c in get_baseline_features() if c in df_exp.columns]
        X = df_exp[feat_cols].fillna(df_exp[feat_cols].median()).values
        y = df_exp["target"].values

        # 按时序顺序划分（不随机，保持时间顺序）
        split = int(len(X) * (1 - TEST_SIZE))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        val_split = int(len(X_train) * 0.9)
        X_tr, X_val = X_train[:val_split], X_train[val_split:]
        y_tr, y_val = y_train[:val_split], y_train[val_split:]

        # LightGBM
        _, _, pred_lgb = train_lgb_proper(X_tr, y_tr, X_val, y_val, X_test)
        if pred_lgb is not None:
            r = evaluate(y_test, pred_lgb, f"LightGBM_baseline")
            r.update({"model": "LightGBM", "features": "SCADA_only", "distance": "N/A", "predict_step": step})
            results.append(r)
            print(f"    LightGBM  RMSE={r['RMSE']:.1f} MAE={r['MAE']:.1f} R²={r['R2']:.4f}")

        # XGBoost
        _, _, pred_xgb = train_xgb_proper(X_tr, y_tr, X_val, y_val, X_test)
        if pred_xgb is not None:
            r = evaluate(y_test, pred_xgb, f"XGBoost_baseline")
            r.update({"model": "XGBoost", "features": "SCADA_only", "distance": "N/A", "predict_step": step})
            results.append(r)
            print(f"    XGBoost   RMSE={r['RMSE']:.1f} MAE={r['MAE']:.1f} R²={r['R2']:.4f}")

    # ── 2. LiDAR 各距离对比（step=1, LightGBM）────────────────
    print()
    print("=" * 60)
    print("【实验 2：逐距离 LiDAR 特征对比（预测步长 +1，即 10 min 超前）】")
    print("=" * 60)

    step = 1
    df_exp1 = df.copy()
    df_exp1["target"] = df_exp1["power_kw"].shift(-step)
    df_exp1 = df_exp1.dropna(subset=["target"])
    split = int(len(df_exp1) * (1 - TEST_SIZE))
    y_test_step1 = df_exp1["target"].values[split:]

    for dist in LIDAR_DISTANCES:
        # 基准特征 + 该距离 LiDAR 特征
        lidar_feats = [c for c in get_lidar_features_for_dist(dist) if c in df_exp1.columns]
        if not lidar_feats:
            continue
        feat_cols = [c for c in get_baseline_features() if c in df_exp1.columns] + lidar_feats

        X = df_exp1[feat_cols].fillna(df_exp1[feat_cols].median()).values
        y = df_exp1["target"].values

        X_train, X_test = X[:split], X[split:]
        y_train = y[:split]
        val_split = int(len(X_train) * 0.9)
        X_tr, X_val = X_train[:val_split], X_train[val_split:]
        y_tr, y_val = y_train[:val_split], y_train[val_split:]

        _, _, pred = train_lgb_proper(X_tr, y_tr, X_val, y_val, X_test)
        if pred is not None:
            r = evaluate(y_test_step1, pred, f"LightGBM_LIDAR_{dist}m")
            r.update({"model": "LightGBM", "features": f"SCADA+LiDAR_{dist}m", "distance": dist, "predict_step": step})
            results.append(r)
            print(f"    {dist:>3}m  RMSE={r['RMSE']:.1f}  MAE={r['MAE']:.1f}  R²={r['R2']:.4f}")

    # ── 3. 全距离 LiDAR（step=1，LightGBM + XGBoost）──────────
    print()
    print("=" * 60)
    print("【实验 3：全距离 LiDAR 特征（预测步长 +1）】")
    print("=" * 60)

    all_lidar = [c for c in get_all_lidar_features() if c in df_exp1.columns]
    feat_cols_all = [c for c in get_baseline_features() if c in df_exp1.columns] + all_lidar
    X_all = df_exp1[feat_cols_all].fillna(df_exp1[feat_cols_all].median()).values
    y_all = df_exp1["target"].values

    X_tr_all, X_val_all = X_all[:split][:int(split * 0.9)], X_all[:split][int(split * 0.9):]
    y_tr_all, y_val_all = y_all[:split][:int(split * 0.9)], y_all[:split][int(split * 0.9):]
    X_test_all = X_all[split:]

    lgb_all_model, _, pred_lgb_all = train_lgb_proper(X_tr_all, y_tr_all, X_val_all, y_val_all, X_test_all)
    if pred_lgb_all is not None:
        r = evaluate(y_test_step1, pred_lgb_all, "LightGBM_all_LiDAR")
        r.update({"model": "LightGBM", "features": "SCADA+all_LiDAR", "distance": "all", "predict_step": step})
        results.append(r)
        print(f"  LightGBM+全距离  RMSE={r['RMSE']:.1f}  MAE={r['MAE']:.1f}  R²={r['R2']:.4f}")

    _, _, pred_xgb_all = train_xgb_proper(X_tr_all, y_tr_all, X_val_all, y_val_all, X_test_all)
    if pred_xgb_all is not None:
        r = evaluate(y_test_step1, pred_xgb_all, "XGBoost_all_LiDAR")
        r.update({"model": "XGBoost", "features": "SCADA+all_LiDAR", "distance": "all", "predict_step": step})
        results.append(r)
        print(f"  XGBoost+全距离   RMSE={r['RMSE']:.1f}  MAE={r['MAE']:.1f}  R²={r['R2']:.4f}")

    # ── 4. LSTM 模型（基准 + 全距离，step=1）──────────────────
    print()
    print("=" * 60)
    print("【实验 4：LSTM 时序模型】")
    print("=" * 60)

    tf = try_import_tf()
    if tf is not None:
        scaler = StandardScaler()

        # 4a. LSTM Baseline（SCADA only）
        feat_base = [c for c in get_baseline_features() if c in df_exp1.columns]
        X_lstm_base = df_exp1[feat_base].fillna(df_exp1[feat_base].median()).values
        y_lstm = df_exp1["target"].values

        X_sc = scaler.fit_transform(X_lstm_base)
        X_seq, y_seq = make_sequences(X_sc, y_lstm)
        split_seq = int(len(X_seq) * (1 - TEST_SIZE))

        _, pred_lstm_base = train_lstm(X_seq[:split_seq], y_seq[:split_seq],
                                       X_seq[split_seq:], y_seq[split_seq:])
        if pred_lstm_base is not None:
            r = evaluate(y_seq[split_seq:], pred_lstm_base, "LSTM_baseline")
            r.update({"model": "LSTM", "features": "SCADA_only", "distance": "N/A", "predict_step": step})
            results.append(r)
            print(f"  LSTM+SCADA  RMSE={r['RMSE']:.1f}  MAE={r['MAE']:.1f}  R²={r['R2']:.4f}")

        # 4b. LSTM with all LiDAR
        scaler2 = StandardScaler()
        X_lstm_all = df_exp1[feat_cols_all].fillna(df_exp1[feat_cols_all].median()).values
        X_sc2 = scaler2.fit_transform(X_lstm_all)
        X_seq2, y_seq2 = make_sequences(X_sc2, y_lstm)
        split_seq2 = int(len(X_seq2) * (1 - TEST_SIZE))

        _, pred_lstm_all = train_lstm(X_seq2[:split_seq2], y_seq2[:split_seq2],
                                      X_seq2[split_seq2:], y_seq2[split_seq2:])
        if pred_lstm_all is not None:
            r = evaluate(y_seq2[split_seq2:], pred_lstm_all, "LSTM_all_LiDAR")
            r.update({"model": "LSTM", "features": "SCADA+all_LiDAR", "distance": "all", "predict_step": step})
            results.append(r)
            print(f"  LSTM+全距离 RMSE={r['RMSE']:.1f}  MAE={r['MAE']:.1f}  R²={r['R2']:.4f}")
    else:
        print("  跳过 LSTM（tensorflow 未安装）")

    return results, lgb_all_model, feat_cols_all, df_exp1, split, y_test_step1, pred_lgb_all


# ──────────────────────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────────────────────

def plot_distance_vs_rmse(results, plt):
    """各距离 LiDAR 对 RMSE 的影响（条形图 + 基准线 + 数值标注）。"""
    df_res = pd.DataFrame(results)

    dist_rows = df_res[
        (df_res["model"] == "LightGBM") &
        (df_res["features"].str.startswith("SCADA+LiDAR")) &
        (df_res["predict_step"] == 1)
    ].copy()
    dist_rows["distance"] = dist_rows["distance"].astype(int)
    dist_rows = dist_rows.sort_values("distance")

    baseline_rmse = df_res[
        (df_res["model"] == "LightGBM") &
        (df_res["features"] == "SCADA_only") &
        (df_res["predict_step"] == 1)
    ]["RMSE"].values
    if len(baseline_rmse) == 0:
        return
    baseline_rmse = baseline_rmse[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(dist_rows["distance"].astype(str), dist_rows["RMSE"],
                  color="steelblue", label="SCADA + 各距离 LiDAR")
    ax.axhline(baseline_rmse, color="red", linestyle="--", linewidth=2,
               label=f"基准（仅 SCADA）RMSE={baseline_rmse:.1f} kW")
    for bar, val in zip(bars, dist_rows["RMSE"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("激光雷达测量距离 (m)", fontsize=12)
    ax.set_ylabel("RMSE (kW)", fontsize=12)
    ax.set_title("不同距离激光雷达特征对功率预测精度的影响\n（LightGBM，预测步长 +10 min）", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(baseline_rmse * 1.12, dist_rows["RMSE"].max() * 1.15))
    plt.tight_layout()
    plt.savefig(FIG_DIST, dpi=150)
    plt.close(fig)
    print(f"  图表已保存：{FIG_DIST}")


def plot_model_comparison(results, plt):
    """各模型在 step=1 下的 RMSE / R² 对比（水平条形图）。"""
    df_res = pd.DataFrame(results)
    key_models = [
        ("LightGBM", "SCADA_only",      "LightGBM\n(仅SCADA)"),
        ("XGBoost",  "SCADA_only",      "XGBoost\n(仅SCADA)"),
        ("LightGBM", "SCADA+all_LiDAR", "LightGBM\n(SCADA+全距离LiDAR)"),
        ("XGBoost",  "SCADA+all_LiDAR", "XGBoost\n(SCADA+全距离LiDAR)"),
        ("LSTM",     "SCADA_only",      "LSTM\n(仅SCADA)"),
        ("LSTM",     "SCADA+all_LiDAR", "LSTM\n(SCADA+全距离LiDAR)"),
    ]
    rows = []
    for model, feats, label in key_models:
        sub = df_res[(df_res["model"] == model) & (df_res["features"] == feats) & (df_res["predict_step"] == 1)]
        if not sub.empty:
            rows.append({"label": label, "RMSE": sub["RMSE"].values[0], "R2": sub["R2"].values[0]})
    if not rows:
        return

    df_plot = pd.DataFrame(rows)
    colors = ["#4C72B0", "#4C72B0", "#DD8452", "#DD8452", "#55A868", "#55A868"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bars1 = axes[0].barh(df_plot["label"], df_plot["RMSE"], color=colors[:len(df_plot)])
    axes[0].set_xlabel("RMSE (kW)", fontsize=12)
    axes[0].set_title("各模型 RMSE（越低越好）", fontsize=13)
    axes[0].grid(axis="x", alpha=0.3)
    for bar, val in zip(bars1, df_plot["RMSE"]):
        axes[0].text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}", va="center", fontsize=9)

    bars2 = axes[1].barh(df_plot["label"], df_plot["R2"], color=colors[:len(df_plot)])
    axes[1].set_xlabel("R²", fontsize=12)
    axes[1].set_title("各模型 R²（越高越好）", fontsize=13)
    axes[1].grid(axis="x", alpha=0.3)
    for bar, val in zip(bars2, df_plot["R2"]):
        axes[1].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=9)

    plt.suptitle("功率预测模型性能对比（预测步长 +10 min）", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_MODEL, dpi=150)
    plt.close(fig)
    print(f"  图表已保存：{FIG_MODEL}")


def plot_step_degradation(results, plt):
    """RMSE 和 R² 随预测步长的变化曲线。"""
    df_res = pd.DataFrame(results)
    lgb_base = df_res[(df_res["model"] == "LightGBM") & (df_res["features"] == "SCADA_only")]
    xgb_base = df_res[(df_res["model"] == "XGBoost") & (df_res["features"] == "SCADA_only")]
    if lgb_base.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(lgb_base["predict_step"] * 10, lgb_base["RMSE"], "o-", label="LightGBM", color="steelblue")
    axes[0].plot(xgb_base["predict_step"] * 10, xgb_base["RMSE"], "s-", label="XGBoost", color="darkorange")
    axes[0].set_xlabel("预测超前时长 (min)", fontsize=12)
    axes[0].set_ylabel("RMSE (kW)", fontsize=12)
    axes[0].set_title("预测时长对精度的影响\n（仅 SCADA 特征）", fontsize=12)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(lgb_base["predict_step"] * 10, lgb_base["R2"], "o-", label="LightGBM", color="steelblue")
    axes[1].plot(xgb_base["predict_step"] * 10, xgb_base["R2"], "s-", label="XGBoost", color="darkorange")
    axes[1].set_xlabel("预测超前时长 (min)", fontsize=12)
    axes[1].set_ylabel("R²", fontsize=12)
    axes[1].set_title("预测时长对精度的影响\n（仅 SCADA 特征）", fontsize=12)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("随预测步长增加，精度逐步降低", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_STEP, dpi=150)
    plt.close(fig)
    print(f"  图表已保存：{FIG_STEP}")


def plot_feature_importance(lgb_model, feat_cols, plt):
    """LightGBM 全距离模型 Top-30 特征重要性。"""
    if lgb_model is None:
        return
    df_imp = pd.DataFrame({"feature": feat_cols, "importance": lgb_model.feature_importances_})
    df_imp = df_imp.sort_values("importance", ascending=False).head(30)

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(df_imp["feature"][::-1], df_imp["importance"][::-1], color="darkorange")
    ax.set_xlabel("Feature Importance (split count)", fontsize=12)
    ax.set_title("LightGBM 全距离模型 Top-30 特征重要性\n（预测步长 +10 min）", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_IMP, dpi=150)
    plt.close(fig)
    print(f"  图表已保存：{FIG_IMP}")


def plot_pred_vs_actual(y_true, y_pred, plt, title="预测值 vs 实际值"):
    """预测值 vs 实际值散点图（含 R² 标注）。"""
    vmin = min(float(y_true.min()), float(y_pred.min()))
    vmax = max(float(y_true.max()), float(y_pred.max()))
    r2 = r2_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, s=5, alpha=0.3, color="steelblue")
    ax.plot([vmin, vmax], [vmin, vmax], "r--", label="完美预测")
    ax.text(0.05, 0.92, f"R² = {r2:.4f}", transform=ax.transAxes,
            fontsize=11, color="darkred")
    ax.set_xlabel("实际功率 (kW)", fontsize=12)
    ax.set_ylabel("预测功率 (kW)", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_PRED, dpi=150)
    plt.close(fig)
    print(f"  图表已保存：{FIG_PRED}")


# ──────────────────────────────────────────────────────────────
# 汇总输出
# ──────────────────────────────────────────────────────────────

def print_summary(results):
    df_res = pd.DataFrame(results)
    print()
    print("=" * 60)
    print("【汇总：各模型 R² 与 RMSE（预测步长 +10 min）】")
    print("=" * 60)
    step1 = df_res[df_res["predict_step"] == 1][["model", "features", "RMSE", "MAE", "R2", "N"]]
    print(step1.to_string(index=False))

    print()
    print("=" * 60)
    print("【汇总：随预测步长变化（LightGBM, SCADA_only）】")
    print("=" * 60)
    lgb_base = df_res[(df_res["model"] == "LightGBM") & (df_res["features"] == "SCADA_only")]
    if not lgb_base.empty:
        print(lgb_base[["predict_step", "RMSE", "MAE", "R2"]].to_string(index=False))

    print()
    print("=" * 60)
    print("【汇总：逐距离 LiDAR 对比（LightGBM, step=1）】")
    print("=" * 60)
    dist_res = df_res[
        (df_res["model"] == "LightGBM") &
        (df_res["features"].str.startswith("SCADA+LiDAR")) &
        (df_res["predict_step"] == 1)
    ].sort_values("distance")
    if not dist_res.empty:
        print(dist_res[["distance", "RMSE", "MAE", "R2"]].to_string(index=False))

    # 保存到 CSV
    df_res.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print()
    print(f"✅ 全部结果已保存至：{OUTPUT_CSV}")


# ──────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 60)
    print("  激光雷达辅助风机功率预测 —— 模型训练与对比")
    print("=" * 60)

    # 1. 加载数据
    df = load_and_prepare_data()

    # 2. 运行实验
    results, lgb_model, feat_cols_all, df_exp1, split, y_test, pred_lgb_all = run_experiments(df)

    # 3. 汇总输出
    print_summary(results)

    # 4. 绘图
    plt = try_import_plt()
    if plt is not None:
        print()
        print("=" * 60)
        print("【生成图表】")
        print("=" * 60)
        plot_distance_vs_rmse(results, plt)
        plot_model_comparison(results, plt)
        plot_step_degradation(results, plt)
        if lgb_model is not None and pred_lgb_all is not None:
            plot_feature_importance(lgb_model, feat_cols_all, plt)
            plot_pred_vs_actual(y_test, pred_lgb_all, plt,
                                title="LightGBM（全距离 LiDAR）预测值 vs 实际值（step+1）")
    else:
        print("  matplotlib 未安装，跳过图表生成")

    print()
    print("🎉 全部实验完成！")


if __name__ == "__main__":
    main()
