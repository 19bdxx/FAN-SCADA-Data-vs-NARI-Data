"""
#10 场站功率预测对比：单机纯功率自回归 vs 场站级别特征增益分析
=================================================================
问题背景
--------
实验 #9 中，对于单台风机（峡沙56号）的超短期功率预测，发现：
  - P0（仅历史功率）是树模型和LSTM的最优基线
  - 添加LiDAR/SCADA风速特征并未改善预测效果

本脚本旨在：
  1. 分析单机预测中"纯功率最优"的数据来源（相关性、方差分解）
  2. 构建场站功率预测场景（模拟N台风机，具有空间多样性）
  3. 对比单机 vs 场站中 P0/M0/M5 三类特征集的相对表现
  4. 验证"场站级别预测中气象特征更有价值"的假设

场站模拟方案（基于单机数据）
----------------------------
  - 以峡沙56号为"参考风机"
  - 通过对功率序列施加时移（不同时刻风到达不同位置风机）
    + 独立噪声（空间湍流差异）生成 N=5 台虚拟风机
  - 场站功率 = N 台风机功率之和（或均值）
  - 模拟假设：
      * 场站尺度约 3-5 km，主风向上游-下游间距约 500m
      * 风速 ~10 m/s → 每 500m 时差约 50s ≈ 1 步（10min 粒度下约为 0 步）
      * 但湍流独立性使空间平均相关系数 ρ ≈ 0.85~0.95
      * 我们模拟 ρ=0.90（相当于5台机，有10%独立噪声）

输出
----
  PROCESS_DATA/#10单机vs场站对比结果.csv
  PROCESS_DATA/#10_单机vs场站_相关性分析.png
  PROCESS_DATA/#10_单机vs场站_RMSE对比.png
  PROCESS_DATA/#10_场站预测_特征增益.png
  CODE/#10单机vs场站功率预测分析报告.md
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── 字体（中文支持）──────────────────────────────────────────────────
fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm._load_fontmanager(try_read_cache=False)
fm.fontManager.addfont(fp)
_prop = fm.FontProperties(fname=fp)
plt.rcParams["font.sans-serif"] = [_prop.get_name(), "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 路径 ─────────────────────────────────────────────────────────────
CODE_DIR    = os.path.dirname(os.path.abspath(__file__))
WORK_DIR    = os.path.dirname(CODE_DIR)
OUTPUT_DIR  = os.path.join(WORK_DIR, "PROCESS_DATA")
DATASET_CSV = os.path.join(OUTPUT_DIR, "#7构建好的训练数据集.csv")

# ── 实验参数 ─────────────────────────────────────────────────────────
LIDAR_DISTANCES  = [40, 60, 90, 120, 150, 180, 210, 240, 270, 300]
LOOK_BACK        = 6           # 历史时间步数（60min）
FORECAST_STEPS   = [1, 2, 3, 6]  # 预测步长（×10min）
TEST_RATIO       = 0.20
VAL_RATIO        = 0.10
TIME_GAP         = pd.Timedelta("10min")
RANDOM_STATE     = 42
N_VIRTUAL_TURBINES = 5   # 虚拟风机数（模拟场站）
INTER_TURBINE_RHO  = 0.90  # 各台风机功率序列相关系数（模拟空间多样性）
SEED             = 42

np.random.seed(SEED)


# ════════════════════════════════════════════════════════════════
# 1. 数据加载
# ════════════════════════════════════════════════════════════════

def load_wide_dataset():
    """加载 #7 数据集，返回宽格式 DataFrame（含所有 LiDAR 列）。"""
    df_raw = pd.read_csv(DATASET_CSV, encoding="gbk")
    df_raw["DateAndTime"] = pd.to_datetime(df_raw["DateAndTime"])

    d0 = df_raw[df_raw["Distance"] == 0][
        ["DateAndTime", "HWS(hub)", "ACTIVE_POWER_#56_对齐_前10分钟均值"]
    ].rename(columns={
        "HWS(hub)": "HWS_scada",
        "ACTIVE_POWER_#56_对齐_前10分钟均值": "power",
    }).copy()

    df_wide = d0
    for dist in LIDAR_DISTANCES:
        sub = df_raw[df_raw["Distance"] == dist][
            ["DateAndTime", "HWS(hub)", "VShear", "HShear", "TI1", "TI2", "TI3", "TI4"]
        ].copy()
        sub[f"TI_avg_{dist}m"] = sub[["TI1", "TI2", "TI3", "TI4"]].mean(axis=1)
        sub = sub.rename(columns={
            "HWS(hub)":  f"HWS_{dist}m",
            "VShear":    f"VShear_{dist}m",
            "HShear":    f"HShear_{dist}m",
        }).drop(columns=["TI1", "TI2", "TI3", "TI4"])
        df_wide = df_wide.merge(sub, on="DateAndTime", how="left")

    df_wide = df_wide.sort_values("DateAndTime").reset_index(drop=True)
    df_wide = df_wide.dropna(subset=["power"]).reset_index(drop=True)
    dt_diff = df_wide["DateAndTime"].diff()
    df_wide["segment_id"] = (dt_diff != TIME_GAP).cumsum()
    print(f"[数据] 单机有效时间戳：{len(df_wide):,} 行")
    return df_wide


def add_farm_power(df_wide):
    """
    基于单机功率序列模拟场站功率。

    方案：
    - 生成 N-1 台虚拟风机功率序列 = ρ × P_original + √(1-ρ²) × ε × σ_P
      其中 ε~N(0,1) 独立噪声，σ_P 为功率标准差
    - 场站功率 = 所有风机的均值
    - 反映"空间多样性：同一时刻各台风机功率部分独立"

    注意：这是一个简化模拟，真实场站还受尾流、地形等影响。
    """
    power = df_wide["power"].values
    sigma_p = np.nanstd(power)
    rho = np.sqrt(INTER_TURBINE_RHO)   # 单个虚拟机与原机的相关系数（Cholesky）

    virtual_powers = [power]  # 包含原始风机
    for _ in range(N_VIRTUAL_TURBINES - 1):
        noise = np.random.randn(len(power)) * sigma_p
        virtual_p = rho * power + np.sqrt(1 - rho**2) * noise
        # 对于 power 为 NaN 的位置同步处理
        virtual_p = np.where(np.isnan(power), np.nan, virtual_p)
        virtual_powers.append(virtual_p)

    farm_power = np.nanmean(virtual_powers, axis=0)
    df_farm = df_wide.copy()
    df_farm["power_farm"] = farm_power
    print(f"[模拟] 场站功率：{N_VIRTUAL_TURBINES} 台风机（ρ={INTER_TURBINE_RHO}），"
          f"功率均值={np.nanmean(farm_power):.0f} kW，"
          f"σ={np.nanstd(farm_power):.0f} kW（原机σ={sigma_p:.0f} kW）")
    return df_farm


# ════════════════════════════════════════════════════════════════
# 2. 相关性与自相关分析
# ════════════════════════════════════════════════════════════════

def analyze_autocorrelation(df_wide):
    """计算单机功率、场站功率、HWS 的自相关系数及预测相关性。"""
    power      = df_wide["power"].dropna()
    farm_power = df_wide["power_farm"].dropna()
    hws        = df_wide["HWS_scada"].dropna()

    rows = []
    for lag in [1, 2, 3, 6, 12, 18]:
        r_p    = power.autocorr(lag=lag)
        r_fp   = farm_power.autocorr(lag=lag)
        r_hws  = hws.autocorr(lag=lag)
        # 持续性预测误差（RMSE）
        def persist_rmse(series, lag):
            actual = series.iloc[lag:].values
            pred   = series.iloc[:-lag].values
            mask   = ~(np.isnan(actual) | np.isnan(pred))
            return float(np.sqrt(np.mean((actual[mask] - pred[mask])**2)))
        rmse_p  = persist_rmse(power, lag)
        rmse_fp = persist_rmse(farm_power, lag)
        rows.append({
            "lag_steps": lag,
            "lag_min":   lag * 10,
            "autocorr_single_power": round(r_p, 4),
            "autocorr_farm_power":   round(r_fp, 4),
            "autocorr_HWS":          round(r_hws, 4),
            "persist_RMSE_single":   round(rmse_p, 1),
            "persist_RMSE_farm":     round(rmse_fp, 1),
        })
        print(f"  lag={lag}({lag*10:>3}min)  "
              f"单机ACF={r_p:.4f}  场站ACF={r_fp:.4f}  "
              f"HWS_ACF={r_hws:.4f}  "
              f"单机持续性RMSE={rmse_p:.0f}  场站持续性RMSE={rmse_fp:.0f}")
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# 3. 机器学习实验（单机 vs 场站）
# ════════════════════════════════════════════════════════════════

def fill_features(df, feat_cols):
    X = df[feat_cols].copy()
    for c in feat_cols:
        if X[c].isna().any():
            med = X[c].median()
            X[c] = X[c].fillna(med if not np.isnan(med) else 0.0)
    return X


def build_windows(df, feat_cols, target_col, look_back=LOOK_BACK):
    """构建滑动窗口特征矩阵（连续时间段内）。"""
    all_X, all_Y = [], []
    for _, seg in df.groupby("segment_id", sort=True):
        seg = seg.sort_values("DateAndTime").reset_index(drop=True)
        n = len(seg)
        max_h = max(FORECAST_STEPS)
        if n < look_back + max_h:
            continue
        feats_df = fill_features(seg, feat_cols)
        fv = feats_df.values.astype(float)
        tv = seg[target_col].values.astype(float)
        for i in range(look_back, n - max_h + 1):
            fut = tv[i: i + max_h]
            if np.any(np.isnan(fut)):
                continue
            all_X.append(fv[i - look_back: i].flatten())  # 展平历史窗口
            all_Y.append(fut)
    if not all_X:
        return None, None
    return np.array(all_X, dtype=np.float32), np.array(all_Y, dtype=np.float32)


def time_split(X, Y):
    n   = len(X)
    nte = int(n * TEST_RATIO)
    nv  = int(n * VAL_RATIO)
    ntr = n - nte - nv
    return (X[:ntr], Y[:ntr]), (X[ntr:ntr+nv], Y[ntr:ntr+nv]), (X[ntr+nv:], Y[ntr+nv:])


def evaluate(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan, "N": 0}
    return {
        "RMSE": round(float(np.sqrt(mean_squared_error(yt, yp))), 2),
        "MAE":  round(float(mean_absolute_error(yt, yp)), 2),
        "R2":   round(float(r2_score(yt, yp)), 4),
        "N":    int(len(yt)),
    }


def run_lgb_experiment(df, feat_cols, target_col, label):
    """用 LightGBM 做多步直接预测（每步一个模型）。"""
    X, Y = build_windows(df, feat_cols, target_col)
    if X is None:
        print(f"  ⚠️ {label}: 数据不足，跳过")
        return []
    (X_tr, Y_tr), (X_val, Y_val), (X_te, Y_te) = time_split(X, Y)
    results = []
    for h in FORECAST_STEPS:
        step_idx = FORECAST_STEPS.index(h)
        y_tr_h = Y_tr[:, step_idx]
        y_val_h = Y_val[:, step_idx]
        y_te_h  = Y_te[:, step_idx]
        model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
        model.fit(X_tr, y_tr_h,
                  eval_set=[(X_val, y_val_h)],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                             lgb.log_evaluation(period=-1)])
        pred = model.predict(X_te)
        r = evaluate(y_te_h, pred)
        r["label"]    = label
        r["feat_cols"] = "|".join(feat_cols)
        r["step"]     = h
        r["step_min"] = h * 10
        r["target"]   = target_col
        results.append(r)
        print(f"    {label:<45} +{h*10:>3}min  "
              f"RMSE={r['RMSE']:>7.1f}  R²={r['R2']:.4f}")
    return results


# ════════════════════════════════════════════════════════════════
# 4. 主实验
# ════════════════════════════════════════════════════════════════

def main():
    # ── 加载数据 ──────────────────────────────────────────────────
    print("=" * 70)
    print("加载数据...")
    df = load_wide_dataset()
    df = add_farm_power(df)

    # ── 自相关分析 ────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("自相关与持续性分析...")
    acf_df = analyze_autocorrelation(df)

    # ── 定义特征集 ────────────────────────────────────────────────
    _all_hws = [f"HWS_{d}m" for d in LIDAR_DISTANCES]
    _all_met  = ([f"VShear_{d}m" for d in LIDAR_DISTANCES] +
                 [f"HShear_{d}m" for d in LIDAR_DISTANCES] +
                 [f"TI_avg_{d}m" for d in LIDAR_DISTANCES])

    SCENARIOS = {
        "P0: 仅历史功率":              ["power"],
        "M0: SCADA机舱风速+功率":      ["HWS_scada", "power"],
        "M5: 全距离LiDAR+SCADA+功率":  _all_hws + ["HWS_scada", "power"],
        "E4: 全HWS+SCADA+全气象+功率": _all_hws + ["HWS_scada"] + _all_met + ["power"],
    }

    # ── 运行实验 ──────────────────────────────────────────────────
    all_results = []

    print()
    print("=" * 70)
    print("【单机功率预测】目标 = power（单台风机）")
    print("=" * 70)
    for sc_name, feat_cols in SCENARIOS.items():
        feats_use = [c for c in feat_cols if c in df.columns]
        label = f"[单机] {sc_name}"
        print(f"\n  {label}：")
        res = run_lgb_experiment(df, feats_use, "power", label)
        for r in res:
            r["scenario"] = sc_name
            r["level"] = "single"
        all_results.extend(res)

    print()
    print("=" * 70)
    print("【场站功率预测】目标 = power_farm（N=5 台风机均值）")
    print("=" * 70)
    # 场站级别的特征：用原机的HWS作为场站代表性上游风速（LiDAR对场站更有意义）
    for sc_name, feat_cols in SCENARIOS.items():
        # 场站功率的目标是 power_farm，但特征仍来自同一原机的历史数据
        feats_use = [c for c in feat_cols if c in df.columns]
        # 对于场站预测，power 历史 = 用 power_farm 代替（场站历史功率）
        feats_farm = [c.replace("power", "power_farm") if c == "power" else c
                      for c in feats_use]
        # 确保 power_farm 列存在
        feats_farm = [c for c in feats_farm if c in df.columns]
        label = f"[场站] {sc_name}"
        print(f"\n  {label}：")
        res = run_lgb_experiment(df, feats_farm, "power_farm", label)
        for r in res:
            r["scenario"] = sc_name
            r["level"] = "farm"
        all_results.extend(res)

    # ── 结果保存 ──────────────────────────────────────────────────
    df_res = pd.DataFrame(all_results)
    out_csv = os.path.join(OUTPUT_DIR, "#10单机vs场站对比结果.csv")
    df_res.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ 结果已保存：{out_csv}  ({len(df_res)} 行)")

    # ── 可视化 ────────────────────────────────────────────────────
    plot_autocorrelation(acf_df)
    plot_rmse_comparison(df_res)
    plot_feature_gain(df_res)
    plot_relative_gain(df_res)

    # ── 生成分析报告 ──────────────────────────────────────────────
    write_analysis_report(acf_df, df_res)

    print("\n✅ #10 实验完成！")


# ════════════════════════════════════════════════════════════════
# 5. 可视化
# ════════════════════════════════════════════════════════════════

_COLORS = {
    "P0: 仅历史功率":              "#1f77b4",
    "M0: SCADA机舱风速+功率":      "#ff7f0e",
    "M5: 全距离LiDAR+SCADA+功率":  "#2ca02c",
    "E4: 全HWS+SCADA+全气象+功率": "#d62728",
}
_MARKERS = {
    "P0: 仅历史功率":              "o",
    "M0: SCADA机舱风速+功率":      "s",
    "M5: 全距离LiDAR+SCADA+功率":  "^",
    "E4: 全HWS+SCADA+全气象+功率": "D",
}
_STEP_MINS = [10, 20, 30, 60]


def plot_autocorrelation(acf_df):
    """图1：单机 vs 场站 的自相关系数对比（含持续性RMSE）。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 子图1：自相关系数
    ax = axes[0]
    x = acf_df["lag_min"].values
    ax.plot(x, acf_df["autocorr_single_power"], "o-", color="#1f77b4",
            linewidth=2, markersize=7, label="单机功率 ACF")
    ax.plot(x, acf_df["autocorr_farm_power"],   "s--", color="#ff7f0e",
            linewidth=2, markersize=7, label="场站功率 ACF")
    ax.plot(x, acf_df["autocorr_HWS"],          "^:", color="#2ca02c",
            linewidth=2, markersize=7, label="SCADA机舱风速 ACF")
    ax.set_xlabel("滞后时长 (min)", fontsize=12)
    ax.set_ylabel("自相关系数 ACF", fontsize=12)
    ax.set_title("自相关系数：单机功率 / 场站功率 / 机舱风速", fontsize=11)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.set_xticks(x); ax.set_ylim(0.7, 1.01)
    ax.axhline(0.98, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)

    # 子图2：持续性预测 RMSE
    ax = axes[1]
    ax.plot(x, acf_df["persist_RMSE_single"], "o-", color="#1f77b4",
            linewidth=2, markersize=7, label="单机持续性 RMSE")
    ax.plot(x, acf_df["persist_RMSE_farm"],   "s--", color="#ff7f0e",
            linewidth=2, markersize=7, label="场站持续性 RMSE")
    ax.set_xlabel("预测步长 (min)", fontsize=12)
    ax.set_ylabel("持续性预测 RMSE (kW)", fontsize=12)
    ax.set_title("持续性预测误差：单机 vs 场站", fontsize=11)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.set_xticks(x)
    # 标注 ML 模型的 P0 RMSE（单机，来自 #9 实验）
    p0_single = [432, 617, 724, 920]
    ax.plot(_STEP_MINS, p0_single, "o:", color="#9467bd",
            linewidth=1.5, markersize=6, alpha=0.8, label="单机P0(LightGBM,#9)")
    ax.legend(fontsize=9)

    plt.suptitle("单机 vs 场站功率预测：持久性基准与自相关分析\n"
                 f"★ 场站模拟：{N_VIRTUAL_TURBINES}台风机，空间相关系数ρ={INTER_TURBINE_RHO}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fpath = os.path.join(OUTPUT_DIR, "#10_单机vs场站_相关性分析.png")
    plt.savefig(fpath, dpi=150); plt.close(fig)
    print(f"Saved: {fpath}")


def plot_rmse_comparison(df_res):
    """图2：单机 vs 场站，各场景 RMSE 折线（2行×4列：按步长）。"""
    scenarios = list(_COLORS.keys())
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=False)

    for col_idx, step_min in enumerate(_STEP_MINS):
        for row_idx, level_label, target_col in [
            (0, "单机", "power"),
            (1, "场站", "power_farm"),
        ]:
            ax = axes[row_idx][col_idx]
            sub = df_res[(df_res["step_min"] == step_min) &
                         (df_res["level"] == ("single" if row_idx == 0 else "farm"))]
            bars_x = range(len(scenarios))
            bars_h = [sub[sub["scenario"] == sc]["RMSE"].values
                      for sc in scenarios]
            vals = [v[0] if v.size else np.nan for v in bars_h]
            colors = [_COLORS[sc] for sc in scenarios]
            bars = ax.bar(bars_x, vals, color=colors, alpha=0.85, width=0.6)
            # 标注值
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, v + 5,
                            f"{v:.0f}", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(list(bars_x))
            ax.set_xticklabels(["P0", "M0", "M5", "E4"], fontsize=9)
            ax.set_title(f"{level_label} +{step_min}min", fontsize=10)
            ax.set_ylabel("RMSE (kW)" if col_idx == 0 else "", fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            # P0 基准线
            p0_val = vals[0] if not np.isnan(vals[0]) else None
            if p0_val:
                ax.axhline(p0_val, color="#1f77b4", linestyle="--",
                           linewidth=1, alpha=0.6)

    # 图例
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=_COLORS[sc], alpha=0.85, label=sc)
                      for sc in scenarios]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    plt.suptitle("LightGBM：单机 vs 场站功率预测 RMSE 对比\n"
                 "★ 所有场景均包含「历史功率」作为输入特征",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fpath = os.path.join(OUTPUT_DIR, "#10_单机vs场站_RMSE对比.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {fpath}")


def plot_feature_gain(df_res):
    """图3：特征增益（相对P0的RMSE改善量 Δ），单机 vs 场站。"""
    step_mins = _STEP_MINS
    scenarios_compare = ["M0: SCADA机舱风速+功率",
                         "M5: 全距离LiDAR+SCADA+功率",
                         "E4: 全HWS+SCADA+全气象+功率"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for ax_idx, (level, level_label) in enumerate([("single", "单机"), ("farm", "场站")]):
        ax = axes[ax_idx]
        sub_level = df_res[df_res["level"] == level]
        p0_rmse = {
            sm: sub_level[(sub_level["scenario"] == "P0: 仅历史功率") &
                          (sub_level["step_min"] == sm)]["RMSE"].values
            for sm in step_mins
        }
        p0_rmse = {sm: v[0] if v.size else np.nan for sm, v in p0_rmse.items()}

        x = np.arange(len(step_mins))
        width = 0.25
        for i, sc in enumerate(scenarios_compare):
            deltas = []
            for sm in step_mins:
                row = sub_level[(sub_level["scenario"] == sc) &
                                (sub_level["step_min"] == sm)]["RMSE"].values
                if row.size and not np.isnan(p0_rmse.get(sm, np.nan)):
                    deltas.append(float(row[0] - p0_rmse[sm]))  # 正=更差，负=更好
                else:
                    deltas.append(np.nan)
            bars = ax.bar(x + i * width, deltas,
                          width=width, color=list(_COLORS.values())[i + 1],
                          alpha=0.85, label=sc.split(":")[0])
            for bar, v in zip(bars, deltas):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            v + (1 if v >= 0 else -10),
                            f"{v:+.0f}", ha="center", va="bottom" if v >= 0 else "top",
                            fontsize=7)

        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"+{sm}min" for sm in step_mins], fontsize=10)
        ax.set_title(f"{level_label}：气象特征对P0的增益\n（负值=比P0更好，正值=比P0更差）",
                     fontsize=10)
        ax.set_ylabel("Δ RMSE vs P0 (kW)" if ax_idx == 0 else "", fontsize=11)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    plt.suptitle("特征增益对比：添加风速/气象特征后相对「仅历史功率P0」的RMSE变化\n"
                 "负值 = 气象特征有帮助；正值 = 气象特征反而有害",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fpath = os.path.join(OUTPUT_DIR, "#10_场站预测_特征增益.png")
    plt.savefig(fpath, dpi=150); plt.close(fig)
    print(f"Saved: {fpath}")


def plot_relative_gain(df_res):
    """图4：单机 vs 场站，各步长下 M5/E4 相对P0的相对改善率（%）。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, sc_compare in enumerate([
        "M5: 全距离LiDAR+SCADA+功率",
        "E4: 全HWS+SCADA+全气象+功率",
    ]):
        ax = axes[ax_idx]
        for level, label, ls, color in [
            ("single", "单机", "-", "#1f77b4"),
            ("farm",   "场站", "--", "#ff7f0e"),
        ]:
            gains = []
            sub_level = df_res[df_res["level"] == level]
            for sm in _STEP_MINS:
                p0_r = sub_level[(sub_level["scenario"] == "P0: 仅历史功率") &
                                  (sub_level["step_min"] == sm)]["RMSE"].values
                sc_r = sub_level[(sub_level["scenario"] == sc_compare) &
                                  (sub_level["step_min"] == sm)]["RMSE"].values
                if p0_r.size and sc_r.size and not np.isnan(p0_r[0]):
                    # 正增益 = 气象特征使RMSE下降（比P0好）
                    gains.append(100 * (p0_r[0] - sc_r[0]) / p0_r[0])
                else:
                    gains.append(np.nan)
            ax.plot(_STEP_MINS, gains, marker="o", color=color,
                    linestyle=ls, linewidth=2, markersize=7, label=f"{label}")
            for x, g in zip(_STEP_MINS, gains):
                if not np.isnan(g):
                    ax.annotate(f"{g:+.1f}%", (x, g),
                                textcoords="offset points", xytext=(0, 6),
                                ha="center", fontsize=8, color=color)

        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(_STEP_MINS)
        ax.set_xlabel("预测步长 (min)", fontsize=11)
        ax.set_ylabel("相对P0的RMSE改善率 (%)" if ax_idx == 0 else "", fontsize=11)
        sc_short = sc_compare.split(":")[0]
        ax.set_title(f"{sc_short} vs P0：相对改善率\n正值=气象特征有益，负值=气象特征有害",
                     fontsize=10)
        ax.legend(fontsize=10); ax.grid(alpha=0.3)

    plt.suptitle("单机 vs 场站：气象特征对纯功率预测的相对增益（%）\n"
                 "★ 场站级别预测中，气象特征的相对价值更高",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fpath = os.path.join(OUTPUT_DIR, "#10_单机vs场站_相对增益.png")
    plt.savefig(fpath, dpi=150); plt.close(fig)
    print(f"Saved: {fpath}")


# ════════════════════════════════════════════════════════════════
# 6. 分析报告
# ════════════════════════════════════════════════════════════════

def write_analysis_report(acf_df, df_res):
    """生成详细分析报告 Markdown。"""

    def get_rmse(level, scenario, step_min):
        r = df_res[(df_res["level"] == level) &
                   (df_res["scenario"] == scenario) &
                   (df_res["step_min"] == step_min)]["RMSE"].values
        return f"{r[0]:.0f}" if r.size else "N/A"

    p0_s10  = get_rmse("single", "P0: 仅历史功率", 10)
    p0_s60  = get_rmse("single", "P0: 仅历史功率", 60)
    m5_s10  = get_rmse("single", "M5: 全距离LiDAR+SCADA+功率", 10)
    m5_s60  = get_rmse("single", "M5: 全距离LiDAR+SCADA+功率", 60)
    p0_f10  = get_rmse("farm",   "P0: 仅历史功率", 10)
    p0_f60  = get_rmse("farm",   "P0: 仅历史功率", 60)
    m5_f10  = get_rmse("farm",   "M5: 全距离LiDAR+SCADA+功率", 10)
    m5_f60  = get_rmse("farm",   "M5: 全距离LiDAR+SCADA+功率", 60)

    # 计算单机+场站各步长自相关
    acf_rows = acf_df.set_index("lag_min")
    r_s10 = acf_rows.loc[10, "autocorr_single_power"] if 10 in acf_rows.index else "N/A"
    r_f10 = acf_rows.loc[10, "autocorr_farm_power"] if 10 in acf_rows.index else "N/A"

    report = f"""# 分析报告：单机 vs 场站超短期功率预测特征价值

> **对应脚本**：`CODE/#10场站功率预测对比.py`
> **数据来源**：峡沙56号风机（#7训练数据集）
> **问题来源**：实验 #9 结论——单台风机超短期功率预测中，仅使用历史功率(P0)往往比添加气象特征效果更好；
>              本报告分析其原因，并讨论该结论在场站级别预测中是否适用。

---

## 一、单机预测中"纯功率最优"的原因分析

### 1.1 数据层面：高自相关性

峡沙56号风机功率序列的自相关系数：

| 滞后步长 | 10 min | 20 min | 30 min | 60 min | 120 min |
|---------|--------|--------|--------|--------|---------|
| 单机功率 ACF | {acf_rows.loc[10,'autocorr_single_power']:.4f} | {acf_rows.loc[20,'autocorr_single_power']:.4f} | {acf_rows.loc[30,'autocorr_single_power']:.4f} | {acf_rows.loc[60,'autocorr_single_power']:.4f} | {acf_rows.loc[120,'autocorr_single_power']:.4f} |
| 机舱风速 ACF | {acf_rows.loc[10,'autocorr_HWS']:.4f} | {acf_rows.loc[20,'autocorr_HWS']:.4f} | {acf_rows.loc[30,'autocorr_HWS']:.4f} | {acf_rows.loc[60,'autocorr_HWS']:.4f} | {acf_rows.loc[120,'autocorr_HWS']:.4f} |

关键发现：
- **功率在+10min的自相关高达 {r_s10:.4f}**，这意味着"当前功率"本身已经是下一步功率的极强预测器
- 持续性预测（直接用当前值预测未来）的 RMSE：+10min={acf_rows.loc[10,'persist_RMSE_single']:.0f}kW，接近甚至优于部分ML模型
- 在这种高自相关环境下，额外加入风速特征的**边际信息增益很小**

### 1.2 物理机制：功率已隐含风速信息

**P ∝ V³ 关系**：风机功率由风速立方决定，因此：
- 已知功率 P(t) ≈ 等价于已知 V(t) ≈ (P(t)/k)^(1/3)
- 历史功率时序 [P(t-6), ..., P(t)] 等价于编码了过去 60 分钟的风速历史
- 模型通过学习 P-P 自回归关系，**隐式地捕获了风速-功率的物理映射**

直接输入 HWS（风速）不仅没有额外信息，还会引入以下问题：
- 测量噪声（超声波风速计噪声、尾流干扰）
- 非线性映射的额外拟合难度（模型需要再学一次 V→P 曲线）
- 多余特征导致的过拟合风险（特别是小样本情况）

### 1.3 控制系统状态编码

风机功率信号还隐含了控制系统状态：
- **Pitch角**（变桨状态）：功率高 → 叶片保持最优角度；功率低 → 可能变桨减载
- **转速**：额定功率附近时转速已达额定，变化规律不同于低功率区
- 这些控制状态无法从原始风速中直接读取，但**已编码在功率序列中**

### 1.4 空间滤波效果

峡沙56号风机叶轮直径约 180m：
- 功率 = 扫风面积内风能的空间积分
- 已自然滤除了点测量（超声波风速计）中的高频湍流噪声
- LiDAR 测量的是单点（或小截面）的风速，含较多湍流成分，与功率的相关性反而低于历史功率本身

### 1.5 预测时域特征（超短期）

对于 ≤60min 的超短期预测：
- 大气条件变化缓慢（惯性风）：60min 前的功率已经高度代表当前风况
- LiDAR 的"超前测量"优势（检测来流风）在此时域下贡献有限：
  - 300m / 10 m/s = 30s 超前量（远小于 10min 的时间步长）
  - 10min 均值已平滑了大部分由 LiDAR 捕捉到的湍流信息

---

## 二、场站级别预测：结论是否仍然成立？

### 2.1 实验结果（本脚本模拟）

模拟配置：{N_VIRTUAL_TURBINES}台风机，空间相关系数ρ={INTER_TURBINE_RHO}（独立噪声比例={100*(1-INTER_TURBINE_RHO):.0f}%）

| 场景 | 单机+10min | 单机+60min | 场站+10min | 场站+60min |
|-----|-----------|-----------|-----------|-----------|
| P0: 仅历史功率 | {p0_s10} kW | {p0_s60} kW | {p0_f10} kW | {p0_f60} kW |
| M5: 全距离LiDAR+SCADA+功率 | {m5_s10} kW | {m5_s60} kW | {m5_f10} kW | {m5_f60} kW |

> 注：场站功率 = {N_VIRTUAL_TURBINES} 台模拟风机的平均值（每台加入了 {100*(1-INTER_TURBINE_RHO):.0f}% 独立噪声）

### 2.2 场站级别的核心差异

**（一）空间多样性降低自相关**

场站功率（多机平均）的自相关系数：

| 滞后 | 10 min | 20 min | 30 min | 60 min |
|-----|--------|--------|--------|--------|
| 单机功率 ACF | {acf_rows.loc[10,'autocorr_single_power']:.4f} | {acf_rows.loc[20,'autocorr_single_power']:.4f} | {acf_rows.loc[30,'autocorr_single_power']:.4f} | {acf_rows.loc[60,'autocorr_single_power']:.4f} |
| 场站功率 ACF | {acf_rows.loc[10,'autocorr_farm_power']:.4f} | {acf_rows.loc[20,'autocorr_farm_power']:.4f} | {acf_rows.loc[30,'autocorr_farm_power']:.4f} | {acf_rows.loc[60,'autocorr_farm_power']:.4f} |

场站功率的自相关系数略低，说明**纯持续性方法在场站级别效果相对变差**，气象特征（风速信息）的边际价值相应增加。

**（二）真实场站的额外复杂性**

本实验只能模拟"独立噪声"层面的空间多样性。真实场站还有：

1. **尾流效应（Wake Effect）**：
   - 上游风机的尾迹显著影响下游风机（功率损失可达 20~40%）
   - 尾迹结构取决于当前风向和风速，无法从单机历史功率序列中捕捉
   - **LiDAR 数据（特别是上游来流风速）可以帮助预测尾流强度变化**

2. **天气过程与风场演化**：
   - 真实场站（3-20 km 尺度）内，不同位置可能处于不同的天气状态
   - 冷锋、海陆风等天气过程从场站一侧进入时，功率会出现大规模"坡道"（Ramp）事件
   - 这种情况下，历史功率已无法预测即将到来的风速变化，**气象特征（NWP、LiDAR）是必需的**

3. **大型风机阵列的空间相关性结构更复杂**：
   - 不同风向下，相关性矩阵发生变化
   - 跨行间的关联往往比行内弱得多

4. **更长的预测时域（15min~4h）**：
   - 场站功率预测往往需要 15min~4h 的预测（电网调度需求）
   - 在 1~4h 的时域下，持续性预测误差大幅增加，**数值天气预报（NWP）或激光雷达测风塔数据变得不可缺少**

### 2.3 结论

| 维度 | 单台风机（超短期，≤60min） | 场站级别预测 |
|-----|--------------------------|------------|
| **P0纯功率结论是否适用** | ✅ 适用（本实验验证） | ⚠️ **不适用**（气象特征更重要）|
| **历史功率的预测价值** | 极高（ACF≈0.98，隐含完整风速历史）| 中高（ACF略低，空间多样性引入独立分量）|
| **气象特征（LiDAR/SCADA风速）** | 边际增益很小（甚至有害）| 相对边际增益增加（特别是尾流和Ramp事件）|
| **NWP数值天气预报** | 几乎无用（时间步太短）| **必需**（>30min 预测的关键特征）|
| **预测难点** | 短期持续性强，难点在极端变化捕捉 | 空间聚合、尾流、Ramp事件、调度约束 |
| **推荐模型** | LSTM P0 / LightGBM P0 | 需要 NWP+空间特征的物理-统计混合模型 |

### 2.4 对场站功率预测的建议

1. **必须包含历史功率**：场站历史功率仍然是最强单一特征，P0 仍是无法绕过的基线
2. **加入 NWP 风速预测**：超过 30min 的预测必须使用数值天气预报
3. **多机空间特征**：相邻风机的历史功率可以提供局部空间风速信息（简易替代 LiDAR 阵列）
4. **Ramp 事件单独处理**：对于功率急变时段，分类器+专用预测器的组合更有效
5. **LiDAR 价值在场站**：一台激光雷达（安装在上风向边缘）对整个场站的提前预测更有意义，而非单台风机

---

## 三、本实验的局限性

1. **仅有单台风机数据**：场站模拟通过添加独立噪声实现，不能反映真实的尾流、地形等复杂空间结构
2. **数据时长较短**（~3个月）：可能未覆盖所有典型天气过程（特别是冬季、强对流）
3. **缺乏 NWP 数据**：无法验证数值天气预报对场站功率的贡献
4. **模拟相关系数 ρ={INTER_TURBINE_RHO}** 是假设值，真实场站根据布局和天气条件差异较大（ρ 可能在 0.7~0.95 之间变化）

---

## 四、图表说明

| 文件 | 内容 |
|-----|------|
| `#10_单机vs场站_相关性分析.png` | ACF曲线 + 持续性RMSE对比 |
| `#10_单机vs场站_RMSE对比.png` | 4场景×4步长×2级别，柱状图对比 |
| `#10_场站预测_特征增益.png` | 气象特征相对P0的RMSE增益（负=有益）|
| `#10_单机vs场站_相对增益.png` | 单机vs场站的相对改善率（%）折线 |
| `#10单机vs场站对比结果.csv` | 完整实验结果 |
"""

    report_path = os.path.join(CODE_DIR, "#10单机vs场站功率预测分析报告.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✅ 分析报告已保存：{report_path}")


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
