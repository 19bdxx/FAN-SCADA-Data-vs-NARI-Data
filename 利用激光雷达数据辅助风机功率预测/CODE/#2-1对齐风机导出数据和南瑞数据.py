"""
align_timestamps.py
===================
峡沙56号风机 - NARI 平台与 SCADA 导出数据时间戳对齐工具

背景
----
经互相关分析发现，南瑞（NARI）平台导出的有功功率数据相对于风机 SCADA
导出数据存在系统性时间戳滞后，且滞后量随时间段而变化：

  ┌──────────────────────────────────────────┬────────────────┐
  │ 时段                                      │ NARI 滞后量    │
  ├──────────────────────────────────────────┼────────────────┤
  │ 2024-08-30 00:00 ~ 2024-09-30 23:59      │ +2 分钟        │
  │ 2024-10-01 00:00 ~ 2024-11-07 17:59      │ +3 分钟        │
  │ 2024-11-07 18:00 ~ 2024-11-20 23:59      │ +4 分钟        │
  └──────────────────────────────────────────┴────────────────┘

"NARI 滞后 N 分钟" 的含义
------------------------
  SCADA 在时刻 T 记录的功率值，与 NARI 在时刻 T+N 记录的功率值，
  对应同一个物理采样点。

  对齐方法：将 NARI 的时间戳整体前移 N 分钟（即 nari_aligned_time = nari_time - N min），
  使其与 SCADA 时间戳在同一时刻对应同一物理事件。

用法
----
  python align_timestamps.py [--input <路径>] [--output <路径>]

  默认输入：  DATA/峡沙56号_合并风机导出数据和南瑞数据.csv
  默认输出：  DATA/峡沙56号_时间戳对齐后数据.csv
"""

import argparse
import os
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────
# 1. 时移规则配置
# ──────────────────────────────────────────────────────────────
LAG_RULES = [
    ("2024-08-30 00:00:00", "2024-09-30 23:59:00", 2),
    ("2024-10-01 00:00:00", "2024-11-07 17:59:00", 3),
    ("2024-11-07 18:00:00", "2024-11-20 23:59:00", 4),
]


def get_lag_for_time(t: pd.Timestamp) -> int:
    """
    给定一个时间戳，返回对应的 NARI 滞后分钟数。
    若不在任何规则区间内，返回 0（不做时移）。
    """
    for start_str, end_str, lag in LAG_RULES:
        start = pd.Timestamp(start_str)
        end = pd.Timestamp(end_str)
        if start <= t <= end:
            return lag
    return 0


def assign_lag_column(df: pd.DataFrame) -> pd.Series:
    """向量化地为每行分配 lag 值。"""
    lag_series = pd.Series(0, index=df.index, dtype=int)
    for start_str, end_str, lag in LAG_RULES:
        start = pd.Timestamp(start_str)
        end = pd.Timestamp(end_str)
        mask = (df["time"] >= start) & (df["time"] <= end)
        lag_series[mask] = lag
    return lag_series


def align(input_path: str, output_path: str) -> pd.DataFrame:
    """
    读取原始合并数据，进行时间戳对齐，返回对齐后的 DataFrame 并保存到 CSV。

    对齐逻辑
    --------
    以 SCADA 数据的时间戳为基准（不变）。
    对每一行：从 NARI 数据中取“当前时刻 + lag”处的 NARI 值，
    即相当于将 NARI 时间戳整体前移 lag 分钟后，与 SCADA 时间对齐。

    输出列
    ------
    时间                     : 对齐后的时间戳（与 SCADA 原始时间相同）
    lag_min                  : 该行使用的 NARI 时移分钟数
    ACTIVE_POWER_#56_原始    : NARI 原始有功功率（未对齐）
    ACTIVE_POWER_#56_对齐    : NARI 时移对齐后的有功功率
    WINDSPEED_#56_原始       : NARI 原始风速（未对齐）
    WINDSPEED_#56_对齐       : NARI 时移对齐后的风速
    WINDDIRECTION_#56_原始   : NARI 原始风向（未对齐）
    WINDDIRECTION_#56_对齐   : NARI 时移对齐后的风向
    平均有功功率_风机导出     : SCADA 有功功率（参考基准，不变）
    功率差_对齐后             : NARI对齐功率 - SCADA功率
    """
    print(f"[1/4] 读取数据: {input_path}")
    df_raw = pd.read_csv(input_path, encoding="gbk", parse_dates=["时间"])
    df_raw = df_raw.rename(columns={
        "时间": "time",
        "ACTIVE_POWER_#56": "nari_power",
        "WINDSPEED_#56": "nari_wind",
        "WINDDIRECTION_#56": "nari_wind_direction",
        "平均有功功率_风机导出": "scada_power",
    })
    df_raw = df_raw.sort_values("time").reset_index(drop=True)
    print(f"    共 {len(df_raw):,} 条记录，时间范围：{df_raw['time'].min()} ~ {df_raw['time'].max()}")

    print("[2/4] 分配各行时移量...")
    df_raw["lag_min"] = assign_lag_column(df_raw)
    print(f"    lag 分布：\n{df_raw['lag_min'].value_counts().sort_index().to_string()}")

    print("[3/4] 对齐 NARI 数据到 SCADA 时间轴...")
    nari_lookup = df_raw.set_index("time")[[
        "nari_power",
        "nari_wind",
        "nari_wind_direction"
    ]]

    aligned_parts = []
    for lag_val, group in df_raw.groupby("lag_min"):
        shifted_time = group["time"] + pd.Timedelta(minutes=lag_val)
        nari_shifted = (
            nari_lookup
            .reindex(shifted_time.values)
            .rename(columns={
                "nari_power": "ACTIVE_POWER_#56_对齐",
                "nari_wind": "WINDSPEED_#56_对齐",
                "nari_wind_direction": "WINDDIRECTION_#56_对齐",
            })
        )
        nari_shifted.index = group.index
        aligned_parts.append(nari_shifted)

    aligned_nari = pd.concat(aligned_parts).sort_index()

    result = pd.DataFrame({
        "时间":                    df_raw["time"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "lag_min":                 df_raw["lag_min"],
        "ACTIVE_POWER_#56_原始":   df_raw["nari_power"],
        "ACTIVE_POWER_#56_对齐":   aligned_nari["ACTIVE_POWER_#56_对齐"].values,
        "WINDSPEED_#56_原始":      df_raw["nari_wind"],
        "WINDSPEED_#56_对齐":      aligned_nari["WINDSPEED_#56_对齐"].values,
        "WINDDIRECTION_#56_原始":  df_raw["nari_wind_direction"],
        "WINDDIRECTION_#56_对齐":  aligned_nari["WINDDIRECTION_#56_对齐"].values,
        "平均有功功率_风机导出":    df_raw["scada_power"],
    })
    result["功率差_对齐后"] = result["ACTIVE_POWER_#56_对齐"] - result["平均有功功率_风机导出"]

    valid = result.dropna(subset=["ACTIVE_POWER_#56_对齐", "平均有功功率_风机导出"])
    r_raw = np.corrcoef(df_raw["nari_power"], df_raw["scada_power"])[0, 1]
    r_align = np.corrcoef(valid["ACTIVE_POWER_#56_对齐"], valid["平均有功功率_风机导出"])[0, 1]
    rmse_raw = np.sqrt(np.mean((df_raw["nari_power"] - df_raw["scada_power"]) ** 2))
    rmse_align = np.sqrt(np.mean(valid["功率差_对齐后"] ** 2))

    print(f"\n    ── 对齐效果汇总 ────────────────────────────────────")
    print(f"    对齐前：r = {r_raw:.6f},  RMSE = {rmse_raw:.2f} kW")
    print(f"    对齐后：r = {r_align:.6f},  RMSE = {rmse_align:.2f} kW")
    print(f"    有效记录（对齐后非NaN）：{len(valid):,} / {len(result):,}")
    print(f"    NaN 记录（边界截断）：{len(result) - len(valid):,}")

    print(f"\n[4/4] 保存对齐数据: {output_path}")
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"    完成！共 {len(result):,} 行写入。")
    return result


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(repo_root, "../PROCESS_DATA", "#1-1合并风机导出数据和南瑞数据.csv")
    default_output = os.path.join(repo_root, "../PROCESS_DATA", "#2-1对齐风机导出数据和南瑞数据.csv")

    parser = argparse.ArgumentParser(
        description="峡沙56号风机 NARI vs SCADA 时间戳对齐工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", default=default_input, help="原始 CSV 路径")
    parser.add_argument("--output", default=default_output, help="对齐后 CSV 输出路径")
    args = parser.parse_args()

    align(args.input, args.output)


if __name__ == "__main__":
    main()