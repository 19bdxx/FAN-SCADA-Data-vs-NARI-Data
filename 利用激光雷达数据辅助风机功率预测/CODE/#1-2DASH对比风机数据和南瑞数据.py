import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go

# 读取并预处理数据
df = pd.read_csv(r'PROCESS_DATA\#1-1合并风机导出数据和南瑞数据.csv', encoding='gbk')
df['时间'] = pd.to_datetime(df['时间'])
df['时间_导出功率'] = df['时间'] + pd.Timedelta(minutes=2)

# 初始化 Dash 应用
app = Dash(__name__)
app.title = "风机功率与风速分析"

# 布局设计
app.layout = html.Div([
    html.H2("风机功率与风速对比分析"),

    dcc.DatePickerRange(
        id='date-range',
        min_date_allowed=df['时间'].min().date(),
        max_date_allowed=df['时间'].max().date(),
        start_date=df['时间'].min().date(),
        end_date=df['时间'].max().date()
    ),

    html.Div([
        html.Label("是否平移导出功率的时间戳："),
        dcc.RadioItems(
            id='shift-toggle',
            options=[
                {'label': '平移2分钟', 'value': 'shift'},
                {'label': '不平移', 'value': 'no-shift'}
            ],
            value='shift',
            inline=True
        )
    ], style={'marginTop': '10px'}),

    html.Div(id='stats-output', style={"marginTop": "10px", "fontWeight": "bold", "whiteSpace": "pre-line"}),

    dcc.Graph(id='power-wind-graph')
])

# 回调逻辑
@app.callback(
    [Output('power-wind-graph', 'figure'),
     Output('stats-output', 'children')],
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('shift-toggle', 'value')]
)
def update_graph(start_date, end_date, shift_option):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    filtered_df = df[(df['时间'] >= start) & (df['时间'] <= end)]

    # 使用平移后的时间还是原时间
    if shift_option == 'shift':
        export_mask = (df['时间_导出功率'] >= start) & (df['时间_导出功率'] <= end)
        export_time = df['时间_导出功率']
    else:
        export_mask = (df['时间'] >= start) & (df['时间'] <= end)
        export_time = df['时间']

    export_df = df[export_mask]
    export_df = export_df.copy()
    export_df['绘图时间'] = export_time[export_mask]

    # 计算统计信息
    wind_avg = filtered_df['WINDSPEED_#56'].mean()
    wind_std = filtered_df['WINDSPEED_#56'].std()

    active_avg = filtered_df['ACTIVE_POWER_#56'].mean()
    active_std = filtered_df['ACTIVE_POWER_#56'].std()

    export_avg = export_df['平均有功功率_风机导出'].mean()
    export_std = export_df['平均有功功率_风机导出'].std()

    if filtered_df.empty:
        stats_text = "无数据可显示"
    else:
        stats_text = (
            f"风速均值：{wind_avg:.2f} m/s，标准差：{wind_std:.2f} m/s\n"
            f"ACTIVE_POWER_#56 均值：{active_avg:.2f}，标准差：{active_std:.2f}\n"
            f"平均有功功率_风机导出 均值：{export_avg:.2f}，标准差：{export_std:.2f}"
        )

    # 图表构建
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=filtered_df['时间'],
        y=filtered_df['ACTIVE_POWER_#56'],
        mode='lines',
        name='ACTIVE_POWER_#56',
        yaxis='y1'
    ))

    fig.add_trace(go.Scatter(
        x=export_df['绘图时间'],
        y=export_df['平均有功功率_风机导出'],
        mode='lines',
        name='平均有功功率_风机导出',
        yaxis='y2'
    ))

    fig.add_trace(go.Scatter(
        x=filtered_df['时间'],
        y=filtered_df['WINDSPEED_#56'],
        mode='lines',
        name='WINDSPEED_#56',
        line=dict(dash='dot'),
        yaxis='y2'
    ))

    fig.update_layout(
        xaxis=dict(title='时间'),
        yaxis=dict(title='ACTIVE_POWER_#56', side='left'),
        yaxis2=dict(title='风速 / 平均有功功率', overlaying='y', side='right'),
        hovermode="x unified",
        legend=dict(x=0, y=1)
    )

    return fig, stats_text

# 启动服务
if __name__ == '__main__':
    app.run_server(debug=True)
