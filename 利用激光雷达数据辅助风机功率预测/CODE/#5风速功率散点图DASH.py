import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px


# ===== 文件路径 =====
data_path = r"PROCESS_DATA\#4峡沙56号_风机-激光雷达数据合并.csv"
curve_path = r"RAW_DATA\MySE6.45-180推力曲线.csv"

# ===== 数据读取 =====
try:
    df = pd.read_csv(data_path, encoding='utf-8-sig')
except UnicodeDecodeError:
    df = pd.read_csv(data_path, encoding='gbk')

df['DateAndTime'] = pd.to_datetime(df['DateAndTime'])

curve_df = pd.read_csv(curve_path, encoding='gbk')
curve_filtered = curve_df[(curve_df["风速"] >= 3) & (curve_df["风速"] <= 20)].copy()
standard_density = curve_filtered["标准空气密度(kg/m3)"].iloc[0]
field_density = curve_filtered["现场空气密度(kg/m3)"].iloc[0]
density_info = f"标准空气密度: {standard_density} kg/m³ | 现场空气密度: {field_density} kg/m³"

distance_values = sorted(df['Distance'].dropna().unique())

# ===== Dash 初始化 =====
app = dash.Dash(__name__)
app.title = "风机功率分析"

# ===== 页面布局 =====
app.layout = html.Div([
    html.H2("风机功率与测风数据分析", style={'textAlign': 'center'}),

    html.Div([
        html.Label("选择时间范围："),
        dcc.DatePickerRange(
            id='date-range',
            start_date=df['DateAndTime'].min().date(),
            end_date=df['DateAndTime'].max().date(),
            display_format='YYYY-MM-DD'
        ),
    ], style={'marginBottom': 20}),

    html.Div([
        html.Label("选择 Distance（可多选）："),
        dcc.Dropdown(
            id='distance-filter',
            options=[{'label': str(d), 'value': d} for d in distance_values],
            multi=True,
            value=[]
        ),
    ], style={'marginBottom': 20}),

    html.Div([
        html.Label("选择显示的功率类型："),
        dcc.Checklist(
            id='power-type-checklist',
            options=[
                {'label': 'SCADA前10分钟均值', 'value': 'scada_mean10'},
                {'label': 'NARI对齐前10分钟均值', 'value': 'nari_mean10'}
            ],
            value=[],
            inline=True
        )
    ], style={'marginBottom': 20}),

    html.Div([
        html.Label("选择风机故障状态："),
        dcc.Checklist(
            id='status-filter',
            options=[
                {'label': '非故障窗口（风机故障=0）', 'value': 0},
                {'label': '故障窗口（风机故障=1）', 'value': 1},
            ],
            value=[0, 1],
            inline=True
        )
    ], style={'marginBottom': 20}),

    dcc.Graph(id='scatter-plot')
])


# ===== 回调函数 =====
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
    Input('distance-filter', 'value'),
    Input('power-type-checklist', 'value'),
    Input('status-filter', 'value')
)
def update_graph(start_date, end_date, selected_distances, selected_types, selected_status):
    if not selected_distances or not selected_types or not start_date or not end_date or not selected_status:
        return {
            'layout': {
                'xaxis': {'title': 'HWS(hub)', 'range': [0, 20]},
                'yaxis': {'title': '平均有功功率 (kW)'},
                'annotations': [{
                    'text': '请先选择时间范围、Distance、功率类型和风机故障状态',
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 16}
                }]
            }
        }

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)

    filtered = df[
        (df['DateAndTime'] >= start_dt) &
        (df['DateAndTime'] < end_dt) &
        (df['Distance'].isin(selected_distances)) &
        (df['风机故障'].isin(selected_status))
    ]

    fig = go.Figure()
    color_map = px.colors.qualitative.Set2

    for idx, dist in enumerate(sorted(selected_distances)):
        df_dist = filtered[filtered['Distance'] == dist]
        color = color_map[idx % len(color_map)]

        for status in selected_status:
            df_sub = df_dist[df_dist['风机故障'] == status]

            symbol_scada = 'circle' if status == 0 else 'x'
            symbol_nari = 'square' if status == 0 else 'diamond-open'

            if 'scada_mean10' in selected_types:
                fig.add_trace(go.Scattergl(
                    x=df_sub['HWS(hub)'],
                    y=df_sub['平均有功功率_风机导出_前10分钟均值'],
                    mode='markers',
                    name=f"Distance {dist} - SCADA10min - 故障{status}",
                    marker=dict(color=color, size=4, symbol=symbol_scada),
                    text=df_sub['DateAndTime'].dt.strftime('%Y-%m-%d %H:%M:%S'),
                    hovertemplate='HWS(hub): %{x}<br>SCADA功率: %{y} kW<br>时间: %{text}<extra></extra>'
                ))

            if 'nari_mean10' in selected_types:
                fig.add_trace(go.Scattergl(
                    x=df_sub['HWS(hub)'],
                    y=df_sub['ACTIVE_POWER_#56_对齐_前10分钟均值'],
                    mode='markers',
                    name=f"Distance {dist} - NARI10min - 故障{status}",
                    marker=dict(color=color, size=4, symbol=symbol_nari),
                    text=df_sub['DateAndTime'].dt.strftime('%Y-%m-%d %H:%M:%S'),
                    hovertemplate='HWS(hub): %{x}<br>NARI功率: %{y} kW<br>时间: %{text}<extra></extra>'
                ))

    fig.add_trace(go.Scattergl(
        x=curve_filtered["风速"], y=curve_filtered["标准静态功率"],
        mode='lines', name="标准静态功率",
        line=dict(dash='solid', width=3, color='black')
    ))
    fig.add_trace(go.Scattergl(
        x=curve_filtered["风速"], y=curve_filtered["标准动态功率"],
        mode='lines', name="标准动态功率",
        line=dict(dash='dash', width=3, color='black')
    ))
    fig.add_trace(go.Scattergl(
        x=curve_filtered["风速"], y=curve_filtered["现场静态功率"],
        mode='lines', name="现场静态功率",
        line=dict(dash='solid', width=3, color='red')
    ))
    fig.add_trace(go.Scattergl(
        x=curve_filtered["风速"], y=curve_filtered["现场动态功率"],
        mode='lines', name="现场动态功率",
        line=dict(dash='dash', width=3, color='red')
    ))

    fig.update_layout(
        title='HWS(hub) 与 平均有功功率关系图',
        xaxis=dict(title='HWS(hub)', range=[0, 20]),
        yaxis=dict(title='平均有功功率 (kW)', range=[0, 6800]),
        template='plotly_white',
        legend_title='图例',
        annotations=[dict(
            text=density_info,
            xref='paper', yref='paper', x=0, y=-0.2,
            showarrow=False,
            font=dict(size=12, color='gray')
        )]
    )
    return fig


# ===== 启动服务器 =====
if __name__ == '__main__':
    app.run(debug=True)