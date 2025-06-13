import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import base64
import io
import json

# 既存の予測システムをインポート（前回作成したクラス）
# from power_prediction_system import PowerPredictionSystem
# ここでは同じファイル内に簡略版を定義

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

class PowerPredictionSystem:
    """電力需要・PV発電量予測システム（Dash用簡略版）"""
    
    def __init__(self):
        self.demand_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.demand_scaler = StandardScaler()
        self.pv_capacity = None
        self.is_trained = False
        self.feature_columns = []
        self.training_data = None
        self.last_forecast = None
        
    def load_and_process_data(self, demand_content, solar_content):
        """データ読み込みと処理"""
        try:
            # Base64デコードしてDataFrame作成
            demand_df = pd.read_excel(io.BytesIO(demand_content))
            solar_df = pd.read_excel(io.BytesIO(solar_content))
            
            # 簡略化：最初のカラムを日時、2番目を値として仮定
            demand_df.columns = ['datetime', 'demand_kw'] + list(demand_df.columns[2:])
            solar_df.columns = ['datetime', 'solar_irradiance'] + list(solar_df.columns[2:])
            
            # 日時変換
            demand_df['datetime'] = pd.to_datetime(demand_df['datetime'])
            solar_df['datetime'] = pd.to_datetime(solar_df['datetime'])
            
            # 15分間隔にリサンプリング
            demand_df = self._resample_to_15min(demand_df)
            solar_df = self._aggregate_solar_data(solar_df)
            
            # 結合
            combined_df = pd.merge(demand_df, solar_df, on='datetime', how='inner')
            
            self.training_data = combined_df
            return True, f"データ読み込み完了: {len(combined_df)}件"
            
        except Exception as e:
            return False, f"データ読み込みエラー: {str(e)}"
    
    def _resample_to_15min(self, df):
        """15分間隔にリサンプリング"""
        df = df.set_index('datetime').sort_index()
        df_15min = df.resample('15min').interpolate(method='linear')
        return df_15min.reset_index()
    
    def _aggregate_solar_data(self, df):
        """日射データを15分平均に集約"""
        df = df.set_index('datetime').sort_index()
        df_15min = df.resample('15min').mean()
        return df_15min.reset_index()
    
    def prepare_features(self, df):
        """特徴量準備"""
        df = df.copy()
        
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['quarter_hour'] = df['hour'] + df['minute'] / 60
        
        # 周期性特徴量
        df['hour_sin'] = np.sin(2 * np.pi * df['quarter_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['quarter_hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # ラグ特徴量
        if 'demand_kw' in df.columns:
            df['demand_lag1'] = df['demand_kw'].shift(1)
            df['demand_lag4'] = df['demand_kw'].shift(4)
            df['demand_lag96'] = df['demand_kw'].shift(96)
        
        if 'solar_irradiance' in df.columns:
            df['solar_lag1'] = df['solar_irradiance'].shift(1)
        
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def train_model(self):
        """モデル学習"""
        if self.training_data is None:
            return False, "学習データがありません"
        
        try:
            df_features = self.prepare_features(self.training_data)
            df_features = df_features.dropna()
            
            feature_cols = [col for col in df_features.columns 
                           if col not in ['datetime', 'demand_kw']]
            
            X = df_features[feature_cols]
            y = df_features['demand_kw']
            self.feature_columns = feature_cols
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            X_train_scaled = self.demand_scaler.fit_transform(X_train)
            X_test_scaled = self.demand_scaler.transform(X_test)
            
            self.demand_model.fit(X_train_scaled, y_train)
            
            test_pred = self.demand_model.predict(X_test_scaled)
            rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            mape = np.mean(np.abs((y_test - test_pred) / y_test)) * 100
            
            self.is_trained = True
            return True, f"学習完了 - RMSE: {rmse:.2f} kW, MAPE: {mape:.2f}%"
            
        except Exception as e:
            return False, f"学習エラー: {str(e)}"
    
    def daily_forecast(self, start_datetime):
        """24時間予測"""
        if not self.is_trained:
            return None, "モデルが学習されていません"
        
        try:
            forecast_times = pd.date_range(
                start=start_datetime, periods=96, freq='15min'
            )
            
            forecast_df = pd.DataFrame({'datetime': forecast_times})
            forecast_df = self.prepare_features(forecast_df)
            
            X_forecast = forecast_df[self.feature_columns].fillna(0)
            X_forecast_scaled = self.demand_scaler.transform(X_forecast)
            
            demand_forecast = self.demand_model.predict(X_forecast_scaled)
            
            results = pd.DataFrame({
                'datetime': forecast_times,
                'demand_forecast_kw': demand_forecast
            })
            
            if self.pv_capacity and 'solar_irradiance' in forecast_df.columns:
                pv_forecast = forecast_df['solar_irradiance'] * self.pv_capacity * 0.15
                results['pv_forecast_kw'] = pv_forecast
            
            self.last_forecast = results
            return results, "予測完了"
            
        except Exception as e:
            return None, f"予測エラー: {str(e)}"

# Dashアプリケーション初期化
app = dash.Dash(__name__)

# グローバル変数
predictor = PowerPredictionSystem()

# アプリケーションスタイル
app.layout = html.Div([
    # ヘッダー
    html.Div([
        html.H1("電力需要・PV発電量予測システム", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'})
    ]),
    
    # メインコンテンツ
    dcc.Tabs(id="main-tabs", value='data-upload', children=[
        
        # データアップロードタブ
        dcc.Tab(label='データアップロード', value='data-upload', children=[
            html.Div([
                html.H3("1. データファイルアップロード"),
                
                html.Div([
                    html.Div([
                        html.H4("需要データ (Excel)"),
                        dcc.Upload(
                            id='upload-demand',
                            children=html.Div([
                                'ファイルをドラッグ&ドロップまたは ',
                                html.A('選択してください')
                            ]),
                            style={
                                'width': '100%', 'height': '60px',
                                'lineHeight': '60px', 'borderWidth': '1px',
                                'borderStyle': 'dashed', 'borderRadius': '5px',
                                'textAlign': 'center', 'margin': '10px'
                            },
                            multiple=False
                        ),
                    ], className='six columns'),
                    
                    html.Div([
                        html.H4("日射データ (Excel)"),
                        dcc.Upload(
                            id='upload-solar',
                            children=html.Div([
                                'ファイルをドラッグ&ドロップまたは ',
                                html.A('選択してください')
                            ]),
                            style={
                                'width': '100%', 'height': '60px',
                                'lineHeight': '60px', 'borderWidth': '1px',
                                'borderStyle': 'dashed', 'borderRadius': '5px',
                                'textAlign': 'center', 'margin': '10px'
                            },
                            multiple=False
                        ),
                    ], className='six columns'),
                ], className='row'),
                
                html.Div(id='upload-status', style={'margin': '20px 0'}),
                
                html.Button('データ処理実行', id='process-data-btn', 
                           style={'backgroundColor': '#3498db', 'color': 'white',
                                 'padding': '10px 20px', 'border': 'none',
                                 'borderRadius': '5px', 'margin': '10px'}),
                
                html.Div(id='data-process-status', style={'margin': '20px 0'}),
            ], style={'padding': '20px'})
        ]),
        
        # モデル学習タブ
        dcc.Tab(label='モデル学習', value='model-training', children=[
            html.Div([
                html.H3("2. モデル学習"),
                
                html.Button('学習開始', id='train-model-btn',
                           style={'backgroundColor': '#e74c3c', 'color': 'white',
                                 'padding': '10px 20px', 'border': 'none',
                                 'borderRadius': '5px', 'margin': '10px'}),
                
                html.Div(id='training-status', style={'margin': '20px 0'}),
                
                # 学習結果表示用
                html.Div(id='training-results', style={'margin': '20px 0'}),
                
            ], style={'padding': '20px'})
        ]),
        
        # 予測実行タブ
        dcc.Tab(label='予測実行', value='prediction', children=[
            html.Div([
                html.H3("3. 需要・PV発電量予測"),
                
                html.Div([
                    html.Div([
                        html.Label("PV容量 (kW):"),
                        dcc.Input(id='pv-capacity-input', type='number', 
                                 value=1000, style={'margin': '10px'})
                    ], className='four columns'),
                    
                    html.Div([
                        html.Label("予測開始日時:"),
                        dcc.DatePickerSingle(
                            id='forecast-date',
                            date=datetime.now().date() + timedelta(days=1),
                            style={'margin': '10px'}
                        )
                    ], className='four columns'),
                    
                    html.Div([
                        html.Button('24時間予測実行', id='forecast-btn',
                                   style={'backgroundColor': '#27ae60', 'color': 'white',
                                         'padding': '10px 20px', 'border': 'none',
                                         'borderRadius': '5px', 'margin': '20px'})
                    ], className='four columns'),
                ], className='row'),
                
                html.Div(id='forecast-status', style={'margin': '20px 0'}),
                
                # 予測結果グラフ
                dcc.Graph(id='forecast-graph'),
                
                # 予測データテーブル
                html.Div(id='forecast-table', style={'margin': '20px 0'}),
                
            ], style={'padding': '20px'})
        ]),
        
        # リアルタイム補正タブ  
        dcc.Tab(label='リアルタイム補正', value='realtime-update', children=[
            html.Div([
                html.H3("4. 実績による予測補正"),
                
                html.Div([
                    html.Div([
                        html.Label("現在時刻の実績需要 (kW):"),
                        dcc.Input(id='actual-demand-input', type='number',
                                 placeholder='例: 850.5', style={'margin': '10px'})
                    ], className='six columns'),
                    
                    html.Div([
                        html.Label("現在の日射量 (kW/m²):"),
                        dcc.Input(id='actual-solar-input', type='number', 
                                 placeholder='例: 0.65', style={'margin': '10px'})
                    ], className='six columns'),
                ], className='row'),
                
                html.Button('予測更新', id='update-forecast-btn',
                           style={'backgroundColor': '#f39c12', 'color': 'white',
                                 'padding': '10px 20px', 'border': 'none',
                                 'borderRadius': '5px', 'margin': '10px'}),
                
                html.Div(id='update-status', style={'margin': '20px 0'}),
                
                # 更新結果グラフ
                dcc.Graph(id='updated-forecast-graph'),
                
            ], style={'padding': '20px'})
        ]),
    ]),
    
    # データ保存用（非表示）
    dcc.Store(id='demand-data-store'),
    dcc.Store(id='solar-data-store'),
    dcc.Store(id='forecast-data-store'),
    
], style={'fontFamily': 'Arial, sans-serif'})

# コールバック関数群

@app.callback(
    Output('upload-status', 'children'),
    [Input('upload-demand', 'contents'),
     Input('upload-solar', 'contents')],
    [State('upload-demand', 'filename'),
     State('upload-solar', 'filename')]
)
def update_upload_status(demand_contents, solar_contents, demand_filename, solar_filename):
    status = []
    if demand_contents:
        status.append(html.P(f"✓ 需要データ: {demand_filename}", style={'color': 'green'}))
    if solar_contents:
        status.append(html.P(f"✓ 日射データ: {solar_filename}", style={'color': 'green'}))
    
    return status

@app.callback(
    [Output('data-process-status', 'children'),
     Output('demand-data-store', 'data'),
     Output('solar-data-store', 'data')],
    [Input('process-data-btn', 'n_clicks')],
    [State('upload-demand', 'contents'),
     State('upload-solar', 'contents')]
)
def process_uploaded_data(n_clicks, demand_contents, solar_contents):
    if n_clicks is None or not demand_contents or not solar_contents:
        return "ファイルをアップロードしてください", None, None
    
    try:
        # Base64デコード
        demand_decoded = base64.b64decode(demand_contents.split(',')[1])
        solar_decoded = base64.b64decode(solar_contents.split(',')[1])
        
        success, message = predictor.load_and_process_data(demand_decoded, solar_decoded)
        
        if success:
            return html.P(f"✓ {message}", style={'color': 'green'}), demand_decoded, solar_decoded
        else:
            return html.P(f"✗ {message}", style={'color': 'red'}), None, None
            
    except Exception as e:
        return html.P(f"✗ 処理エラー: {str(e)}", style={'color': 'red'}), None, None

@app.callback(
    Output('training-status', 'children'),
    [Input('train-model-btn', 'n_clicks')]
)
def train_model(n_clicks):
    if n_clicks is None:
        return "学習ボタンをクリックしてください"
    
    success, message = predictor.train_model()
    
    if success:
        return html.P(f"✓ {message}", style={'color': 'green'})
    else:
        return html.P(f"✗ {message}", style={'color': 'red'})

@app.callback(
    [Output('forecast-status', 'children'),
     Output('forecast-graph', 'figure'),
     Output('forecast-table', 'children'),
     Output('forecast-data-store', 'data')],
    [Input('forecast-btn', 'n_clicks')],
    [State('pv-capacity-input', 'value'),
     State('forecast-date', 'date')]
)
def run_forecast(n_clicks, pv_capacity, forecast_date):
    if n_clicks is None:
        return "予測ボタンをクリックしてください", {}, "", None
    
    if pv_capacity:
        predictor.pv_capacity = pv_capacity
    
    forecast_datetime = datetime.strptime(forecast_date, '%Y-%m-%d')
    forecast_results, message = predictor.daily_forecast(forecast_datetime)
    
    if forecast_results is None:
        return html.P(f"✗ {message}", style={'color': 'red'}), {}, "", None
    
    # グラフ作成
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=forecast_results['datetime'],
        y=forecast_results['demand_forecast_kw'],
        mode='lines',
        name='需要予測',
        line=dict(color='blue', width=2)
    ))
    
    if 'pv_forecast_kw' in forecast_results.columns:
        fig.add_trace(go.Scatter(
            x=forecast_results['datetime'],
            y=forecast_results['pv_forecast_kw'],
            mode='lines',
            name='PV発電予測',
            line=dict(color='orange', width=2)
        ))
    
    fig.update_layout(
        title='24時間電力需要・PV発電量予測',
        xaxis_title='時刻',
        yaxis_title='電力 (kW)',
        hovermode='x unified'
    )
    
    # テーブル作成
    table = dash_table.DataTable(
        data=forecast_results.head(24).to_dict('records'),
        columns=[{'name': col, 'id': col} for col in forecast_results.columns],
        style_cell={'textAlign': 'center'},
        style_header={'backgroundColor': 'lightblue'},
        page_size=10
    )
    
    status = html.P(f"✓ {message}", style={'color': 'green'})
    
    return status, fig, table, forecast_results.to_json(date_format='iso')

if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8050)
