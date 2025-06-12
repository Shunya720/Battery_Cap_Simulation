"""
容量シミュレーション専用アプリケーション - デバッグ版
コアロジックを参照して複数容量での効果比較を実行
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# コアロジック読み込み（エラーハンドリング追加）
try:
    from battery_core_logic import (
        BatteryControlEngine, PeakBottomOptimizer, BatterySOCManager, 
        DemandSmoothnessOptimizer, create_time_series, OPTIMIZATION_AVAILABLE
    )
    CORE_LOGIC_AVAILABLE = True
except ImportError as e:
    st.error(f"コアロジックの読み込みに失敗: {e}")
    CORE_LOGIC_AVAILABLE = False
    # デフォルト値を設定
    OPTIMIZATION_AVAILABLE = False


class BatteryCapacityComparator:
    """バッテリー容量別シミュレーション比較クラス"""
    
    def __init__(self):
        self.comparison_results = {}
    
    def run_capacity_comparison(self, demand_forecast, capacity_list, cycle_target_ratio=1.0, 
                              cycle_tolerance=1500, optimization_trials=50, power_scaling_method='capacity_ratio',
                              manual_scaling_ratio=16.0, manual_base_power=0, manual_powers=None):
        """複数容量でのシミュレーション比較実行"""
        self.comparison_results = {}
        
        # 入力検証
        if not isinstance(demand_forecast, (list, np.ndarray)) or len(demand_forecast) < 96:
            raise ValueError("demand_forecast は96以上の要素を持つ配列である必要があります")
        
        if not capacity_list or len(capacity_list) == 0:
            raise ValueError("capacity_list が空です")
        
        # NaN値の確認と処理
        demand_forecast = np.array(demand_forecast)
        if np.any(np.isnan(demand_forecast)):
            st.warning("需要予測データにNaN値が含まれています。0で補完します。")
            demand_forecast = np.nan_to_num(demand_forecast, nan=0.0)
        
        for i, capacity in enumerate(capacity_list):
            try:
                print(f"容量 {capacity:,}kWh の最適化開始 ({i+1}/{len(capacity_list)})")
                
                # 容量に比例したサイクル目標設定
                cycle_target = int(capacity * cycle_target_ratio)
                
                # 容量に応じた最大出力設定
                if power_scaling_method == 'capacity_ratio':
                    max_power = capacity / 16
                elif power_scaling_method == 'manual':
                    if manual_powers and i < len(manual_powers):
                        # 個別設定がある場合
                        max_power = manual_powers[i]
                    else:
                        # 数式設定の場合
                        max_power = capacity / manual_scaling_ratio + manual_base_power
                else:
                    max_power = capacity / 16
                
                print(f"   - 容量: {capacity:,}kWh, 最大出力: {max_power:.0f}kW/15分")
                
                # バッテリー制御エンジン初期化
                if not CORE_LOGIC_AVAILABLE:
                    # コアロジックが利用できない場合のダミー処理
                    st.warning("コアロジックが利用できません。ダミーデータで処理します。")
                    dummy_result = self._create_dummy_result(demand_forecast, capacity, max_power, cycle_target)
                    self.comparison_results[capacity] = dummy_result
                    continue
                
                engine = BatteryControlEngine(
                    battery_capacity=capacity,
                    max_power=max_power
                )
                
                # 最適化実行
                if OPTIMIZATION_AVAILABLE:
                    optimization_result = engine.run_optimization(
                        demand_forecast,
                        cycle_target=cycle_target,
                        cycle_tolerance=cycle_tolerance,
                        method='optuna',
                        n_trials=optimization_trials
                    )
                    
                    optimized_params = optimization_result.get('best_params')
                    
                    if optimized_params is None:
                        print(f"容量 {capacity:,}kWh の最適化に失敗")
                        continue
                    
                    control_result = engine.run_control_simulation(
                        demand_forecast, **optimized_params
                    )
                    
                else:
                    default_params = {
                        'peak_percentile': 80,
                        'bottom_percentile': 20,
                        'peak_power_ratio': 1.0,
                        'bottom_power_ratio': 1.0,
                        'flattening_power_ratio': 0.3
                    }
                    
                    optimized_params = default_params
                    control_result = engine.run_control_simulation(
                        demand_forecast, **default_params
                    )
                
                # 制御後需要
                demand_after_control = control_result['demand_after_battery']
                battery_output = control_result['battery_output']
                soc_profile = control_result['soc_profile']
                control_info = control_result['control_info']
                
                # 滑らかさ指標計算
                smoothness_optimizer = DemandSmoothnessOptimizer(
                    PeakBottomOptimizer, BatterySOCManager, capacity, max_power
                )
                smoothness_metrics = smoothness_optimizer.calculate_demand_smoothness_metrics(
                    demand_forecast, demand_after_control
                )
                
                # 結果保存
                self.comparison_results[capacity] = {
                    'capacity': capacity,
                    'max_power': max_power,
                    'cycle_target': cycle_target,
                    'optimized_params': optimized_params,
                    'battery_output': battery_output,
                    'soc_profile': soc_profile,
                    'demand_after_control': demand_after_control,
                    'control_info': control_info,
                    'smoothness_metrics': smoothness_metrics,
                    'peak_reduction': np.max(demand_forecast) - np.max(demand_after_control),
                    'range_improvement': (np.max(demand_forecast) - np.min(demand_forecast)) - 
                                       (np.max(demand_after_control) - np.min(demand_after_control)),
                    'actual_discharge': -np.sum(battery_output[battery_output < 0]),
                    'cycle_constraint_satisfied': abs(-np.sum(battery_output[battery_output < 0]) - cycle_target) <= cycle_tolerance
                }
                
                print(f"容量 {capacity:,}kWh の最適化完了")
                
            except Exception as e:
                print(f"容量 {capacity:,}kWh でエラー: {e}")
                st.error(f"容量 {capacity:,}kWh でエラーが発生しました: {e}")
                import traceback
                st.text(traceback.format_exc())
                continue
        
        return self.comparison_results
    
    def _create_dummy_result(self, demand_forecast, capacity, max_power, cycle_target):
        """コアロジックが利用できない場合のダミー結果生成"""
        # 簡単なダミーデータを生成
        battery_output = np.random.uniform(-max_power/2, max_power/2, 96)
        demand_after_control = demand_forecast + battery_output
        soc_profile = np.random.uniform(20, 80, 96)
        
        return {
            'capacity': capacity,
            'max_power': max_power,
            'cycle_target': cycle_target,
            'optimized_params': {'peak_percentile': 80, 'bottom_percentile': 20, 'peak_power_ratio': 1.0, 'bottom_power_ratio': 1.0, 'flattening_power_ratio': 0.3},
            'battery_output': battery_output,
            'soc_profile': soc_profile,
            'demand_after_control': demand_after_control,
            'control_info': {},
            'smoothness_metrics': {'smoothness_improvement': 0.1, 'max_jump_improvement': 0.1},
            'peak_reduction': np.max(demand_forecast) - np.max(demand_after_control),
            'range_improvement': 100.0,
            'actual_discharge': np.sum(np.abs(battery_output[battery_output < 0])),
            'cycle_constraint_satisfied': True
        }
    
    def get_comparison_summary(self):
        """比較結果のサマリーを取得"""
        if not self.comparison_results:
            return None
        
        summary = []
        for capacity, result in self.comparison_results.items():
            # 安全な値の取得
            smoothness_improvement = result.get('smoothness_metrics', {}).get('smoothness_improvement', 0) * 100
            max_jump_improvement = result.get('smoothness_metrics', {}).get('max_jump_improvement', 0) * 100
            
            summary.append({
                '容量(kWh)': f"{capacity:,}",
                '最大出力(kW)': f"{result['max_power']:.0f}",
                'ピーク削減(kW)': f"{result['peak_reduction']:.1f}",
                '需要幅改善(kW)': f"{result['range_improvement']:.1f}",
                '隣接変動改善(%)': f"{smoothness_improvement:.1f}",
                '最大変動抑制(%)': f"{max_jump_improvement:.1f}",
                'サイクル制約': 'OK' if result['cycle_constraint_satisfied'] else 'NG',
                '実際放電(kWh)': f"{result['actual_discharge']:.0f}",
                'ピーク制御比率': f"{result['optimized_params'].get('peak_power_ratio', 1.0):.2f}",
                'ボトム制御比率': f"{result['optimized_params'].get('bottom_power_ratio', 1.0):.2f}"
            })
        
        return pd.DataFrame(summary)


def safe_create_time_series(start_time=None):
    """安全な時系列作成関数"""
    if start_time is None:
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    try:
        if CORE_LOGIC_AVAILABLE:
            return create_time_series(start_time)
        else:
            # ダミーの時系列を作成
            return [start_time + timedelta(minutes=15*i) for i in range(96)]
    except Exception as e:
        st.error(f"時系列作成エラー: {e}")
        return [start_time + timedelta(minutes=15*i) for i in range(96)]


def main():
    st.title("バッテリー容量別シミュレーション比較システム")
    st.write("複数のバッテリー容量で需要平準化効果を比較し、最適容量を検討")
    
    # コアロジック利用可能性の表示
    if not CORE_LOGIC_AVAILABLE:
        st.error("⚠️ コアロジック（battery_core_logic）が利用できません。ダミーデータでの動作となります。")
    
    # CSVアップロード
    st.header("1. 需要予測データアップロード")
    uploaded_file = st.file_uploader("需要予測CSV（96ステップ、15分間隔）", type=['csv'])
    
    demand_forecast = None
    
    if uploaded_file is not None:
        try:
            encodings = ['utf-8', 'shift-jis', 'cp932', 'euc-jp', 'iso-2022-jp']
            df = None
            
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    st.success(f"エンコーディング: {encoding} で読み込み成功")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                st.error("サポートされているエンコーディングで読み込めませんでした")
            elif len(df.columns) >= 2:
                st.subheader("データプレビュー")
                st.dataframe(df.head(10))
                
                time_column = st.selectbox("時刻列を選択", df.columns, index=0)
                demand_column = st.selectbox("需要データ列を選択", df.columns, index=1)
                
                if len(df) >= 96:
                    try:
                        demand_values = pd.to_numeric(df[demand_column], errors='coerce').values
                        demand_forecast = demand_values[:96]
                        valid_count = np.sum(~np.isnan(demand_forecast))
                        
                        st.success(f"需要予測データ読み込み完了（{valid_count}/96ステップ有効）")
                        
                        # NaN値の処理
                        if valid_count < 96:
                            st.warning(f"NaN値が{96-valid_count}個含まれています。平均値で補完します。")
                            mean_value = np.nanmean(demand_forecast)
                            demand_forecast = np.nan_to_num(demand_forecast, nan=mean_value)
                        
                        valid_demands = demand_forecast[~np.isnan(demand_forecast)]
                        if len(valid_demands) > 0:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("最小値", f"{valid_demands.min():.0f} kW")
                            with col2:
                                st.metric("平均値", f"{valid_demands.mean():.0f} kW")
                            with col3:
                                st.metric("最大値", f"{valid_demands.max():.0f} kW")
                            with col4:
                                st.metric("需要幅", f"{valid_demands.max() - valid_demands.min():.0f} kW")
                        
                    except Exception as e:
                        st.error(f"需要データの変換エラー: {e}")
                        import traceback
                        st.text(traceback.format_exc())
                else:
                    st.error(f"データが96ステップ未満です（現在: {len(df)}ステップ）")
            else:
                st.error("CSVファイルに最低2列（時刻、需要）が必要です")
                
        except Exception as e:
            st.error(f"ファイル読み込みエラー: {e}")
            import traceback
            st.text(traceback.format_exc())
    
    if demand_forecast is not None:
        
        st.header("2. 容量別シミュレーション設定")
        
        with st.expander("容量別比較設定", expanded=True):
            st.subheader("比較対象の容量設定")
            
            # 容量数の選択
            num_capacities = st.selectbox(
                "比較する容量の数",
                options=[2, 3, 4],
                index=1,  # デフォルトは3つ
                help="比較したい容量の数を選択してください"
            )
            
            # 容量設定の入力欄を動的に表示
            capacity_list = []
            cols = st.columns(4)
            
            # 容量1（必須）
            with cols[0]:
                capacity1 = st.number_input("容量1 (kWh)", value=24000, min_value=10000, max_value=200000, step=6000)
                capacity_list.append(capacity1)
            
            # 容量2（必須）
            with cols[1]:
                capacity2 = st.number_input("容量2 (kWh)", value=48000, min_value=10000, max_value=200000, step=6000)
                capacity_list.append(capacity2)
            
            # 容量3（オプション）
            if num_capacities >= 3:
                with cols[2]:
                    capacity3 = st.number_input("容量3 (kWh)", value=100000, min_value=10000, max_value=200000, step=6000)
                    capacity_list.append(capacity3)
            else:
                with cols[2]:
                    st.text_input("容量3 (kWh)", value="未使用", disabled=True)
            
            # 容量4（オプション）
            if num_capacities >= 4:
                with cols[3]:
                    capacity4 = st.number_input("容量4 (kWh)", value=200000, min_value=10000, max_value=200000, step=6000)
                    capacity_list.append(capacity4)
            else:
                with cols[3]:
                    st.text_input("容量4 (kWh)", value="未使用", disabled=True)
            
            # 重複チェック（選択された容量のみ）
            if len(set(capacity_list)) != len(capacity_list):
                st.warning("重複する容量があります。異なる容量を設定してください。")
            
            # 選択された容量の表示
            st.info(f"選択された容量: {', '.join([f'{cap:,}kWh' for cap in capacity_list])}")
            
            st.subheader("最大出力スケーリング設定")
            power_scaling_method = st.selectbox(
                "最大出力の決定方法",
                ["capacity_ratio", "manual"],
                index=0,
                format_func=lambda x: {
                    "capacity_ratio": "容量比例（容量÷16）",
                    "manual": "手動入力"
                }[x],
                help="バッテリー容量に対する最大出力の算出方法"
            )
            
            # 手動入力モードの場合のパラメータ設定
            manual_scaling_ratio = 16.0
            manual_base_power = 0
            manual_override = False
            
            if power_scaling_method == "manual":
                st.subheader("手動最大出力設定（15分値）")
                col1, col2 = st.columns(2)
                with col1:
                    manual_scaling_ratio = st.number_input(
                        "容量比率（容量÷X）", 
                        value=16.0, min_value=1.0, max_value=50.0, step=1.0,
                        help="容量をこの値で割った値を最大出力とする"
                    )
                with col2:
                    manual_base_power = st.number_input(
                        "ベース出力 (kW)", 
                        value=0, min_value=0, max_value=20000, step=100,
                        help="全容量に共通で加算するベース出力"
                    )
                
                # 個別設定オプション
                st.write("**個別設定（オプション）:**")
                manual_override = st.checkbox("容量別に個別の最大出力を設定", value=False)
                
                manual_powers_dict = {}
                if manual_override:
                    st.write("選択された容量に対して個別の最大出力を設定:")
                    cols = st.columns(4)
                    
                    for i, capacity in enumerate(capacity_list):
                        if i < 4:  # 最大4列まで
                            with cols[i]:
                                manual_power = st.number_input(
                                    f"容量{i+1}の最大出力 (kW)", 
                                    value=int(capacity / manual_scaling_ratio + manual_base_power), 
                                    min_value=100, max_value=50000, step=100,
                                    key=f"manual_power_{i}"
                                )
                                manual_powers_dict[i] = manual_power
            
            # 最大出力計算関数
            def calculate_max_power(capacity, method):
                if method == "capacity_ratio":
                    return capacity / 16
                elif method == "manual":
                    return capacity / manual_scaling_ratio + manual_base_power
                return capacity / 16
            
            # 最大出力表示
            st.subheader("容量別最大出力（15分値）")
            cols = st.columns(4)
            
            for i, capacity in enumerate(capacity_list):
                if i < 4:  # 最大4列まで
                    with cols[i]:
                        if power_scaling_method == "manual" and manual_override and i in manual_powers_dict:
                            max_power = manual_powers_dict[i]
                        else:
                            max_power = calculate_max_power(capacity, power_scaling_method)
                        st.info(f"容量{i+1}: {max_power:.0f}kW/15分")
            
            # 使用されていない容量欄は空白表示
            for i in range(len(capacity_list), 4):
                with cols[i]:
                    st.write("")  # 空白
            
            st.subheader("容量別最適化設定")
            col1, col2 = st.columns(2)
            
            with col1:
                capacity_cycle_ratio = st.slider(
                    "サイクル目標比率", 
                    min_value=0.5, max_value=2.0, value=1.0, step=0.1,
                    help="容量に対するサイクル目標の比率（1.0 = 容量と同じkWh）"
                )
                
                capacity_cycle_tolerance = st.number_input(
                    "サイクル許容範囲 (kWh)", 
                    value=1500, min_value=500, max_value=20000, step=500,
                    help="各容量でのサイクル制約許容範囲"
                )
            
            with col2:
                capacity_optimization_trials = st.slider(
                    "最適化試行回数（容量別）",
                    min_value=30, max_value=100, value=50, step=10,
                    help="容量別最適化の試行回数"
                )
                
                num_capacities = len(capacity_list)  # 修正: 実際に選択された容量数を使用
                st.info(f"""
                予想計算時間:
                {num_capacities}容量 × {capacity_optimization_trials}試行
                最大出力: {power_scaling_method}方式
                約 {capacity_optimization_trials * num_capacities * 0.5:.0f}秒 〜 {capacity_optimization_trials * num_capacities * 2:.0f}秒
                """)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("容量別シミュレーション実行", use_container_width=True):
                
                if len(set(capacity_list)) != len(capacity_list):
                    st.error("重複する容量があります。異なる容量を設定してください。")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        status_text.text("容量別シミュレーションシステム初期化中...")
                        progress_bar.progress(5)
                        
                        capacity_comparator = BatteryCapacityComparator()
                        
                        status_text.text("需要データ準備中...")
                        progress_bar.progress(10)
                        
                        status_text.text("容量別最適化実行中...")
                        progress_bar.progress(15)
                        
                        # 手動設定の場合のパラメータ準備
                        manual_powers_list = None
                        if power_scaling_method == "manual" and manual_override and manual_powers_dict:
                            manual_powers_list = []
                            for i in range(len(capacity_list)):
                                if i in manual_powers_dict:
                                    manual_powers_list.append(manual_powers_dict[i])
                                else:
                                    manual_powers_list.append(calculate_max_power(capacity_list[i], power_scaling_method))
                        
                        # パラメータ辞書作成
                        comparison_params = {
                            'demand_forecast': demand_forecast,
                            'capacity_list': capacity_list,
                            'cycle_target_ratio': capacity_cycle_ratio,
                            'cycle_tolerance': capacity_cycle_tolerance,
                            'optimization_trials': capacity_optimization_trials,
                            'power_scaling_method': power_scaling_method
                        }
                        
                        # 手動設定のパラメータを追加
                        if power_scaling_method == "manual":
                            comparison_params['manual_scaling_ratio'] = manual_scaling_ratio
                            comparison_params['manual_base_power'] = manual_base_power
                            if manual_powers_list:
                                comparison_params['manual_powers'] = manual_powers_list
                        
                        comparison_results = capacity_comparator.run_capacity_comparison(**comparison_params)
                        
                        progress_bar.progress(90)
                        status_text.text("結果分析中...")
                        
                        if comparison_results:
                            st.session_state.capacity_comparison_results = comparison_results
                            st.session_state.capacity_list = capacity_list
                            st.session_state.demand_forecast = demand_forecast
                            
                            progress_bar.progress(100)
                            status_text.text("容量別シミュレーション完了！")
                            
                            st.success(f"{len(comparison_results)}種類の容量でシミュレーションが完了しました！")
                            
                            st.session_state.show_capacity_results = True
                            
                            time.sleep(1)
                            st.rerun()
                        
                        else:
                            st.error("容量別シミュレーションで有効な結果が得られませんでした。")
                    
                    except Exception as e:
                        st.error(f"容量別シミュレーションエラー: {e}")
                        import traceback
                        st.text(traceback.format_exc())
                    
                    finally:
                        progress_bar.empty()
                        status_text.empty()
        
        # 結果表示
        if hasattr(st.session_state, 'show_capacity_results') and st.session_state.show_capacity_results and hasattr(st.session_state, 'capacity_comparison_results'):
            results = st.session_state.capacity_comparison_results
            capacity_list = st.session_state.capacity_list
            demand_forecast = st.session_state.demand_forecast
            
            st.header("3. 容量別シミュレーション結果")
            
            capacity_comparator = BatteryCapacityComparator()
            capacity_comparator.comparison_results = results
            summary_df = capacity_comparator.get_comparison_summary()
            
            if summary_df is not None:
                st.write("容量別効果サマリー:")
                st.dataframe(summary_df, use_container_width=True)
            
            st.subheader("容量別需要カーブ比較")
            
            fig_capacity_demand = go.Figure()
            
            try:
                time_series = safe_create_time_series(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0))
                fig_capacity_demand.add_trace(go.Scatter(
                    x=time_series, y=demand_forecast,
                    name="元需要予測", line=dict(color="lightgray", dash="dash", width=2),
                    opacity=0.7
                ))
                
                colors = ['red', 'blue', 'green', 'orange']
                for i, (capacity, result) in enumerate(results.items()):
                    fig_capacity_demand.add_trace(go.Scatter(
                        x=time_series, y=result['demand_after_control'],
                        name=f"容量{capacity:,}kWh制御後",
                        line=dict(color=colors[i % len(colors)], width=3)
                    ))
                
                fig_capacity_demand.update_layout(
                    title="容量別需要平準化効果比較",
                    xaxis_title="時刻",
                    yaxis_title="需要 (kW)",
                    height=500,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                st.plotly_chart(fig_capacity_demand, use_container_width=True)
                
            except Exception as e:
                st.error(f"グラフ作成エラー: {e}")
                st.text("グラフの表示をスキップしました")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("容量別電池出力比較")
                try:
                    fig_capacity_battery = go.Figure()
                    
                    for i, (capacity, result) in enumerate(results.items()):
                        fig_capacity_battery.add_trace(go.Scatter(
                            x=time_series, y=result['battery_output'],
                            name=f"容量{capacity:,}kWh",
                            line=dict(color=colors[i % len(colors)], width=2),
                            mode='lines'
                        ))
                    
                    fig_capacity_battery.update_layout(
                        title="容量別電池出力パターン",
                        xaxis_title="時刻",
                        yaxis_title="出力 (kWh)",
                        height=400
                    )
                    st.plotly_chart(fig_capacity_battery, use_container_width=True)
                except Exception as e:
                    st.error(f"電池出力グラフエラー: {e}")
            
            with col2:
                st.subheader("容量別SOCプロファイル")
                try:
                    fig_capacity_soc = go.Figure()
                    
                    for i, (capacity, result) in enumerate(results.items()):
                        fig_capacity_soc.add_trace(go.Scatter(
                            x=time_series, y=result['soc_profile'],
                            name=f"容量{capacity:,}kWh",
                            line=dict(color=colors[i % len(colors)], width=2)
                        ))
                    
                    fig_capacity_soc.add_hline(y=10, line_dash="dot", line_color="red", opacity=0.5)
                    fig_capacity_soc.add_hline(y=90, line_dash="dot", line_color="blue", opacity=0.5)
                    
                    fig_capacity_soc.update_layout(
                        title="容量別SOC変化",
                        xaxis_title="時刻",
                        yaxis_title="SOC (%)",
                        height=400,
                        yaxis=dict(range=[0, 100])
                    )
                    st.plotly_chart(fig_capacity_soc, use_container_width=True)
                except Exception as e:
                    st.error(f"SOCグラフエラー: {e}")
            
            # 推奨容量の判定
            st.subheader("推奨容量の判定")
            
            try:
                best_capacity = None
                best_score = -1
                
                for capacity, result in results.items():
                    # 安全なスコア計算
                    peak_reduction = result.get('peak_reduction', 0)
                    range_improvement = result.get('range_improvement', 0)
                    smoothness_improvement = result.get('smoothness_metrics', {}).get('smoothness_improvement', 0)
                    cycle_satisfied = result.get('cycle_constraint_satisfied', False)
                    
                    score = (
                        peak_reduction * 0.3 +
                        range_improvement * 0.2 +
                        smoothness_improvement * 100 * 0.3 +
                        (100 if cycle_satisfied else 0) * 0.2
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_capacity = capacity
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if best_capacity is not None:
                        st.success(f"""
                        総合評価1位:
                        
                        容量: {best_capacity:,}kWh
                        
                        総合スコア: {best_score:.1f}点
                        推奨理由: 最もバランスの取れた効果
                        """)
                    else:
                        st.warning("総合評価の計算ができませんでした")
                
                with col2:
                    try:
                        best_peak_capacity = max(results.keys(), 
                                               key=lambda x: results[x].get('peak_reduction', 0))
                        peak_reduction_value = results[best_peak_capacity].get('peak_reduction', 0)
                        st.info(f"""
                        ピーク削減効果1位:
                        
                        容量: {best_peak_capacity:,}kWh
                        
                        ピーク削減: {peak_reduction_value:.1f}kW
                        特徴: 最大需要抑制に優秀
                        """)
                    except Exception as e:
                        st.error(f"ピーク削減評価エラー: {e}")
                
                with col3:
                    try:
                        best_smooth_capacity = max(results.keys(), 
                                                 key=lambda x: results[x].get('smoothness_metrics', {}).get('smoothness_improvement', 0))
                        smoothness_best = results[best_smooth_capacity].get('smoothness_metrics', {}).get('smoothness_improvement', 0) * 100
                        st.info(f"""
                        滑らかさ改善1位:
                        
                        容量: {best_smooth_capacity:,}kWh
                        
                        隣接変動改善: {smoothness_best:.1f}%
                        特徴: 最も滑らかな需要カーブ
                        """)
                    except Exception as e:
                        st.error(f"滑らかさ評価エラー: {e}")
                        
            except Exception as e:
                st.error(f"推奨容量判定エラー: {e}")
            
            # 結果のダウンロード
            st.subheader("容量別シミュレーション結果ダウンロード")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if summary_df is not None:
                    try:
                        summary_csv = summary_df.to_csv(index=False)
                        st.download_button(
                            label="容量別サマリーCSVダウンロード",
                            data=summary_csv,
                            file_name=f"capacity_comparison_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"サマリーCSV生成エラー: {e}")
            
            with col2:
                if st.button("容量別詳細結果CSVダウンロード", use_container_width=True):
                    try:
                        detailed_data = []
                        time_series = safe_create_time_series(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0))
                        
                        for capacity, result in results.items():
                            for i in range(96):
                                detailed_data.append({
                                    '容量(kWh)': capacity,
                                    '最大出力(kWh/15分)': result.get('max_power', 0),
                                    'ステップ': i + 1,
                                    '時刻': time_series[i].strftime('%H:%M') if i < len(time_series) else f"{i*15//60:02d}:{i*15%60:02d}",
                                    '元需要': demand_forecast[i] if i < len(demand_forecast) else 0,
                                    '制御後需要': result['demand_after_control'][i] if i < len(result.get('demand_after_control', [])) else 0,
                                    '電池出力': result['battery_output'][i] if i < len(result.get('battery_output', [])) else 0,
                                    'SOC(%)': result['soc_profile'][i] if i < len(result.get('soc_profile', [])) else 0,
                                    '需要削減': (demand_forecast[i] if i < len(demand_forecast) else 0) - (result['demand_after_control'][i] if i < len(result.get('demand_after_control', [])) else 0)
                                })
                        
                        detailed_df = pd.DataFrame(detailed_data)
                        detailed_csv = detailed_df.to_csv(index=False)
                        
                        st.download_button(
                            label="詳細結果をダウンロード",
                            data=detailed_csv,
                            file_name=f"capacity_comparison_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"詳細結果生成エラー: {e}")
                        import traceback
                        st.text(traceback.format_exc())

if __name__ == "__main__":
    st.warning("需要予測データをアップロードしてください")
    st.info("CSVファイルには時刻列と需要列（96ステップ、15分間隔）が必要です")
    
    main()
