"""
年間容量シミュレーション専用アプリケーション
複数容量での年間需要平準化効果比較を実行
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import warnings
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
warnings.filterwarnings('ignore')

# コアロジック読み込み（エラーハンドリング追加）
try:
    from battery_core_logic import (
        BatteryControlEngine, PeakBottomOptimizer, BatterySOCManager, 
        DemandSmoothnessOptimizer, create_time_series, OPTIMIZATION_AVAILABLE
    )
    CORE_LOGIC_AVAILABLE = True
    print("✅ コアロジック読み込み成功")
except ImportError as e:
    print(f"⚠️ コアロジックの読み込みに失敗: {e}")
    print("ダミーデータモードで動作します")
    CORE_LOGIC_AVAILABLE = False
    OPTIMIZATION_AVAILABLE = False
except SyntaxError as e:
    print(f"⚠️ コアロジックに構文エラー: {e}")
    print("ダミーデータモードで動作します")
    CORE_LOGIC_AVAILABLE = False
    OPTIMIZATION_AVAILABLE = False


class AnnualBatteryCapacityComparator:
    """年間バッテリー容量別シミュレーション比較クラス"""
    
    def __init__(self):
        self.comparison_results = {}
        self.monthly_results = {}
        
    def validate_annual_data(self, demand_forecast):
        """年間データの検証"""
        if not isinstance(demand_forecast, (list, np.ndarray)):
            raise ValueError("demand_forecast は配列である必要があります")
        
        demand_array = np.array(demand_forecast)
        expected_steps = 365 * 96  # 35,040ステップ
        
        if len(demand_array) < expected_steps:
            if len(demand_array) >= 96:
                # 日単位データの場合、年間に拡張
                days_available = len(demand_array) // 96
                if days_available < 7:
                    raise ValueError(f"最低7日分のデータが必要です（現在: {days_available}日分）")
                
                # 週単位パターンで年間拡張
                weekly_pattern = demand_array[:days_available*96]
                extended_data = []
                
                for week in range(53):  # 年間53週
                    if len(extended_data) + len(weekly_pattern) <= expected_steps:
                        extended_data.extend(weekly_pattern)
                    else:
                        remaining = expected_steps - len(extended_data)
                        extended_data.extend(weekly_pattern[:remaining])
                        break
                
                demand_array = np.array(extended_data)
                st.info(f"データを{days_available}日パターンから年間{len(demand_array)}ステップに拡張しました")
            else:
                raise ValueError(f"データが不足しています（現在: {len(demand_array)}、必要: {expected_steps}）")
        
        # NaN値の処理
        if np.any(np.isnan(demand_array)):
            nan_count = np.sum(np.isnan(demand_array))
            st.warning(f"年間データにNaN値が{nan_count:,}個含まれています。補間処理を実行します。")
            
            # 線形補間でNaN値を埋める
            mask = ~np.isnan(demand_array)
            indices = np.arange(len(demand_array))
            demand_array[~mask] = np.interp(indices[~mask], indices[mask], demand_array[mask])
        
        return demand_array[:expected_steps]  # 必要なステップ数に切り取り
    
    def create_monthly_batches(self, annual_demand):
        """年間データを月別バッチに分割"""
        monthly_batches = []
        days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        
        start_idx = 0
        for month, days in enumerate(days_per_month):
            end_idx = start_idx + (days * 96)
            if end_idx <= len(annual_demand):
                monthly_data = annual_demand[start_idx:end_idx]
                monthly_batches.append({
                    'month': month + 1,
                    'month_name': ['1月', '2月', '3月', '4月', '5月', '6月',
                                 '7月', '8月', '9月', '10月', '11月', '12月'][month],
                    'data': monthly_data,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
                start_idx = end_idx
            else:
                break
        
        return monthly_batches
    
    def run_monthly_simulation(self, monthly_data, capacity, max_power, 
                             cycle_target, cycle_tolerance, optimization_trials):
        """月別シミュレーション実行"""
        try:
            if not CORE_LOGIC_AVAILABLE:
                return self._create_dummy_monthly_result(monthly_data, capacity, max_power, cycle_target)
            
            # 月別サイクル目標（年間目標を12で割る）
            monthly_cycle_target = cycle_target // 12
            
            engine = BatteryControlEngine(
                battery_capacity=capacity,
                max_power=max_power
            )
            
            # 最適化実行
            if OPTIMIZATION_AVAILABLE:
                optimization_result = engine.run_optimization(
                    monthly_data,
                    cycle_target=monthly_cycle_target,
                    cycle_tolerance=cycle_tolerance // 12,
                    method='optuna',
                    n_trials=optimization_trials
                )
                
                optimized_params = optimization_result.get('best_params')
                if optimized_params is None:
                    # デフォルトパラメータで実行
                    optimized_params = {
                        'peak_percentile': 80,
                        'bottom_percentile': 20,
                        'peak_power_ratio': 1.0,
                        'bottom_power_ratio': 1.0,
                        'flattening_power_ratio': 0.3
                    }
            else:
                optimized_params = {
                    'peak_percentile': 80,
                    'bottom_percentile': 20,
                    'peak_power_ratio': 1.0,
                    'bottom_power_ratio': 1.0,
                    'flattening_power_ratio': 0.3
                }
            
            control_result = engine.run_control_simulation(
                monthly_data, **optimized_params
            )
            
            return {
                'optimized_params': optimized_params,
                'battery_output': control_result['battery_output'],
                'soc_profile': control_result['soc_profile'],
                'demand_after_control': control_result['demand_after_battery'],
                'control_info': control_result['control_info'],
                'monthly_discharge': -np.sum(control_result['battery_output'][control_result['battery_output'] < 0]),
                'peak_reduction': np.max(monthly_data) - np.max(control_result['demand_after_battery']),
                'range_improvement': (np.max(monthly_data) - np.min(monthly_data)) - 
                                   (np.max(control_result['demand_after_battery']) - np.min(control_result['demand_after_battery']))
            }
            
        except Exception as e:
            st.warning(f"月別シミュレーションでエラー: {e}")
            return self._create_dummy_monthly_result(monthly_data, capacity, max_power, cycle_target)
    
    def _create_dummy_monthly_result(self, monthly_data, capacity, max_power, cycle_target):
        """ダミー月別結果生成"""
        battery_output = np.random.uniform(-max_power/2, max_power/2, len(monthly_data))
        demand_after_control = monthly_data + battery_output
        soc_profile = np.random.uniform(20, 80, len(monthly_data))
        
        return {
            'optimized_params': {'peak_percentile': 80, 'bottom_percentile': 20, 'peak_power_ratio': 1.0, 'bottom_power_ratio': 1.0, 'flattening_power_ratio': 0.3},
            'battery_output': battery_output,
            'soc_profile': soc_profile,
            'demand_after_control': demand_after_control,
            'control_info': {},
            'monthly_discharge': np.sum(np.abs(battery_output[battery_output < 0])),
            'peak_reduction': np.max(monthly_data) - np.max(demand_after_control),
            'range_improvement': 100.0
        }
    
    def run_annual_capacity_comparison(self, annual_demand, capacity_list, 
                                     cycle_target_ratio=1.0, cycle_tolerance=1500,
                                     optimization_trials=30, power_scaling_method='capacity_ratio',
                                     manual_scaling_ratio=16.0, manual_base_power=0, 
                                     manual_powers=None, use_parallel=True):
        """年間容量別シミュレーション実行"""
        
        # データ検証
        validated_demand = self.validate_annual_data(annual_demand)
        
        # 月別バッチ作成
        monthly_batches = self.create_monthly_batches(validated_demand)
        st.info(f"年間データを{len(monthly_batches)}ヶ月のバッチに分割しました")
        
        self.comparison_results = {}
        self.monthly_results = {}
        
        total_operations = len(capacity_list) * len(monthly_batches)
        completed_operations = 0
        
        for i, capacity in enumerate(capacity_list):
            try:
                st.write(f"容量 {capacity:,}kWh の年間最適化開始 ({i+1}/{len(capacity_list)})")
                
                # 容量に応じた設定
                annual_cycle_target = int(capacity * cycle_target_ratio)
                
                # 最大出力設定
                if power_scaling_method == 'capacity_ratio':
                    max_power = capacity / 16
                elif power_scaling_method == 'custom':
                    max_power = capacity / 20
                elif power_scaling_method == 'fixed':
                    max_power = 3000
                elif power_scaling_method == 'manual':
                    if manual_powers and i < len(manual_powers):
                        max_power = manual_powers[i]
                    else:
                        max_power = capacity / manual_scaling_ratio + manual_base_power
                else:
                    max_power = capacity / 16
                
                # 月別結果を保存するリスト
                monthly_results_for_capacity = {}
                annual_battery_output = []
                annual_soc_profile = []
                annual_demand_after_control = []
                
                # 月別シミュレーション（並列処理オプション）
                if use_parallel and len(monthly_batches) > 3:
                    # 並列処理
                    with ThreadPoolExecutor(max_workers=min(4, len(monthly_batches))) as executor:
                        future_to_month = {
                            executor.submit(
                                self.run_monthly_simulation,
                                batch['data'], capacity, max_power, 
                                annual_cycle_target, cycle_tolerance, optimization_trials
                            ): batch for batch in monthly_batches
                        }
                        
                        for future in as_completed(future_to_month):
                            batch = future_to_month[future]
                            try:
                                result = future.result()
                                monthly_results_for_capacity[batch['month']] = result
                                completed_operations += 1
                                
                                # プログレス更新
                                progress = completed_operations / total_operations
                                st.progress(progress)
                                
                            except Exception as e:
                                st.error(f"{batch['month_name']}の処理でエラー: {e}")
                
                else:
                    # 逐次処理
                    for batch in monthly_batches:
                        try:
                            result = self.run_monthly_simulation(
                                batch['data'], capacity, max_power,
                                annual_cycle_target, cycle_tolerance, optimization_trials
                            )
                            monthly_results_for_capacity[batch['month']] = result
                            completed_operations += 1
                            
                            # プログレス更新
                            progress = completed_operations / total_operations
                            st.progress(progress)
                            
                            st.write(f"  - {batch['month_name']} 完了")
                            
                        except Exception as e:
                            st.error(f"{batch['month_name']}の処理でエラー: {e}")
                            continue
                
                # 月別結果を年間結果に統合
                for month in sorted(monthly_results_for_capacity.keys()):
                    result = monthly_results_for_capacity[month]
                    annual_battery_output.extend(result['battery_output'])
                    annual_soc_profile.extend(result['soc_profile'])
                    annual_demand_after_control.extend(result['demand_after_control'])
                
                # 年間統計計算
                annual_battery_output = np.array(annual_battery_output)
                annual_demand_after_control = np.array(annual_demand_after_control)
                annual_soc_profile = np.array(annual_soc_profile)
                
                # 年間滑らかさ指標（サンプリングして計算負荷軽減）
                sample_indices = np.random.choice(len(validated_demand), 
                                                min(len(validated_demand), 10000), 
                                                replace=False)
                sample_original = validated_demand[sample_indices]
                sample_controlled = annual_demand_after_control[sample_indices]
                
                smoothness_metrics = {
                    'smoothness_improvement': np.std(np.diff(sample_original)) - np.std(np.diff(sample_controlled)),
                    'max_jump_improvement': np.max(np.abs(np.diff(sample_original))) - np.max(np.abs(np.diff(sample_controlled)))
                }
                
                # 年間結果保存
                self.comparison_results[capacity] = {
                    'capacity': capacity,
                    'max_power': max_power,
                    'annual_cycle_target': annual_cycle_target,
                    'battery_output': annual_battery_output,
                    'soc_profile': annual_soc_profile,
                    'demand_after_control': annual_demand_after_control,
                    'smoothness_metrics': smoothness_metrics,
                    'annual_peak_reduction': np.max(validated_demand) - np.max(annual_demand_after_control),
                    'annual_range_improvement': (np.max(validated_demand) - np.min(validated_demand)) - 
                                              (np.max(annual_demand_after_control) - np.min(annual_demand_after_control)),
                    'annual_discharge': -np.sum(annual_battery_output[annual_battery_output < 0]),
                    'annual_cycle_constraint_satisfied': abs(-np.sum(annual_battery_output[annual_battery_output < 0]) - annual_cycle_target) <= cycle_tolerance,
                    'monthly_results': monthly_results_for_capacity,
                    # 季節別統計
                    'seasonal_stats': self._calculate_seasonal_stats(validated_demand, annual_demand_after_control, monthly_results_for_capacity)
                }
                
                self.monthly_results[capacity] = monthly_results_for_capacity
                
                st.success(f"容量 {capacity:,}kWh の年間最適化完了")
                
                # メモリクリーンアップ
                gc.collect()
                
            except Exception as e:
                st.error(f"容量 {capacity:,}kWh でエラー: {e}")
                import traceback
                st.text(traceback.format_exc())
                continue
        
        return self.comparison_results
    
    def _calculate_seasonal_stats(self, original_demand, controlled_demand, monthly_results):
        """季節別統計計算"""
        seasons = {
            'spring': [3, 4, 5],    # 春
            'summer': [6, 7, 8],    # 夏  
            'autumn': [9, 10, 11],  # 秋
            'winter': [12, 1, 2]    # 冬
        }
        
        seasonal_stats = {}
        
        for season_name, months in seasons.items():
            seasonal_original = []
            seasonal_controlled = []
            seasonal_discharge = 0
            
            days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            
            start_idx = 0
            for month in range(1, 13):
                end_idx = start_idx + (days_per_month[month-1] * 96)
                if month in months and month in monthly_results:
                    if end_idx <= len(original_demand):
                        seasonal_original.extend(original_demand[start_idx:end_idx])
                        seasonal_controlled.extend(controlled_demand[start_idx:end_idx])
                        seasonal_discharge += monthly_results[month]['monthly_discharge']
                start_idx = end_idx
            
            if seasonal_original:
                seasonal_stats[season_name] = {
                    'peak_reduction': np.max(seasonal_original) - np.max(seasonal_controlled),
                    'average_reduction': np.mean(seasonal_original) - np.mean(seasonal_controlled),
                    'total_discharge': seasonal_discharge
                }
            else:
                seasonal_stats[season_name] = {
                    'peak_reduction': 0,
                    'average_reduction': 0,
                    'total_discharge': 0
                }
        
        return seasonal_stats
    
    def get_annual_comparison_summary(self):
        """年間比較結果のサマリー取得"""
        if not self.comparison_results:
            return None
        
        summary = []
        for capacity, result in self.comparison_results.items():
            smoothness_improvement = result.get('smoothness_metrics', {}).get('smoothness_improvement', 0)
            max_jump_improvement = result.get('smoothness_metrics', {}).get('max_jump_improvement', 0)
            
            # サイクル制約の目標と実績
            cycle_target = result.get('annual_cycle_target', 0)
            cycle_actual = result.get('annual_discharge', 0)
            
            summary.append({
                '容量(kWh)': f"{capacity:,}",
                '最大出力(kW)': f"{result['max_power']:.0f}",
                '年間ピーク削減(kW)': f"{result['annual_peak_reduction']:.1f}",
                '年間需要幅改善(kW)': f"{result['annual_range_improvement']:.1f}",
                '年間放電量(MWh)': f"{result['annual_discharge']/1000:.1f}",
                'サイクル制約目標(MWh)': f"{cycle_target/1000:.1f}",
                'サイクル制約実績(MWh)': f"{cycle_actual/1000:.1f}",
                'サイクル目標/実績': f"{cycle_target/1000:.1f}/{cycle_actual/1000:.1f}",
                '年間サイクル制約': 'OK' if result['annual_cycle_constraint_satisfied'] else 'NG',
                '春ピーク削減(kW)': f"{result['seasonal_stats']['spring']['peak_reduction']:.1f}",
                '夏ピーク削減(kW)': f"{result['seasonal_stats']['summer']['peak_reduction']:.1f}",
                '秋ピーク削減(kW)': f"{result['seasonal_stats']['autumn']['peak_reduction']:.1f}",
                '冬ピーク削減(kW)': f"{result['seasonal_stats']['winter']['peak_reduction']:.1f}"
            })
        
        return pd.DataFrame(summary)


def create_annual_time_series(start_date=None):
    """年間時系列作成"""
    if start_date is None:
        start_date = datetime(2024, 1, 1, 0, 0, 0)
    
    time_series = []
    current_time = start_date
    
    for i in range(365 * 96):  # 年間35,040ステップ
        time_series.append(current_time)
        current_time += timedelta(minutes=15)
    
    return time_series


# セッション状態の初期化
def initialize_session_state():
    """セッション状態の初期化"""
    if 'annual_demand' not in st.session_state:
        st.session_state.annual_demand = None
    if 'annual_comparison_results' not in st.session_state:
        st.session_state.annual_comparison_results = None
    if 'annual_capacity_list' not in st.session_state:
        st.session_state.annual_capacity_list = []
    if 'annual_comparator' not in st.session_state:
        st.session_state.annual_comparator = None
    if 'show_annual_results' not in st.session_state:
        st.session_state.show_annual_results = False
    if 'simulation_stage' not in st.session_state:
        st.session_state.simulation_stage = 'data_upload'  # 'data_upload', 'simulation_config', 'results'
    # シミュレーション設定用のセッション状態も初期化
    if 'sim_num_capacities' not in st.session_state:
        st.session_state.sim_num_capacities = 2  # デフォルトを2に変更
    if 'sim_power_scaling_method' not in st.session_state:
        st.session_state.sim_power_scaling_method = "capacity_ratio"
    if 'sim_annual_cycle_ratio' not in st.session_state:
        st.session_state.sim_annual_cycle_ratio = 1.0
    if 'sim_annual_cycle_tolerance' not in st.session_state:
        st.session_state.sim_annual_cycle_tolerance = 5000
    if 'sim_monthly_optimization_trials' not in st.session_state:
        st.session_state.sim_monthly_optimization_trials = 20
    if 'sim_use_parallel' not in st.session_state:
        st.session_state.sim_use_parallel = True


def main():
    initialize_session_state()
    
    st.title("年間バッテリー容量別シミュレーション比較システム")
    st.write("複数のバッテリー容量での年間需要平準化効果を比較し、最適容量を検討")
    
    # コアロジック利用可能性の表示
    if not CORE_LOGIC_AVAILABLE:
        st.error("⚠️ コアロジック（battery_core_logic）が利用できません。ダミーデータでの動作となります。")
    
    # ステージ表示
    stage_names = {
        'data_upload': '1. データアップロード', 
        'simulation_config': '2. シミュレーション設定', 
        'results': '3. 結果表示'
    }
    current_stage_name = stage_names.get(st.session_state.simulation_stage, st.session_state.simulation_stage)
    st.subheader(f"現在のステージ: {current_stage_name}")
    
    # ステージ1: 年間データアップロード
    if st.session_state.simulation_stage == 'data_upload' or st.session_state.annual_demand is None:
        show_data_upload_section()
    
    # ステージ2: シミュレーション設定
    elif st.session_state.simulation_stage == 'simulation_config':
        show_simulation_config_section()
    
    # ステージ3: 結果表示
    elif st.session_state.simulation_stage == 'results':
        display_annual_results()


def show_data_upload_section():
    """データアップロードセクション"""
    st.header("1. 年間需要予測データアップロード")
    
    tab1, tab2 = st.tabs(["CSVアップロード", "サンプルデータ生成"])
    
    with tab1:
        st.subheader("年間需要予測CSVアップロード")
        uploaded_file = st.file_uploader(
            "年間需要予測CSV（35,040ステップ推奨、15分間隔）", 
            type=['csv'],
            help="365日×96ステップ/日=35,040ステップの年間データ、または7日以上の短期データ（年間拡張します）"
        )
        
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
                    
                    if st.button("年間データとして読み込み", key="upload_csv_data"):
                        try:
                            demand_values = pd.to_numeric(df[demand_column], errors='coerce').values
                            
                            # データ長の確認
                            data_days = len(demand_values) // 96
                            st.info(f"アップロードデータ: {len(demand_values):,}ステップ（約{data_days}日分）")
                            
                            # 年間データへの拡張処理は後でvalidate_annual_dataで実行
                            st.session_state.annual_demand = demand_values
                            st.session_state.simulation_stage = 'simulation_config'
                            st.success("年間需要データとして設定しました")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"データ読み込みエラー: {e}")
                
                else:
                    st.error("CSVファイルに最低2列（時刻、需要）が必要です")
                    
            except Exception as e:
                st.error(f"ファイル読み込みエラー: {e}")
                import traceback
                st.text(traceback.format_exc())
    
    with tab2:
        st.subheader("サンプル年間データ生成")
        
        col1, col2 = st.columns(2)
        
        with col1:
            base_demand = st.number_input("ベース需要 (kW)", value=5000, min_value=1000, max_value=20000, step=500, key="sample_base_demand")
            seasonal_variation = st.slider("季節変動 (%)", min_value=10, max_value=50, value=20, step=5, key="sample_seasonal")
        
        with col2:
            daily_variation = st.slider("日内変動 (%)", min_value=10, max_value=50, value=30, step=5, key="sample_daily")
            noise_level = st.slider("ランダムノイズ (%)", min_value=1, max_value=10, value=5, step=1, key="sample_noise")
        
        if st.button("サンプル年間データ生成", key="generate_sample_data"):
            with st.spinner("年間データ生成中..."):
                # 年間サンプルデータ生成
                np.random.seed(42)
                
                # 基本パターン
                time_of_year = np.linspace(0, 2*np.pi, 365)
                seasonal_pattern = np.sin(time_of_year - np.pi/2) * (seasonal_variation/100)  # 夏がピーク
                
                annual_demand_sample = []
                
                for day in range(365):
                    # 日内パターン（2つのピーク：朝、夕方）
                    time_of_day = np.linspace(0, 2*np.pi, 96)
                    daily_pattern = (
                        np.sin(time_of_day - np.pi/3) * 0.3 +  # 夕方ピーク
                        np.sin(time_of_day * 2 - np.pi/6) * 0.2  # 朝ピーク
                    ) * (daily_variation/100)
                    
                    # 季節×日内の組み合わせ
                    daily_demand = base_demand * (
                        1 + seasonal_pattern[day] + daily_pattern + 
                        np.random.normal(0, noise_level/100, 96)
                    )
                    
                    # 最小値制限
                    daily_demand = np.maximum(daily_demand, base_demand * 0.3)
                    annual_demand_sample.extend(daily_demand)
                
                st.session_state.annual_demand = np.array(annual_demand_sample)
                st.session_state.simulation_stage = 'simulation_config'
                
                st.success(f"年間サンプルデータ生成完了: {len(annual_demand_sample):,}ステップ")
                
                # 統計表示
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("最小需要", f"{st.session_state.annual_demand.min():.0f} kW")
                with col2:
                    st.metric("平均需要", f"{st.session_state.annual_demand.mean():.0f} kW")
                with col3:
                    st.metric("最大需要", f"{st.session_state.annual_demand.max():.0f} kW")
                with col4:
                    st.metric("需要幅", f"{st.session_state.annual_demand.max() - st.session_state.annual_demand.min():.0f} kW")
                
                st.rerun()


def show_simulation_config_section():
    """シミュレーション設定セクション"""
    st.header("2. 年間容量別シミュレーション設定")
    
    # データ確認表示
    if st.session_state.annual_demand is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("データ長", f"{len(st.session_state.annual_demand):,}ステップ")
        with col2:
            st.metric("平均需要", f"{st.session_state.annual_demand.mean():.0f}kW")
        with col3:
            st.metric("最大需要", f"{st.session_state.annual_demand.max():.0f}kW")
        with col4:
            st.metric("需要幅", f"{st.session_state.annual_demand.max() - st.session_state.annual_demand.min():.0f}kW")
    
    # データ再設定ボタン
    if st.button("📝 データを再設定", key="reset_data"):
        st.session_state.simulation_stage = 'data_upload'
        st.session_state.annual_demand = None
        st.rerun()
    
    with st.expander("年間シミュレーション設定", expanded=True):
        
        # 容量設定（個別入力のみ）
        st.subheader("比較容量設定")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.sim_num_capacities = st.selectbox(
                "比較容量数", 
                [2, 3, 4, 5], 
                index=[2, 3, 4, 5].index(st.session_state.sim_num_capacities) if st.session_state.sim_num_capacities in [2, 3, 4, 5] else 0,
                help="比較したいバッテリー容量の数を選択してください",
                key="num_capacities_select"
            )
        
        with col2:
            st.info("各容量を個別に入力してください")
        
        # 容量入力欄
        cols = st.columns(5)
        
        # セッション状態で個別容量を保存
        if 'sim_individual_capacities' not in st.session_state:
            st.session_state.sim_individual_capacities = [30000, 60000, 120000, 200000, 300000]
        
        capacity_list = []
        
        for i in range(st.session_state.sim_num_capacities):
            with cols[i]:
                st.session_state.sim_individual_capacities[i] = st.number_input(
                    f"容量{i+1} (kWh)", 
                    value=st.session_state.sim_individual_capacities[i],
                    min_value=10000, max_value=500000, step=10000,
                    key=f"manual_capacity_{i}_input"
                )
                capacity_list.append(st.session_state.sim_individual_capacities[i])
        
        # 未使用の列は空白
        for i in range(st.session_state.sim_num_capacities, 5):
            with cols[i]:
                st.text_input(f"容量{i+1} (kWh)", value="未使用", disabled=True, key=f"unused_capacity_{i}")
        
        # 重複チェック
        if len(set(capacity_list)) != len(capacity_list):
            st.warning("⚠️ 重複する容量があります。異なる容量を設定してください。")
        else:
            st.success(f"✅ 選択容量: {', '.join([f'{cap:,}kWh' for cap in capacity_list])}")
        
        # 最大出力設定
        st.subheader("最大出力設定")
        st.session_state.sim_power_scaling_method = st.selectbox(
            "最大出力決定方法",
            ["capacity_ratio", "fixed", "custom"],
            index=["capacity_ratio", "fixed", "custom"].index(st.session_state.sim_power_scaling_method),
            format_func=lambda x: {
                "capacity_ratio": "容量比例（容量÷16）",
                "fixed": "固定値（3000kW）",
                "custom": "カスタム比率（容量÷20）"
            }[x],
            key="power_scaling_select"
        )
        
        # 年間最適化設定
        st.subheader("年間最適化設定")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state.sim_annual_cycle_ratio = st.slider(
                "年間サイクル比率", 
                min_value=0.5, max_value=3.0, value=st.session_state.sim_annual_cycle_ratio, step=0.1,
                help="容量に対する年間サイクル目標の比率",
                key="annual_cycle_ratio_slider"
            )
            
        with col2:
            st.session_state.sim_annual_cycle_tolerance = st.number_input(
                "年間サイクル許容範囲 (kWh)", 
                value=st.session_state.sim_annual_cycle_tolerance, 
                min_value=1000, max_value=50000, step=1000,
                help="年間サイクル制約の許容範囲",
                key="annual_cycle_tolerance_input"
            )
        
        with col3:
            st.session_state.sim_monthly_optimization_trials = st.slider(
                "月別最適化試行回数",
                min_value=10, max_value=50, value=st.session_state.sim_monthly_optimization_trials, step=5,
                help="各月の最適化試行回数（少なくすると高速化）",
                key="monthly_optimization_trials_slider"
            )
        
        # 処理方式設定
        st.subheader("処理設定")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.sim_use_parallel = st.checkbox(
                "並列処理を使用", 
                value=st.session_state.sim_use_parallel,
                help="月別処理を並列実行（高速化、但しメモリ使用量増加）",
                key="use_parallel_checkbox"
            )
        
        with col2:
            # 予想計算時間
            estimated_time = len(capacity_list) * 12 * st.session_state.sim_monthly_optimization_trials * (0.5 if st.session_state.sim_use_parallel else 2)
            st.info(f"""
            **予想処理時間:**
            - 容量数: {len(capacity_list)}
            - 月数: 12ヶ月
            - 並列処理: {'有効' if st.session_state.sim_use_parallel else '無効'}
            
            約 {estimated_time/60:.1f}分 〜 {estimated_time/30:.1f}分
            """)
    
    # 年間シミュレーション実行ボタン
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 年間シミュレーション実行", use_container_width=True, key="run_simulation_button"):
            
            if len(set(capacity_list)) != len(capacity_list):
                st.error("重複する容量があります。設定を確認してください。")
            else:
                # プログレスバーと状態表示
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_text = st.empty()
                
                start_time = time.time()
                
                try:
                    status_text.text("年間シミュレーションシステム初期化中...")
                    progress_bar.progress(5)
                    
                    annual_comparator = AnnualBatteryCapacityComparator()
                    
                    status_text.text("年間需要データ検証・準備中...")
                    progress_bar.progress(10)
                    
                    # 年間シミュレーション実行
                    status_text.text("年間容量別最適化実行中...")
                    time_text.text(f"経過時間: {time.time() - start_time:.0f}秒")
                    
                    annual_results = annual_comparator.run_annual_capacity_comparison(
                        annual_demand=st.session_state.annual_demand,
                        capacity_list=capacity_list,
                        cycle_target_ratio=st.session_state.sim_annual_cycle_ratio,
                        cycle_tolerance=st.session_state.sim_annual_cycle_tolerance,
                        optimization_trials=st.session_state.sim_monthly_optimization_trials,
                        power_scaling_method=st.session_state.sim_power_scaling_method,
                        use_parallel=st.session_state.sim_use_parallel
                    )
                    
                    progress_bar.progress(95)
                    status_text.text("結果分析中...")
                    
                    if annual_results:
                        # セッション状態に保存
                        st.session_state.annual_comparison_results = annual_results
                        st.session_state.annual_capacity_list = capacity_list
                        st.session_state.annual_comparator = annual_comparator
                        st.session_state.simulation_stage = 'results'
                        
                        progress_bar.progress(100)
                        elapsed_time = time.time() - start_time
                        status_text.text(f"年間シミュレーション完了！（処理時間: {elapsed_time/60:.1f}分）")
                        time_text.empty()
                        
                        st.success(f"🎉 {len(annual_results)}種類の容量で年間シミュレーションが完了しました！")
                        
                        time.sleep(2)
                        st.rerun()
                    
                    else:
                        st.error("年間シミュレーションで有効な結果が得られませんでした。")
                
                except Exception as e:
                    st.error(f"年間シミュレーションエラー: {e}")
                    import traceback
                    st.text(traceback.format_exc())
                
                finally:
                    progress_bar.empty()
                    status_text.empty()
                    time_text.empty()


def display_annual_results():
    """年間結果表示"""
    results = st.session_state.annual_comparison_results
    capacity_list = st.session_state.annual_capacity_list
    annual_demand = st.session_state.annual_demand
    annual_comparator = st.session_state.annual_comparator
    
    st.header("3. 年間シミュレーション結果")
    
    # 設定変更ボタン
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("結果が表示されています。設定を変更して再実行することも可能です。")
    with col2:
        if st.button("⚙️ 設定変更", key="change_settings"):
            st.session_state.simulation_stage = 'simulation_config'
            st.rerun()
    
    # サマリーテーブル
    st.subheader("📊 年間効果サマリー")
    summary_df = annual_comparator.get_annual_comparison_summary()
    
    if summary_df is not None:
        st.dataframe(summary_df, use_container_width=True)
        
        # サイクル制約の詳細説明
        with st.expander("📋 サイクル制約について", expanded=False):
            st.write("""
            **サイクル制約とは:**
            - バッテリーの年間使用量（放電量）の目標値
            - 容量 × サイクル比率で計算されます
            - 実績が目標±許容範囲内であれば「OK」、範囲外であれば「NG」
            
            **表示項目:**
            - **サイクル制約目標**: 設定された年間放電目標値
            - **サイクル制約実績**: シミュレーション結果の実際の年間放電量
            - **サイクル目標/実績**: 目標値/実績値の対比表示
            - **年間サイクル制約**: 制約条件を満たしているかの判定
            """)
            
            # 設定値の表示
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("年間サイクル比率", f"{st.session_state.sim_annual_cycle_ratio:.1f}")
            with col2:
                st.metric("サイクル許容範囲", f"±{st.session_state.sim_annual_cycle_tolerance/1000:.1f} MWh")
            with col3:
                cycle_range_percent = (st.session_state.sim_annual_cycle_tolerance / (capacity_list[0] * st.session_state.sim_annual_cycle_ratio)) * 100
                st.metric("許容範囲（%）", f"±{cycle_range_percent:.1f}%")
    
    # タブで結果を整理
    tab1, tab2, tab3, tab4 = st.tabs(["年間需要比較", "季節別分析", "月別詳細", "推奨容量"])
    
    with tab1:
        st.subheader("年間需要カーブ比較")
        
        # データサンプリング（表示用）
        sample_size = min(len(annual_demand), 8760)  # 最大1週間分を表示
        sample_indices = np.linspace(0, len(annual_demand)-1, sample_size, dtype=int)
        
        fig_annual = go.Figure()
        
        try:
            # サンプル時系列作成
            time_series = create_annual_time_series()
            sample_times = [time_series[i] for i in sample_indices]
            
            # 元需要
            fig_annual.add_trace(go.Scatter(
                x=sample_times,
                y=annual_demand[sample_indices],
                name="元需要予測",
                line=dict(color="lightgray", width=1),
                opacity=0.8
            ))
            
            # 各容量の制御後需要
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for i, (capacity, result) in enumerate(results.items()):
                fig_annual.add_trace(go.Scatter(
                    x=sample_times,
                    y=result['demand_after_control'][sample_indices],
                    name=f"容量{capacity:,}kWh制御後",
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
            
            fig_annual.update_layout(
                title="年間需要平準化効果比較（サンプル表示）",
                xaxis_title="日時",
                yaxis_title="需要 (kW)",
                height=600,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig_annual, use_container_width=True)
            
        except Exception as e:
            st.error(f"年間グラフ作成エラー: {e}")
        
        # 年間統計
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("年間ピーク削減効果")
            peak_data = []
            for capacity, result in results.items():
                peak_data.append({
                    'capacity': f"{capacity:,}kWh",
                    'peak_reduction': result['annual_peak_reduction']
                })
            
            fig_peak = px.bar(
                pd.DataFrame(peak_data),
                x='capacity', y='peak_reduction',
                title="容量別年間ピーク削減量"
            )
            st.plotly_chart(fig_peak, use_container_width=True)
        
        with col2:
            st.subheader("年間放電量")
            discharge_data = []
            for capacity, result in results.items():
                discharge_data.append({
                    'capacity': f"{capacity:,}kWh",
                    'discharge': result['annual_discharge'] / 1000  # MWh換算
                })
            
            fig_discharge = px.bar(
                pd.DataFrame(discharge_data),
                x='capacity', y='discharge',
                title="容量別年間放電量 (MWh)"
            )
            st.plotly_chart(fig_discharge, use_container_width=True)
        
        with col3:
            st.subheader("容量効率")
            efficiency_data = []
            for capacity, result in results.items():
                efficiency = result['annual_peak_reduction'] / (capacity / 1000)  # kW削減/MWh容量
                efficiency_data.append({
                    'capacity': f"{capacity:,}kWh",
                    'efficiency': efficiency
                })
            
            fig_efficiency = px.bar(
                pd.DataFrame(efficiency_data),
                x='capacity', y='efficiency',
                title="容量効率 (kW削減/MWh容量)"
            )
            st.plotly_chart(fig_efficiency, use_container_width=True)
    
    with tab2:
        st.subheader("🌸 季節別分析")
        
        # 季節別ピーク削減比較
        seasonal_data = []
        seasons = ['spring', 'summer', 'autumn', 'winter']
        season_names = ['春', '夏', '秋', '冬']
        
        for capacity, result in results.items():
            for season, season_name in zip(seasons, season_names):
                seasonal_data.append({
                    '容量': f"{capacity:,}kWh",
                    '季節': season_name,
                    'ピーク削減': result['seasonal_stats'][season]['peak_reduction'],
                    '平均削減': result['seasonal_stats'][season]['average_reduction'],
                    '放電量': result['seasonal_stats'][season]['total_discharge']
                })
        
        seasonal_df = pd.DataFrame(seasonal_data)
        
        if not seasonal_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_seasonal_peak = px.bar(
                    seasonal_df, x='季節', y='ピーク削減', color='容量',
                    title="季節別ピーク削減効果",
                    barmode='group'
                )
                st.plotly_chart(fig_seasonal_peak, use_container_width=True)
            
            with col2:
                fig_seasonal_avg = px.bar(
                    seasonal_df, x='季節', y='平均削減', color='容量',
                    title="季節別平均削減効果",
                    barmode='group'
                )
                st.plotly_chart(fig_seasonal_avg, use_container_width=True)
            
            # 季節別詳細テーブル
            st.subheader("季節別詳細データ")
            pivot_peak = seasonal_df.pivot(index='容量', columns='季節', values='ピーク削減')
            st.dataframe(pivot_peak, use_container_width=True)
    
    with tab3:
        st.subheader("📅 月別詳細分析")
        
        # 容量選択
        selected_capacity = st.selectbox(
            "詳細表示する容量を選択",
            capacity_list,
            format_func=lambda x: f"{x:,}kWh",
            key="monthly_detail_capacity_select"
        )
        
        if selected_capacity in results and 'monthly_results' in results[selected_capacity]:
            monthly_results = results[selected_capacity]['monthly_results']
            
            # 月別統計テーブル
            monthly_data = []
            month_names = ['1月', '2月', '3月', '4月', '5月', '6月',
                          '7月', '8月', '9月', '10月', '11月', '12月']
            
            for month in range(1, 13):
                if month in monthly_results:
                    result = monthly_results[month]
                    monthly_data.append({
                        '月': month_names[month-1],
                        'ピーク削減(kW)': f"{result['peak_reduction']:.1f}",
                        '需要幅改善(kW)': f"{result['range_improvement']:.1f}",
                        '月間放電(kWh)': f"{result['monthly_discharge']:.0f}",
                        'ピーク制御比率': f"{result['optimized_params'].get('peak_power_ratio', 1.0):.2f}",
                        'ボトム制御比率': f"{result['optimized_params'].get('bottom_power_ratio', 1.0):.2f}"
                    })
            
            monthly_df = pd.DataFrame(monthly_data)
            st.dataframe(monthly_df, use_container_width=True)
            
            # 月別トレンド
            col1, col2 = st.columns(2)
            
            with col1:
                monthly_peak_data = []
                for month in range(1, 13):
                    if month in monthly_results:
                        monthly_peak_data.append({
                            'month': month_names[month-1],
                            'peak_reduction': monthly_results[month]['peak_reduction']
                        })
                
                if monthly_peak_data:
                    fig_monthly_peak = px.line(
                        pd.DataFrame(monthly_peak_data),
                        x='month', y='peak_reduction',
                        title=f"月別ピーク削減トレンド（容量{selected_capacity:,}kWh）"
                    )
                    st.plotly_chart(fig_monthly_peak, use_container_width=True)
            
            with col2:
                monthly_discharge_data = []
                for month in range(1, 13):
                    if month in monthly_results:
                        monthly_discharge_data.append({
                            'month': month_names[month-1],
                            'discharge': monthly_results[month]['monthly_discharge']
                        })
                
                if monthly_discharge_data:
                    fig_monthly_discharge = px.line(
                        pd.DataFrame(monthly_discharge_data),
                        x='month', y='discharge',
                        title=f"月別放電量トレンド（容量{selected_capacity:,}kWh）"
                    )
                    st.plotly_chart(fig_monthly_discharge, use_container_width=True)
    
    with tab4:
        st.subheader("🏆 推奨容量判定")
        
        # 推奨容量の総合評価
        try:
            best_capacity = None
            best_score = -1
            evaluation_results = []
            
            for capacity, result in results.items():
                # 各指標のスコア計算
                peak_score = result.get('annual_peak_reduction', 0) * 0.3
                efficiency_score = (result.get('annual_peak_reduction', 0) / (capacity / 1000)) * 0.25
                cycle_score = 100 if result.get('annual_cycle_constraint_satisfied', False) else 0
                seasonal_balance_score = np.std([
                    result['seasonal_stats']['spring']['peak_reduction'],
                    result['seasonal_stats']['summer']['peak_reduction'],
                    result['seasonal_stats']['autumn']['peak_reduction'],
                    result['seasonal_stats']['winter']['peak_reduction']
                ]) * (-0.2)  # 標準偏差が小さい方が良い
                
                total_score = peak_score + efficiency_score + cycle_score * 0.2 + seasonal_balance_score
                
                evaluation_results.append({
                    '容量(kWh)': f"{capacity:,}",
                    'ピーク削減スコア': f"{peak_score:.1f}",
                    '容量効率スコア': f"{efficiency_score:.1f}",
                    'サイクル制約スコア': f"{cycle_score * 0.2:.1f}",
                    '季節バランススコア': f"{seasonal_balance_score:.1f}",
                    '総合スコア': f"{total_score:.1f}"
                })
                
                if total_score > best_score:
                    best_score = total_score
                    best_capacity = capacity
            
            # 評価結果テーブル
            st.dataframe(pd.DataFrame(evaluation_results), use_container_width=True)
            
            # 推奨容量の詳細
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if best_capacity is not None:
                    st.success(f"""
                    **🥇 総合推奨容量**
                    
                    **{best_capacity:,}kWh**
                    
                    総合スコア: {best_score:.1f}点
                    
                    推奨理由:
                    - 年間通して安定した効果
                    - 容量効率が優秀
                    - サイクル制約を満足
                    """)
            
            with col2:
                # 最大ピーク削減容量
                best_peak_capacity = max(results.keys(), 
                                       key=lambda x: results[x].get('annual_peak_reduction', 0))
                peak_value = results[best_peak_capacity].get('annual_peak_reduction', 0)
                
                st.info(f"""
                **📈 最大ピーク削減**
                
                **{best_peak_capacity:,}kWh**
                
                年間ピーク削減: {peak_value:.1f}kW
                
                特徴:
                - 最大需要の大幅削減
                - 電力契約容量削減効果大
                """)
            
            with col3:
                # 最高効率容量
                best_efficiency_capacity = max(results.keys(), 
                                             key=lambda x: results[x].get('annual_peak_reduction', 0) / (x / 1000))
                efficiency_value = results[best_efficiency_capacity].get('annual_peak_reduction', 0) / (best_efficiency_capacity / 1000)
                
                st.info(f"""
                **⚡ 最高効率**
                
                **{best_efficiency_capacity:,}kWh**
                
                容量効率: {efficiency_value:.2f}kW/MWh
                
                特徴:
                - 投資効率が最も良好
                - コストパフォーマンス重視
                """)
        
        except Exception as e:
            st.error(f"推奨容量判定エラー: {e}")
    
    # ダウンロードセクション
    st.header("4. 結果ダウンロード")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if summary_df is not None:
            try:
                summary_csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="📊 年間サマリーCSV",
                    data=summary_csv,
                    file_name=f"annual_capacity_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_summary_csv"
                )
            except Exception as e:
                st.error(f"サマリーCSV生成エラー: {e}")
    
    with col2:
        if st.button("📅 月別詳細CSV", use_container_width=True, key="download_monthly_detail_btn"):
            try:
                monthly_detail_data = []
                
                for capacity, result in results.items():
                    if 'monthly_results' in result:
                        for month, monthly_result in result['monthly_results'].items():
                            monthly_detail_data.append({
                                '容量(kWh)': capacity,
                                '月': month,
                                'ピーク削減(kW)': monthly_result['peak_reduction'],
                                '需要幅改善(kW)': monthly_result['range_improvement'],
                                '月間放電(kWh)': monthly_result['monthly_discharge'],
                                'ピーク制御比率': monthly_result['optimized_params'].get('peak_power_ratio', 1.0),
                                'ボトム制御比率': monthly_result['optimized_params'].get('bottom_power_ratio', 1.0)
                            })
                
                monthly_detail_df = pd.DataFrame(monthly_detail_data)
                monthly_csv = monthly_detail_df.to_csv(index=False)
                
                st.download_button(
                    label="月別詳細をダウンロード",
                    data=monthly_csv,
                    file_name=f"annual_monthly_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_monthly_detail_csv"
                )
            except Exception as e:
                st.error(f"月別詳細CSV生成エラー: {e}")
    
    with col3:
        if st.button("🌍 季節別統計CSV", use_container_width=True, key="download_seasonal_detail_btn"):
            try:
                seasonal_detail_data = []
                seasons = ['spring', 'summer', 'autumn', 'winter']
                season_names = ['春', '夏', '秋', '冬']
                
                for capacity, result in results.items():
                    for season, season_name in zip(seasons, season_names):
                        seasonal_detail_data.append({
                            '容量(kWh)': capacity,
                            '季節': season_name,
                            'ピーク削減(kW)': result['seasonal_stats'][season]['peak_reduction'],
                            '平均削減(kW)': result['seasonal_stats'][season]['average_reduction'],
                            '放電量(kWh)': result['seasonal_stats'][season]['total_discharge']
                        })
                
                seasonal_detail_df = pd.DataFrame(seasonal_detail_data)
                seasonal_csv = seasonal_detail_df.to_csv(index=False)
                
                st.download_button(
                    label="季節別統計をダウンロード",
                    data=seasonal_csv,
                    file_name=f"annual_seasonal_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_seasonal_detail_csv"
                )
            except Exception as e:
                st.error(f"季節別CSV生成エラー: {e}")


# デバッグ機能
def debug_annual_test():
    """年間データ用デバッグ機能"""
    st.sidebar.header("🔧 年間デバッグモード")
    
    if st.sidebar.button("年間テストデータ生成", key="debug_generate_data"):
        with st.sidebar:
            with st.spinner("年間テストデータ生成中..."):
                # 簡易年間データ生成
                np.random.seed(42)
                base_demand = 5000
                
                annual_test_data = []
                for day in range(365):
                    # 季節変動
                    seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * day / 365 - np.pi/2)
                    
                    # 日内パターン
                    daily_pattern = []
                    for hour in range(24):
                        for quarter in range(4):
                            time_factor = 1 + 0.3 * np.sin(2 * np.pi * (hour + quarter/4) / 24 - np.pi/3)
                            noise = np.random.normal(0, 0.05)
                            demand = base_demand * seasonal_factor * time_factor * (1 + noise)
                            daily_pattern.append(max(demand, base_demand * 0.5))
                    
                    annual_test_data.extend(daily_pattern)
                
                st.session_state.annual_test_demand = np.array(annual_test_data)
                st.sidebar.success(f"年間テストデータ生成完了: {len(annual_test_data):,}ステップ")
    
    if hasattr(st.session_state, 'annual_test_demand'):
        if st.sidebar.button("テストデータを年間データに適用", key="debug_apply_data"):
            st.session_state.annual_demand = st.session_state.annual_test_demand
            st.session_state.simulation_stage = 'simulation_config'
            st.sidebar.success("年間テストデータを適用しました")
            st.rerun()


if __name__ == "__main__":
    # デバッグモードの表示
    if st.sidebar.checkbox("🔧 年間デバッグモード", value=False, key="debug_mode_checkbox"):
        debug_annual_test()
    
    main()
