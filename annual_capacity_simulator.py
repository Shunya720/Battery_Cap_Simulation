"""
年間容量シミュレーション専用アプリケーション（SOC引き継ぎ対応版・デバッグ済み）
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
from plotly.subplots import make_subplots

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
    """年間バッテリー容量別シミュレーション比較クラス（SOC引き継ぎ対応）"""
    
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
            if np.sum(mask) == 0:
                raise ValueError("すべてのデータがNaN値です")
            
            indices = np.arange(len(demand_array))
            demand_array[~mask] = np.interp(indices[~mask], indices[mask], demand_array[mask])
        
        return demand_array[:expected_steps]  # 必要なステップ数に切り取り
    
    def create_daily_batches(self, annual_demand):
        """年間データを日別バッチに分割"""
        daily_batches = []
        
        for day in range(365):
            start_idx = day * 96
            end_idx = start_idx + 96
            if end_idx <= len(annual_demand):
                daily_data = annual_demand[start_idx:end_idx]
                
                # 月を計算
                month = self._get_month_from_day(day)
                
                daily_batches.append({
                    'day': day + 1,
                    'month': month,
                    'day_name': f"{month}月{self._get_day_in_month(day)}日",
                    'data': daily_data,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
            else:
                break
        
        return daily_batches
    
    def _get_month_from_day(self, day_of_year):
        """年間通算日から月を取得"""
        days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        cumulative_days = 0
        for month, days in enumerate(days_per_month):
            cumulative_days += days
            if day_of_year < cumulative_days:
                return month + 1
        return 12
    
    def _get_day_in_month(self, day_of_year):
        """年間通算日から月内日付を取得"""
        days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        cumulative_days = 0
        for month, days in enumerate(days_per_month):
            if day_of_year < cumulative_days + days:
                return day_of_year - cumulative_days + 1
            cumulative_days += days
        return 31
    
    def run_daily_simulation_with_soc(self, daily_data, capacity, max_power, 
                                    daily_cycle_target, cycle_tolerance, optimization_trials,
                                    initial_soc=50.0):
        """SOC引き継ぎ対応日別シミュレーション実行（1日=96ステップ）"""
        try:
            if not CORE_LOGIC_AVAILABLE:
                return self._create_dummy_daily_result_with_soc(daily_data, capacity, max_power, 
                                                              daily_cycle_target, initial_soc)
            
            # バッテリーエンジンの初期化（SOC指定対応）
            engine = BatteryControlEngine(
                battery_capacity=capacity,
                max_power=max_power
            )
            
            # SOC初期値を設定（エンジンがサポートしている場合）
            if hasattr(engine, 'set_initial_soc'):
                engine.set_initial_soc(initial_soc)
            elif hasattr(engine, 'soc_manager') and hasattr(engine.soc_manager, 'current_soc'):
                engine.soc_manager.current_soc = initial_soc
            
            # 日別最適化実行
            if OPTIMIZATION_AVAILABLE:
                optimization_result = engine.run_optimization(
                    daily_data,  # 96ステップの日別データ
                    cycle_target=daily_cycle_target,
                    cycle_tolerance=cycle_tolerance,
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
                daily_data, **optimized_params
            )
            
            # 安全な配列アクセス
            battery_output = control_result.get('battery_output', np.zeros(len(daily_data)))
            demand_after_battery = control_result.get('demand_after_battery', daily_data)
            soc_profile = control_result.get('soc_profile', np.linspace(initial_soc, initial_soc, len(daily_data)))
            
            # 最終SOCを取得
            final_soc = soc_profile[-1] if len(soc_profile) > 0 else initial_soc
            
            return {
                'optimized_params': optimized_params,
                'battery_output': battery_output,
                'soc_profile': soc_profile,
                'demand_after_control': demand_after_battery,
                'control_info': control_result.get('control_info', {}),
                'daily_discharge': -np.sum(battery_output[battery_output < 0]) if len(battery_output) > 0 else 0,
                'peak_reduction': np.max(daily_data) - np.max(demand_after_battery) if len(demand_after_battery) > 0 else 0,
                'range_improvement': (np.max(daily_data) - np.min(daily_data)) - 
                                   (np.max(demand_after_battery) - np.min(demand_after_battery)) if len(demand_after_battery) > 0 else 0,
                'initial_soc': initial_soc,
                'final_soc': final_soc
            }
            
        except Exception as e:
            st.warning(f"日別シミュレーションでエラー: {e}")
            return self._create_dummy_daily_result_with_soc(daily_data, capacity, max_power, 
                                                          daily_cycle_target, initial_soc)
    
    def _create_dummy_daily_result_with_soc(self, daily_data, capacity, max_power, daily_cycle_target, initial_soc):
        """SOC引き継ぎ対応ダミー日別結果生成"""
        np.random.seed(42)  # 再現性のため
        battery_output = np.random.uniform(-max_power/2, max_power/2, len(daily_data))
        demand_after_control = daily_data + battery_output
        
        # SOCプロファイル生成（初期SOCから開始）
        soc_changes = np.cumsum(battery_output) / (capacity / 100)  # 容量に対する変化率
        soc_profile = np.clip(initial_soc + soc_changes, 10, 90)  # 10-90%の範囲
        final_soc = soc_profile[-1]
        
        return {
            'optimized_params': {
                'peak_percentile': 80, 
                'bottom_percentile': 20, 
                'peak_power_ratio': 1.0, 
                'bottom_power_ratio': 1.0, 
                'flattening_power_ratio': 0.3
            },
            'battery_output': battery_output,
            'soc_profile': soc_profile,
            'demand_after_control': demand_after_control,
            'control_info': {},
            'daily_discharge': np.sum(np.abs(battery_output[battery_output < 0])),
            'peak_reduction': np.max(daily_data) - np.max(demand_after_control),
            'range_improvement': 100.0,
            'initial_soc': initial_soc,
            'final_soc': final_soc
        }
    
    def run_annual_capacity_comparison(self, annual_demand, capacity_list, 
                                     cycle_target_ratio=365.0, cycle_tolerance=5000,
                                     optimization_trials=20, power_scaling_method='capacity_ratio',
                                     use_parallel=True, initial_soc=50.0):
        """SOC引き継ぎ対応年間容量別シミュレーション実行"""
        
        # データ検証
        validated_demand = self.validate_annual_data(annual_demand)
        
        # 日別バッチ作成
        daily_batches = self.create_daily_batches(validated_demand)
        st.info(f"年間データを{len(daily_batches)}日のバッチに分割しました（SOC引き継ぎあり）")
        
        self.comparison_results = {}
        self.daily_results = {}
        
        total_operations = len(capacity_list) * len(daily_batches)
        completed_operations = 0
        
        # プログレスバーの初期化
        progress_bar = st.progress(0)
        
        for i, capacity in enumerate(capacity_list):
            try:
                st.write(f"容量 {capacity:,}kWh の年間最適化開始 ({i+1}/{len(capacity_list)}) - SOC引き継ぎあり")
                
                # 容量に応じた設定
                annual_cycle_target = int(capacity * cycle_target_ratio)
                daily_cycle_target = annual_cycle_target / 365  # 日別サイクル目標
                daily_cycle_tolerance = cycle_tolerance / 365   # 日別許容範囲
                
                # 最大出力設定
                if power_scaling_method == 'capacity_ratio':
                    max_power = capacity / 16
                elif power_scaling_method == 'custom':
                    max_power = capacity / 20
                elif power_scaling_method == 'individual':
                    # 個別入力から対応する出力を取得
                    if hasattr(st.session_state, 'sim_individual_powers') and i < len(st.session_state.sim_individual_powers):
                        max_power = st.session_state.sim_individual_powers[i]
                    else:
                        max_power = capacity / 16  # フォールバック
                else:
                    max_power = capacity / 16
                
                # 年間シミュレーション用変数
                daily_results_for_capacity = {}
                monthly_summary = {}
                annual_battery_output = []
                annual_soc_profile = []
                annual_demand_after_control = []
                
                # SOC引き継ぎのための変数
                current_soc = initial_soc  # 年間開始時のSOC
                soc_history = [initial_soc]  # SOC履歴
                
                # 並列処理は使用しない（SOC引き継ぎのため逐次処理必須）
                st.info("SOC引き継ぎのため逐次処理で実行します")
                
                # 日別シミュレーション（逐次処理・SOC引き継ぎ）
                for day_idx, batch in enumerate(daily_batches):
                    try:
                        # SOC引き継ぎありで日別シミュレーション実行
                        result = self.run_daily_simulation_with_soc(
                            batch['data'], capacity, max_power,
                            daily_cycle_target, daily_cycle_tolerance, optimization_trials,
                            initial_soc=current_soc  # 前日の最終SOCを引き継ぎ
                        )
                        
                        daily_results_for_capacity[batch['day']] = result
                        completed_operations += 1
                        
                        # 翌日のためにSOCを更新
                        current_soc = result['final_soc']
                        soc_history.append(current_soc)
                        
                        # プログレス更新（5日毎に表示）
                        if completed_operations % 5 == 0:
                            progress = completed_operations / total_operations
                            progress_bar.progress(progress)
                            st.write(f"  - {batch['day_name']} 完了 ({len(daily_results_for_capacity)}/365日), SOC: {current_soc:.1f}%")
                        
                    except Exception as e:
                        st.error(f"{batch['day_name']}の処理でエラー: {e}")
                        # エラー時もSOCは前の値を維持
                        continue
                
                # 日別結果を年間結果に統合
                for day in sorted(daily_results_for_capacity.keys()):
                    result = daily_results_for_capacity[day]
                    annual_battery_output.extend(result['battery_output'])
                    annual_soc_profile.extend(result['soc_profile'])
                    annual_demand_after_control.extend(result['demand_after_control'])
                
                # 月別サマリー作成
                for month in range(1, 13):
                    month_days = [day for day in daily_results_for_capacity.keys() 
                                if len(daily_batches) > day-1 and daily_batches[day-1]['month'] == month]
                    
                    if month_days:
                        monthly_discharge = sum(daily_results_for_capacity[day]['daily_discharge'] 
                                              for day in month_days)
                        monthly_peak_reduction = np.mean([daily_results_for_capacity[day]['peak_reduction'] 
                                                        for day in month_days])
                        # 月末SOC
                        month_end_soc = daily_results_for_capacity[max(month_days)]['final_soc']
                        
                        monthly_summary[month] = {
                            'monthly_discharge': monthly_discharge,
                            'peak_reduction': monthly_peak_reduction,
                            'days_count': len(month_days),
                            'month_end_soc': month_end_soc
                        }
                
                # 年間統計計算
                annual_battery_output = np.array(annual_battery_output)
                annual_demand_after_control = np.array(annual_demand_after_control)
                annual_soc_profile = np.array(annual_soc_profile)
                
                # SOC統計
                soc_stats = {
                    'initial_soc': initial_soc,
                    'final_soc': current_soc,
                    'soc_range': np.max(annual_soc_profile) - np.min(annual_soc_profile),
                    'soc_average': np.mean(annual_soc_profile),
                    'soc_daily_history': soc_history
                }
                
                # 年間滑らかさ指標
                sample_size = min(len(validated_demand), 10000)
                sample_indices = np.random.choice(len(validated_demand), sample_size, replace=False)
                sample_original = validated_demand[sample_indices]
                sample_controlled = annual_demand_after_control[sample_indices] if len(annual_demand_after_control) > 0 else sample_original
                
                smoothness_metrics = {
                    'smoothness_improvement': np.std(np.diff(sample_original)) - np.std(np.diff(sample_controlled)),
                    'max_jump_improvement': np.max(np.abs(np.diff(sample_original))) - np.max(np.abs(np.diff(sample_controlled)))
                }
                
                # 年間結果保存
                self.comparison_results[capacity] = {
                    'capacity': capacity,
                    'max_power': max_power,
                    'annual_cycle_target': annual_cycle_target,
                    'daily_cycle_target': daily_cycle_target,
                    'battery_output': annual_battery_output,
                    'soc_profile': annual_soc_profile,
                    'demand_after_control': annual_demand_after_control,
                    'smoothness_metrics': smoothness_metrics,
                    'annual_peak_reduction': (np.max(validated_demand) - np.max(annual_demand_after_control)) if len(annual_demand_after_control) > 0 else 0,
                    'annual_range_improvement': ((np.max(validated_demand) - np.min(validated_demand)) - 
                                              (np.max(annual_demand_after_control) - np.min(annual_demand_after_control))) if len(annual_demand_after_control) > 0 else 0,
                    'annual_discharge': -np.sum(annual_battery_output[annual_battery_output < 0]) if len(annual_battery_output) > 0 else 0,
                    'annual_cycle_constraint_satisfied': abs(-np.sum(annual_battery_output[annual_battery_output < 0]) - annual_cycle_target) <= cycle_tolerance if len(annual_battery_output) > 0 else False,
                    'daily_results': daily_results_for_capacity,
                    'monthly_summary': monthly_summary,
                    'seasonal_stats': self._calculate_seasonal_stats(validated_demand, annual_demand_after_control, monthly_summary),
                    'soc_stats': soc_stats  # SOC統計を追加
                }
                
                self.daily_results[capacity] = daily_results_for_capacity
                
                st.success(f"容量 {capacity:,}kWh の年間最適化完了（{len(daily_results_for_capacity)}日処理, 初期SOC: {initial_soc:.1f}% → 最終SOC: {current_soc:.1f}%）")
                
                # メモリクリーンアップ
                gc.collect()
                
            except Exception as e:
                st.error(f"容量 {capacity:,}kWh でエラー: {e}")
                import traceback
                st.text(traceback.format_exc())
                continue
        
        # プログレスバーの終了
        progress_bar.progress(1.0)
        
        return self.comparison_results
    
    def _calculate_seasonal_stats(self, original_demand, controlled_demand, monthly_summary):
        """季節別統計計算（月別サマリーから算出）"""
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
                if month in months and month in monthly_summary:
                    if end_idx <= len(original_demand) and end_idx <= len(controlled_demand):
                        seasonal_original.extend(original_demand[start_idx:end_idx])
                        seasonal_controlled.extend(controlled_demand[start_idx:end_idx])
                        seasonal_discharge += monthly_summary[month]['monthly_discharge']
                start_idx = end_idx
            
            if seasonal_original and seasonal_controlled:
                seasonal_stats[season_name] = {
                    'peak_reduction': max(0, np.max(seasonal_original) - np.max(seasonal_controlled)),
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
        """年間比較結果のサマリー取得（SOC情報含む）"""
        if not self.comparison_results:
            return None
        
        summary = []
        for capacity, result in self.comparison_results.items():
            # サイクル制約の目標と実績
            cycle_target = result.get('annual_cycle_target', 0)
            cycle_actual = result.get('annual_discharge', 0)
            
            # サイクル数計算（放電量 ÷ 容量）
            target_cycles = cycle_target / capacity if capacity > 0 else 0
            actual_cycles = cycle_actual / capacity if capacity > 0 else 0
            
            # SOC統計
            soc_stats = result.get('soc_stats', {})
            
            summary.append({
                '容量(kWh)': f"{capacity:,}",
                '最大出力(kW)': f"{result['max_power']:.0f}",
                '年間ピーク削減(kW)': f"{result['annual_peak_reduction']:.1f}",
                '年間需要幅改善(kW)': f"{result['annual_range_improvement']:.1f}",
                '年間放電量(MWh)': f"{result['annual_discharge']/1000:.1f}",
                'サイクル制約目標(MWh)': f"{cycle_target/1000:.1f}",
                'サイクル制約実績(MWh)': f"{cycle_actual/1000:.1f}",
                'サイクル目標/実績': f"{cycle_target/1000:.1f}/{cycle_actual/1000:.1f}",
                'サイクル数目標': f"{target_cycles:.0f}回",
                'サイクル数実績': f"{actual_cycles:.0f}回",
                'サイクル数達成率(%)': f"{(actual_cycles/target_cycles*100):.1f}" if target_cycles > 0 else "0.0",
                '年間サイクル制約': 'OK' if result['annual_cycle_constraint_satisfied'] else 'NG',
                '初期SOC(%)': f"{soc_stats.get('initial_soc', 50):.1f}",
                '最終SOC(%)': f"{soc_stats.get('final_soc', 50):.1f}",
                'SOC変化': f"{soc_stats.get('final_soc', 50) - soc_stats.get('initial_soc', 50):+.1f}",
                'SOC範囲(%)': f"{soc_stats.get('soc_range', 0):.1f}",
                '平均SOC(%)': f"{soc_stats.get('soc_average', 50):.1f}",
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
    session_vars = {
        'annual_demand': None,
        'annual_comparison_results': None,
        'annual_capacity_list': [],
        'annual_comparator': None,
        'show_annual_results': False,
        'simulation_stage': 'data_upload',
        'sim_num_capacities': 2,
        'sim_power_scaling_method': "capacity_ratio",
        'sim_annual_cycle_ratio': 365.0,
        'sim_annual_cycle_tolerance': 5000,
        'sim_monthly_optimization_trials': 20,
        'sim_use_parallel': False,  # SOC引き継ぎのため並列処理は無効
        'sim_individual_capacities': [30000, 60000, 120000, 200000, 300000],
        'sim_individual_powers': [],
        'sim_initial_soc': 50.0  # 初期SOC設定を追加
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value


def main():
    """メイン関数"""
    initialize_session_state()
    
    st.title("年間バッテリー容量別シミュレーション比較システム（SOC引き継ぎ対応）")
    st.write("複数のバッテリー容量での年間需要平準化効果を比較し、最適容量を検討（SOC引き継ぎあり）")
    
    # コアロジック利用可能性の表示
    if not CORE_LOGIC_AVAILABLE:
        st.error("⚠️ コアロジック（battery_core_logic）が利用できません。ダミーデータでの動作となります。")
    
    # SOC引き継ぎ対応の表示
    st.info("🔋 SOC引き継ぎ機能が有効です：前日の最終SOCが翌日の初期SOCとして使用されます")
    
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
    """シミュレーション設定セクション（SOC引き継ぎ対応）"""
    st.header("2. 年間容量別シミュレーション設定（SOC引き継ぎ対応）")
    
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
    
    # SOC引き継ぎ設定セクション
    st.subheader("🔋 SOC引き継ぎ設定")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.sim_initial_soc = st.slider(
            "年間開始時の初期SOC (%)",
            min_value=10.0, max_value=90.0, value=st.session_state.sim_initial_soc, step=5.0,
            help="年間シミュレーション開始時のバッテリーSOC（各日は前日の最終SOCから開始）",
            key="initial_soc_slider"
        )
    
    with col2:
        st.info(f"""
        **SOC引き継ぎ機能:**
        - 1日目: {st.session_state.sim_initial_soc:.0f}%からスタート
        - 2日目以降: 前日の最終SOCから開始
        - より現実的なバッテリー運用をシミュレーション
        """)
    
    # 年間シミュレーション設定セクション
    st.subheader("年間シミュレーション設定")
    
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
    if len(st.session_state.sim_individual_capacities) < 5:
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
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.sim_power_scaling_method = st.selectbox(
            "最大出力決定方法",
            ["capacity_ratio", "custom", "individual"],
            index=["capacity_ratio", "custom", "individual"].index(st.session_state.sim_power_scaling_method) if st.session_state.sim_power_scaling_method in ["capacity_ratio", "custom", "individual"] else 0,
            format_func=lambda x: {
                "capacity_ratio": "容量比例（容量÷16）",
                "custom": "カスタム比率（容量÷20）",
                "individual": "個別入力"
            }[x],
            key="power_scaling_select"
        )
    
    with col2:
        if st.session_state.sim_power_scaling_method == "capacity_ratio":
            st.info("各容量を16で割った値を最大出力とします")
        elif st.session_state.sim_power_scaling_method == "custom":
            st.info("各容量を20で割った値を最大出力とします")
        elif st.session_state.sim_power_scaling_method == "individual":
            st.info("各容量に対して個別に最大出力を設定します")
    
    # 個別入力の場合の設定欄
    if st.session_state.sim_power_scaling_method == "individual":
        st.write("**各容量の最大出力を個別設定:**")
        
        # セッション状態で個別最大出力を保存
        if len(st.session_state.sim_individual_powers) < st.session_state.sim_num_capacities:
            # デフォルト値として容量÷16を設定
            st.session_state.sim_individual_powers = [
                cap // 16 for cap in st.session_state.sim_individual_capacities[:st.session_state.sim_num_capacities]
            ]
        
        power_cols = st.columns(5)
        
        for i in range(st.session_state.sim_num_capacities):
            with power_cols[i]:
                # 対応する容量を取得
                capacity = st.session_state.sim_individual_capacities[i]
                
                # デフォルト値を容量÷16に設定（まだ設定されていない場合）
                if i >= len(st.session_state.sim_individual_powers):
                    st.session_state.sim_individual_powers.append(capacity // 16)
                
                st.session_state.sim_individual_powers[i] = st.number_input(
                    f"出力{i+1} (kW)\n容量: {capacity:,}kWh",
                    value=st.session_state.sim_individual_powers[i],
                    min_value=100, max_value=50000, step=100,
                    key=f"individual_power_{i}_input",
                    help=f"容量{capacity:,}kWh に対する最大出力"
                )
        
        # 未使用の列は空白
        for i in range(st.session_state.sim_num_capacities, 5):
            with power_cols[i]:
                st.text_input(f"出力{i+1} (kW)", value="未使用", disabled=True, key=f"unused_power_{i}")
        
        # 出力/容量比の表示
        st.write("**出力/容量比 確認:**")
        ratio_data = []
        for i in range(st.session_state.sim_num_capacities):
            capacity = st.session_state.sim_individual_capacities[i]
            power = st.session_state.sim_individual_powers[i]
            ratio = capacity / power if power > 0 else 0
            ratio_data.append({
                f'容量{i+1}': f"{capacity:,}kWh",
                f'出力{i+1}': f"{power:,}kW",
                f'比率{i+1}': f"1:{ratio:.1f}" if ratio > 0 else "設定エラー"
            })
        
        ratio_df = pd.DataFrame(ratio_data)
        st.dataframe(ratio_df, use_container_width=True)
    
    # 年間最適化設定
    st.subheader("年間最適化設定")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.session_state.sim_annual_cycle_ratio = st.slider(
            "年間サイクル数", 
            min_value=300.0, max_value=400.0, value=st.session_state.sim_annual_cycle_ratio, step=5.0,
            help="年間のバッテリーサイクル数（350-365回推奨）",
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
            "日別最適化試行回数",
            min_value=5, max_value=30, value=min(st.session_state.sim_monthly_optimization_trials, 15), step=2,
            help="各日の最適化試行回数（少なくすると高速化）",
            key="daily_optimization_trials_slider"
        )
    
    # 処理方式設定
    st.subheader("処理設定")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # SOC引き継ぎのため並列処理は無効
        st.session_state.sim_use_parallel = False
        st.info("🔒 SOC引き継ぎのため並列処理は無効化されています")
        st.checkbox(
            "並列処理を使用", 
            value=False,
            disabled=True,
            help="SOC引き継ぎ機能のため、逐次処理で実行されます",
            key="use_parallel_checkbox_disabled"
        )
    
    with col2:
        # 予想計算時間（逐次処理）
        estimated_time = len(capacity_list) * 365 * st.session_state.sim_monthly_optimization_trials * 0.3
        st.info(f"""
        **予想処理時間（SOC引き継ぎ・逐次処理）:**
        - 容量数: {len(capacity_list)}
        - 日数: 365日
        - 処理方式: 逐次（SOC引き継ぎのため）
        
        約 {estimated_time/60:.1f}分 〜 {estimated_time/20:.1f}分
        
        ※SOC引き継ぎにより正確な年間運用をシミュレーション
        """)
    
    # SOC引き継ぎ処理の説明
    with st.expander("🔋 SOC引き継ぎ処理について", expanded=False):
        st.write("""
        **SOC引き継ぎ処理の特徴:**
        - **現実的なバッテリー運用**: 前日の最終SOCが翌日の初期SOCとして使用
        - **エネルギー収支の整合性**: 日をまたぐ充放電計画が可能
        - **年間通しての最適化**: 季節変動やSOC推移を考慮した運用
        
        **処理の流れ:**
        - 1日目: 設定した初期SOC（{st.session_state.sim_initial_soc:.0f}%）からスタート
        - 2日目以降: 前日の最終SOCから開始
        - 各日で独立した最適化を実行
        - SOC履歴を記録し、年間推移を分析
        
        **従来の日別独立処理との違い:**
        - ❌ 従来: 毎日同じSOCからリセット → 非現実的
        - ✅ SOC引き継ぎ: 前日の状態を継承 → 現実的
        
        **注意事項:**
        - 逐次処理のため計算時間が増加
        - より正確だが、処理負荷が高い
        - SOC履歴により詳細な分析が可能
        """)
    
    # 年間シミュレーション実行ボタン
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 年間シミュレーション実行（SOC引き継ぎ）", use_container_width=True, key="run_simulation_button"):
            
            if len(set(capacity_list)) != len(capacity_list):
                st.error("重複する容量があります。設定を確認してください。")
            else:
                # プログレスバーと状態表示
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_text = st.empty()
                
                start_time = time.time()
                
                try:
                    status_text.text("年間シミュレーションシステム初期化中（SOC引き継ぎ対応）...")
                    progress_bar.progress(5)
                    
                    annual_comparator = AnnualBatteryCapacityComparator()
                    
                    status_text.text("年間需要データ検証・準備中...")
                    progress_bar.progress(10)
                    
                    # 年間シミュレーション実行（SOC引き継ぎあり）
                    status_text.text("年間容量別最適化実行中（SOC引き継ぎ処理）...")
                    time_text.text(f"経過時間: {time.time() - start_time:.0f}秒")
                    
                    annual_results = annual_comparator.run_annual_capacity_comparison(
                        annual_demand=st.session_state.annual_demand,
                        capacity_list=capacity_list,
                        cycle_target_ratio=st.session_state.sim_annual_cycle_ratio,
                        cycle_tolerance=st.session_state.sim_annual_cycle_tolerance,
                        optimization_trials=st.session_state.sim_monthly_optimization_trials,
                        power_scaling_method=st.session_state.sim_power_scaling_method,
                        use_parallel=False,  # SOC引き継ぎのため強制的に無効
                        initial_soc=st.session_state.sim_initial_soc
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
                        
                        st.success(f"🎉 {len(annual_results)}種類の容量で年間シミュレーションが完了しました（SOC引き継ぎあり）！")
                        
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
    """年間結果表示（SOC引き継ぎ対応）"""
    # 個別チェックでNumPy配列エラーを回避
    if (not st.session_state.annual_comparison_results or
        not st.session_state.annual_capacity_list or
        st.session_state.annual_demand is None or
        len(st.session_state.annual_demand) == 0 or
        not st.session_state.annual_comparator):
        
        st.error("結果データが不完全です。シミュレーションを再実行してください。")
        if st.button("設定に戻る"):
            st.session_state.simulation_stage = 'simulation_config'
            st.rerun()
        return
    
    results = st.session_state.annual_comparison_results
    capacity_list = st.session_state.annual_capacity_list
    annual_demand = st.session_state.annual_demand
    annual_comparator = st.session_state.annual_comparator
    
    st.header("3. 年間シミュレーション結果（SOC引き継ぎ対応）")
    
    # 設定変更ボタン
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("結果が表示されています。設定を変更して再実行することも可能です。")
    with col2:
        if st.button("⚙️ 設定変更", key="change_settings"):
            st.session_state.simulation_stage = 'simulation_config'
            st.rerun()
    
    # SOC引き継ぎ結果の表示
    st.subheader("🔋 SOC引き継ぎ結果概要")
    
    soc_summary_data = []
    for capacity, result in results.items():
        soc_stats = result.get('soc_stats', {})
        soc_summary_data.append({
            '容量(kWh)': f"{capacity:,}",
            '初期SOC(%)': f"{soc_stats.get('initial_soc', 50):.1f}",
            '最終SOC(%)': f"{soc_stats.get('final_soc', 50):.1f}",
            'SOC変化': f"{soc_stats.get('final_soc', 50) - soc_stats.get('initial_soc', 50):+.1f}",
            'SOC範囲(%)': f"{soc_stats.get('soc_range', 0):.1f}",
            '平均SOC(%)': f"{soc_stats.get('soc_average', 50):.1f}"
        })
    
    soc_summary_df = pd.DataFrame(soc_summary_data)
    st.dataframe(soc_summary_df, use_container_width=True)
    
    # サマリーテーブル
    st.subheader("📊 年間効果サマリー（SOC引き継ぎ対応）")
    summary_df = annual_comparator.get_annual_comparison_summary()
    
    if summary_df is not None:
        st.dataframe(summary_df, use_container_width=True)
        
        # SOC引き継ぎとサイクル制約の詳細説明
        st.subheader("SOC引き継ぎとサイクル制約の詳細説明")
        with st.expander("🔋 SOC引き継ぎ・サイクル制約について", expanded=False):
            st.write("""
            **SOC引き継ぎ機能:**
            - **現実的なバッテリー運用**: 前日の最終SOCが翌日の初期SOCとして使用
            - **エネルギー収支の整合性**: 日をまたぐ充放電計画が可能
            - **年間通しての最適化**: 季節変動やSOC推移を考慮した運用
            
            **サイクル制約とは:**
            - バッテリーの年間使用量（放電量）の目標値
            - 年間サイクル数 × バッテリー容量で計算されます
            - 実績が目標±許容範囲内であれば「OK」、範囲外であれば「NG」
            
            **サイクル数の計算方法:**
            - 1サイクル = 設定容量と同量の放電
            - サイクル数 = 年間放電量 ÷ バッテリー容量
            - 年間目標: 350-365サイクル（ほぼ毎日1回の使用）
            - 例：容量50MWh、年間放電量18,250MWh → 365.0サイクル
            
            **表示項目:**
            - **初期/最終SOC**: 年間開始時と終了時のSOC状態
            - **SOC変化**: 年間を通したSOCの変化量
            - **SOC範囲**: 年間で最大・最小SOCの差
            - **平均SOC**: 年間平均SOC
            - **サイクル制約目標/実績**: 設定目標値とシミュレーション結果
            - **年間サイクル制約**: 制約条件を満たしているかの判定
            """)
            
            # 設定値の表示
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("年間サイクル数", f"{st.session_state.sim_annual_cycle_ratio:.0f}回")
            with col2:
                st.metric("サイクル許容範囲", f"±{st.session_state.sim_annual_cycle_tolerance/1000:.1f} MWh")
            with col3:
                st.metric("初期SOC設定", f"{st.session_state.sim_initial_soc:.0f}%")
            with col4:
                st.metric("1日あたり", f"{st.session_state.sim_annual_cycle_ratio/365:.2f}回")
            
            # サイクル数の例
            st.write("**サイクル数の例:**")
            example_data = []
            for capacity in capacity_list:
                target_discharge = capacity * st.session_state.sim_annual_cycle_ratio
                tolerance_cycles = st.session_state.sim_annual_cycle_tolerance / capacity
                example_data.append({
                    '容量(kWh)': f"{capacity:,}",
                    '目標放電量(MWh)': f"{target_discharge/1000:.1f}",
                    '目標サイクル数': f"{st.session_state.sim_annual_cycle_ratio:.0f}回",
                    '許容範囲(±サイクル)': f"±{tolerance_cycles:.1f}回",
                    '許容範囲(MWh)': f"±{st.session_state.sim_annual_cycle_tolerance/1000:.1f}"
                })
            
            example_df = pd.DataFrame(example_data)
            st.dataframe(example_df, use_container_width=True)
    
    # タブで結果を整理
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["年間需要比較", "SOC推移分析", "季節別分析", "月別詳細", "推奨容量"])
    
    with tab1:
        (results, capacity_list, annual_demand)
    
    with tab2:
        show_soc_analysis(results, capacity_list)
    
    with tab3:
        show_seasonal_analysis(results)
    
    with tab4:
        show_monthly_detail_analysis(results, capacity_list, annual_comparator)
    
    with tab5:
        show_capacity_recommendation(results, capacity_list)
    
    # ダウンロードセクション
    show_download_section(summary_df, results, annual_comparator)


def show_annual_demand_comparison(results, capacity_list, annual_demand):
    """年間需要比較タブの内容（SOC引き継ぎ対応）"""
    st.subheader("年間需要カーブ比較")
    
    # === 変更1: グラフ表示期間選択部分 ===
    col1, col2, col3, col4 = st.columns(4)  # 3列から4列に変更
    with col1:
        graph_period = st.selectbox(
            "表示期間",
            ["1週間", "1ヶ月", "3ヶ月", "全年間（サンプル）"],
            index=0,
            key="graph_period_select"
        )
    
    with col2:
        if graph_period in ["1週間", "1ヶ月", "3ヶ月"]:
            start_month = st.selectbox(
                "開始月",
                list(range(1, 13)),
                index=0,
                format_func=lambda x: f"{x}月",
                key="start_month_select"
            )
        else:
            start_month = 1
    
    with col3:
        selected_capacity_graph = st.selectbox(
            "表示する容量",
            capacity_list,
            index=0,
            format_func=lambda x: f"{x:,}kWh",
            key="selected_capacity_graph"
        )
    
    # === 追加: SOC表示オプション ===
    with col4:
        show_soc = st.checkbox(
            "SOC推移を表示",
            value=True,
            help="需要グラフと一緒にSOC推移を表示",
            key="show_soc_checkbox"
        )

    
    # データ期間とサンプリング設定
    try:
        if graph_period == "1週間":
            # 指定月の第1週
            days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            start_idx = sum(days_per_month[:start_month-1]) * 96
            end_idx = start_idx + (7 * 96)  # 1週間分
            period_title = f"{start_month}月第1週"
        elif graph_period == "1ヶ月":
            # 指定月全体
            days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            start_idx = sum(days_per_month[:start_month-1]) * 96
            end_idx = start_idx + (days_per_month[start_month-1] * 96)
            period_title = f"{start_month}月"
        elif graph_period == "3ヶ月":
            # 指定月から3ヶ月
            days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            start_idx = sum(days_per_month[:start_month-1]) * 96
            end_month = min(start_month + 2, 12)
            end_idx = sum(days_per_month[:end_month]) * 96
            period_title = f"{start_month}月〜{end_month}月"
        else:
            # 全年間（サンプル表示）
            start_idx = 0
            end_idx = len(annual_demand)
            # サンプリング（表示負荷軽減のため）
            sample_size = min(8760, end_idx - start_idx)  # 最大1週間分相当
            sample_indices = np.linspace(start_idx, end_idx-1, sample_size, dtype=int)
            period_title = "全年間（サンプル表示）"
        
        # データ抽出
        if graph_period != "全年間（サンプル）":
            # 指定期間のデータを抽出
            end_idx = min(end_idx, len(annual_demand))
            period_demand = annual_demand[start_idx:end_idx]
            
            if selected_capacity_graph in results:
                period_controlled = results[selected_capacity_graph]['demand_after_control'][start_idx:end_idx]
                period_soc = results[selected_capacity_graph]['soc_profile'][start_idx:end_idx]  # SOC追加
            else:
                period_controlled = period_demand  # フォールバック
                period_soc = np.full(len(period_demand), 50)  # デフォルトSOC
            
            # 時系列作成
            time_series = create_annual_time_series()
            period_times = time_series[start_idx:end_idx]
        else:
            # 全年間サンプル表示
            period_demand = annual_demand[sample_indices]
            
            if selected_capacity_graph in results:
                period_controlled = results[selected_capacity_graph]['demand_after_control'][sample_indices]
                period_soc = results[selected_capacity_graph]['soc_profile'][sample_indices]  # SOC追加
            else:
                period_controlled = period_demand
                period_soc = np.full(len(period_demand), 50)  # デフォルトSOC
            
            time_series = create_annual_time_series()
            period_times = [time_series[i] for i in sample_indices]
        
        # 需要比較グラフ
        if show_soc:
            # サブプロット作成（需要とSOCを縦に並べる）
            fig_demand = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=[
                    f"需要カーブ比較 - {period_title}（SOC引き継ぎ対応）",
                    f"SOC推移 - 容量{selected_capacity_graph:,}kWh"
                ],
                specs=[[{"secondary_y": False}],
                       [{"secondary_y": False}]]
            )
            
            # 上段：需要データ
            fig_demand.add_trace(
                go.Scatter(
                    x=period_times,
                    y=period_demand,
                    name="元需要予測",
                    line=dict(color="lightblue", width=2),
                    opacity=0.8
                ),
                row=1, col=1
            )
            
            fig_demand.add_trace(
                go.Scatter(
                    x=period_times,
                    y=period_controlled,
                    name=f"電池制御後（{selected_capacity_graph:,}kWh）",
                    line=dict(color="red", width=2)
                ),
                row=1, col=1
            )
            
            # 下段：SOCデータ
            fig_demand.add_trace(
                go.Scatter(
                    x=period_times,
                    y=period_soc,
                    name="SOC推移",
                    line=dict(color="green", width=2),
                    fill='tonexty',
                    fillcolor='rgba(0,255,0,0.1)'
                ),
                row=2, col=1
            )
            
            # SOC限界値の表示
            fig_demand.add_hline(y=90, line_dash="dash", line_color="red", 
                               annotation_text="SOC上限(90%)", row=2, col=1)
            fig_demand.add_hline(y=10, line_dash="dash", line_color="red", 
                               annotation_text="SOC下限(10%)", row=2, col=1)
            fig_demand.add_hline(y=50, line_dash="dot", line_color="gray", 
                               annotation_text="SOC中央(50%)", row=2, col=1)
            
            # レイアウト更新
            fig_demand.update_xaxes(title_text="日時", row=2, col=1)
            fig_demand.update_yaxes(title_text="需要 (kW)", row=1, col=1)
            fig_demand.update_yaxes(title_text="SOC (%)", range=[0, 100], row=2, col=1)
            
            fig_demand.update_layout(
                height=800,
                hovermode='x unified',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
        else:
            # 従来の需要のみグラフ
            fig_demand = go.Figure()
            
            # 元需要予測
            fig_demand.add_trace(go.Scatter(
                x=period_times,
                y=period_demand,
                name="元需要予測",
                line=dict(color="lightblue", width=2),
                opacity=0.8
            ))
            
            # 電池制御後需要
            fig_demand.add_trace(go.Scatter(
                x=period_times,
                y=period_controlled,
                name=f"電池制御後（{selected_capacity_graph:,}kWh・SOC引き継ぎ）",
                line=dict(color="red", width=2)
            ))
            
            fig_demand.update_layout(
                title=f"需要カーブ比較 - {period_title}（SOC引き継ぎ対応）",
                xaxis_title="日時",
                yaxis_title="需要 (kW)",
                height=500,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                hovermode='x unified'
            )
        
        st.plotly_chart(fig_demand, use_container_width=True)
        
        # 効果統計表示
        if selected_capacity_graph in results:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                peak_reduction = np.max(period_demand) - np.max(period_controlled)
                st.metric("ピーク削減", f"{peak_reduction:.1f} kW")
            
            with col2:
                avg_reduction = np.mean(period_demand) - np.mean(period_controlled)
                st.metric("平均削減", f"{avg_reduction:.1f} kW")
            
            with col3:
                range_original = np.max(period_demand) - np.min(period_demand)
                range_controlled = np.max(period_controlled) - np.min(period_controlled)
                range_improvement = range_original - range_controlled
                st.metric("需要幅改善", f"{range_improvement:.1f} kW")
            
            with col4:
                smoothness_original = np.std(np.diff(period_demand))
                smoothness_controlled = np.std(np.diff(period_controlled))
                smoothness_improvement = smoothness_original - smoothness_controlled
                st.metric("変動改善", f"{smoothness_improvement:.1f} kW")
        
        # 全容量比較グラフ（年間データのサンプル表示）
       # 表示オプション
        col1, col2 = st.columns(2)
        with col1:
            comparison_mode = st.radio(
                "比較表示モード",
                ["需要のみ", "需要+SOC"],
                index=0,
                key="comparison_mode_select"
            )
        with col2:
            if comparison_mode == "需要+SOC":
                soc_capacity = st.selectbox(
                    "SOC表示する容量",
                    capacity_list,
                    index=0,
                    format_func=lambda x: f"{x:,}kWh",
                    key="soc_display_capacity"
                )
        
        # データサンプリング（表示用）
        sample_size = min(len(annual_demand), 4320)  # 約3日分を表示
        sample_indices = np.linspace(0, len(annual_demand)-1, sample_size, dtype=int)
        
        # サンプル時系列作成
        time_series = create_annual_time_series()
        sample_times = [time_series[i] for i in sample_indices]
        
        if comparison_mode == "需要のみ":
            # 需要のみの比較グラフ
            fig_annual = go.Figure()
            
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
                    name=f"容量{capacity:,}kWh制御後（SOC引き継ぎ）",
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
            
            fig_annual.update_layout(
                title="年間需要平準化効果比較（全容量・SOC引き継ぎ・サンプル表示）",
                xaxis_title="日時",
                yaxis_title="需要 (kW)",
                height=600,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
        else:
            # 需要+SOCの比較グラフ（サブプロット）
            fig_annual = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=[
                    "年間需要平準化効果比較（全容量・SOC引き継ぎ）",
                    f"SOC推移 - 容量{soc_capacity:,}kWh（年間サンプル）"
                ],
                specs=[[{"secondary_y": False}],
                       [{"secondary_y": False}]]
            )
            
            # 上段：需要データ
            fig_annual.add_trace(
                go.Scatter(
                    x=sample_times,
                    y=annual_demand[sample_indices],
                    name="元需要予測",
                    line=dict(color="lightgray", width=1),
                    opacity=0.8
                ),
                row=1, col=1
            )
            
            # 各容量の制御後需要
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for i, (capacity, result) in enumerate(results.items()):
                fig_annual.add_trace(
                    go.Scatter(
                        x=sample_times,
                        y=result['demand_after_control'][sample_indices],
                        name=f"容量{capacity:,}kWh制御後",
                        line=dict(color=colors[i % len(colors)], width=2)
                    ),
                    row=1, col=1
                )
            
            # 下段：選択容量のSOCデータ
            if soc_capacity in results:
                soc_data = results[soc_capacity]['soc_profile'][sample_indices]
                fig_annual.add_trace(
                    go.Scatter(
                        x=sample_times,
                        y=soc_data,
                        name=f"SOC推移（{soc_capacity:,}kWh）",
                        line=dict(color="green", width=2),
                        fill='tonexty',
                        fillcolor='rgba(0,255,0,0.1)'
                    ),
                    row=2, col=1
                )
                
                # SOC限界値の表示
                fig_annual.add_hline(y=90, line_dash="dash", line_color="red", 
                                   annotation_text="SOC上限(90%)", row=2, col=1)
                fig_annual.add_hline(y=10, line_dash="dash", line_color="red", 
                                   annotation_text="SOC下限(10%)", row=2, col=1)
                fig_annual.add_hline(y=50, line_dash="dot", line_color="gray", 
                                   annotation_text="SOC中央(50%)", row=2, col=1)
            
            # レイアウト更新
            fig_annual.update_xaxes(title_text="日時", row=2, col=1)
            fig_annual.update_yaxes(title_text="需要 (kW)", row=1, col=1)
            fig_annual.update_yaxes(title_text="SOC (%)", range=[0, 100], row=2, col=1)
            
            fig_annual.update_layout(
                height=800,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
        
        st.plotly_chart(fig_annual, use_container_width=True)

    except Exception as e:
        st.error(f"年間グラフ作成エラー: {e}")
        import traceback
        st.text(traceback.format_exc())
    
    # 年間統計
    st.subheader("年間統計とSOC分析")  # タイトル変更
    col1, col2, col3, col4 = st.columns(4)
    
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
            title="容量別年間ピーク削減量（SOC引き継ぎ）"
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
    
    with col4:
        st.subheader("サイクル数実績")
        cycle_data = []
        for capacity, result in results.items():
            actual_cycles = result['annual_discharge'] / capacity if capacity > 0 else 0
            cycle_data.append({
                'capacity': f"{capacity:,}kWh",
                'cycles': actual_cycles
            })
        
        fig_cycles = px.bar(
            pd.DataFrame(cycle_data),
            x='capacity', y='cycles',
            title="容量別年間サイクル数"
        )
        # 目標サイクル数の水平線を追加
        fig_cycles.add_hline(
            y=st.session_state.sim_annual_cycle_ratio, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"目標: {st.session_state.sim_annual_cycle_ratio:.0f}回/年"
        )
        st.plotly_chart(fig_cycles, use_container_width=True)
        
def show_soc_analysis(results, capacity_list):
    """SOC推移分析タブの内容"""
    st.subheader("🔋 SOC推移分析")
    
    # 容量選択
    selected_capacity_soc = st.selectbox(
        "SOC分析する容量を選択",
        capacity_list,
        format_func=lambda x: f"{x:,}kWh",
        key="soc_analysis_capacity_select"
    )
    
    if selected_capacity_soc in results:
        result = results[selected_capacity_soc]
        soc_stats = result.get('soc_stats', {})
        
        # SOC統計表示
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("初期SOC", f"{soc_stats.get('initial_soc', 50):.1f}%")
        with col2:
            st.metric("最終SOC", f"{soc_stats.get('final_soc', 50):.1f}%")
        with col3:
            soc_change = soc_stats.get('final_soc', 50) - soc_stats.get('initial_soc', 50)
            st.metric("SOC変化", f"{soc_change:+.1f}%")
        with col4:
            st.metric("SOC範囲", f"{soc_stats.get('soc_range', 0):.1f}%")
        with col5:
            st.metric("平均SOC", f"{soc_stats.get('soc_average', 50):.1f}%")
        
        # 年間SOC推移グラフ
            st.subheader("SOC統計比較")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # SOC変化の比較
        soc_change_data = []
        for capacity, result in results.items():
            soc_stats = result.get('soc_stats', {})
            soc_change = soc_stats.get('final_soc', 50) - soc_stats.get('initial_soc', 50)
            soc_change_data.append({
                'capacity': f"{capacity:,}kWh",
                'soc_change': soc_change,
                'soc_change_abs': abs(soc_change)
            })
        
        fig_soc_change = px.bar(
            pd.DataFrame(soc_change_data),
            x='capacity', y='soc_change',
            title="容量別年間SOC変化（SOC引き継ぎ）",
            color='soc_change',
            color_continuous_scale='RdYlGn_r'
        )
        fig_soc_change.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_soc_change, use_container_width=True)
    
    with col2:
        # SOC変動範囲の比較
        soc_range_data = []
        for capacity, result in results.items():
            soc_stats = result.get('soc_stats', {})
            soc_range_data.append({
                'capacity': f"{capacity:,}kWh",
                'soc_range': soc_stats.get('soc_range', 0),
                'avg_soc': soc_stats.get('soc_average', 50)
            })
        
        fig_soc_range = px.bar(
            pd.DataFrame(soc_range_data),
            x='capacity', y='soc_range',
            title="容量別SOC変動範囲（SOC引き継ぎ）",
            color='avg_soc',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_soc_range, use_container_width=True)
        
        # SOCプロファイルのサンプリング（表示用）
        soc_profile = result.get('soc_profile', [])
        if len(soc_profile) > 0:
            sample_size = min(len(soc_profile), 8760)  # 約1週間分を表示
            sample_indices = np.linspace(0, len(soc_profile)-1, sample_size, dtype=int)
            
            time_series = create_annual_time_series()
            sample_times = [time_series[i] for i in sample_indices]
            sample_soc = soc_profile[sample_indices]
            
            fig_soc = go.Figure()
            
            fig_soc.add_trace(go.Scatter(
                x=sample_times,
                y=sample_soc,
                name=f"SOC推移（{selected_capacity_soc:,}kWh）",
                line=dict(color="green", width=2),
                fill='tonexty' if len(sample_soc) > 0 else None
            ))
            
            # SOC限界値の表示
            fig_soc.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="SOC上限(90%)")
            fig_soc.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="SOC下限(10%)")
            fig_soc.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="SOC中央(50%)")
            
            fig_soc.update_layout(
                title=f"年間SOC推移 - 容量{selected_capacity_soc:,}kWh（SOC引き継ぎあり）",
                xaxis_title="日時",
                yaxis_title="SOC (%)",
                yaxis=dict(range=[0, 100]),
                height=500
            )
            
            st.plotly_chart(fig_soc, use_container_width=True)
        
        # 日別SOC変化履歴
        st.subheader("日別SOC変化履歴")
        
        soc_daily_history = soc_stats.get('soc_daily_history', [])
        if len(soc_daily_history) > 1:
            # 月単位でグループ化
            days = list(range(len(soc_daily_history)))
            months = [((day-1) // 30) + 1 for day in days if day > 0]  # 簡易月計算
            
            fig_daily_soc = go.Figure()
            
            fig_daily_soc.add_trace(go.Scatter(
                x=days,
                y=soc_daily_history,
                name="日別SOC",
                line=dict(color="blue", width=2),
                mode='lines+markers',
                marker=dict(size=4)
            ))
            
            fig_daily_soc.update_layout(
                title=f"日別SOC履歴 - 容量{selected_capacity_soc:,}kWh（365日間）",
                xaxis_title="日数",
                yaxis_title="SOC (%)",
                yaxis=dict(range=[0, 100]),
                height=400
            )
            
            st.plotly_chart(fig_daily_soc, use_container_width=True)
            
            # SOC変化の統計
            if len(soc_daily_history) > 1:
                daily_soc_changes = np.diff(soc_daily_history)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("最大日別SOC増加", f"{np.max(daily_soc_changes):.1f}%")
                with col2:
                    st.metric("最大日別SOC減少", f"{np.min(daily_soc_changes):.1f}%")
                with col3:
                    st.metric("日別SOC変化平均", f"{np.mean(daily_soc_changes):.2f}%")
                with col4:
                    st.metric("日別SOC変化標準偏差", f"{np.std(daily_soc_changes):.2f}%")
    
    # 全容量のSOC比較
    st.subheader("全容量SOC比較")
    
    soc_comparison_data = []
    for capacity, result in results.items():
        soc_stats = result.get('soc_stats', {})
        soc_comparison_data.append({
            '容量': f"{capacity:,}kWh",
            '初期SOC': soc_stats.get('initial_soc', 50),
            '最終SOC': soc_stats.get('final_soc', 50),
            'SOC変化': soc_stats.get('final_soc', 50) - soc_stats.get('initial_soc', 50),
            'SOC範囲': soc_stats.get('soc_range', 0),
            '平均SOC': soc_stats.get('soc_average', 50)
        })
    
    soc_comparison_df = pd.DataFrame(soc_comparison_data)
    
    if not soc_comparison_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_soc_change = px.bar(
                soc_comparison_df, x='容量', y='SOC変化',
                title="容量別年間SOC変化",
                color='SOC変化',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_soc_change, use_container_width=True)
        
        with col2:
            fig_soc_range = px.bar(
                soc_comparison_df, x='容量', y='SOC範囲',
                title="容量別SOC変動範囲",
                color='SOC範囲',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_soc_range, use_container_width=True)
        
        # SOC比較詳細テーブル
        st.subheader("SOC比較詳細データ")
        st.dataframe(soc_comparison_df, use_container_width=True)


def show_seasonal_analysis(results):
    """季節別分析タブの内容（SOC引き継ぎ対応）"""
    st.subheader("🌸 季節別分析（SOC引き継ぎ対応）")
    
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
                title="季節別ピーク削減効果（SOC引き継ぎ）",
                barmode='group'
            )
            st.plotly_chart(fig_seasonal_peak, use_container_width=True)
        
        with col2:
            fig_seasonal_avg = px.bar(
                seasonal_df, x='季節', y='平均削減', color='容量',
                title="季節別平均削減効果（SOC引き継ぎ）",
                barmode='group'
            )
            st.plotly_chart(fig_seasonal_avg, use_container_width=True)
        
        # 季節別詳細テーブル
        st.subheader("季節別詳細データ")
        pivot_peak = seasonal_df.pivot(index='容量', columns='季節', values='ピーク削減')
        st.dataframe(pivot_peak, use_container_width=True)


def show_monthly_detail_analysis(results, capacity_list, annual_comparator):
    """月別詳細分析タブの内容（SOC引き継ぎ対応）"""
    st.subheader("📅 日別・月別詳細分析（SOC引き継ぎ対応）")
    
    # 容量選択
    selected_capacity = st.selectbox(
        "詳細表示する容量を選択",
        capacity_list,
        format_func=lambda x: f"{x:,}kWh",
        key="daily_detail_capacity_select"
    )
    
    # 表示モード選択
    detail_mode = st.radio(
        "表示モード",
        ["月別サマリー", "日別詳細"],
        index=0,
        key="detail_mode_select"
    )
    
    if selected_capacity in results:
        if detail_mode == "月別サマリー" and 'monthly_summary' in results[selected_capacity]:
            # 月別サマリー表示
            monthly_summary = results[selected_capacity]['monthly_summary']
            
            monthly_data = []
            month_names = ['1月', '2月', '3月', '4月', '5月', '6月',
                          '7月', '8月', '9月', '10月', '11月', '12月']
            
            for month in range(1, 13):
                if month in monthly_summary:
                    summary = monthly_summary[month]
                    monthly_data.append({
                        '月': month_names[month-1],
                        'ピーク削減(kW)': f"{summary['peak_reduction']:.1f}",
                        '月間放電(kWh)': f"{summary['monthly_discharge']:.0f}",
                        '処理日数': f"{summary['days_count']}日"
                    })
            
            monthly_df = pd.DataFrame(monthly_data)
            st.dataframe(monthly_df, use_container_width=True)
            
            # 月別トレンド
            col1, col2 = st.columns(2)
            
            with col1:
                monthly_peak_data = []
                for month in range(1, 13):
                    if month in monthly_summary:
                        monthly_peak_data.append({
                            'month': month_names[month-1],
                            'peak_reduction': monthly_summary[month]['peak_reduction']
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
                    if month in monthly_summary:
                        monthly_discharge_data.append({
                            'month': month_names[month-1],
                            'discharge': monthly_summary[month]['monthly_discharge']
                        })
                
                if monthly_discharge_data:
                    fig_monthly_discharge = px.line(
                        pd.DataFrame(monthly_discharge_data),
                        x='month', y='discharge',
                        title=f"月別放電量トレンド（容量{selected_capacity:,}kWh）"
                    )
                    st.plotly_chart(fig_monthly_discharge, use_container_width=True)
        
        elif detail_mode == "日別詳細" and 'daily_results' in results[selected_capacity]:
            # 日別詳細表示
            daily_results = results[selected_capacity]['daily_results']
            
            # 月選択
            selected_month = st.selectbox(
                "表示する月",
                list(range(1, 13)),
                index=0,
                format_func=lambda x: f"{x}月",
                key="selected_month_detail"
            )
            
            # 選択月の日別データ抽出
            month_daily_data = []
            for day, result in daily_results.items():
                # 日から月を計算（簡易版）
                day_month = annual_comparator._get_month_from_day(day - 1)
                if day_month == selected_month:
                    month_daily_data.append({
                        '日': day,
                        '日付': f"{selected_month}月{annual_comparator._get_day_in_month(day - 1)}日",
                        'ピーク削減(kW)': f"{result['peak_reduction']:.1f}",
                        '日別放電(kWh)': f"{result['daily_discharge']:.0f}",
                        '需要幅改善(kW)': f"{result['range_improvement']:.1f}"
                    })
            
            if month_daily_data:
                daily_df = pd.DataFrame(month_daily_data)
                st.dataframe(daily_df, use_container_width=True)
                
                # 日別トレンド（選択月）
                fig_daily = px.line(
                    daily_df,
                    x='日付', y='ピーク削減(kW)',
                    title=f"{selected_month}月の日別ピーク削減トレンド"
                )
                fig_daily.update_xaxes(tickangle=45)
                st.plotly_chart(fig_daily, use_container_width=True)
            else:
                st.info(f"{selected_month}月のデータがありません")


def show_capacity_recommendation(results, capacity_list):
    """推奨容量判定タブの内容（SOC引き継ぎ対応）"""
    st.subheader("🏆 推奨容量判定（SOC引き継ぎ対応）")
    
    # 推奨容量の総合評価
    try:
        best_capacity = None
        best_score = -1
        evaluation_results = []
        
        for capacity, result in results.items():
            # 各指標のスコア計算
            peak_score = result.get('annual_peak_reduction', 0) * 0.3
            efficiency_score = (result.get('annual_peak_reduction', 0) / (capacity / 1000)) * 0.25 if capacity > 0 else 0
            cycle_score = 100 if result.get('annual_cycle_constraint_satisfied', False) else 0
            
            # SOC安定性スコア（SOC変化が小さい方が良い）
            soc_stats = result.get('soc_stats', {})
            soc_change = abs(soc_stats.get('final_soc', 50) - soc_stats.get('initial_soc', 50))
            soc_stability_score = max(0, 10 - soc_change) * 0.1  # SOC変化10%以下で満点
            
            # 季節バランススコア（標準偏差が小さい方が良い）
            seasonal_values = [
                result['seasonal_stats']['spring']['peak_reduction'],
                result['seasonal_stats']['summer']['peak_reduction'],
                result['seasonal_stats']['autumn']['peak_reduction'],
                result['seasonal_stats']['winter']['peak_reduction']
            ]
            seasonal_balance_score = -np.std(seasonal_values) * 0.1
            
            total_score = peak_score + efficiency_score + cycle_score * 0.2 + soc_stability_score + seasonal_balance_score
            
            evaluation_results.append({
                '容量(kWh)': f"{capacity:,}",
                'ピーク削減スコア': f"{peak_score:.1f}",
                '容量効率スコア': f"{efficiency_score:.1f}",
                'サイクル制約スコア': f"{cycle_score * 0.2:.1f}",
                'SOC安定性スコア': f"{soc_stability_score:.1f}",
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
                best_result = results[best_capacity]
                best_soc_stats = best_result.get('soc_stats', {})
                st.success(f"""
                **🥇 総合推奨容量（SOC引き継ぎ対応）**
                
                **{best_capacity:,}kWh**
                
                総合スコア: {best_score:.1f}点
                
                推奨理由:
                - 年間通して安定した効果
                - 容量効率が優秀
                - サイクル制約を満足
                - SOC安定性が良好
                
                **SOC特性:**
                - 初期: {best_soc_stats.get('initial_soc', 50):.1f}%
                - 最終: {best_soc_stats.get('final_soc', 50):.1f}%
                - 変化: {best_soc_stats.get('final_soc', 50) - best_soc_stats.get('initial_soc', 50):+.1f}%
                """)
        
        with col2:
            # 最大ピーク削減容量
            best_peak_capacity = max(results.keys(), 
                                   key=lambda x: results[x].get('annual_peak_reduction', 0))
            peak_value = results[best_peak_capacity].get('annual_peak_reduction', 0)
            peak_soc_stats = results[best_peak_capacity].get('soc_stats', {})
            
            st.info(f"""
            **📈 最大ピーク削減**
            
            **{best_peak_capacity:,}kWh**
            
            年間ピーク削減: {peak_value:.1f}kW
            
            特徴:
            - 最大需要の大幅削減
            - 電力契約容量削減効果大
            
            **SOC特性:**
            - SOC変化: {peak_soc_stats.get('final_soc', 50) - peak_soc_stats.get('initial_soc', 50):+.1f}%
            - 平均SOC: {peak_soc_stats.get('soc_average', 50):.1f}%
            """)
        
        with col3:
            # 最高効率容量
            best_efficiency_capacity = max(results.keys(), 
                                         key=lambda x: results[x].get('annual_peak_reduction', 0) / (x / 1000) if x > 0 else 0)
            efficiency_value = results[best_efficiency_capacity].get('annual_peak_reduction', 0) / (best_efficiency_capacity / 1000) if best_efficiency_capacity > 0 else 0
            efficiency_soc_stats = results[best_efficiency_capacity].get('soc_stats', {})
            
            st.info(f"""
            **⚡ 最高効率**
            
            **{best_efficiency_capacity:,}kWh**
            
            容量効率: {efficiency_value:.2f}kW/MWh
            
            特徴:
            - 投資効率が最も良好
            - コストパフォーマンス重視
            
            **SOC特性:**
            - SOC範囲: {efficiency_soc_stats.get('soc_range', 0):.1f}%
            - 最終SOC: {efficiency_soc_stats.get('final_soc', 50):.1f}%
            """)
    
    except Exception as e:
        st.error(f"推奨容量判定エラー: {e}")
        import traceback
        st.text(traceback.format_exc())


def show_download_section(summary_df, results, annual_comparator):
    """ダウンロードセクション（SOC引き継ぎ対応）"""
    st.header("4. 結果ダウンロード（SOC引き継ぎ対応）")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if summary_df is not None:
            try:
                summary_csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="📊 年間サマリーCSV（SOC引き継ぎ）",
                    data=summary_csv,
                    file_name=f"annual_capacity_summary_soc_carryover_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_summary_csv"
                )
            except Exception as e:
                st.error(f"サマリーCSV生成エラー: {e}")
    
    with col2:
        if st.button("📅 日別・月別詳細CSV（SOC含む）", use_container_width=True, key="download_monthly_detail_btn"):
            try:
                detail_data = []
                
                for capacity, result in results.items():
                    # 月別サマリー（SOC情報追加）
                    if 'monthly_summary' in result:
                        for month, monthly_result in result['monthly_summary'].items():
                            detail_data.append({
                                '容量(kWh)': capacity,
                                '分析レベル': '月別',
                                '期間': f"{month}月",
                                'ピーク削減(kW)': monthly_result['peak_reduction'],
                                '放電量(kWh)': monthly_result['monthly_discharge'],
                                '処理日数': monthly_result['days_count'],
                                '月末SOC(%)': monthly_result.get('month_end_soc', 50)
                            })
                    
                    # 日別詳細（SOC情報追加、サンプル：最初の50日）
                    if 'daily_results' in result:
                        for day, daily_result in list(result['daily_results'].items())[:50]:
                            month = annual_comparator._get_month_from_day(day - 1)
                            day_in_month = annual_comparator._get_day_in_month(day - 1)
                            detail_data.append({
                                '容量(kWh)': capacity,
                                '分析レベル': '日別',
                                '期間': f"{month}月{day_in_month}日",
                                'ピーク削減(kW)': daily_result['peak_reduction'],
                                '放電量(kWh)': daily_result['daily_discharge'],
                                '需要幅改善(kW)': daily_result['range_improvement'],
                                '初期SOC(%)': daily_result.get('initial_soc', 50),
                                '最終SOC(%)': daily_result.get('final_soc', 50),
                                'SOC変化(%)': daily_result.get('final_soc', 50) - daily_result.get('initial_soc', 50)
                            })
                
                detail_df = pd.DataFrame(detail_data)
                detail_csv = detail_df.to_csv(index=False)
                
                st.download_button(
                    label="日別・月別詳細をダウンロード（SOC含む）",
                    data=detail_csv,
                    file_name=f"annual_daily_monthly_details_soc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_detail_csv"
                )
            except Exception as e:
                st.error(f"詳細CSV生成エラー: {e}")
    
    with col3:
        if st.button("🔋 SOC統計CSV", use_container_width=True, key="download_soc_stats_btn"):
            try:
                soc_stats_data = []
                
                for capacity, result in results.items():
                    soc_stats = result.get('soc_stats', {})
                    soc_stats_data.append({
                        '容量(kWh)': capacity,
                        '初期SOC(%)': soc_stats.get('initial_soc', 50),
                        '最終SOC(%)': soc_stats.get('final_soc', 50),
                        'SOC変化(%)': soc_stats.get('final_soc', 50) - soc_stats.get('initial_soc', 50),
                        'SOC範囲(%)': soc_stats.get('soc_range', 0),
                        '平均SOC(%)': soc_stats.get('soc_average', 50),
                        '年間ピーク削減(kW)': result['annual_peak_reduction'],
                        '年間放電量(MWh)': result['annual_discharge'] / 1000,
                        'サイクル制約達成': 'OK' if result['annual_cycle_constraint_satisfied'] else 'NG'
                    })
                
                soc_stats_df = pd.DataFrame(soc_stats_data)
                soc_stats_csv = soc_stats_df.to_csv(index=False)
                
                st.download_button(
                    label="SOC統計をダウンロード",
                    data=soc_stats_csv,
                    file_name=f"annual_soc_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_soc_stats_csv"
                )
            except Exception as e:
                st.error(f"SOC統計CSV生成エラー: {e}")


if __name__ == "__main__":
    main() 
