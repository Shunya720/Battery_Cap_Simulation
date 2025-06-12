"""
バッテリー制御システム コアロジック
共通ロジックを提供し、複数のアプリケーションから参照可能
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union

# 自動最適化用ライブラリ
try:
    import optuna
    from scipy.optimize import differential_evolution
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# === 誤差データ（96ステップ） ===
ERROR_DATA = np.array([
    -41,149,340,530,720,860,1000,1139,1279,1640,2002,2363,2724,2806,2887,2969,
    3050,2978,2905,2833,2760,1949,1138,326,-485,-678,-871,-1063,-1256,-1449,
    -1641,-1834,-2026,-1797,-1568,-1339,-1110,-457,197,850,1503,1383,1263,1143,
    1023,1032,1041,1050,1059,1068,1076,1085,1093,1630,2167,2703,3240,4036,4832,
    5628,6424,5620,4816,4012,3208,2598,1987,1377,766,1796,2825,3855,4884,3869,
    2855,1840,825,615,404,194,-17,-69,-120,-172,-223,-319,-414,-510,-605,-418,
    -231,-44,149,340,530,720
])


class PeakBottomOptimizer:
    """ピーク・ボトム同時最適化制御クラス（分離制御対応版）"""
    
    def __init__(self, daily_cycle_target=48000, peak_percentile=80, bottom_percentile=20, 
                 battery_capacity=48000, max_power=3000, peak_power_ratio=1.0, bottom_power_ratio=1.0,
                 flattening_power_ratio=0.3):
        self.daily_cycle_target = daily_cycle_target  # kWh/day
        self.peak_percentile = peak_percentile  # 上位何%をピークとするか
        self.bottom_percentile = bottom_percentile  # 下位何%をボトムとするか
        self.battery_capacity = battery_capacity  # バッテリー容量 kWh
        self.max_power = max_power  # 最大出力 kW
        self.peak_power_ratio = peak_power_ratio  # ピーク時間帯での最大出力比率
        self.bottom_power_ratio = bottom_power_ratio  # ボトム時間帯での最大出力比率
        self.flattening_power_ratio = flattening_power_ratio  # 平準化制御での最大出力比率
        
    def optimize_battery_output(self, demand_forecast: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """ピーク・ボトム同時最適化制御のメイン関数（分離制御版）"""
        
        # Step 1: ピーク・ボトム領域の特定
        peak_threshold = np.percentile(demand_forecast, self.peak_percentile)
        bottom_threshold = np.percentile(demand_forecast, self.bottom_percentile)
        
        peak_indices = np.where(demand_forecast >= peak_threshold)[0]
        bottom_indices = np.where(demand_forecast <= bottom_threshold)[0]
        
        # Step 2: 現在のピーク・ボトム値
        current_peak = np.max(demand_forecast[peak_indices]) if len(peak_indices) > 0 else np.max(demand_forecast)
        current_bottom = np.min(demand_forecast[bottom_indices]) if len(bottom_indices) > 0 else np.min(demand_forecast)
        
        # Step 3: 目標レベルの計算（バランスモード固定）
        ideal_target = (current_peak + current_bottom) / 2
        
        # Step 4: 分離制御による電池制御
        battery_output = self.allocate_battery_output_separated_control(
            demand_forecast, peak_indices, bottom_indices, ideal_target
        )
        
        # Step 5: 理論的なエネルギー量計算（参考値）
        theoretical_discharge_energy = -np.sum(battery_output[battery_output < 0])
        theoretical_charge_energy = np.sum(battery_output[battery_output > 0])
        
        # Step 6: 制御情報の整理
        control_info = {
            'peak_threshold': peak_threshold,
            'bottom_threshold': bottom_threshold,
            'ideal_target': ideal_target,
            'current_peak': current_peak,
            'current_bottom': current_bottom,
            'peak_reduction': current_peak - np.max(demand_forecast + battery_output),
            'bottom_elevation': np.min(demand_forecast + battery_output) - current_bottom,
            'demand_range_before': current_peak - current_bottom,
            'demand_range_after': np.max(demand_forecast + battery_output) - np.min(demand_forecast + battery_output),
            'peak_excess_energy': self.calculate_theoretical_peak_excess(demand_forecast, peak_indices, ideal_target),
            'bottom_deficit_energy': self.calculate_theoretical_bottom_deficit(demand_forecast, bottom_indices, ideal_target),
            'total_discharge_energy': theoretical_discharge_energy,
            'total_charge_energy': theoretical_charge_energy,
            'peak_indices': peak_indices,
            'bottom_indices': bottom_indices,
            'peak_control_count': len(peak_indices),
            'bottom_control_count': len(bottom_indices),
            'theoretical_max_discharge': len(peak_indices) * self.max_power * self.peak_power_ratio,
            'theoretical_max_charge': len(bottom_indices) * self.max_power * self.bottom_power_ratio,
            'peak_power_ratio': self.peak_power_ratio,
            'bottom_power_ratio': self.bottom_power_ratio
        }
        
        return battery_output, control_info
    
    def allocate_battery_output_separated_control(self, demand_forecast: np.ndarray, 
                                                peak_indices: np.ndarray, bottom_indices: np.ndarray, 
                                                target: float) -> np.ndarray:
        """電池出力の配分（分離制御対応版）"""
        battery_output = np.zeros(96)
        
        # ピーク時間帯：需要偏差に応じた可変放電（独立制御強度）
        for idx in peak_indices:
            excess = max(0, demand_forecast[idx] - target)
            discharge_amount = min(excess, self.max_power * self.peak_power_ratio)
            battery_output[idx] = -discharge_amount
        
        # ボトム時間帯：需要偏差に応じた可変充電（独立制御強度）
        for idx in bottom_indices:
            deficit = max(0, target - demand_forecast[idx])
            charge_amount = min(deficit, self.max_power * self.bottom_power_ratio)
            battery_output[idx] = charge_amount
        
        # その他時間帯：通常の平準化制御
        other_indices = []
        for i in range(96):
            if i not in peak_indices and i not in bottom_indices:
                other_indices.append(i)
        
        # 平準化制御のための目標レベル（全需要の平均値を使用）
        flattening_target = np.mean(demand_forecast)
        
        # その他時間帯での平準化制御
        for idx in other_indices:
            demand_deviation = demand_forecast[idx] - flattening_target
            
            # 需要が平均より高い場合は放電、低い場合は充電
            if demand_deviation > 0:
                discharge_amount = min(demand_deviation, self.max_power * self.flattening_power_ratio)
                battery_output[idx] = -discharge_amount
            elif demand_deviation < 0:
                charge_amount = min(abs(demand_deviation), self.max_power * self.flattening_power_ratio)
                battery_output[idx] = charge_amount
        
        return battery_output
    
    def calculate_theoretical_peak_excess(self, demand_forecast: np.ndarray, 
                                        peak_indices: np.ndarray, target: float) -> float:
        """理論的なピーク超過エネルギー計算（参考値）"""
        if len(peak_indices) == 0:
            return 0
        
        excess_power = demand_forecast[peak_indices] - target
        excess_power = np.maximum(excess_power, 0)
        excess_energy = np.sum(excess_power)
        
        return excess_energy
    
    def calculate_theoretical_bottom_deficit(self, demand_forecast: np.ndarray, 
                                           bottom_indices: np.ndarray, target: float) -> float:
        """理論的なボトム不足エネルギー計算（参考値）"""
        if len(bottom_indices) == 0:
            return 0
        
        deficit_power = target - demand_forecast[bottom_indices]
        deficit_power = np.maximum(deficit_power, 0)
        deficit_energy = np.sum(deficit_power)
        
        return deficit_energy


class BatterySOCManager:
    """SOC管理クラス（既存SOC目標との協調）"""
    
    def __init__(self, battery_capacity=48000, max_power=3000, efficiency=1.0, initial_soc=4.5):
        self.battery_capacity = battery_capacity
        self.max_power = max_power
        self.efficiency = efficiency
        self.initial_soc = initial_soc
        self.current_soc = initial_soc
        
        # 制御履歴
        self.confirmed_battery_output = np.full(96, np.nan)
        self.confirmed_soc_profile = np.full(96, np.nan)
        self.confirmed_soc_profile[0] = initial_soc
    
    def apply_soc_constraints_with_cycle_coordination(self, battery_output_raw: np.ndarray, 
                                                    current_step: int, target_discharge=48000, 
                                                    tolerance=1500) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """理論出力→残量計算→SOC制約確認→実際出力算出の正しいロジック（分離制御対応）"""
        battery_output = battery_output_raw.copy()
        soc_profile = np.zeros(96)
        battery_remaining_kwh = np.zeros(96)
        shortage_output = np.zeros(96)
        
        # 初期状態設定
        soc_profile[0] = self.initial_soc
        battery_remaining_kwh[0] = (self.initial_soc / 100) * self.battery_capacity
        
        # Step 1: 現在時刻までの確定値をコピー
        for step in range(min(current_step + 1, 96)):
            if not np.isnan(self.confirmed_battery_output[step]):
                battery_output[step] = self.confirmed_battery_output[step]
                soc_profile[step] = self.confirmed_soc_profile[step]
                battery_remaining_kwh[step] = (soc_profile[step] / 100) * self.battery_capacity
        
        # Step 2: 全ステップで正しいロジック適用
        for step in range(1, 96):
            previous_remaining = battery_remaining_kwh[step - 1]
            
            if step <= current_step and not np.isnan(self.confirmed_battery_output[step]):
                continue
            
            theoretical_output = battery_output_raw[step]
            theoretical_next_remaining = previous_remaining + theoretical_output
            
            # SOC制約を確認して実際出力を決定
            if theoretical_next_remaining < 0:
                actual_output = -previous_remaining
                actual_next_remaining = 0
            elif theoretical_next_remaining > self.battery_capacity:
                actual_output = self.battery_capacity - previous_remaining
                actual_next_remaining = self.battery_capacity
            else:
                actual_output = theoretical_output
                actual_next_remaining = theoretical_next_remaining
            
            # 出力制約適用
            max_energy_per_step = self.max_power
            actual_output = np.clip(actual_output, -max_energy_per_step, max_energy_per_step)
            
            # 最終的な残量を計算
            final_remaining = previous_remaining + actual_output
            final_remaining = np.clip(final_remaining, 0, self.battery_capacity)
            
            # 結果を記録
            battery_output[step] = actual_output
            battery_remaining_kwh[step] = final_remaining
            soc_profile[step] = (final_remaining / self.battery_capacity) * 100
            shortage_output[step] = theoretical_output - actual_output
        
        return battery_output, soc_profile, battery_remaining_kwh, shortage_output
    
    def update_confirmed_values(self, step: int, battery_output: float, soc: float):
        """確定値の更新"""
        if 0 <= step < 96:
            self.confirmed_battery_output[step] = battery_output
            self.confirmed_soc_profile[step] = soc
            self.current_soc = soc
    
    def reset_simulation(self, initial_soc: Optional[float] = None):
        """シミュレーションリセット（動的SOC対応）"""
        if initial_soc is not None:
            self.initial_soc = initial_soc
            self.current_soc = initial_soc  # ← 現在SOCも更新
        
        self.confirmed_battery_output = np.full(96, np.nan)
        self.confirmed_soc_profile = np.full(96, np.nan)
        self.confirmed_soc_profile[0] = self.initial_soc  # ← 更新されたSOCを使用


class DemandSmoothnessOptimizer:
    """需要滑らかさ特化型自動最適化クラス（平準化重視・分離制御対応）"""
    
    def __init__(self, peak_bottom_optimizer_class, soc_manager_class, 
                 battery_capacity=48000, max_power=3000):
        self.PeakBottomOptimizer = peak_bottom_optimizer_class
        self.BatterySOCManager = soc_manager_class
        self.battery_capacity = battery_capacity
        self.max_power = max_power
        
        # 最適化履歴
        self.optimization_history = []
        self.best_result = None
        
    def calculate_demand_smoothness_metrics(self, demand_original: np.ndarray, 
                                          demand_after_control: np.ndarray) -> Dict:
        """需要滑らかさの詳細計算（平準化重視）"""
        
        # 1. 隣接ステップ間の差分（ギザギザ度）
        diff_original = np.abs(np.diff(demand_original))
        diff_after = np.abs(np.diff(demand_after_control))
        smoothness_improvement = (np.sum(diff_original) - np.sum(diff_after)) / np.sum(diff_original) if np.sum(diff_original) > 0 else 0
        
        # 2. 2ステップ間の変動（より滑らか性）
        diff2_original = np.abs(np.diff(demand_original, n=2)) if len(demand_original) >= 3 else np.array([0])
        diff2_after = np.abs(np.diff(demand_after_control, n=2)) if len(demand_after_control) >= 3 else np.array([0])
        smoothness2_improvement = (np.sum(diff2_original) - np.sum(diff2_after)) / np.sum(diff2_original) if np.sum(diff2_original) > 0 else 0
        
        # 3. 最大変動幅の改善（急激な変化の抑制）
        max_jump_original = np.max(diff_original) if len(diff_original) > 0 else 0
        max_jump_after = np.max(diff_after) if len(diff_after) > 0 else 0
        max_jump_improvement = (max_jump_original - max_jump_after) / max_jump_original if max_jump_original > 0 else 0
        
        # 4. 変動の標準偏差（変動の安定性）
        variation_std_original = np.std(diff_original) if len(diff_original) > 0 else 0
        variation_std_after = np.std(diff_after) if len(diff_after) > 0 else 0
        variation_stability = (variation_std_original - variation_std_after) / variation_std_original if variation_std_original > 0 else 0
        
        # 5. 連続する急変回数の削減
        def count_sharp_changes(data, threshold_percentile=90):
            """急激な変化の回数をカウント"""
            if len(data) < 2:
                return 0
            diffs = np.abs(np.diff(data))
            if len(diffs) == 0:
                return 0
            threshold = np.percentile(diffs, threshold_percentile)
            return np.sum(diffs > threshold)
        
        sharp_changes_original = count_sharp_changes(demand_original)
        sharp_changes_after = count_sharp_changes(demand_after_control)
        sharp_change_reduction = (sharp_changes_original - sharp_changes_after) / sharp_changes_original if sharp_changes_original > 0 else 0
        
        # 6. 標準偏差の改善度
        std_original = np.std(demand_original)
        std_after = np.std(demand_after_control)
        std_improvement = (std_original - std_after) / std_original if std_original > 0 else 0
        
        # 7. レンジ改善
        range_original = np.max(demand_original) - np.min(demand_original)
        range_after = np.max(demand_after_control) - np.min(demand_after_control)
        range_improvement = (range_original - range_after) / range_original if range_original > 0 else 0
        
        # 8. 変動係数の改善度
        cv_original = std_original / np.mean(demand_original) if np.mean(demand_original) > 0 else 0
        cv_after = std_after / np.mean(demand_after_control) if np.mean(demand_after_control) > 0 else 0
        cv_improvement = (cv_original - cv_after) / cv_original if cv_original > 0 else 0
        
        return {
            # 滑らかさ関連指標（最重要）
            'smoothness_improvement': smoothness_improvement,
            'smoothness2_improvement': smoothness2_improvement,
            'max_jump_improvement': max_jump_improvement,
            'variation_stability': variation_stability,
            'sharp_change_reduction': sharp_change_reduction,
            
            # 従来指標（補助的）
            'std_improvement': std_improvement,
            'range_improvement': range_improvement,
            'cv_improvement': cv_improvement,
            
            # 詳細値
            'diff_original_sum': np.sum(diff_original),
            'diff_after_sum': np.sum(diff_after),
            'max_jump_original': max_jump_original,
            'max_jump_after': max_jump_after,
            'sharp_changes_original': sharp_changes_original,
            'sharp_changes_after': sharp_changes_after,
            'std_original': std_original,
            'std_after': std_after,
            'range_original': range_original,
            'range_after': range_after
        }
    
    def objective_function_smoothness_focused(self, params: List[float], demand_forecast: np.ndarray, 
                                            cycle_target=48000, cycle_tolerance=1500) -> float:
        """需要滑らかさ特化型目的関数（平準化最優先・分離制御対応）"""
        try:
            # パラメータ取得（5次元に拡張）
            peak_percentile = max(50, min(100, params[0]))
            bottom_percentile = max(0, min(50, params[1]))
            peak_power_ratio = max(0.1, min(1.0, params[2]))
            bottom_power_ratio = max(0.1, min(1.0, params[3]))
            flattening_power_ratio = max(0.1, min(1.0, params[4]))
            
            # オプティマイザー設定（分離制御対応）
            optimizer = self.PeakBottomOptimizer(
                daily_cycle_target=cycle_target,
                peak_percentile=peak_percentile,
                bottom_percentile=bottom_percentile,
                battery_capacity=self.battery_capacity,
                max_power=self.max_power,
                peak_power_ratio=peak_power_ratio,
                bottom_power_ratio=bottom_power_ratio,
                flattening_power_ratio=flattening_power_ratio
            )
            
            # SOCマネージャー設定
            soc_manager = self.BatterySOCManager(
                self.battery_capacity, self.max_power, efficiency=1.0, initial_soc=4.5
            )
            
            # 制御実行
            battery_output_raw, control_info = optimizer.optimize_battery_output(demand_forecast)
            battery_output, soc_profile, battery_remaining_kwh, shortage_output = \
                soc_manager.apply_soc_constraints_with_cycle_coordination(
                    battery_output_raw, 95, cycle_target, tolerance=cycle_tolerance
                )
            
            # 制御後需要
            demand_after_control = demand_forecast + battery_output
            
            # サイクル制約チェック
            actual_discharge = -np.sum(battery_output[battery_output < 0])
            cycle_deviation = abs(actual_discharge - cycle_target)
            cycle_constraint_satisfied = cycle_deviation <= cycle_tolerance
            
            # 需要滑らかさ指標計算
            smoothness_metrics = self.calculate_demand_smoothness_metrics(
                demand_forecast, demand_after_control
            )
            
            # 滑らかさ重視の総合スコア計算（重み付き）
            smoothness_score = (
                0.40 * smoothness_metrics['smoothness_improvement'] +
                0.20 * smoothness_metrics['smoothness2_improvement'] +
                0.15 * smoothness_metrics['max_jump_improvement'] +
                0.10 * smoothness_metrics['variation_stability'] +
                0.10 * smoothness_metrics['sharp_change_reduction'] +
                0.03 * smoothness_metrics['std_improvement'] +
                0.02 * smoothness_metrics['range_improvement']
            )
            
            # 分離制御のバランス評価
            peak_bottom_balance = 1.0 - abs(peak_power_ratio - bottom_power_ratio) * 0.1
            smoothness_score *= peak_bottom_balance
            
            # SOC制約違反ペナルティ
            soc_violations = np.sum((soc_profile < 0) | (soc_profile > 100))
            shortage_penalty = np.sum(np.abs(shortage_output)) / 1000
            soc_penalty = soc_violations + shortage_penalty
            
            # 目的関数の計算
            if cycle_constraint_satisfied:
                objective = smoothness_score - 0.1 * soc_penalty
            else:
                cycle_penalty = (cycle_deviation / cycle_target) * 10
                objective = smoothness_score - cycle_penalty - 0.1 * soc_penalty
            
            # 結果記録
            result = {
                'params': params.copy(),
                'objective': objective,
                'smoothness_score': smoothness_score,
                'cycle_constraint_satisfied': cycle_constraint_satisfied,
                'cycle_deviation': cycle_deviation,
                'actual_discharge': actual_discharge,
                'soc_penalty': soc_penalty,
                'smoothness_metrics': smoothness_metrics,
                'control_info': control_info,
                'demand_after_control': demand_after_control.copy(),
                'peak_bottom_balance': peak_bottom_balance
            }
            self.optimization_history.append(result)
            
            return -objective  # 最小化問題として返す
            
        except Exception as e:
            print(f"目的関数計算エラー: {e}")
            return 1000
    
    def optimize_for_demand_smoothness(self, demand_forecast: np.ndarray, cycle_target=48000, 
                                     cycle_tolerance=1500, method='optuna', n_trials=100) -> Union[optuna.Study, object]:
        """需要滑らかさのための最適化実行（平準化重視）"""
        print(f"🎯 需要滑らかさ最適化開始（平準化重視）")
        print(f"   サイクル目標: {cycle_target:,} ± {cycle_tolerance:,} kWh")
        print(f"   最適化手法: {method}")
        
        # 初期状態の滑らかさ指標
        initial_metrics = self.calculate_demand_smoothness_metrics(demand_forecast, demand_forecast)
        print(f"   初期隣接変動合計: {initial_metrics['diff_original_sum']:.1f} kW")
        print(f"   初期最大変動: {initial_metrics['max_jump_original']:.1f} kW")
        print(f"   初期急変回数: {initial_metrics['sharp_changes_original']} 回")
        
        self.optimization_history = []  # 履歴リセット
        
        if method == 'optuna' and OPTIMIZATION_AVAILABLE:
            result = self._optimize_with_optuna(demand_forecast, cycle_target, cycle_tolerance, n_trials)
        elif method == 'differential_evolution' and OPTIMIZATION_AVAILABLE:
            result = self._optimize_with_differential_evolution(demand_forecast, cycle_target, cycle_tolerance)
        else:
            if not OPTIMIZATION_AVAILABLE:
                raise ValueError("必要な最適化ライブラリが不足しています: pip install optuna scipy")
            else:
                raise ValueError(f"サポートされていない最適化手法: {method}")
        
        # 最良結果の選択と詳細分析
        self._analyze_best_result()
        
        return result
    
    def _optimize_with_optuna(self, demand_forecast: np.ndarray, cycle_target: float, 
                            cycle_tolerance: float, n_trials: int) -> optuna.Study:
        """Optunaによる最適化（分離制御対応）"""
        
        def optuna_objective(trial):
            params = [
                trial.suggest_float('peak_percentile', 60, 95),
                trial.suggest_float('bottom_percentile', 5, 40),
                trial.suggest_float('peak_power_ratio', 0.3, 1.0),
                trial.suggest_float('bottom_power_ratio', 0.3, 1.0),
                trial.suggest_float('flattening_power_ratio', 0.1, 0.8)
            ]
            return self.objective_function_smoothness_focused(
                params, demand_forecast, cycle_target, cycle_tolerance
            )
        
        study = optuna.create_study(direction='minimize')
        study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=False)
        
        return study
    
    def _optimize_with_differential_evolution(self, demand_forecast: np.ndarray, 
                                            cycle_target: float, cycle_tolerance: float) -> object:
        """差分進化による最適化（分離制御対応）"""
        
        bounds = [
            (60, 95),    # peak_percentile
            (5, 40),     # bottom_percentile  
            (0.3, 1.0),  # peak_power_ratio
            (0.3, 1.0),  # bottom_power_ratio
            (0.1, 0.8)   # flattening_power_ratio
        ]
        
        result = differential_evolution(
            self.objective_function_smoothness_focused,
            bounds,
            args=(demand_forecast, cycle_target, cycle_tolerance),
            maxiter=50,
            popsize=15,
            seed=42,
            atol=1e-4,
            polish=True
        )
        
        return result
    
    def _analyze_best_result(self):
        """最良結果の詳細分析（滑らかさ重視）"""
        if not self.optimization_history:
            return
        
        # サイクル制約を満たす結果の中から最良を選択
        valid_results = [r for r in self.optimization_history if r['cycle_constraint_satisfied']]
        
        if valid_results:
            self.best_result = max(valid_results, key=lambda x: x['objective'])
        else:
            # サイクル制約を満たす解が無い場合は全体から最良を選択
            self.best_result = max(self.optimization_history, key=lambda x: x['objective'])
    
    def get_optimized_parameters(self) -> Optional[Dict]:
        """最適化されたパラメータを取得（分離制御対応）"""
        if self.best_result is None:
            return None
        
        params = self.best_result['params']
        return {
            'peak_percentile': params[0],
            'bottom_percentile': params[1], 
            'peak_power_ratio': params[2],
            'bottom_power_ratio': params[3],
            'flattening_power_ratio': params[4],
            'daily_cycle_target': 48000  # 固定値
        }
    
    def compare_before_after(self, demand_forecast: np.ndarray) -> Optional[Dict]:
        """最適化前後の比較分析"""
        if self.best_result is None:
            return None
        
        original_metrics = self.calculate_demand_smoothness_metrics(demand_forecast, demand_forecast)
        optimized_metrics = self.best_result['smoothness_metrics']
        demand_after = self.best_result['demand_after_control']
        
        return {
            'before': {
                'demand': demand_forecast,
                'metrics': original_metrics
            },
            'after': {
                'demand': demand_after,
                'metrics': optimized_metrics
            }
        }
    
    def generate_optimization_report(self, demand_forecast: np.ndarray) -> str:
        """滑らかさ最適化レポート生成（平準化重視・分離制御対応）"""
        if self.best_result is None:
            return "最適化が実行されていません"
        
        report = []
        report.append("=" * 70)
        report.append("🎯 需要滑らかさ最適化レポート（平準化重視・分離制御対応）")
        report.append("=" * 70)
        
        # 最適パラメータ
        params = self.best_result['params']
        report.append(f"\n📋 最適パラメータ:")
        report.append(f"  ピーク判定閾値: {params[0]:.1f}% (上位{100-params[0]:.1f}%をピークとして扱う)")
        report.append(f"  ボトム判定閾値: {params[1]:.1f}% (下位{params[1]:.1f}%をボトムとして扱う)")
        report.append(f"  ピーク制御強度: {params[2]:.1%} (最大{params[2]*3000:.0f}kWh)")
        report.append(f"  ボトム制御強度: {params[3]:.1%} (最大{params[3]*3000:.0f}kWh)")
        report.append(f"  平準化制御強度: {params[4]:.1%} (最大{params[4]*3000:.0f}kWh)")
        
        # 制御比率バランス表示
        balance_ratio = abs(params[2] - params[3])
        if balance_ratio < 0.1:
            report.append(f"  制御バランス: ✅ 均衡 (差異{balance_ratio:.2f})")
        elif balance_ratio < 0.3:
            report.append(f"  制御バランス: ⚖️ 軽微差異 (差異{balance_ratio:.2f})")
        else:
            report.append(f"  制御バランス: ⚠️ 大きな差異 (差異{balance_ratio:.2f})")
        
        # 滑らかさ効果
        metrics = self.best_result['smoothness_metrics']
        report.append(f"\n📈 滑らかさ効果（平準化）:")
        report.append(f"  隣接変動改善: {metrics['smoothness_improvement']*100:.1f}%")
        report.append(f"    → 変動合計: {metrics['diff_original_sum']:.1f}kW → {metrics['diff_after_sum']:.1f}kW")
        
        if 'smoothness2_improvement' in metrics:
            report.append(f"  2次差分改善: {metrics['smoothness2_improvement']*100:.1f}%")
        
        report.append(f"  最大変動抑制: {metrics['max_jump_improvement']*100:.1f}%")
        report.append(f"    → 最大変動: {metrics['max_jump_original']:.1f}kW → {metrics['max_jump_after']:.1f}kW")
        
        if 'sharp_change_reduction' in metrics:
            report.append(f"  急変回数削減: {metrics['sharp_change_reduction']*100:.1f}%")
            report.append(f"    → 急変回数: {metrics['sharp_changes_original']}回 → {metrics['sharp_changes_after']}回")
        
        if 'variation_stability' in metrics:
            report.append(f"  変動安定性向上: {metrics['variation_stability']*100:.1f}%")
        
        report.append(f"  標準偏差改善: {metrics['std_improvement']*100:.1f}%")
        report.append(f"  需要レンジ改善: {metrics['range_improvement']*100:.1f}%")
        
        # サイクル制約
        report.append(f"\n🔄 サイクル制約:")
        report.append(f"  実際放電量: {self.best_result['actual_discharge']:.0f} kWh")
        report.append(f"  目標値: 48,000 ± 1,500 kWh")
        report.append(f"  制約満足: {'✅ 満足' if self.best_result['cycle_constraint_satisfied'] else '❌ 違反'}")
        
        # 分離制御効果
        report.append(f"\n⚡ 分離制御効果:")
        report.append(f"  ピーク制御強度: {params[2]:.1%} → 最大{params[2]*3000:.0f}kWh放電")
        report.append(f"  ボトム制御強度: {params[3]:.1%} → 最大{params[3]*3000:.0f}kWh充電")
        report.append(f"  制御バランス効果: {self.best_result['peak_bottom_balance']*100:.1f}%")
        
        # 推奨事項
        report.append(f"\n💡 推奨事項:")
        if metrics['smoothness_improvement'] > 0.4:
            report.append(f"  🎉 優秀な滑らかさ改善効果 (40%以上の変動削減)")
        elif metrics['smoothness_improvement'] > 0.2:
            report.append(f"  👍 良好な滑らかさ改善効果 (20%以上の変動削減)")
        else:
            report.append(f"  📝 滑らかさ改善効果が限定的")
        
        if 'sharp_changes_after' in metrics and metrics['sharp_changes_after'] == 0:
            report.append(f"  🌟 完璧！急激な変動を完全に排除")
        elif 'sharp_change_reduction' in metrics and metrics['sharp_change_reduction'] > 0.5:
            report.append(f"  ✨ 急激な変動を大幅削減")
        
        # 制御バランス推奨
        if balance_ratio < 0.1:
            report.append(f"  ⚖️ 均衡な制御バランスで安定した制御を実現")
        elif params[2] > params[3] + 0.2:
            report.append(f"  📈 ピーク重視制御：急激な需要増加に強い対応")
        elif params[3] > params[2] + 0.2:
            report.append(f"  📉 ボトム重視制御：需要低下時の効率的充電")
        
        if self.best_result['cycle_constraint_satisfied']:
            report.append(f"  ✅ サイクル制約を満たしながら効果的な滑らかさ改善を実現")
            report.append(f"  ✅ このパラメータでの運用を推奨")
        else:
            report.append(f"  ⚠️ サイクル制約違反が発生")
            report.append(f"  💡 バッテリー容量またはサイクル目標の見直しを検討")
        
        report.append("=" * 70)
        
        return "\n".join(report)


# === ユーティリティ関数 ===

def correct_forecast(original_forecast: np.ndarray, actual_data: np.ndarray, 
                    current_step: int, weight_factor=0.7) -> np.ndarray:
    """需要予測補正（加重平均）"""
    corrected = original_forecast.copy()
    
    for i in range(len(corrected)):
        if i <= current_step and not np.isnan(actual_data[i]):
            # 実績がある断面は実績を使用
            corrected[i] = actual_data[i]
        elif i > current_step:
            # 将来断面は加重平均で補正
            if current_step >= 0 and not np.isnan(actual_data[current_step]):
                recent_error = actual_data[current_step] - original_forecast[current_step]
                distance_weight = np.exp(-(i - current_step) / 10)  # 距離による重み減衰
                corrected[i] = original_forecast[i] + recent_error * weight_factor * distance_weight
    
    return corrected


def create_time_series(start_time: datetime, steps=96) -> List[datetime]:
    """時系列データ作成"""
    return [start_time + timedelta(minutes=15*i) for i in range(steps)]


class BatteryControlEngine:
    """バッテリー制御エンジン - メイン処理クラス"""
    
    def __init__(self, battery_capacity=48000, max_power=3000, efficiency=1.0, initial_soc=50.0):
        self.battery_capacity = battery_capacity
        self.max_power = max_power
        self.efficiency = efficiency
        self.initial_soc = initial_soc  # ← 可変値として保持
        
        # コアコンポーネント
        self.peak_bottom_optimizer = None
        self.soc_manager = None
        self.smoothness_optimizer = None
        
        # 状態管理
        self.simulation_started = False
        self.current_step = -1
        self.actual_data = np.full(96, np.nan)
        self.original_forecast = None
    
    def set_initial_soc(self, soc_percent):
        """初期SOC設定メソッド（年間シミュレーションから呼び出し用）"""
        self.initial_soc = soc_percent
        # 既存のSOCマネージャーがあれば更新
        if self.soc_manager:
            self.soc_manager.initial_soc = soc_percent
            self.soc_manager.current_soc = soc_percent
            # SOCプロファイルの最初の値も更新
            if hasattr(self.soc_manager, 'confirmed_soc_profile'):
                self.soc_manager.confirmed_soc_profile[0] = soc_percent
    
    def initialize_components(self, **params):
        """コンポーネント初期化（動的SOC対応）"""
        self.peak_bottom_optimizer = PeakBottomOptimizer(
            battery_capacity=self.battery_capacity,
            max_power=self.max_power,
            **params
        )
        
        # SOCマネージャーを動的SOCで初期化
        self.soc_manager = BatterySOCManager(
            self.battery_capacity, 
            self.max_power, 
            self.efficiency, 
            self.initial_soc  # ← 動的な値を使用
        )
        
        self.smoothness_optimizer = DemandSmoothnessOptimizer(
            PeakBottomOptimizer, BatterySOCManager,
            self.battery_capacity, self.max_power
        )
    
    def run_optimization(self, demand_forecast: np.ndarray, **optimization_params) -> Dict:
        """最適化実行"""
        if self.smoothness_optimizer is None:
            self.initialize_components()
        
        result = self.smoothness_optimizer.optimize_for_demand_smoothness(
            demand_forecast, **optimization_params
        )
        
        return {
            'optimization_result': result,
            'best_params': self.smoothness_optimizer.get_optimized_parameters(),
            'comparison': self.smoothness_optimizer.compare_before_after(demand_forecast),
            'report': self.smoothness_optimizer.generate_optimization_report(demand_forecast)
        }
    
    def run_control_simulation(self, demand_forecast: np.ndarray, **control_params) -> Dict:
        """制御シミュレーション実行"""
        self.initialize_components(**control_params)
        
        # 需要予測補正
        corrected_forecast = correct_forecast(
            self.original_forecast or demand_forecast,
            self.actual_data,
            self.current_step
        )
        
        # 制御実行
        battery_output_raw, control_info = self.peak_bottom_optimizer.optimize_battery_output(corrected_forecast)
        
        # SOC制約適用
        battery_output, soc_profile, battery_remaining_kwh, shortage_output = \
            self.soc_manager.apply_soc_constraints_with_cycle_coordination(
                battery_output_raw, self.current_step, 
                control_params.get('daily_cycle_target', 48000),
                tolerance=control_params.get('cycle_tolerance', 1500)
            )
        
        # 制御後需要
        demand_after_battery = corrected_forecast + battery_output
        
        return {
            'corrected_forecast': corrected_forecast,
            'battery_output_raw': battery_output_raw,
            'battery_output': battery_output,
            'soc_profile': soc_profile,
            'battery_remaining_kwh': battery_remaining_kwh,
            'shortage_output': shortage_output,
            'demand_after_battery': demand_after_battery,
            'control_info': control_info
        }
    
    def update_simulation_state(self, step: int = None, actual_value: float = None):
        """シミュレーション状態更新"""
        if step is not None:
            self.current_step = step
        
        if actual_value is not None and self.current_step >= 0:
            self.actual_data[self.current_step] = actual_value
    
    def reset_simulation(self):
        """シミュレーションリセット"""
        self.simulation_started = False
        self.current_step = -1
        self.actual_data = np.full(96, np.nan)
        if self.soc_manager:
            self.soc_manager.reset_simulation(self.initial_soc)
