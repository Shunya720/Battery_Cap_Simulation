import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple
import io

# ページ設定
st.set_page_config(
    page_title="発電機構成計算ツール",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .generator-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class GeneratorConfig:
    def __init__(self, name: str, min_output: float, max_output: float, 
                 priority: int, min_run_time: float, min_stop_time: float, 
                 is_must_run: bool = False, unit_type: str = "DG",
                 heat_rate_a: float = 0.0, heat_rate_b: float = 10.0, 
                 heat_rate_c: float = 0.0, fuel_price: float = 60354.0,
                 startup_cost: float = 0.0, shutdown_cost: float = 0.0):
        self.name = name
        self.min_output = min_output
        self.max_output = max_output
        self.priority = priority
        self.min_run_time = min_run_time  # 時間
        self.min_stop_time = min_stop_time  # 時間
        self.is_must_run = is_must_run
        self.unit_type = unit_type  # "DG" or "GT"
        # 燃費特性係数 (Heat Rate = a*P^2 + b*P + c の形)
        self.heat_rate_a = heat_rate_a  # 2次係数
        self.heat_rate_b = heat_rate_b  # 1次係数
        self.heat_rate_c = heat_rate_c  # 定数項
        self.fuel_price = fuel_price    # 燃料単価 [円/kL]
        self.startup_cost = startup_cost  # 起動費 [円]
        self.shutdown_cost = shutdown_cost  # 停止費 [円]

class EconomicDispatchSolver:
    def __init__(self):
        self.lambda_min = 0.0
        self.lambda_max = 100.0
        self.lambda_tolerance = 0.001  # kW
        self.max_iterations = 50
    
    def calculate_output_from_lambda(self, generator: GeneratorConfig, lambda_val: float) -> float:
        """λ値から発電機出力を計算（最小出力制約を厳密に適用）"""
        lambda_per_fuel = lambda_val / generator.fuel_price * 1000  # 単位調整
        
        if generator.heat_rate_a == 0:
            # 2次係数が0の場合は線形
            if generator.heat_rate_b == 0:
                output = generator.min_output
            else:
                output = lambda_per_fuel / generator.heat_rate_b
        else:
            output = (lambda_per_fuel - generator.heat_rate_b) / (2 * generator.heat_rate_a)
        
        # 上下限制約の厳密な適用
        return max(generator.min_output, min(generator.max_output, output))
    
    def calculate_total_power(self, generators: List[GeneratorConfig], lambda_val: float, 
                            status_flags: np.ndarray) -> float:
        """λ値から総出力を計算（運転中発電機のみ最小出力以上を保証）"""
        total_power = 0.0
        
        for i, gen in enumerate(generators):
            if status_flags[i] == 1:  # 運転中
                output = self.calculate_output_from_lambda(gen, lambda_val)
                total_power += max(output, gen.min_output)
        
        return total_power
    
    def find_lambda_binary_search(self, generators: List[GeneratorConfig], 
                                 demand: float, status_flags: np.ndarray) -> float:
        """バイナリサーチでλを探索（最小出力制約考慮）"""
        running_generators = [(i, gen) for i, gen in enumerate(generators) if status_flags[i] == 1]
        
        if not running_generators:
            return self.lambda_min
        
        min_total = sum(gen.min_output for _, gen in running_generators)
        max_total = sum(gen.max_output for _, gen in running_generators)
        
        # 需要が実現可能範囲外の場合
        if demand <= min_total:
            min_lambda = float('inf')
            for _, gen in running_generators:
                marginal_cost = (2 * gen.heat_rate_a * gen.min_output + gen.heat_rate_b) * gen.fuel_price / 1000
                min_lambda = min(min_lambda, marginal_cost)
            return max(self.lambda_min, min_lambda)
        
        if demand >= max_total:
            max_lambda = 0
            for _, gen in running_generators:
                marginal_cost = (2 * gen.heat_rate_a * gen.max_output + gen.heat_rate_b) * gen.fuel_price / 1000
                max_lambda = max(max_lambda, marginal_cost)
            return min(self.lambda_max, max_lambda)
        
        # 通常のバイナリサーチ
        lambda_low = self.lambda_min
        lambda_high = self.lambda_max
        
        for iteration in range(self.max_iterations):
            lambda_mid = (lambda_low + lambda_high) / 2
            total_power = self.calculate_total_power(generators, lambda_mid, status_flags)
            gap = total_power - demand
            
            if abs(gap) <= self.lambda_tolerance:
                return lambda_mid
            
            if gap > 0:
                lambda_high = lambda_mid
            else:
                lambda_low = lambda_mid
        
        return lambda_mid
    
    def solve_economic_dispatch(self, generators: List[GeneratorConfig], 
                              demand_data: np.ndarray, output_flags: np.ndarray) -> Dict:
        """経済配分計算（最小出力制約を厳密に適用）"""
        time_steps = len(demand_data)
        gen_count = len(generators)
        
        # λ値と出力の保存配列
        lambda_values = np.zeros(time_steps)
        power_outputs = np.zeros((gen_count, time_steps))
        
        # 各時刻での計算
        for t in range(time_steps):
            demand = demand_data[t]
            status_flags = output_flags[:, t]
            
            # 運転中発電機の出力範囲チェック
            running_generators = []
            min_total_output = 0.0
            max_total_output = 0.0
            
            for i, gen in enumerate(generators):
                if status_flags[i] == 1:  # 運転中
                    running_generators.append((i, gen))
                    min_total_output += gen.min_output
                    max_total_output += gen.max_output
            
            # 需要が最小出力合計を下回る場合
            if demand < min_total_output:
                for i, gen in running_generators:
                    power_outputs[i, t] = gen.min_output
                
                if running_generators:
                    best_gen = min(running_generators, key=lambda x: x[1].heat_rate_b)[1]
                    lambda_values[t] = (2 * best_gen.heat_rate_a * best_gen.min_output + best_gen.heat_rate_b) * best_gen.fuel_price / 1000
                else:
                    lambda_values[t] = 0.0
                
            # 需要が最大出力合計を上回る場合
            elif demand > max_total_output:
                for i, gen in running_generators:
                    power_outputs[i, t] = gen.max_output
                
                if running_generators:
                    worst_gen = max(running_generators, key=lambda x: x[1].heat_rate_b)[1]
                    lambda_values[t] = (2 * worst_gen.heat_rate_a * worst_gen.max_output + worst_gen.heat_rate_b) * worst_gen.fuel_price / 1000
                else:
                    lambda_values[t] = 0.0
                
            else:
                # 通常の経済配分計算
                lambda_val = self.find_lambda_binary_search(generators, demand, status_flags)
                lambda_values[t] = lambda_val
                
                for i, gen in enumerate(generators):
                    if status_flags[i] == 1:  # 運転中
                        calculated_output = self.calculate_output_from_lambda(gen, lambda_val)
                        power_outputs[i, t] = max(calculated_output, gen.min_output)
                    else:
                        power_outputs[i, t] = 0.0
        
        return {
            'lambda_values': lambda_values,
            'power_outputs': power_outputs,
            'total_costs': self.calculate_fuel_costs(generators, power_outputs, output_flags)
        }
    
    def calculate_fuel_costs(self, generators: List[GeneratorConfig], 
                           power_outputs: np.ndarray, output_flags: np.ndarray) -> Dict:
        """燃料費・起動停止費計算"""
        time_steps = power_outputs.shape[1]
        gen_count = len(generators)
        
        fuel_costs = np.zeros((gen_count, time_steps))
        startup_costs = np.zeros((gen_count, time_steps))
        shutdown_costs = np.zeros((gen_count, time_steps))
        total_fuel_cost = 0.0
        total_startup_cost = 0.0
        total_shutdown_cost = 0.0
        
        for i, gen in enumerate(generators):
            for t in range(time_steps):
                # 燃料費計算（運転中のみ）
                if output_flags[i, t] == 1:  # 運転中のみ
                    power = power_outputs[i, t]
                    # 燃料費 = (a*P^2 + b*P + c) * u * 0.25 (15分間隔なので1/4時間)
                    fuel_consumption = gen.heat_rate_a * power**2 + gen.heat_rate_b * power + gen.heat_rate_c
                    cost = fuel_consumption * gen.fuel_price * 0.25
                    fuel_costs[i, t] = cost
                    total_fuel_cost += cost
                
                # 起動費計算
                if t > 0 and output_flags[i, t-1] == 0 and output_flags[i, t] >= 1:  # 停止→運転（起動中含む）
                    startup_costs[i, t] = gen.startup_cost
                    total_startup_cost += gen.startup_cost
                elif t == 0 and output_flags[i, t] >= 1:  # 初期時刻で運転開始
                    startup_costs[i, t] = gen.startup_cost
                    total_startup_cost += gen.startup_cost
                
                # 停止費計算
                if t > 0 and output_flags[i, t-1] >= 1 and output_flags[i, t] == 0:  # 運転→停止
                    shutdown_costs[i, t] = gen.shutdown_cost
                    total_shutdown_cost += gen.shutdown_cost
        
        total_cost = total_fuel_cost + total_startup_cost + total_shutdown_cost
        
        return {
            'individual_costs': fuel_costs,
            'startup_costs': startup_costs,
            'shutdown_costs': shutdown_costs,
            'total_cost': total_cost,
            'total_fuel_cost': total_fuel_cost,
            'total_startup_cost': total_startup_cost,
            'total_shutdown_cost': total_shutdown_cost,
            'average_cost_per_hour': total_cost / 24
        }

class UnitCommitmentSolver:
    def __init__(self):
        self.generators = []
        self.demand_data = None
        self.time_steps = 96  # 15分間隔、24時間
        self.margin_rate_dg = 0.1  # DG用マージン率
        self.margin_rate_gt = 0.15  # GT用マージン率
        self.stop_margin_rate_dg = 0.05  # DG用解列マージン率
        self.stop_margin_rate_gt = 0.08  # GT用解列マージン率
        
    def add_generator(self, gen_config: GeneratorConfig):
        self.generators.append(gen_config)
        
    def set_demand_data(self, demand_data: np.ndarray):
        self.demand_data = demand_data[:self.time_steps]
    
    def calculate_minimum_units_required(self, demand: float, sorted_generators: List[GeneratorConfig], 
                                       margin_rate: float = 0.0) -> Tuple[int, List[int], Dict]:
        """
        優先順位に基づいて需要を満たす最小台数の発電機を選択
        """
        target_capacity = demand * (1 + margin_rate)
        selected_units = []
        cumulative_capacity = 0.0
        cumulative_min_output = 0.0
        
        analysis = {
            'demand': demand,
            'target_capacity': target_capacity,
            'margin_rate': margin_rate,
            'selection_process': [],
            'feasibility_check': True,
            'selection_complete': False
        }
        
        # Step 1: マストラン発電機を必須選択
        for i, gen in enumerate(sorted_generators):
            if gen.is_must_run:
                selected_units.append(i)
                cumulative_capacity += gen.max_output
                cumulative_min_output += gen.min_output
                analysis['selection_process'].append({
                    'step': len(selected_units),
                    'unit': gen.name,
                    'priority': gen.priority,
                    'reason': 'マストラン（必須選択）',
                    'capacity_added': gen.max_output,
                    'cumulative_capacity': cumulative_capacity,
                    'target_met': cumulative_capacity >= target_capacity
                })
        
        # Step 2: 優先順位順に必要最小限の発電機を追加
        for i, gen in enumerate(sorted_generators):
            # 既に選択済み（マストラン）の場合はスキップ
            if gen.is_must_run:
                continue
            
            # 目標容量に達している場合は選択を停止
            if cumulative_capacity >= target_capacity:
                analysis['selection_complete'] = True
                break
            
            # 優先順位に従って発電機を追加
            selected_units.append(i)
            cumulative_capacity += gen.max_output
            cumulative_min_output += gen.min_output
            
            target_met = cumulative_capacity >= target_capacity
            analysis['selection_process'].append({
                'step': len(selected_units),
                'unit': gen.name,
                'priority': gen.priority,
                'reason': f'容量不足解消（{cumulative_capacity - gen.max_output:.0f} → {cumulative_capacity:.0f} kW）',
                'capacity_added': gen.max_output,
                'cumulative_capacity': cumulative_capacity,
                'target_met': target_met
            })
            
            # 目標容量に達したら選択完了
            if target_met:
                analysis['selection_complete'] = True
                break
        
        # Step 3: 最小出力制約の実現可能性チェック
        analysis['feasibility_check'] = cumulative_min_output <= demand
        analysis['final_capacity'] = cumulative_capacity
        analysis['final_min_output'] = cumulative_min_output
        analysis['capacity_shortage'] = max(0, target_capacity - cumulative_capacity)
        analysis['min_output_excess'] = max(0, cumulative_min_output - demand)
        
        return len(selected_units), selected_units, analysis
    
    def validate_unit_commitment_feasibility(self, demand_data: np.ndarray, 
                                           output_flags: np.ndarray) -> Dict:
        """構成計算結果の実現可能性を検証"""
        sorted_generators = sorted(self.generators, key=lambda x: x.priority)
        validation_results = {
            'overall_feasible': True,
            'infeasible_periods': [],
            'statistics': {
                'total_periods': len(demand_data),
                'feasible_periods': 0,
                'min_output_violations': 0,
                'capacity_shortages': 0
            }
        }
        
        for t in range(len(demand_data)):
            demand = demand_data[t]
            period_analysis = {
                'time_step': t,
                'hour': (t * 0.25) % 24,
                'demand': demand,
                'issues': []
            }
            
            # 運転中発電機の容量チェック
            total_min_output = 0.0
            total_max_output = 0.0
            
            for i, gen in enumerate(sorted_generators):
                if output_flags[i, t] == 1:  # 運転中
                    total_min_output += gen.min_output
                    total_max_output += gen.max_output
            
            # 実現可能性チェック
            is_feasible = True
            
            if total_max_output < demand:
                period_analysis['issues'].append(
                    f"容量不足: 最大出力{total_max_output:.0f}kW < 需要{demand:.0f}kW"
                )
                validation_results['statistics']['capacity_shortages'] += 1
                is_feasible = False
            
            if total_min_output > demand:
                period_analysis['issues'].append(
                    f"最小出力超過: 最小出力{total_min_output:.0f}kW > 需要{demand:.0f}kW"
                )
                validation_results['statistics']['min_output_violations'] += 1
                is_feasible = False
            
            if is_feasible:
                validation_results['statistics']['feasible_periods'] += 1
            else:
                validation_results['overall_feasible'] = False
                validation_results['infeasible_periods'].append(period_analysis)
        
        # 統計情報の追加
        total_periods = validation_results['statistics']['total_periods']
        feasible_rate = (validation_results['statistics']['feasible_periods'] / total_periods) * 100
        validation_results['statistics']['feasibility_rate'] = feasible_rate
        
        return validation_results
    
    def get_time_based_margin(self, time_step: int) -> Tuple[float, float]:
        """時間帯別マージン設定（17:00-22:00がピーク）"""
        hour = (time_step * 0.25) % 24
        is_peak_hour = 17 <= hour < 22
        
        if is_peak_hour:
            return self.margin_rate_dg, self.margin_rate_gt
        else:
            return self.margin_rate_dg / 2, self.margin_rate_gt / 2
            
    def get_stop_margin(self, time_step: int) -> Tuple[float, float]:
        """解列用マージン設定"""
        hour = (time_step * 0.25) % 24
        is_peak_hour = 17 <= hour < 22
        
        if is_peak_hour:
            return self.stop_margin_rate_dg, self.stop_margin_rate_gt
        else:
            return self.stop_margin_rate_dg / 2, self.stop_margin_rate_gt / 2
    
    def _find_last_stop_time(self, output_flags: np.ndarray, gen_index: int, current_time: int) -> int:
        """指定発電機の最後の停止開始時刻を探索"""
        for back in range(current_time - 1, -1, -1):
            if output_flags[gen_index, back] == 1:
                return back + 1  # 停止開始時刻
        return 0  # 初期から停止
    
    def solve_unit_commitment(self) -> Dict:
        """発電機構成計算のメイン処理（最小台数構成重視）"""
        if self.demand_data is None or len(self.generators) == 0:
            return {}
            
        # 発電機を優先順位でソート
        sorted_generators = sorted(self.generators, key=lambda x: x.priority)
        gen_count = len(sorted_generators)
        
        # 状態配列初期化 (0:停止, 1:運転, 2:起動中)
        output_flags = np.zeros((gen_count, self.time_steps), dtype=int)
        prev_flags = np.zeros(gen_count, dtype=int)
        last_start = np.full(gen_count, -100, dtype=int)
        
        # 最小運転・停止時間を15分単位に変換
        min_run_steps = [int(gen.min_run_time * 4) for gen in sorted_generators]
        min_stop_steps = [int(gen.min_stop_time * 4) for gen in sorted_generators]
        
        # デバッグ情報保存用
        debug_info = []
        
        # 各時間断面での計算
        for i in range(self.time_steps):
            demand = self.demand_data[i]
            current_hour = (i * 0.25) % 24
            
            # 将来需要（2断面後）
            future_demand = self.demand_data[min(i + 2, self.time_steps - 1)]
            
            step_debug = {
                'time_step': i,
                'hour': current_hour,
                'demand': demand,
                'future_demand': future_demand,
                'actions': []
            }
            
            # === 最小台数構成決定ロジック ===
            margin_dg, margin_gt = self.get_time_based_margin(i)
            current_margin = max(margin_dg, margin_gt)
            
            # 現在および将来需要での最小構成を計算
            margin_dg, margin_gt = self.get_time_based_margin(i)
            current_margin = max(margin_dg, margin_gt)
            
            _, current_required, current_analysis = self.calculate_minimum_units_required(
                demand, sorted_generators, current_margin
            )
            _, future_required, future_analysis = self.calculate_minimum_units_required(
                future_demand, sorted_generators, current_margin
            )
            
            # 現在と将来の要求を統合（優先順位が高い方を優先）
            required_units = set(current_required + future_required)
            
            step_debug['capacity_analysis'] = {
                'current': current_analysis,
                'future': future_analysis,
                'required_units': [sorted_generators[idx].name for idx in sorted(required_units)]
            }
            
            # 起動判定（最小構成ベース）
            target_flags = np.zeros(gen_count, dtype=int)
            
            for j, gen in enumerate(sorted_generators):
                should_start = False
                start_reason = ""
                
                # マストラン発電機は常時運転
                if gen.is_must_run:
                    should_start = True
                    start_reason = "マストラン（必須運転）"
                
                # 最小構成に含まれる場合
                elif j in required_units:
                    # 最小停止時間制約チェック
                    can_start_now = True
                    
                    if prev_flags[j] == 0:  # 現在停止中
                        last_stop_step = self._find_last_stop_time(output_flags, j, i)
                        stop_duration = i - last_stop_step
                        
                        if stop_duration < min_stop_steps[j]:
                            # 深刻な容量不足時のみ制約無視
                            capacity_shortage = current_analysis.get('capacity_shortage', 0)
                            if capacity_shortage > 2000:  # 2MW以上の不足
                                start_reason = f"緊急起動（容量不足{capacity_shortage:.0f}kW）"
                                step_debug['actions'].append(
                                    f"{gen.name}: {start_reason} [停止時間制約無視: {stop_duration}ステップ < {min_stop_steps[j]}ステップ]"
                                )
                            else:
                                can_start_now = False
                                step_debug['actions'].append(
                                    f"{gen.name}: 最小構成だが停止時間制約により見送り（{stop_duration}ステップ < {min_stop_steps[j]}ステップ）"
                                )
                    
                    if can_start_now and not start_reason:
                        should_start = True
                        if j in current_required and j in future_required:
                            start_reason = "現在・将来需要の最小構成"
                        elif j in current_required:
                            start_reason = "現在需要の最小構成"
                        else:
                            start_reason = "将来需要の最小構成（予防起動）"
                
                # 最小構成外の緊急起動判定
                elif not should_start:
                    # 現在選択済みの容量を計算
                    current_selected_capacity = sum(
                        sorted_generators[k].max_output for k in range(j) if target_flags[k] == 1
                    )
                    
                    # GT急激需要上昇対応
                    if i >= 1 and gen.unit_type == "GT":
                        prev_demand = self.demand_data[i - 1]
                        margin_dg_check, margin_gt_check = self.get_time_based_margin(i)
                        margin_check = max(margin_dg_check, margin_gt_check)
                        if (demand - prev_demand) > 3000 and current_selected_capacity < demand * (1 + margin_check):
                            should_start = True
                            start_reason = "GT急激需要上昇対応"
                    
                    # 最終予備力不足対応
                    elif current_selected_capacity > 0:
                        reserve_margin = current_selected_capacity - demand
                        if reserve_margin < 500:
                            should_start = True
                            start_reason = f"予備力不足対応（{reserve_margin:.0f}kW）"
                
                    # 起動決定
                if should_start:
                    target_flags[j] = 1
                    step_debug['actions'].append(f"{gen.name}: {start_reason}")
            
                # 初回断面の処理（起動時間無視）
                if i == 0:
                    for j in range(gen_count):
                        if sorted_generators[j].is_must_run:
                            output_flags[j, i] = 1
                            prev_flags[j] = 1
                        elif target_flags[j] == 1:
                            # 初期断面では起動時間を無視して即座に運転状態にする
                            output_flags[j, i] = 1
                            prev_flags[j] = 1
                            last_start[j] = i
                        else:
                            output_flags[j, i] = 0
                            prev_flags[j] = 0
                    continue
            
            # === 解列判定処理 ===
            final_flags = target_flags.copy()
            stop_margin_dg, stop_margin_gt = self.get_stop_margin(i)
            
            for j, gen in enumerate(sorted_generators):
                stop_margin = stop_margin_gt if gen.unit_type == "GT" else stop_margin_dg
                
                # マストランは解列しない
                if gen.is_must_run:
                    final_flags[j] = 1
                    continue
                
                # 最小構成に含まれる場合は解列しない
                if j in required_units:
                    final_flags[j] = 1
                    continue
                
                # 現在運転中で最小構成に含まれない場合の解列判定
                if prev_flags[j] == 1 and target_flags[j] == 0:
                    # 最小運転時間チェック
                    active_steps = 0
                    for back in range(i - 1, -1, -1):
                        if output_flags[j, back] == 1:
                            active_steps += 1
                        else:
                            break
                    
                    can_stop = active_steps >= min_run_steps[j]
                    
                    if can_stop:
                        final_flags[j] = 0
                        step_debug['actions'].append(f"{gen.name}: 最小構成外のため解列")
                    else:
                        final_flags[j] = 1
                        step_debug['actions'].append(f"{gen.name}: 最小運転時間未達のため運転継続")
                else:
                    final_flags[j] = target_flags[j]
            
            debug_info.append(step_debug)
            
            # === 状態遷移処理 ===
            for j, gen in enumerate(sorted_generators):
                if gen.is_must_run:
                    output_flags[j, i] = 1
                    prev_flags[j] = 1
                    continue
                
                # 状態遷移ロジック
                if prev_flags[j] == 0 and final_flags[j] == 1:
                    # 起動処理
                    output_flags[j, i] = 2
                    if i + 1 < self.time_steps:
                        output_flags[j, i + 1] = 2
                    if i + 2 < self.time_steps:
                        output_flags[j, i + 2] = 1
                    last_start[j] = i
                    prev_flags[j] = 2
                
                elif prev_flags[j] == 2:
                    # 起動中の継続処理
                    if i - last_start[j] >= 2:
                        output_flags[j, i] = 1
                        prev_flags[j] = 1
                    else:
                        output_flags[j, i] = 2
                        prev_flags[j] = 2
                
                elif prev_flags[j] == 1 and final_flags[j] == 0:
                    # 停止処理
                    output_flags[j, i] = 0
                    prev_flags[j] = 0
                
                else:
                    # 状態維持
                    output_flags[j, i] = prev_flags[j]
                    prev_flags[j] = output_flags[j, i]
        
        # 構成計算結果の実現可能性検証
        feasibility_validation = self.validate_unit_commitment_feasibility(self.demand_data, output_flags)
        
        # 結果をまとめて返す
        result = {
            'generators': sorted_generators,
            'output_flags': output_flags,
            'demand_data': self.demand_data,
            'time_steps': self.time_steps,
            'debug_info': debug_info,
            'feasibility_validation': feasibility_validation,
            'margins': {
                'dg_start': self.margin_rate_dg,
                'gt_start': self.margin_rate_gt,
                'dg_stop': self.stop_margin_rate_dg,
                'gt_stop': self.stop_margin_rate_gt
            }
        }
        
        return result

def create_unit_commitment_chart(result: Dict) -> go.Figure:
    """発電機構成のチャートを作成"""
    if not result:
        return go.Figure()
    
    generators = result['generators']
    output_flags = result['output_flags']
    demand_data = result['demand_data']
    time_steps = result['time_steps']
    
    # 時間軸作成（15分間隔）
    time_labels = []
    for i in range(time_steps):
        hour = (i * 15) // 60
        minute = (i * 15) % 60
        time_labels.append(f"{hour:02d}:{minute:02d}")
    
    # 積み上げ面グラフ用のデータ準備
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('発電機構成・需要', '発電機状態'),
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1
    )
    
    # 色設定
    colors = px.colors.qualitative.Set3
    
    # 発電機出力の積み上げ
    y_stack = np.zeros(time_steps)
    
    for i, gen in enumerate(generators):
        y_values = []
        for t in range(time_steps):
            if output_flags[i, t] == 1:  # 運転中
                y_values.append(gen.max_output)
            else:
                y_values.append(0)
        
        # 積み上げ計算
        y_upper = y_stack + np.array(y_values)
        
        fig.add_trace(
            go.Scatter(
                x=time_labels,
                y=y_upper,
                fill='tonexty' if i > 0 else 'tozeroy',
                mode='none',
                name=gen.name,
                fillcolor=colors[i % len(colors)],
                hovertemplate=f'{gen.name}: %{{y:.0f}} kW<br>時刻: %{{x}}<extra></extra>'
            ),
            row=1, col=1
        )
        
        y_stack = y_upper
    
    # 需要ライン
    fig.add_trace(
        go.Scatter(
            x=time_labels,
            y=demand_data,
            mode='lines',
            name='需要',
            line=dict(color='red', width=3),
            hovertemplate='需要: %{y:.0f} kW<br>時刻: %{x}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 発電機状態表示（下段）
    for i, gen in enumerate(generators):
        status_text = []
        for t in range(time_steps):
            if output_flags[i, t] == 0:
                status_text.append('停止')
            elif output_flags[i, t] == 1:
                status_text.append('運転')
            elif output_flags[i, t] == 2:
                status_text.append('起動中')
        
        fig.add_trace(
            go.Scatter(
                x=time_labels,
                y=[i] * time_steps,
                mode='markers',
                marker=dict(
                    color=[0 if s == '停止' else 1 if s == '運転' else 0.5 for s in status_text],
                    colorscale=[[0, 'gray'], [0.5, 'orange'], [1, 'green']],
                    size=8,
                    symbol='square'
                ),
                name=f'{gen.name}_状態',
                text=status_text,
                hovertemplate=f'{gen.name}: %{{text}}<br>時刻: %{{x}}<extra></extra>',
                showlegend=False
            ),
            row=2, col=1
        )
    
    # レイアウト設定
    fig.update_layout(
        title='発電機構成計算結果',
        height=800,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="時刻", row=2, col=1)
    fig.update_yaxes(title_text="出力 (kW)", row=1, col=1)
    fig.update_yaxes(
        title_text="発電機",
        row=2, col=1,
        tickmode='array',
        tickvals=list(range(len(generators))),
        ticktext=[gen.name for gen in generators]
    )
    
    return fig

def create_economic_dispatch_chart(uc_result: Dict, ed_result: Dict) -> go.Figure:
    """経済配分結果のチャートを作成"""
    if not uc_result or not ed_result:
        return go.Figure()
    
    generators = uc_result['generators']
    power_outputs = ed_result['power_outputs']
    lambda_values = ed_result['lambda_values']
    demand_data = uc_result['demand_data']
    time_steps = uc_result['time_steps']
    
    # 時間軸作成（15分間隔）
    time_labels = []
    for i in range(time_steps):
        hour = (i * 15) // 60
        minute = (i * 15) % 60
        time_labels.append(f"{hour:02d}:{minute:02d}")
    
    # サブプロット作成
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('発電機出力配分', 'λ値推移', '燃料費'),
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.08
    )
    
    # 色設定
    colors = px.colors.qualitative.Set3
    
    # 1. 発電機出力の積み上げ棒グラフ
    for i, gen in enumerate(generators):
        y_values = power_outputs[i, :]
        
        fig.add_trace(
            go.Bar(
                x=time_labels,
                y=y_values,
                name=gen.name,
                marker_color=colors[i % len(colors)],
                hovertemplate=f'{gen.name}: %{{y:.1f}} kW<br>時刻: %{{x}}<extra></extra>',
                opacity=0.8
            ),
            row=1, col=1
        )
    
    # 需要ライン
    fig.add_trace(
        go.Scatter(
            x=time_labels,
            y=demand_data,
            mode='lines',
            name='需要',
            line=dict(color='red', width=3, dash='dash'),
            hovertemplate='需要: %{y:.1f} kW<br>時刻: %{x}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. λ値推移
    fig.add_trace(
        go.Scatter(
            x=time_labels,
            y=lambda_values,
            mode='lines+markers',
            name='λ値',
            line=dict(color='purple', width=2),
            marker=dict(size=4),
            hovertemplate='λ値: %{y:.3f}<br>時刻: %{x}<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 3. 燃料費（時間あたり）
    fuel_costs = ed_result.get('total_costs', {})
    if fuel_costs:
        hourly_costs = []
        for t in range(time_steps):
            hour_cost = 0
            for i in range(len(generators)):
                if 'individual_costs' in fuel_costs:
                    hour_cost += fuel_costs['individual_costs'][i, t]
            hourly_costs.append(hour_cost * 4)  # 15分→1時間換算
        
        fig.add_trace(
            go.Scatter(
                x=time_labels,
                y=hourly_costs,
                mode='lines',
                name='燃料費',
                line=dict(color='orange', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 165, 0, 0.3)',
                hovertemplate='燃料費: %{y:.0f} 円/h<br>時刻: %{x}<extra></extra>',
                showlegend=False
            ),
            row=3, col=1
        )
    
    # レイアウト設定
    fig.update_layout(
        title='経済配分計算結果',
        height=900,
        hovermode='x unified',
        barmode='stack'  # 積み上げ棒グラフ設定
    )
    
    fig.update_xaxes(title_text="時刻", row=3, col=1)
    fig.update_yaxes(title_text="出力 (kW)", row=1, col=1)
    fig.update_yaxes(title_text="λ値", row=2, col=1)
    fig.update_yaxes(title_text="燃料費 (円/h)", row=3, col=1)
    
    return fig

def generate_detailed_report(uc_result: Dict, ed_result: Dict = None) -> str:
    """詳細レポートを生成"""
    if not uc_result:
        return "計算結果がありません。"
    
    generators = uc_result['generators']
    output_flags = uc_result['output_flags']
    demand_data = uc_result['demand_data']
    time_steps = uc_result['time_steps']
    
    # レポート作成開始
    report = []
    report.append("# 🔍 発電機構成・経済配分 詳細レポート")
    report.append(f"**生成日時**: {pd.Timestamp.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
    report.append("---")
    
    # 1. エグゼクティブサマリー
    report.append("## 📋 エグゼクティブサマリー")
    
    # 需要統計
    valid_demands = demand_data[~np.isnan(demand_data)]
    min_demand = valid_demands.min()
    max_demand = valid_demands.max()
    avg_demand = valid_demands.mean()
    
    report.append(f"- **分析期間**: 24時間 (96 x 15分間隔)")
    report.append(f"- **需要統計**: 最小 {min_demand:.0f} kW / 平均 {avg_demand:.0f} kW / 最大 {max_demand:.0f} kW")
    report.append(f"- **発電機台数**: {len(generators)}台")
    
    # 総発電容量
    total_capacity = sum(gen.max_output for gen in generators)
    utilization_rate = (max_demand / total_capacity) * 100
    report.append(f"- **総発電容量**: {total_capacity:.0f} kW")
    report.append(f"- **最大需要時容量利用率**: {utilization_rate:.1f}%")
    
    # 最小台数構成分析
    running_hours_by_unit = []
    for i, gen in enumerate(generators):
        running_steps = np.sum(output_flags[i, :] == 1)
        running_hours = running_steps * 0.25
        running_hours_by_unit.append(running_hours)
    
    avg_running_hours = np.mean(running_hours_by_unit)
    report.append(f"- **平均発電機稼働時間**: {avg_running_hours:.1f} 時間")
    
    # 経済配分結果がある場合
    if ed_result:
        total_cost = ed_result['total_costs']['total_cost']
        total_fuel_cost = ed_result['total_costs'].get('total_fuel_cost', 0)      # 追加
        total_startup_cost = ed_result['total_costs'].get('total_startup_cost', 0)  # 追加
        total_shutdown_cost = ed_result['total_costs'].get('total_shutdown_cost', 0)  # 追加
        avg_cost_per_hour = ed_result['total_costs']['average_cost_per_hour']
        total_generation = np.sum(ed_result['power_outputs']) * 0.25  # kWh
        avg_cost_per_kwh = total_cost / total_generation if total_generation > 0 else 0
        
        report.append(f"- **総コスト**: {total_cost:,.0f} 円")                    # 変更
        report.append(f"  - 燃料費: {total_fuel_cost:,.0f} 円 ({total_fuel_cost/total_cost*100:.1f}%)")    # 追加
        report.append(f"  - 起動費: {total_startup_cost:,.0f} 円 ({total_startup_cost/total_cost*100:.1f}%)")  # 追加
        report.append(f"  - 停止費: {total_shutdown_cost:,.0f} 円 ({total_shutdown_cost/total_cost*100:.1f}%)")  # 追加
        report.append(f"- **平均コスト**: {avg_cost_per_hour:,.0f} 円/時")        # 変更
        report.append(f"- **総発電量**: {total_generation:,.0f} kWh")
        report.append(f"- **平均発電コスト**: {avg_cost_per_kwh:.2f} 円/kWh")
    
    report.append("")
    
    # 2. 最小台数構成分析
    report.append("## ⚙️ 最小台数構成分析")
    
    # 各時間帯での運転台数統計
    running_units_per_time = []
    for t in range(time_steps):
        running_count = np.sum(output_flags[:, t] == 1)
        running_units_per_time.append(running_count)
    
    min_running_units = min(running_units_per_time)
    max_running_units = max(running_units_per_time)
    avg_running_units = np.mean(running_units_per_time)
    
    report.append(f"- **最小運転台数**: {min_running_units} 台")
    report.append(f"- **最大運転台数**: {max_running_units} 台")
    report.append(f"- **平均運転台数**: {avg_running_units:.1f} 台")
    
    # 優先順位による効果分析
    report.append("### 優先順位効果")
    priority_effectiveness = []
    for i, gen in enumerate(generators):
        running_steps = np.sum(output_flags[i, :] == 1)
        utilization = (running_steps / 96) * 100
        priority_effectiveness.append((gen.priority, gen.name, utilization))
    
    priority_effectiveness.sort(key=lambda x: x[0])  # 優先順位順
    
    for priority, name, util in priority_effectiveness:
        if util > 80:
            status = "高稼働"
        elif util > 50:
            status = "中稼働"
        elif util > 20:
            status = "低稼働"
        else:
            status = "待機"
        report.append(f"- **{name}** (優先順位{priority}): {util:.1f}% ({status})")
    
    report.append("")
    
    # 3. 発電機別運転実績
    report.append("## ⚡ 発電機別運転実績")
    
    for i, gen in enumerate(generators):
        running_steps = np.sum(output_flags[i, :] == 1)
        starting_steps = np.sum(output_flags[i, :] == 2)
        running_hours = running_steps * 0.25
        utilization = (running_steps / 96) * 100
        
        # 起動回数計算
        start_count = 0
        for j in range(1, 96):
            if output_flags[i, j] == 2 and output_flags[i, j-1] == 0:
                start_count += 1
        
        report.append(f"### {gen.name} ({gen.unit_type}) - 優先順位{gen.priority}")
        report.append(f"- **容量**: {gen.min_output:.0f} - {gen.max_output:.0f} kW")
        report.append(f"- **運転時間**: {running_hours:.1f} 時間 ({utilization:.1f}%)")
        report.append(f"- **起動回数**: {start_count} 回")
        report.append(f"- **マストラン**: {'はい' if gen.is_must_run else 'いいえ'}")
        
        if ed_result:
            power_outputs = ed_result['power_outputs']
            gen_outputs = power_outputs[i, :]
            running_outputs = gen_outputs[gen_outputs > 0]
            
            if len(running_outputs) > 0:
                avg_output = np.mean(running_outputs)
                max_output_actual = np.max(gen_outputs)
                min_output_actual = np.min(running_outputs)
                total_generation_gen = np.sum(gen_outputs) * 0.25
                
                # 燃料費計算
                fuel_costs = ed_result['total_costs']['individual_costs']
                gen_fuel_cost = np.sum(fuel_costs[i, :])
                
                report.append(f"- **平均出力**: {avg_output:.1f} kW")
                report.append(f"- **出力範囲**: {min_output_actual:.1f} - {max_output_actual:.1f} kW")
                report.append(f"- **発電量**: {total_generation_gen:,.1f} kWh")
                report.append(f"- **燃料費**: {gen_fuel_cost:,.0f} 円")
                
                if total_generation_gen > 0:
                    unit_cost = gen_fuel_cost / total_generation_gen
                    report.append(f"- **単位発電コスト**: {unit_cost:.2f} 円/kWh")
        
        report.append("")
    
    # 4. 時間帯別分析
    report.append("## 🕐 時間帯別分析")
    
    # ピーク時間帯の定義
    peak_hours = list(range(68, 88))  # 17:00-22:00 (17*4 to 22*4)
    off_peak_hours = [i for i in range(96) if i not in peak_hours]
    
    peak_demand = np.mean([demand_data[i] for i in peak_hours if i < len(demand_data)])
    off_peak_demand = np.mean([demand_data[i] for i in off_peak_hours if i < len(demand_data)])
    
    report.append(f"### ピーク時間帯 (17:00-22:00)")
    report.append(f"- **平均需要**: {peak_demand:.0f} kW")
    
    # ピーク時の運転台数
    peak_running_units = []
    for i in peak_hours:
        if i < len(demand_data):
            running_count = np.sum(output_flags[:, i] == 1)
            peak_running_units.append(running_count)
    
    if peak_running_units:
        avg_peak_units = np.mean(peak_running_units)
        report.append(f"- **平均運転台数**: {avg_peak_units:.1f} 台")
    
    report.append(f"### オフピーク時間帯")
    report.append(f"- **平均需要**: {off_peak_demand:.0f} kW")
    
    # オフピーク時の運転台数
    off_peak_running_units = []
    for i in off_peak_hours:
        if i < len(demand_data):
            running_count = np.sum(output_flags[:, i] == 1)
            off_peak_running_units.append(running_count)
    
    if off_peak_running_units:
        avg_off_peak_units = np.mean(off_peak_running_units)
        report.append(f"- **平均運転台数**: {avg_off_peak_units:.1f} 台")
    
    load_factor = off_peak_demand / peak_demand if peak_demand > 0 else 0
    report.append(f"- **負荷率**: {load_factor:.2f}")
    report.append("")
    
    # 5. 経済性分析（経済配分結果がある場合）
    if ed_result:
        report.append("## 💰 経済性分析")
        
        lambda_values = ed_result['lambda_values']
        power_outputs = ed_result['power_outputs']
        fuel_costs = ed_result['total_costs']['individual_costs']
        
        # λ値分析
        report.append("### λ値分析")
        report.append(f"- **最小λ値**: {lambda_values.min():.3f}")
        report.append(f"- **最大λ値**: {lambda_values.max():.3f}")
        report.append(f"- **平均λ値**: {lambda_values.mean():.3f}")
        report.append(f"- **λ値標準偏差**: {lambda_values.std():.3f}")
        
        # 時間帯別λ値
        peak_lambda = np.mean([lambda_values[i] for i in peak_hours if i < len(lambda_values)])
        off_peak_lambda = np.mean([lambda_values[i] for i in off_peak_hours if i < len(lambda_values)])
        
        report.append(f"- **ピーク時平均λ値**: {peak_lambda:.3f}")
        report.append(f"- **オフピーク時平均λ値**: {off_peak_lambda:.3f}")
        report.append("")
        
        # コスト分析
        report.append("### 燃料費分析")
        
        # 発電機別コスト効率
        report.append("#### 発電機別コスト効率")
        cost_efficiency = []
        for i, gen in enumerate(generators):
            gen_outputs = power_outputs[i, :]
            gen_costs = fuel_costs[i, :]
            total_gen_output = np.sum(gen_outputs) * 0.25  # kWh
            total_gen_cost = np.sum(gen_costs)
            
            if total_gen_output > 0:
                unit_cost = total_gen_cost / total_gen_output
                cost_efficiency.append((gen.name, unit_cost, total_gen_output, total_gen_cost))
        
        # コスト効率でソート
        cost_efficiency.sort(key=lambda x: x[1])
        
        for name, unit_cost, total_output, total_cost in cost_efficiency:
            report.append(f"- **{name}**: {unit_cost:.2f} 円/kWh (発電量: {total_output:,.1f} kWh, 燃料費: {total_cost:,.0f} 円)")
        
        report.append("")
    
    # 6. 運用制約分析
    report.append("## ⚙️ 運用制約分析")
    
    # 最小運転・停止時間制約違反チェック
    constraint_violations = []
    
    for i, gen in enumerate(generators):
        min_run_steps = int(gen.min_run_time * 4)
        min_stop_steps = int(gen.min_stop_time * 4)
        
        # 運転期間分析
        current_run = 0
        current_stop = 0
        run_violations = 0
        stop_violations = 0
        
        for t in range(96):
            if output_flags[i, t] == 1:  # 運転中
                if current_stop > 0 and current_stop < min_stop_steps:
                    stop_violations += 1
                current_run += 1
                current_stop = 0
            else:  # 停止中
                if current_run > 0 and current_run < min_run_steps:
                    run_violations += 1
                current_stop += 1
                current_run = 0
        
        if run_violations > 0 or stop_violations > 0:
            constraint_violations.append(f"- **{gen.name}**: 最小運転時間違反 {run_violations}回, 最小停止時間違反 {stop_violations}回")
    
    if constraint_violations:
        report.append("### 制約違反")
        report.extend(constraint_violations)
    else:
        report.append("### 制約遵守状況")
        report.append("- ✅ すべての発電機で最小運転・停止時間制約が遵守されています")
    
    report.append("")
    
    # 7. 改善提案
    report.append("## 💡 改善提案")
    
    suggestions = []
    
    # 最小台数構成の効率性評価
    if avg_running_units <= len(generators) * 0.6:
        suggestions.append("### 最小台数構成の効果")
        efficiency_rate = (1 - avg_running_units / len(generators)) * 100
        suggestions.append(f"- ✅ **優秀**: 平均{avg_running_units:.1f}台/{len(generators)}台運転で効率性{efficiency_rate:.1f}%を実現")
    
    # 稼働率の低い発電機
    low_utilization_gens = []
    for i, gen in enumerate(generators):
        running_steps = np.sum(output_flags[i, :] == 1)
        utilization = (running_steps / 96) * 100
        if utilization < 20 and not gen.is_must_run:
            low_utilization_gens.append((gen.name, utilization))
    
    if low_utilization_gens:
        suggestions.append("### 稼働率改善")
        for name, util in low_utilization_gens:
            suggestions.append(f"- **{name}**: 稼働率{util:.1f}%と低く、優先順位の見直しを検討")
    
    # コスト効率の改善
    if ed_result and cost_efficiency:
        if len(cost_efficiency) > 1:
            highest_cost_gen = cost_efficiency[-1]  # 最もコストが高い
            lowest_cost_gen = cost_efficiency[0]   # 最もコストが低い
            
            suggestions.append("### コスト効率改善")
            suggestions.append(f"- **{highest_cost_gen[0]}**: 発電コスト{highest_cost_gen[1]:.2f}円/kWh と高く、運用見直しを検討")
            suggestions.append(f"- **{lowest_cost_gen[0]}**: 発電コスト{lowest_cost_gen[1]:.2f}円/kWh と効率的、優先的活用を推奨")
    
    if suggestions:
        report.extend(suggestions)
    else:
        report.append("- ✅ 現在の運用計画は効率的で、最小台数構成が適切に機能しています")
    
    report.append("")
    report.append("---")
    report.append("*このレポートは発電機構成計算ツール（最小台数構成版）により自動生成されました*")
    
    return "\n".join(report)

def create_summary_metrics(uc_result: Dict, ed_result: Dict = None) -> Dict:
    """サマリーメトリクスを作成"""
    if not uc_result:
        return {}
    
    generators = uc_result['generators']
    output_flags = uc_result['output_flags']
    demand_data = uc_result['demand_data']
    
    metrics = {}
    
    # 基本統計
    valid_demands = demand_data[~np.isnan(demand_data)]
    metrics['demand_min'] = valid_demands.min()
    metrics['demand_max'] = valid_demands.max()
    metrics['demand_avg'] = valid_demands.mean()
    metrics['total_capacity'] = sum(gen.max_output for gen in generators)
    metrics['peak_utilization'] = (metrics['demand_max'] / metrics['total_capacity']) * 100
    
    # 運転統計
    total_running_hours = 0
    total_starts = 0
    
    for i, gen in enumerate(generators):
        running_steps = np.sum(output_flags[i, :] == 1)
        running_hours = running_steps * 0.25
        total_running_hours += running_hours
        
        # 起動回数
        start_count = 0
        for j in range(1, 96):
            if output_flags[i, j] == 2 and output_flags[i, j-1] == 0:
                start_count += 1
        total_starts += start_count
    
    metrics['total_running_hours'] = total_running_hours
    metrics['total_starts'] = total_starts
    metrics['avg_running_hours_per_unit'] = total_running_hours / len(generators)
    
    # 最小台数構成指標
    running_units_per_time = []
    for t in range(96):
        running_count = np.sum(output_flags[:, t] == 1)
        running_units_per_time.append(running_count)
    
    metrics['min_running_units'] = min(running_units_per_time)
    metrics['max_running_units'] = max(running_units_per_time)
    metrics['avg_running_units'] = np.mean(running_units_per_time)
    
    # 経済指標
    if ed_result:
        metrics['total_cost'] = ed_result['total_costs']['total_cost']
        metrics['avg_cost_per_hour'] = ed_result['total_costs']['average_cost_per_hour']
        
        total_generation = np.sum(ed_result['power_outputs']) * 0.25
        metrics['total_generation'] = total_generation
        metrics['avg_cost_per_kwh'] = metrics['total_cost'] / total_generation if total_generation > 0 else 0
        
        # λ値統計
        lambda_values = ed_result['lambda_values']
        metrics['lambda_min'] = lambda_values.min()
        metrics['lambda_max'] = lambda_values.max()
        metrics['lambda_avg'] = lambda_values.mean()
        metrics['lambda_std'] = lambda_values.std()
    
    return metrics

def get_default_generator_config(index: int) -> dict:
    """デフォルト発電機設定を取得"""
    defaults = {
        0: {"name": "DG3", "type": "DG", "min": 5000, "max": 10000, "priority": 1, 
            "heat_a": 4.8e-06, "heat_b": 0.1120, "heat_c": 420,
            "startup_cost": 25893, "shutdown_cost": 42084},
        1: {"name": "DG4", "type": "DG", "min": 5000, "max": 10000, "priority": 2, 
            "heat_a": 1.0e-07, "heat_b": 0.1971, "heat_c": 103,
            "startup_cost": 23116, "shutdown_cost": 50116},
        2: {"name": "DG5", "type": "DG", "min": 7500, "max": 15000, "priority": 3, 
            "heat_a": 3.2e-06, "heat_b": 0.1430, "heat_c": 300,
            "startup_cost": 50630, "shutdown_cost": 65729},
        3: {"name": "DG6", "type": "DG", "min": 6000, "max": 12000, "priority": 4, 
            "heat_a": 1.0e-06, "heat_b": 0.1900, "heat_c": 216,
            "startup_cost": 13580, "shutdown_cost": 13097},
        4: {"name": "DG7", "type": "DG", "min": 6000, "max": 12000, "priority": 5, 
            "heat_a": 5.0e-06, "heat_b": 0.1100, "heat_c": 612,
            "startup_cost": 13580, "shutdown_cost": 13097},
        5: {"name": "GT1", "type": "GT", "min": 2500, "max": 5000, "priority": 6, 
            "heat_a": 2.0e-06, "heat_b": 0.1500, "heat_c": 800,
            "startup_cost": 12748, "shutdown_cost": 26643},
        6: {"name": "GT2", "type": "GT", "min": 2500, "max": 5000, "priority": 7, 
            "heat_a": 2.0e-06, "heat_b": 0.1500, "heat_c": 800,
            "startup_cost": 12748, "shutdown_cost": 26643},
        7: {"name": "GT3", "type": "GT", "min": 2500, "max": 5000, "priority": 8, 
            "heat_a": 2.0e-06, "heat_b": 0.1500, "heat_c": 800,
            "startup_cost": 12748, "shutdown_cost": 26643}
    }
    
    if index in defaults:
        return defaults[index]
    else:
        return {"name": f"発電機{index+1}", "type": "DG", "min": 1000, "max": 5000, "priority": index+1,
                "heat_a": 1.0e-06, "heat_b": 0.1500, "heat_c": 300,
                "startup_cost": 10000, "shutdown_cost": 10000}

def main():
    st.markdown('<div class="main-header"><h1>⚡ 発電機構成計算ツール</h1></div>', 
                unsafe_allow_html=True)
    
    # セッション状態初期化
    if 'solver' not in st.session_state:
        st.session_state.solver = UnitCommitmentSolver()
    if 'ed_solver' not in st.session_state:
        st.session_state.ed_solver = EconomicDispatchSolver()
    if 'demand_loaded' not in st.session_state:
        st.session_state.demand_loaded = False
    if 'generators_configured' not in st.session_state:
        st.session_state.generators_configured = False
    
    # サイドバー：計算設定
    with st.sidebar:
        st.header("⚙️ 計算設定")
        
        # Unit Commitment設定
        st.subheader("📋 構成計算設定")
        margin_dg = st.slider("DGマージン率 (%)", 0, 30, 10, key="margin_dg_slider") / 100
        margin_gt = st.slider("GTマージン率 (%)", 0, 30, 15, key="margin_gt_slider") / 100
        stop_margin_dg = st.slider("DG解列マージン率 (%)", 0, 20, 5, key="stop_margin_dg_slider") / 100
        stop_margin_gt = st.slider("GT解列マージン率 (%)", 0, 20, 8, key="stop_margin_gt_slider") / 100
        
        st.session_state.solver.margin_rate_dg = margin_dg
        st.session_state.solver.margin_rate_gt = margin_gt
        st.session_state.solver.stop_margin_rate_dg = stop_margin_dg
        st.session_state.solver.stop_margin_rate_gt = stop_margin_gt
        
        # Economic Dispatch設定
        st.subheader("⚡ 経済配分設定")
        lambda_min = st.number_input("λ最小値", value=0.0, step=1.0, key="lambda_min_input")
        lambda_max = st.number_input("λ最大値", value=100.0, step=1.0, key="lambda_max_input")
        lambda_tolerance = st.number_input("λ許容誤差 (kW)", value=0.001, step=0.001, format="%.3f", key="lambda_tolerance_input")
        
        st.session_state.ed_solver.lambda_min = lambda_min
        st.session_state.ed_solver.lambda_max = lambda_max
        st.session_state.ed_solver.lambda_tolerance = lambda_tolerance
    
    # 1. 需要データアップロード
    st.header("📊 需要予測データアップロード")
    uploaded_file = st.file_uploader("需要予測CSV（96ステップ、15分間隔）", type=['csv'], key="demand_csv_uploader")
    
    if uploaded_file is not None:
        try:
            # エンコーディング自動検出
            encodings = ['utf-8', 'shift-jis', 'cp932', 'euc-jp', 'iso-2022-jp']
            df = None
            
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    st.success(f"✅ エンコーディング: {encoding} で読み込み成功")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                st.error("❌ サポートされているエンコーディングで読み込めませんでした")
            elif len(df.columns) >= 2:
                # データプレビュー
                st.subheader("📋 データプレビュー")
                st.dataframe(df.head(10))
                
                # 列選択
                time_column = st.selectbox("時刻列を選択", df.columns, index=0, key="time_column_select")
                demand_column = st.selectbox("需要データ列を選択", df.columns, index=1, key="demand_column_select")
                
                if len(df) >= 96:
                    try:
                        demand_values = pd.to_numeric(df[demand_column], errors='coerce').values
                        demand_data = demand_values[:96]
                        st.session_state.solver.set_demand_data(demand_data)
                        st.session_state.demand_loaded = True
                        
                        valid_count = np.sum(~np.isnan(demand_data))
                        st.success(f"✅ 需要予測データ読み込み完了（{valid_count}/96ステップ有効）")
                        
                        # 需要データ統計
                        valid_demands = demand_data[~np.isnan(demand_data)]
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
                        st.error(f"❌ 需要データの変換エラー: {e}")
                else:
                    st.error(f"❌ データが96ステップ未満です（現在: {len(df)}ステップ）")
            else:
                st.error("❌ CSVファイルに最低2列（時刻、需要）が必要です")
                
        except Exception as e:
            st.error(f"❌ ファイル読み込みエラー: {e}")
    
    # 2. 発電機設定
    st.header("🔧 発電機設定")
    
    # 発電機数設定
    num_generators = st.number_input("発電機台数", min_value=1, max_value=20, value=8, key="num_generators_input")
    
    # 発電機設定フォーム
    generators_config = []
    
    cols = st.columns(2)
    for i in range(num_generators):
        default_config = get_default_generator_config(i)
        
        with cols[i % 2]:
            with st.expander(f"発電機 {i+1}", expanded=True):
                name = st.text_input(f"名前", value=default_config["name"], key=f"name_{i}")
                unit_type = st.selectbox(f"タイプ", ["DG", "GT"], 
                                       index=0 if default_config["type"] == "DG" else 1, key=f"type_{i}")
                
                # 基本設定
                col1, col2 = st.columns(2)
                with col1:
                    min_output = st.number_input(f"最小出力 (kW)", min_value=0.0, 
                                               value=float(default_config["min"]), key=f"min_{i}")
                    max_output = st.number_input(f"最大出力 (kW)", min_value=0.0, 
                                               value=float(default_config["max"]), key=f"max_{i}")
                    priority = st.number_input(f"優先順位", min_value=1, max_value=100, 
                                             value=default_config["priority"], key=f"priority_{i}")
                
                with col2:
                    min_run_time = st.number_input(f"最小運転時間 (時間)", min_value=0.0, value=2.0, key=f"run_time_{i}")
                    min_stop_time = st.number_input(f"最小停止時間 (時間)", min_value=0.0, value=1.0, key=f"stop_time_{i}")
                    is_must_run = st.checkbox(f"マストラン", key=f"must_run_{i}")
                
                # 燃費特性設定
                st.write("**🔥 燃料消費量特性係数**")
                st.write("*燃料消費量 = a×P² + b×P + c [kL/h]*")
                
                col3, col4, col5 = st.columns(3)
                with col3:
                    heat_rate_a = st.number_input(f"a係数 (2次)", value=default_config["heat_a"], 
                                                step=1e-07, format="%.2e", key=f"heat_a_{i}")
                with col4:
                    heat_rate_b = st.number_input(f"b係数 (1次)", value=default_config["heat_b"], 
                                                step=0.001, format="%.4f", key=f"heat_b_{i}")
                with col5:
                    heat_rate_c = st.number_input(f"c係数 (定数)", value=float(default_config["heat_c"]), 
                                                step=1.0, key=f"heat_c_{i}")
                
                st.write("**💰 燃料単価・起動停止費**")
                col6, col7, col8 = st.columns(3)
                with col6:
                    fuel_price = st.number_input(f"燃料単価 (円/kL)", value=60354.0, step=100.0, key=f"fuel_price_{i}")
                with col7:
                    startup_cost = st.number_input(f"起動費 (円)", value=float(default_config.get("startup_cost", 10000)), step=100.0, key=f"startup_cost_{i}")
                with col8:
                    shutdown_cost = st.number_input(f"停止費 (円)", value=float(default_config.get("shutdown_cost", 10000)), step=100.0, key=f"shutdown_cost_{i}")
                
                generator = GeneratorConfig(
                    name=name,
                    min_output=min_output,
                    max_output=max_output,
                    priority=priority,
                    min_run_time=min_run_time,
                    min_stop_time=min_stop_time,
                    is_must_run=is_must_run,
                    unit_type=unit_type,
                    heat_rate_a=heat_rate_a,
                    heat_rate_b=heat_rate_b,
                    heat_rate_c=heat_rate_c,
                    fuel_price=fuel_price,
                    startup_cost=startup_cost,      # 追加
                    shutdown_cost=shutdown_cost     # 追加
                )
                generators_config.append(generator)
    
    # 発電機設定を保存
    if st.button("発電機設定を保存"):
        st.session_state.solver.generators = generators_config
        st.session_state.generators_configured = True
        st.success("✅ 発電機設定を保存しました")
    
    # 3. 計算実行
    st.header("⚡ 構成計算・経済配分実行")
    
    if st.session_state.demand_loaded and st.session_state.generators_configured:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔧 構成計算のみ実行", type="secondary"):
                with st.spinner("構成計算中..."):
                    try:
                        result = st.session_state.solver.solve_unit_commitment()
                        st.session_state.uc_result = result
                        st.success("✅ 構成計算完了！")
                    except Exception as e:
                        st.error(f"❌ 構成計算エラー: {e}")
        
        with col2:
            if st.button("🚀 構成計算＋経済配分実行", type="primary"):
                with st.spinner("計算中..."):
                    try:
                        # 構成計算
                        uc_result = st.session_state.solver.solve_unit_commitment()
                        st.session_state.uc_result = uc_result
                        
                        # 経済配分計算
                        ed_result = st.session_state.ed_solver.solve_economic_dispatch(
                            uc_result['generators'],
                            uc_result['demand_data'],
                            uc_result['output_flags']
                        )
                        st.session_state.ed_result = ed_result
                        
                        st.success("✅ 構成計算＋経済配分完了！")
                    except Exception as e:
                        st.error(f"❌ 計算エラー: {e}")
    else:
        missing = []
        if not st.session_state.demand_loaded:
            missing.append("需要データ")
        if not st.session_state.generators_configured:
            missing.append("発電機設定")
        st.warning(f"⚠️ 以下の設定が必要です: {', '.join(missing)}")
    
    # 4. 結果表示
    if 'uc_result' in st.session_state and st.session_state.uc_result:
        st.header("📈 計算結果")
        
        uc_result = st.session_state.uc_result
        
        # タブで結果を分離
        if 'ed_result' in st.session_state and st.session_state.ed_result:
            tab1, tab2 = st.tabs(["📊 構成計算結果", "⚡ 経済配分結果"])
            
            with tab1:
                # 構成計算チャート
                fig_uc = create_unit_commitment_chart(uc_result)
                st.plotly_chart(fig_uc, use_container_width=True)

            # 最小台数構成分析
                st.subheader("⚙️ 最小台数構成分析")
                
                running_units_per_time = []
                for t in range(96):
                    running_count = np.sum(uc_result['output_flags'][:, t] == 1)
                    running_units_per_time.append(running_count)
                
                min_running_units = min(running_units_per_time)
                max_running_units = max(running_units_per_time)
                avg_running_units = np.mean(running_units_per_time)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("最小運転台数", f"{min_running_units} 台")
                with col2:
                    st.metric("最大運転台数", f"{max_running_units} 台")
                with col3:
                    st.metric("平均運転台数", f"{avg_running_units:.1f} 台")
                with col4:
                    efficiency = (1 - avg_running_units / len(uc_result['generators'])) * 100
                    st.metric("構成効率", f"{efficiency:.1f}%")
                    
            with tab2:
                # 経済配分チャート
                ed_result = st.session_state.ed_result
                fig_ed = create_economic_dispatch_chart(uc_result, ed_result)
                st.plotly_chart(fig_ed, use_container_width=True)
                
                # 経済配分統計
                st.subheader("💰 経済配分統計")
                
                lambda_stats = {
                    'λ最小値': f"{ed_result['lambda_values'].min():.3f}",
                    'λ最大値': f"{ed_result['lambda_values'].max():.3f}",
                    'λ平均値': f"{ed_result['lambda_values'].mean():.3f}",
                    'λ標準偏差': f"{ed_result['lambda_values'].std():.3f}"
                }
                
                col1, col2, col3, col4 = st.columns(4)
                for i, (key, value) in enumerate(lambda_stats.items()):
                    with [col1, col2, col3, col4][i]:
                        st.metric(key, value)
                
                # 燃料費統計
                if 'total_costs' in ed_result:
                    costs = ed_result['total_costs']
                    st.subheader("🔥 コスト統計")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("総燃料費", f"{costs.get('total_fuel_cost', 0):.0f} 円")
                        st.metric("総起動費", f"{costs.get('total_startup_cost', 0):.0f} 円")
                    with col2:
                        st.metric("総停止費", f"{costs.get('total_shutdown_cost', 0):.0f} 円")
                        st.metric("総コスト", f"{costs['total_cost']:.0f} 円")
                    with col3:                                    # 追加
                        st.metric("平均コスト", f"{costs['average_cost_per_hour']:.0f} 円/時")  # 追加
                        fuel_ratio = (costs.get('total_fuel_cost', 0) / costs['total_cost']) * 100 if costs['total_cost'] > 0 else 0  # 追加
                        st.metric("燃料費比率", f"{fuel_ratio:.1f}%")  # 追加
        else:
            # 構成計算結果のみ
            fig_uc = create_unit_commitment_chart(uc_result)
            st.plotly_chart(fig_uc, use_container_width=True)
        
        # 統計情報
        st.subheader("📊 運転統計")
        
        generators = uc_result['generators']
        output_flags = uc_result['output_flags']
        
        stats_data = []
        for i, gen in enumerate(generators):
            running_steps = np.sum(output_flags[i, :] == 1)
            starting_steps = np.sum(output_flags[i, :] == 2)
            running_hours = running_steps * 0.25
            utilization = (running_steps / 96) * 100
            
            # 起動回数計算
            start_count = 0
            for j in range(1, 96):
                if output_flags[i, j] == 2 and output_flags[i, j-1] == 0:
                    start_count += 1
            
            # 経済配分結果がある場合は出力統計も追加
            if 'ed_result' in st.session_state and st.session_state.ed_result:
                ed_result = st.session_state.ed_result
                power_outputs = ed_result['power_outputs']
                avg_output = np.mean(power_outputs[i, power_outputs[i, :] > 0]) if np.any(power_outputs[i, :] > 0) else 0
                max_output = np.max(power_outputs[i, :])
                total_generation = np.sum(power_outputs[i, :]) * 0.25  # kWh
                
                stats_data.append({
                    '発電機': gen.name,
                    'タイプ': gen.unit_type,
                    '優先順位': gen.priority,
                    '運転時間': f"{running_hours:.1f}h",
                    '稼働率': f"{utilization:.1f}%",
                    '起動回数': start_count,
                    '平均出力': f"{avg_output:.1f} kW",
                    '最大出力': f"{max_output:.1f} kW",
                    '総発電量': f"{total_generation:.1f} kWh",
                    'マストラン': '○' if gen.is_must_run else '×'
                })
            else:
                stats_data.append({
                    '発電機': gen.name,
                    'タイプ': gen.unit_type,
                    '優先順位': gen.priority,
                    '運転時間': f"{running_hours:.1f}h",
                    '稼働率': f"{utilization:.1f}%",
                    '起動回数': start_count,
                    'マストラン': '○' if gen.is_must_run else '×'
                })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # デバッグ情報表示
        if st.checkbox("🔍 詳細計算ログを表示"):
            st.subheader("📝 計算プロセス詳細")
            
            # 時間範囲選択
            start_hour = st.number_input("開始時刻", min_value=0, max_value=23, value=0, step=1, key="debug_start_hour")
            end_hour = st.number_input("終了時刻", min_value=0, max_value=23, value=23, step=1, key="debug_end_hour")
            
            debug_info = uc_result.get('debug_info', [])
            
            for debug_step in debug_info:
                hour = debug_step['hour']
                if start_hour <= hour <= end_hour and debug_step['actions']:
                    with st.expander(f"⏰ {hour:.2f}時 (ステップ {debug_step['time_step']})"):
                        st.write(f"**需要**: {debug_step['demand']:.0f} kW")
                        st.write(f"**将来需要**: {debug_step['future_demand']:.0f} kW")
                        
                        # 最小構成分析
                        if 'capacity_analysis' in debug_step:
                            analysis = debug_step['capacity_analysis']
                            if 'required_units' in analysis:
                                st.write(f"**最小構成**: {', '.join(analysis['required_units'])}")
                        
                        # 経済配分結果があればλ値も表示
                        if 'ed_result' in st.session_state and st.session_state.ed_result:
                            lambda_val = st.session_state.ed_result['lambda_values'][debug_step['time_step']]
                            st.write(f"**λ値**: {lambda_val:.3f}")
                        
                        st.write("**アクション**:")
                        for action in debug_step['actions']:
                            st.write(f"- {action}")
        
        # 計算パラメータ表示
        with st.expander("⚙️ 計算パラメータ"):
            margins = uc_result.get('margins', {})
            col1, col2 = st.columns(2)
            with col1:
                st.write("**構成計算パラメータ**")
                st.write(f"- DG起動マージン: {margins.get('dg_start', 0)*100:.1f}%")
                st.write(f"- GT起動マージン: {margins.get('gt_start', 0)*100:.1f}%")
                st.write(f"- DG解列マージン: {margins.get('dg_stop', 0)*100:.1f}%")
                st.write(f"- GT解列マージン: {margins.get('gt_stop', 0)*100:.1f}%")
            
            with col2:
                if 'ed_result' in st.session_state:
                    st.write("**経済配分パラメータ**")
                    st.write(f"- λ探索範囲: {st.session_state.ed_solver.lambda_min} - {st.session_state.ed_solver.lambda_max}")
                    st.write(f"- λ許容誤差: {st.session_state.ed_solver.lambda_tolerance} kW")
                    st.write(f"- 最大反復回数: {st.session_state.ed_solver.max_iterations}")
        
        # CSVダウンロード
        st.subheader("💾 結果ダウンロード")
        
        # 結果をCSV形式で準備
        time_labels = [f"{(i*15)//60:02d}:{(i*15)%60:02d}" for i in range(96)]
        
        # ダウンロードボタンのレイアウト
        download_col1, download_col2, download_col3 = st.columns(3)
        
        if 'ed_result' in st.session_state and st.session_state.ed_result:
            # 経済配分結果を含むCSV
            ed_result = st.session_state.ed_result
            
            # 発電機出力データ
            output_df = pd.DataFrame(ed_result['power_outputs'].T, columns=[gen.name for gen in generators])
            output_df.insert(0, '時刻', time_labels)
            output_df.insert(1, '需要', uc_result['demand_data'])
            output_df.insert(2, 'λ値', ed_result['lambda_values'])
            
            # 発電機状態データ
            status_df = pd.DataFrame(output_flags.T, columns=[f"{gen.name}_状態" for gen in generators])
            
            # 結合
            result_df = pd.concat([output_df, status_df], axis=1)
            
            # 燃料費データ
            if 'total_costs' in ed_result and 'individual_costs' in ed_result['total_costs']:
                fuel_costs = ed_result['total_costs']['individual_costs']
                fuel_df = pd.DataFrame(fuel_costs.T, columns=[f"{gen.name}_燃料費" for gen in generators])
                result_df = pd.concat([result_df, fuel_df], axis=1)
            
            csv_buffer = io.StringIO()
            result_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            
            with download_col1:
                st.download_button(
                    label="📥 経済配分結果CSV",
                    data=csv_buffer.getvalue(),
                    file_name="economic_dispatch_result.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # λ値のみのダウンロード
            lambda_df = pd.DataFrame({
                '時刻': time_labels,
                'λ値': ed_result['lambda_values']
            })
            
            lambda_buffer = io.StringIO()
            lambda_df.to_csv(lambda_buffer, index=False, encoding='utf-8-sig')
            
            with download_col2:
                st.download_button(
                    label="📊 λ値データCSV",
                    data=lambda_buffer.getvalue(),
                    file_name="lambda_values.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # 詳細レポート
            with download_col3:
                detailed_report = generate_detailed_report(uc_result, ed_result)
                st.download_button(
                    label="📋 詳細レポート",
                    data=detailed_report,
                    file_name="detailed_report.md",
                    mime="text/markdown",
                    use_container_width=True
                )
        else:
            # 構成計算結果のみ
            output_df = pd.DataFrame(output_flags.T, columns=[gen.name for gen in generators])
            output_df.insert(0, '時刻', time_labels)
            output_df.insert(1, '需要', uc_result['demand_data'])
            
            csv_buffer = io.StringIO()
            output_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            
            with download_col1:
                st.download_button(
                    label="📥 構成計算結果CSV",
                    data=csv_buffer.getvalue(),
                    file_name="unit_commitment_result.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # 詳細レポート（構成計算のみ）
            with download_col2:
                detailed_report = generate_detailed_report(uc_result)
                st.download_button(
                    label="📋 詳細レポート",
                    data=detailed_report,
                    file_name="detailed_report.md",
                    mime="text/markdown",
                    use_container_width=True
                )
        
        # レポートプレビュー機能
        st.subheader("📄 レポートプレビュー")
        
        # 実現可能性検証結果の表示
        if 'feasibility_validation' in uc_result:
            validation = uc_result['feasibility_validation']
            
            st.subheader("🔍 構成計算実現可能性検証")
            
            # 全体結果
            if validation['overall_feasible']:
                st.success("✅ 全期間で実現可能な構成計算結果です")
            else:
                st.error(f"❌ {len(validation['infeasible_periods'])}期間で実現不可能な構成があります")
            
            # 統計情報
            stats = validation['statistics']
            val_col1, val_col2, val_col3, val_col4 = st.columns(4)
            
            with val_col1:
                st.metric("総期間", f"{stats['total_periods']} 期間")
            with val_col2:
                st.metric("実現可能期間", f"{stats['feasible_periods']} 期間")
            with val_col3:
                st.metric("実現可能率", f"{stats['feasibility_rate']:.1f}%")
            with val_col4:
                feasible_periods = stats['feasible_periods']
                total_periods = stats['total_periods']
                delta = feasible_periods - (total_periods - feasible_periods)
                st.metric("実現性指標", "良好" if stats['feasibility_rate'] > 95 else "要改善", 
                         delta=f"{delta} 期間差")
            
            # 問題期間の詳細表示
            if validation['infeasible_periods']:
                with st.expander(f"⚠️ 問題期間の詳細 ({len(validation['infeasible_periods'])}件)"):
                    for period in validation['infeasible_periods'][:10]:  # 最初の10件のみ表示
                        st.write(f"**{period['hour']:.2f}時 (ステップ{period['time_step']})**: 需要{period['demand']:.0f}kW")
                        for issue in period['issues']:
                            st.write(f"  - {issue}")
                    
                    if len(validation['infeasible_periods']) > 10:
                        st.write(f"... 他{len(validation['infeasible_periods']) - 10}件")
        
        if st.button("🔍 詳細レポートをプレビュー", use_container_width=True):
            with st.spinner("レポート生成中..."):
                if 'ed_result' in st.session_state and st.session_state.ed_result:
                    report_content = generate_detailed_report(uc_result, st.session_state.ed_result)
                else:
                    report_content = generate_detailed_report(uc_result)
                
                # レポートを表示
                st.markdown(report_content)
        
        # サマリーメトリクス表示
        st.subheader("📊 サマリーメトリクス")
        
        if 'ed_result' in st.session_state and st.session_state.ed_result:
            metrics = create_summary_metrics(uc_result, st.session_state.ed_result)
        else:
            metrics = create_summary_metrics(uc_result)
        
        if metrics:
            # KPIカード表示
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            
            with kpi_col1:
                st.metric(
                    label="需要ピーク", 
                    value=f"{metrics['demand_max']:.0f} kW",
                    delta=f"平均から +{metrics['demand_max'] - metrics['demand_avg']:.0f} kW"
                )
                st.metric(
                    label="総発電容量", 
                    value=f"{metrics['total_capacity']:.0f} kW"
                )
            
            with kpi_col2:
                st.metric(
                    label="容量利用率", 
                    value=f"{metrics['peak_utilization']:.1f}%"
                )
                st.metric(
                    label="平均運転台数", 
                    value=f"{metrics['avg_running_units']:.1f} 台"
                )
            
            with kpi_col3:
                if 'total_cost' in metrics:
                    st.metric(
                        label="総燃料費", 
                        value=f"{metrics['total_cost']:,.0f} 円"
                    )
                    st.metric(
                        label="発電コスト", 
                        value=f"{metrics['avg_cost_per_kwh']:.2f} 円/kWh"
                    )
                else:
                    st.metric(
                        label="総運転時間", 
                        value=f"{metrics['total_running_hours']:.1f} h"
                    )
                    st.metric(
                        label="最小運転台数", 
                        value=f"{metrics['min_running_units']} 台"
                    )
            
            with kpi_col4:
                if 'lambda_avg' in metrics:
                    st.metric(
                        label="平均λ値", 
                        value=f"{metrics['lambda_avg']:.3f}"
                    )
                    st.metric(
                        label="λ値変動幅", 
                        value=f"{metrics['lambda_max'] - metrics['lambda_min']:.3f}"
                    )
                else:
                    st.metric(
                        label="最大運転台数", 
                        value=f"{metrics['max_running_units']} 台"
                    )
                    efficiency = (1 - metrics['avg_running_units'] / len(generators)) * 100
                    st.metric(
                        label="構成効率", 
                        value=f"{efficiency:.1f}%"
                    )

if __name__ == "__main__":
    main()       
