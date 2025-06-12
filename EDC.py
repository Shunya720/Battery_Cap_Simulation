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
                 heat_rate_c: float = 0.0, fuel_price: float = 60354.0):
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

class EconomicDispatchSolver:
    def __init__(self):
        self.lambda_min = 0.0
        self.lambda_max = 100.0
        self.lambda_tolerance = 0.001  # kW
        self.max_iterations = 50
    
    def calculate_output_from_lambda(self, generator: GeneratorConfig, lambda_val: float) -> float:
        """λ値から発電機出力を計算"""
        # λ式: dC/dP = λ より、2*a*P + b = λ/u なので P = (λ/u - b) / (2*a)
        # ここでλ/uは単位変換後のλ値
        lambda_per_fuel = lambda_val / generator.fuel_price * 1000  # 単位調整
        
        if generator.heat_rate_a == 0:
            # 2次係数が0の場合は線形
            if generator.heat_rate_b == 0:
                return generator.min_output
            output = lambda_per_fuel / generator.heat_rate_b
        else:
            output = (lambda_per_fuel - generator.heat_rate_b) / (2 * generator.heat_rate_a)
        
        # 上下限制約
        output = max(generator.min_output, min(generator.max_output, output))
        return output
    
    def calculate_total_power(self, generators: List[GeneratorConfig], lambda_val: float, 
                            status_flags: np.ndarray) -> float:
        """λ値から総出力を計算"""
        total_power = 0.0
        
        for i, gen in enumerate(generators):
            status = status_flags[i]
            
            if status == 0 or status == 2:  # 停止中または起動中
                output = 0.0
            elif status == 1:  # 運転中
                output = self.calculate_output_from_lambda(gen, lambda_val)
            else:
                output = 0.0
            
            total_power += output
        
        return total_power
    
    def find_lambda_binary_search(self, generators: List[GeneratorConfig], 
                                 demand: float, status_flags: np.ndarray) -> float:
        """バイナリサーチでλを探索"""
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
        """経済配分計算"""
        time_steps = len(demand_data)
        gen_count = len(generators)
        
        # λ値と出力の保存配列
        lambda_values = np.zeros(time_steps)
        power_outputs = np.zeros((gen_count, time_steps))
        
        # 各時刻での計算
        for t in range(time_steps):
            demand = demand_data[t]
            status_flags = output_flags[:, t]
            
            # λ探索
            lambda_val = self.find_lambda_binary_search(generators, demand, status_flags)
            lambda_values[t] = lambda_val
            
            # 各発電機の出力計算
            for i, gen in enumerate(generators):
                status = status_flags[i]
                
                if status == 0 or status == 2:  # 停止中または起動中
                    power_outputs[i, t] = 0.0
                elif status == 1:  # 運転中
                    power_outputs[i, t] = self.calculate_output_from_lambda(gen, lambda_val)
                else:
                    power_outputs[i, t] = 0.0
        
        return {
            'lambda_values': lambda_values,
            'power_outputs': power_outputs,
            'total_costs': self.calculate_fuel_costs(generators, power_outputs, output_flags)
        }
    
    def calculate_fuel_costs(self, generators: List[GeneratorConfig], 
                           power_outputs: np.ndarray, output_flags: np.ndarray) -> Dict:
        """燃料費計算"""
        time_steps = power_outputs.shape[1]
        gen_count = len(generators)
        
        fuel_costs = np.zeros((gen_count, time_steps))
        total_fuel_cost = 0.0
        
        for i, gen in enumerate(generators):
            for t in range(time_steps):
                if output_flags[i, t] == 1:  # 運転中のみ
                    power = power_outputs[i, t]
                    # 燃料費 = (a*P^2 + b*P + c) * u * 0.25 (15分間隔なので1/4時間)
                    fuel_consumption = gen.heat_rate_a * power**2 + gen.heat_rate_b * power + gen.heat_rate_c
                    cost = fuel_consumption * gen.fuel_price * 0.25
                    fuel_costs[i, t] = cost
                    total_fuel_cost += cost
        
        return {
            'individual_costs': fuel_costs,
            'total_cost': total_fuel_cost,
            'average_cost_per_hour': total_fuel_cost / 24
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
        self.short_stop_threshold = 12  # 3時間断面解列判定用（15分×12=3時間）
        
    def add_generator(self, gen_config: GeneratorConfig):
        self.generators.append(gen_config)
        
    def set_demand_data(self, demand_data: np.ndarray):
        self.demand_data = demand_data[:self.time_steps]
        
    def get_time_based_margin(self, time_step: int) -> Tuple[float, float]:
        """時間帯別マージン設定（17:00-22:00がピーク）"""
        hour = (time_step * 0.25) % 24  # 15分間隔から時間を計算
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
    
    def solve_unit_commitment(self) -> Dict:
        """発電機構成計算のメイン処理"""
        if self.demand_data is None or len(self.generators) == 0:
            return {}
            
        # 発電機を優先順位でソート
        sorted_generators = sorted(self.generators, key=lambda x: x.priority)
        gen_count = len(sorted_generators)
        
        # 状態配列初期化 (0:停止, 1:運転, 2:起動中)
        output_flags = np.zeros((gen_count, self.time_steps), dtype=int)
        prev_flags = np.zeros(gen_count, dtype=int)
        runtime_steps = np.zeros(gen_count, dtype=int)
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
            
            # 時間帯別マージン
            margin_dg, margin_gt = self.get_time_based_margin(i)
            stop_margin_dg, stop_margin_gt = self.get_stop_margin(i)
            
            target_flags = np.zeros(gen_count, dtype=int)
            total_cap = 0
            
            step_debug = {
                'time_step': i,
                'hour': current_hour,
                'demand': demand,
                'future_demand': future_demand,
                'actions': []
            }
            
            # === 起動判定処理 ===
            for j, gen in enumerate(sorted_generators):
                margin = margin_gt if gen.unit_type == "GT" else margin_dg
                
                # マストランユニットは常時運転
                if gen.is_must_run:
                    target_flags[j] = 1
                    total_cap += gen.max_output
                    continue
                
                # 最小停止時間チェック
                if prev_flags[j] == 0:
                    # 最後に運転していた時刻を探索
                    last_run_step = -1
                    for back in range(i - 1, -1, -1):
                        if output_flags[j, back] == 1:
                            last_run_step = back
                            break
                    
                    if last_run_step > 0 and (i - last_run_step) < min_stop_steps[j]:
                        # 最小停止時間未達の場合、需要不足でなければスキップ
                        if total_cap >= future_demand * (1 + margin):
                            continue
                
                # 起動判定条件
                started = False
                
                # 条件1: 急上昇によるGT起動
                if i >= 1 and gen.unit_type == "GT":
                    prev_demand = self.demand_data[i - 1]
                    if (demand - prev_demand) > 3000 and total_cap < demand * (1 + margin):
                        target_flags[j] = 1
                        total_cap += gen.max_output
                        started = True
                
                # 条件2: 通常起動（将来需要予測ベース）
                if not started:
                    if total_cap < future_demand * (1 + margin):
                        target_flags[j] = 1
                        total_cap += gen.max_output
                        started = True
                
                # 条件3: 緊急起動（予備力不足）
                if not started:
                    reserve_margin = total_cap - demand
                    if reserve_margin < 1000:
                        target_flags[j] = 1
                        total_cap += gen.max_output
                        started = True
            
            # 初回断面の処理
            if i == 0:
                for j in range(gen_count):
                    if sorted_generators[j].is_must_run:
                        output_flags[j, i] = 1
                        prev_flags[j] = 1
                    else:
                        output_flags[j, i] = target_flags[j]
                        prev_flags[j] = target_flags[j]
                        if target_flags[j] == 1:
                            last_start[j] = i
                continue
            
            # === 解列判定処理 ===
            final_flags = target_flags.copy()
            lower_sum = 0
            upper_sum = 0
            
            # 現在の運転中ユニットの上限・下限出力を計算
            for j, gen in enumerate(sorted_generators):
                if prev_flags[j] == 1:
                    lower_sum += gen.min_output
                    upper_sum += gen.max_output
            
            for j, gen in enumerate(sorted_generators):
                stop_margin = stop_margin_gt if gen.unit_type == "GT" else stop_margin_dg
                
                # マストランは解列しない
                if gen.is_must_run:
                    final_flags[j] = 1
                    continue
                
                # 起動判定されたユニットは解列しない
                if target_flags[j] == 1:
                    final_flags[j] = 1
                    continue
                
                # 現在運転中または起動中の場合
                if prev_flags[j] in [1, 2]:
                    # 最小運転時間チェック
                    active_steps = 0
                    for back in range(i - 1, -1, -1):
                        if output_flags[j, back] == 1:
                            active_steps += 1
                        else:
                            break
                    
                    can_stop = active_steps >= min_run_steps[j]
                    
                    # 停止中ユニットの最小停止時間チェック
                    is_still_cooldown = False
                    if prev_flags[j] == 0:
                        stop_step = -1
                        for back in range(i - 1, -1, -1):
                            if output_flags[j, back] == 1:
                                stop_step = back
                                break
                        
                        if stop_step > 0:
                            stop_duration = i - stop_step
                            if stop_duration <= min_stop_steps[j]:
                                is_still_cooldown = True
                    
                    if not can_stop or is_still_cooldown:
                        final_flags[j] = 1
                    else:
                        # 将来需要に対する供給余剰判定
                        if (upper_sum - gen.max_output) > future_demand * (1 + stop_margin):
                            final_flags[j] = 0
                            lower_sum -= gen.min_output
                            upper_sum -= gen.max_output
                            step_debug['actions'].append(
                                f"{gen.name}: 解列 (供給余剰: {upper_sum - gen.max_output:.0f} > {future_demand * (1 + stop_margin):.0f})"
                            )
                        else:
                            final_flags[j] = 1
                            step_debug['actions'].append(f"{gen.name}: 運転継続 (供給不足のため)")
                else:
                    final_flags[j] = 0
            
            # 強制解列（下限出力が需要を上回る場合）
            if lower_sum > demand:
                for j in range(gen_count - 1, -1, -1):  # 優先順位の低い順から
                    if (prev_flags[j] == 1 and final_flags[j] == 1 and 
                        not sorted_generators[j].is_must_run):
                        final_flags[j] = 0
                        step_debug['actions'].append(
                            f"{sorted_generators[j].name}: 強制解列 (下限出力過剰: {lower_sum:.0f} > {demand:.0f})"
                        )
                        break
            
            debug_info.append(step_debug)
            
            # === 状態遷移処理 ===
            for j, gen in enumerate(sorted_generators):
                if gen.is_must_run:
                    output_flags[j, i] = 1
                    prev_flags[j] = 1
                    continue
                
                # 状態0 → 状態2への遷移（起動）
                if prev_flags[j] == 0 and final_flags[j] == 1:
                    # 最小停止時間の最終チェック
                    stop_count = 0
                    for back in range(i - 1, -1, -1):
                        if output_flags[j, back] == 0:
                            stop_count += 1
                        else:
                            break
                    
                    if stop_count < min_stop_steps[j]:
                        output_flags[j, i] = 0
                        prev_flags[j] = 0
                    else:
                        # 起動処理（状態2 → 2 → 1）
                        output_flags[j, i] = 2
                        if i + 1 < self.time_steps:
                            output_flags[j, i + 1] = 2
                        if i + 2 < self.time_steps:
                            output_flags[j, i + 2] = 1
                        last_start[j] = i
                        prev_flags[j] = 2
                
                # 状態2の継続処理
                elif prev_flags[j] == 2:
                    # 3断面続いたら状態1へ
                    if (i >= 3 and output_flags[j, i-1] == 2 and 
                        output_flags[j, i-2] == 2 and output_flags[j, i-3] == 2):
                        output_flags[j, i] = 1
                        prev_flags[j] = 1
                    else:
                        output_flags[j, i] = 2
                        prev_flags[j] = 2
                
                # 状態1 → 停止への遷移
                elif prev_flags[j] == 1 and final_flags[j] == 0:
                    output_flags[j, i] = 2
                    if i + 1 < self.time_steps:
                        output_flags[j, i + 1] = 0
                    prev_flags[j] = 0
                
                # 状態維持
                else:
                    output_flags[j, i] = prev_flags[j]
                    prev_flags[j] = output_flags[j, i]
        
        # 結果をまとめて返す
        result = {
            'generators': sorted_generators,
            'output_flags': output_flags,
            'demand_data': self.demand_data,
            'time_steps': self.time_steps,
            'debug_info': debug_info,
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
    
    # 経済配分結果がある場合
    if ed_result:
        total_cost = ed_result['total_costs']['total_cost']
        avg_cost_per_hour = ed_result['total_costs']['average_cost_per_hour']
        total_generation = np.sum(ed_result['power_outputs']) * 0.25  # kWh
        avg_cost_per_kwh = total_cost / total_generation if total_generation > 0 else 0
        
        report.append(f"- **総燃料費**: {total_cost:,.0f} 円")
        report.append(f"- **平均燃料費**: {avg_cost_per_hour:,.0f} 円/時")
        report.append(f"- **総発電量**: {total_generation:,.0f} kWh")
        report.append(f"- **平均発電コスト**: {avg_cost_per_kwh:.2f} 円/kWh")
    
    report.append("")
    
    # 2. 発電機別運転実績
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
        
        report.append(f"### {gen.name} ({gen.unit_type})")
        report.append(f"- **容量**: {gen.min_output:.0f} - {gen.max_output:.0f} kW")
        report.append(f"- **優先順位**: {gen.priority}")
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
    
    # 3. 時間帯別分析
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
    
    # 4. 経済性分析（経済配分結果がある場合）
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
        
        # 時間帯別コスト
        peak_costs = []
        off_peak_costs = []
        
        for i in range(96):
            hour_cost = np.sum(fuel_costs[:, i]) * 4  # 15分→1時間換算
            if i in peak_hours:
                peak_costs.append(hour_cost)
            else:
                off_peak_costs.append(hour_cost)
        
        avg_peak_cost = np.mean(peak_costs) if peak_costs else 0
        avg_off_peak_cost = np.mean(off_peak_costs) if off_peak_costs else 0
        
        report.append("#### 時間帯別燃料費")
        report.append(f"- **ピーク時平均**: {avg_peak_cost:,.0f} 円/時")
        report.append(f"- **オフピーク時平均**: {avg_off_peak_cost:,.0f} 円/時")
        report.append("")
    
    # 5. 運用制約分析
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
    
    # 6. 予備力分析
    report.append("## 🔋 予備力分析")
    
    reserve_margins = []
    for t in range(96):
        available_capacity = 0
        for i, gen in enumerate(generators):
            if output_flags[i, t] == 1:  # 運転中
                available_capacity += gen.max_output
        
        reserve = available_capacity - demand_data[t]
        reserve_rate = (reserve / demand_data[t]) * 100 if demand_data[t] > 0 else 0
        reserve_margins.append((reserve, reserve_rate))
    
    reserves = [r[0] for r in reserve_margins]
    reserve_rates = [r[1] for r in reserve_margins]
    
    min_reserve = min(reserves)
    max_reserve = max(reserves)
    avg_reserve = np.mean(reserves)
    min_reserve_rate = min(reserve_rates)
    avg_reserve_rate = np.mean(reserve_rates)
    
    report.append(f"- **最小予備力**: {min_reserve:.0f} kW ({min_reserve_rate:.1f}%)")
    report.append(f"- **最大予備力**: {max_reserve:.0f} kW")
    report.append(f"- **平均予備力**: {avg_reserve:.0f} kW ({avg_reserve_rate:.1f}%)")
    
    # 予備力不足の警告
    if min_reserve < 1000:
        report.append("- ⚠️ **警告**: 予備力が1,000kWを下回る時間帯があります")
    
    report.append("")
    
    # 7. 改善提案
    report.append("## 💡 改善提案")
    
    suggestions = []
    
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
        highest_cost_gen = cost_efficiency[-1]  # 最もコストが高い
        lowest_cost_gen = cost_efficiency[0]   # 最もコストが低い
        
        if len(cost_efficiency) > 1:
            suggestions.append("### コスト効率改善")
            suggestions.append(f"- **{highest_cost_gen[0]}**: 発電コスト{highest_cost_gen[1]:.2f}円/kWh と高く、運用見直しを検討")
            suggestions.append(f"- **{lowest_cost_gen[0]}**: 発電コスト{lowest_cost_gen[1]:.2f}円/kWh と効率的、優先的活用を推奨")
    
    # 予備力の改善
    if min_reserve_rate < 10:
        suggestions.append("### 予備力改善")
        suggestions.append(f"- 最小予備力率{min_reserve_rate:.1f}%と低く、信頼性向上のため追加容量確保を検討")
    
    if suggestions:
        report.extend(suggestions)
    else:
        report.append("- ✅ 現在の運用計画は効率的で、大きな改善点は見つかりませんでした")
    
    report.append("")
    report.append("---")
    report.append("*このレポートは発電機構成計算ツールにより自動生成されました*")
    
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
            "heat_a": 4.8e-06, "heat_b": 0.1120, "heat_c": 420},
        1: {"name": "DG4", "type": "DG", "min": 5000, "max": 10000, "priority": 2, 
            "heat_a": 1.0e-07, "heat_b": 0.1971, "heat_c": 103},
        2: {"name": "DG5", "type": "DG", "min": 7500, "max": 15000, "priority": 3, 
            "heat_a": 3.2e-06, "heat_b": 0.1430, "heat_c": 300},
        3: {"name": "DG6", "type": "DG", "min": 6000, "max": 12000, "priority": 4, 
            "heat_a": 1.0e-06, "heat_b": 0.1900, "heat_c": 216},
        4: {"name": "DG7", "type": "DG", "min": 6000, "max": 12000, "priority": 5, 
            "heat_a": 5.0e-06, "heat_b": 0.1100, "heat_c": 612},
        5: {"name": "GT1", "type": "GT", "min": 3000, "max": 10000, "priority": 6, 
            "heat_a": 2.0e-06, "heat_b": 0.1500, "heat_c": 800},
        6: {"name": "GT2", "type": "GT", "min": 3000, "max": 10000, "priority": 7, 
            "heat_a": 2.0e-06, "heat_b": 0.1500, "heat_c": 800},
        7: {"name": "GT3", "type": "GT", "min": 3000, "max": 10000, "priority": 8, 
            "heat_a": 2.0e-06, "heat_b": 0.1500, "heat_c": 800}
    }
    
    if index in defaults:
        return defaults[index]
    else:
        return {"name": f"発電機{index+1}", "type": "DG", "min": 1000, "max": 5000, "priority": index+1,
                "heat_a": 1.0e-06, "heat_b": 0.1500, "heat_c": 300}

def main():
    """デフォルト発電機設定を取得"""
    defaults = {
        0: {"name": "DG3", "type": "DG", "min": 5000, "max": 10000, "priority": 1, 
            "heat_a": 4.8e-06, "heat_b": 0.1120, "heat_c": 420},
        1: {"name": "DG4", "type": "DG", "min": 5000, "max": 10000, "priority": 2, 
            "heat_a": 1.0e-07, "heat_b": 0.1971, "heat_c": 103},
        2: {"name": "DG5", "type": "DG", "min": 7500, "max": 15000, "priority": 3, 
            "heat_a": 3.2e-06, "heat_b": 0.1430, "heat_c": 300},
        3: {"name": "DG6", "type": "DG", "min": 6000, "max": 12000, "priority": 4, 
            "heat_a": 1.0e-06, "heat_b": 0.1900, "heat_c": 216},
        4: {"name": "DG7", "type": "DG", "min": 6000, "max": 12000, "priority": 5, 
            "heat_a": 5.0e-06, "heat_b": 0.1100, "heat_c": 612},
        5: {"name": "GT1", "type": "GT", "min": 3000, "max": 10000, "priority": 6, 
            "heat_a": 2.0e-06, "heat_b": 0.1500, "heat_c": 800},
        6: {"name": "GT2", "type": "GT", "min": 3000, "max": 10000, "priority": 7, 
            "heat_a": 2.0e-06, "heat_b": 0.1500, "heat_c": 800},
        7: {"name": "GT3", "type": "GT", "min": 3000, "max": 10000, "priority": 8, 
            "heat_a": 2.0e-06, "heat_b": 0.1500, "heat_c": 800}
    }
    
    if index in defaults:
        return defaults[index]
    else:
        return {"name": f"発電機{index+1}", "type": "DG", "min": 1000, "max": 5000, "priority": index+1,
                "heat_a": 1.0e-06, "heat_b": 0.1500, "heat_c": 300}

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
                
                # 燃料単価
                fuel_price = st.number_input(f"燃料単価 (円/kL)", value=60354.0, step=100.0, key=f"fuel_price_{i}")
                
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
                    fuel_price=fuel_price
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
                    st.subheader("🔥 燃料費統計")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("総燃料費", f"{costs['total_cost']:.0f} 円")
                    with col2:
                        st.metric("平均燃料費", f"{costs['average_cost_per_hour']:.0f} 円/時")
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
                    label="総起動回数", 
                    value=f"{metrics['total_starts']} 回"
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
                        label="平均運転時間", 
                        value=f"{metrics['avg_running_hours_per_unit']:.1f} h/台"
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
                        label="需要変動", 
                        value=f"{metrics['demand_max'] - metrics['demand_min']:.0f} kW"
                    )
                    st.metric(
                        label="平均需要", 
                        value=f"{metrics['demand_avg']:.0f} kW"
                    )
                    

if __name__ == "__main__":
    main()