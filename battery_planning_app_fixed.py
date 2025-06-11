"""
バッテリー計画専用アプリケーション
コアロジックを参照してバッテリー制御計画を実行
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

# コアロジック読み込み
from battery_core_logic import (
    BatteryControlEngine, ERROR_DATA, create_time_series,
    OPTIMIZATION_AVAILABLE
)


def update_sidebar_with_optimized_params():
    """最適化されたパラメータでサイドバーを更新"""
    
    # 強制更新パラメータがある場合（最優先）
    if hasattr(st.session_state, 'force_update_params'):
        params = st.session_state.force_update_params
        return params
    
    # セッション状態に最適パラメータがある場合
    if hasattr(st.session_state, 'optimized_params'):
        return st.session_state.optimized_params
    
    return None


def main():
    st.title("🔋 バッテリー制御計画システム")
    st.write("**需要予測に基づく分離制御最適化とリアルタイムシミュレーション**")
    
    # サイドバー設定
    st.sidebar.header("⚙️ 制御パラメータ")
    
    # 最適化パラメータの自動適用チェック
    optimized_params = update_sidebar_with_optimized_params()
    
    # 最適化パラメータ適用状況の表示
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 パラメータ適用状況")
    
    # optimized_paramsの存在確認
    has_optimized_params = (optimized_params is not None and 
                           isinstance(optimized_params, dict) and 
                           'peak_percentile' in optimized_params)
    
    if has_optimized_params:
        # 適用済みフラグの確認
        is_applied = (hasattr(st.session_state, 'params_applied_flag') and 
                     st.session_state.params_applied_flag) or \
                    (hasattr(st.session_state, 'auto_applied') and 
                     st.session_state.auto_applied)
        
        if is_applied:
            st.sidebar.success("✅ 最適パラメータが適用されました")
        else:
            st.sidebar.info("🎯 最適パラメータが取得されました")
            st.sidebar.write("「最適パラメータを適用」ボタンで適用してください")
        
        st.sidebar.info(f"""
        **最適パラメータ（分離制御）:**
        • ピーク閾値: {optimized_params['peak_percentile']:.1f}%
        • ボトム閾値: {optimized_params['bottom_percentile']:.1f}%
        • ピーク制御比率: {optimized_params.get('peak_power_ratio', 1.0):.2f}
        • ボトム制御比率: {optimized_params.get('bottom_power_ratio', 1.0):.2f}
        • 平準化比率: {optimized_params['flattening_power_ratio']:.2f}
        """)
        
        # デフォルト値を最適化結果に設定
        default_peak = optimized_params['peak_percentile']
        default_bottom = optimized_params['bottom_percentile']
        default_peak_ratio = optimized_params.get('peak_power_ratio', 1.0)
        default_bottom_ratio = optimized_params.get('bottom_power_ratio', 1.0)
        default_flat_ratio = optimized_params['flattening_power_ratio']
        default_cycle = optimized_params['daily_cycle_target']
    else:
        st.sidebar.info("ℹ️ 手動設定モード")
        st.sidebar.write("自動最適化を実行して最適パラメータを取得できます")
        
        # 通常のデフォルト値
        default_peak = 80
        default_bottom = 20
        default_peak_ratio = 1.0
        default_bottom_ratio = 1.0
        default_flat_ratio = 0.3
        default_cycle = 48000
    
    # パラメータ設定
    st.sidebar.subheader("🏔️ ピーク・ボトム制御設定")
    peak_percentile = st.sidebar.slider(
        "ピーク判定閾値（上位%）", 
        min_value=50, max_value=100, value=int(default_peak), step=5,
        help="上位何%をピークとして扱うか（分離制御で放電）"
    )
    
    bottom_percentile = st.sidebar.slider(
        "ボトム判定閾値（下位%）", 
        min_value=0, max_value=50, value=int(default_bottom), step=5,
        help="下位何%をボトムとして扱うか（分離制御で充電）"
    )
    
    # ピーク・ボトム制御比率を分離
    st.sidebar.markdown("**⚡ 分離制御強度設定**")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        peak_power_ratio = st.sidebar.slider(
            "ピーク制御強度", 
            min_value=0.1, max_value=1.0, value=default_peak_ratio, step=0.1,
            help="ピーク時間帯での最大出力比率（放電）"
        )
    
    with col2:
        bottom_power_ratio = st.sidebar.slider(
            "ボトム制御強度", 
            min_value=0.1, max_value=1.0, value=default_bottom_ratio, step=0.1,
            help="ボトム時間帯での最大出力比率（充電）"
        )
    
    flattening_power_ratio = st.sidebar.slider(
        "平準化制御最大出力比率", 
        min_value=0.1, max_value=1.0, value=default_flat_ratio, step=0.1,
        help="その他時間帯での平準化制御時の最大出力比率"
    )
    
    daily_cycle_target = st.sidebar.number_input(
        "1日サイクル目標 (kWh)", 
        value=int(default_cycle), min_value=10000, max_value=100000, step=1000,
        help="参考値（分離制御では実際の出力がこの値と異なる場合があります）"
    )
    
    # 蓄電池基本設定
    st.sidebar.subheader("🔋 蓄電池基本設定")
    battery_capacity = st.sidebar.number_input("蓄電池容量 (kWh)", value=48000, min_value=10000)
    max_power = st.sidebar.number_input("最大出力 (15分エネルギー値)", value=3000, min_value=100)
    efficiency = st.sidebar.number_input("効率", value=1.0, min_value=0.1, max_value=1.0)
    initial_soc = st.sidebar.number_input("初期SOC (%)", value=10.0, min_value=0.0, max_value=100.0, step=0.1)
    
    # リセットボタン（最適化パラメータをクリア）
    if has_optimized_params:
        st.sidebar.markdown("---")
        if st.sidebar.button("🔄 手動設定モードに戻る"):
            # 最適化関連のセッション状態をクリア
            for key in ['optimized_params', 'auto_applied', 'params_applied_flag', 'force_update_params']:
                if hasattr(st.session_state, key):
                    delattr(st.session_state, key)
            
            st.sidebar.success("🔄 手動設定モードに戻りました")
            st.rerun()
    
    # バッテリー制御エンジン初期化
    if 'battery_engine' not in st.session_state:
        st.session_state.battery_engine = BatteryControlEngine(
            battery_capacity, max_power, efficiency, initial_soc
        )
    
    # セッション状態初期化
    if 'original_forecast' not in st.session_state:
        st.session_state.original_forecast = None
    if 'simulation_started' not in st.session_state:
        st.session_state.simulation_started = False
    
    # CSVアップロード
    st.header("1. 📊 需要予測データアップロード")
    uploaded_file = st.file_uploader("需要予測CSV（96ステップ、15分間隔）", type=['csv'])
    
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
                time_column = st.selectbox("時刻列を選択", df.columns, index=0)
                demand_column = st.selectbox("需要データ列を選択", df.columns, index=1)
                
                if len(df) >= 96:
                    try:
                        demand_values = pd.to_numeric(df[demand_column], errors='coerce').values
                        st.session_state.original_forecast = demand_values[:96]
                        st.session_state.battery_engine.original_forecast = demand_values[:96]
                        valid_count = np.sum(~np.isnan(st.session_state.original_forecast))
                        
                        st.success(f"✅ 需要予測データ読み込み完了（{valid_count}/96ステップ有効）")
                        
                        # 需要データ統計
                        valid_demands = demand_values[~np.isnan(demand_values)]
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
    
    if st.session_state.original_forecast is not None:
        
        # 自動最適化セクション
        st.header("2. 🤖 自動最適化（需要ギザギザ最小化特化・分離制御対応）")
        st.write("**サイクル制約を満たしながら需要の滑らかさを最大化するパラメータを自動探索**")
        
        with st.expander("⚙️ 自動最適化設定", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                auto_cycle_target = st.number_input(
                    "サイクル目標 (kWh)", 
                    value=48000, min_value=30000, max_value=70000, step=1000,
                    help="自動最適化で達成したいサイクル目標"
                )
                
                auto_cycle_tolerance = st.number_input(
                    "サイクル許容範囲 (kWh)", 
                    value=1500, min_value=500, max_value=5000, step=500,
                    help="サイクル目標からの許容偏差"
                )
            
            with col2:
                if OPTIMIZATION_AVAILABLE:
                    optimization_method = st.selectbox(
                        "最適化手法",
                        ["optuna", "differential_evolution"],
                        index=0,
                        help="最適化アルゴリズムの選択"
                    )
                else:
                    st.error("⚠️ OptunaまたはScipyが見つかりません。最適化ライブラリをインストールしてください: pip install optuna scipy")
                    optimization_method = None
                
                n_trials = st.slider(
                    "試行回数（Optuna）",
                    min_value=50, max_value=300, value=100, step=10,
                    help="Optuna使用時の最適化試行回数"
                )
        
        # 自動最適化実行ボタン
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🚀 分離制御自動最適化実行", use_container_width=True) and optimization_method:
                
                # プログレスバー表示
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("🔍 分離制御最適化システム初期化中...")
                    progress_bar.progress(10)
                    
                    status_text.text("📊 需要データ分析中...")
                    progress_bar.progress(20)
                    
                    status_text.text(f"⚡ 分離制御最適化実行中（{optimization_method}）...")
                    progress_bar.progress(30)
                    
                    # 最適化実行
                    optimization_result = st.session_state.battery_engine.run_optimization(
                        st.session_state.original_forecast,
                        cycle_target=auto_cycle_target,
                        cycle_tolerance=auto_cycle_tolerance,
                        method=optimization_method,
                        n_trials=n_trials
                    )
                    
                    progress_bar.progress(80)
                    status_text.text("📋 最適化結果分析中...")
                    
                    if optimization_result and optimization_result.get('best_params'):
                        best_params = optimization_result['best_params']
                        
                        # セッション状態に最適パラメータを保存
                        st.session_state.optimized_params = best_params
                        st.session_state.auto_applied = True
                        st.session_state.optimization_result = optimization_result
                        
                        progress_bar.progress(100)
                        status_text.text("✅ 分離制御最適化完了！")
                        
                        # 成功メッセージ
                        st.success("🎉 分離制御最適化が完了しました！サイドバーのパラメータが自動更新されています。")
                        
                        # 1秒待機してから画面更新
                        time.sleep(1)
                        st.rerun()
                    
                    else:
                        st.error("❌ 最適化に失敗しました。パラメータ範囲やサイクル制約を調整してください。")
                
                except Exception as e:
                    st.error(f"❌ 最適化エラー: {e}")
                    import traceback
                    st.text(traceback.format_exc())
                
                finally:
                    progress_bar.empty()
                    status_text.empty()
        
        # 最適化結果の表示
        if hasattr(st.session_state, 'optimization_result') and st.session_state.optimization_result:
            result = st.session_state.optimization_result
            
            st.subheader("📊 最適化結果")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "最適化スコア", 
                    f"{result.get('best_score', 0):.3f}",
                    help="需要の滑らかさ指標（小さいほど良い）"
                )
            
            with col2:
                best_params = result.get('best_params', {})
                actual_cycle = best_params.get('actual_cycle', 0)
                target_cycle = best_params.get('daily_cycle_target', auto_cycle_target)
                cycle_diff = abs(actual_cycle - target_cycle)
                
                st.metric(
                    "サイクル達成", 
                    f"{actual_cycle:.0f}kWh",
                    delta=f"目標差: {cycle_diff:.0f}kWh"
                )
            
            with col3:
                peak_reduction = best_params.get('peak_reduction', 0)
                st.metric(
                    "ピーク削減", 
                    f"{peak_reduction:.1f}kW",
                    help="最大需要の削減量"
                )
            
            with col4:
                range_reduction = best_params.get('range_reduction', 0)
                st.metric(
                    "需要幅改善", 
                    f"{range_reduction:.1f}kW",
                    help="需要の最大-最小の改善量"
                )
            
            # 最適パラメータの詳細表示
            with st.expander("🔍 最適パラメータ詳細", expanded=False):
                param_df = pd.DataFrame([
                    {"パラメータ": "ピーク閾値", "値": f"{best_params.get('peak_percentile', 0):.1f}%"},
                    {"パラメータ": "ボトム閾値", "値": f"{best_params.get('bottom_percentile', 0):.1f}%"},
                    {"パラメータ": "ピーク制御比率", "値": f"{best_params.get('peak_power_ratio', 1.0):.2f}"},
                    {"パラメータ": "ボトム制御比率", "値": f"{best_params.get('bottom_power_ratio', 1.0):.2f}"},
                    {"パラメータ": "平準化比率", "値": f"{best_params.get('flattening_power_ratio', 0.3):.2f}"},
                    {"パラメータ": "サイクル目標", "値": f"{best_params.get('daily_cycle_target', 48000):,}kWh"}
                ])
                st.dataframe(param_df, use_container_width=True, hide_index=True)
        
        # 制御シミュレーション
        st.header("3. ⚡ 制御シミュレーション実行")
        
        # シミュレーション実行ボタン
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎯 分離制御シミュレーション実行", use_container_width=True):
                
                # プログレスバー表示
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("🔄 バッテリー制御エンジン更新中...")
                    progress_bar.progress(20)
                    
                    # エンジンの設定更新
                    st.session_state.battery_engine.update_settings(
                        battery_capacity, max_power, efficiency, initial_soc
                    )
                    
                    status_text.text("⚡ 分離制御シミュレーション実行中...")
                    progress_bar.progress(40)
                    
                    # 制御シミュレーション実行
                    simulation_result = st.session_state.battery_engine.run_control_simulation(
                        st.session_state.original_forecast,
                        peak_percentile=peak_percentile,
                        bottom_percentile=bottom_percentile,
                        peak_power_ratio=peak_power_ratio,
                        bottom_power_ratio=bottom_power_ratio,
                        flattening_power_ratio=flattening_power_ratio,
                        daily_cycle_target=daily_cycle_target
                    )
                    
                    progress_bar.progress(80)
                    status_text.text("📊 結果分析中...")
                    
                    # 結果をセッション状態に保存
                    st.session_state.simulation_result = simulation_result
                    st.session_state.simulation_started = True
                    
                    progress_bar.progress(100)
                    status_text.text("✅ シミュレーション完了！")
                    
                    st.success("🎉 分離制御シミュレーションが完了しました！")
                    time.sleep(1)
                    st.rerun()
                
                except Exception as e:
                    st.error(f"❌ シミュレーションエラー: {e}")
                    import traceback
                    st.text(traceback.format_exc())
                
                finally:
                    progress_bar.empty()
                    status_text.empty()
        
        # シミュレーション結果の表示
        if st.session_state.simulation_started and hasattr(st.session_state, 'simulation_result'):
            result = st.session_state.simulation_result
            
            st.header("4. 📊 シミュレーション結果")
            
            # KPI表示
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                original_peak = np.max(st.session_state.original_forecast)
                controlled_peak = np.max(result['demand_after_battery'])
                peak_reduction = original_peak - controlled_peak
                
                st.metric(
                    "ピーク削減",
                    f"{peak_reduction:.1f}kW",
                    delta=f"{controlled_peak:.0f}kW (制御後)",
                    delta_color="inverse"
                )
            
            with col2:
                total_discharge = -np.sum(result['battery_output'][result['battery_output'] < 0])
                st.metric(
                    "総放電量",
                    f"{total_discharge:.0f}kWh",
                    help="1日の総放電エネルギー"
                )
            
            with col3:
                total_charge = np.sum(result['battery_output'][result['battery_output'] > 0])
                st.metric(
                    "総充電量",
                    f"{total_charge:.0f}kWh",
                    help="1日の総充電エネルギー"
                )
            
            with col4:
                cycle_count = min(total_discharge, total_charge)
                st.metric(
                    "実際サイクル",
                    f"{cycle_count:.0f}kWh",
                    delta=f"目標: {daily_cycle_target:,}kWh"
                )
            
            with col5:
                final_soc = result['soc_profile'][-1]
                soc_change = final_soc - initial_soc
                st.metric(
                    "最終SOC",
                    f"{final_soc:.1f}%",
                    delta=f"{soc_change:+.1f}%"
                )
            
            # グラフ表示
            st.subheader("📈 需要・制御結果グラフ")
            
            # 時系列作成
            time_series = create_time_series(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0))
            
            # 需要比較グラフ
            fig_demand = go.Figure()
            
            fig_demand.add_trace(go.Scatter(
                x=time_series, y=st.session_state.original_forecast,
                name="元需要予測", line=dict(color="lightgray", width=2, dash="dash")
            ))
            
            fig_demand.add_trace(go.Scatter(
                x=time_series, y=result['demand_after_battery'],
                name="制御後需要", line=dict(color="blue", width=3)
            ))
            
            fig_demand.update_layout(
                title="需要カーブ比較（分離制御）",
                xaxis_title="時刻",
                yaxis_title="需要 (kW)",
                height=400,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_demand, use_container_width=True)
            
            # バッテリー出力・SOCグラフ
            col1, col2 = st.columns(2)
            
            with col1:
                fig_battery = go.Figure()
                
                fig_battery.add_trace(go.Scatter(
                    x=time_series, y=result['battery_output'],
                    name="バッテリー出力", line=dict(color="green", width=2),
                    mode='lines'
                ))
                
                fig_battery.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
                
                fig_battery.update_layout(
                    title="バッテリー出力パターン",
                    xaxis_title="時刻",
                    yaxis_title="出力 (kWh)",
                    height=350
                )
                st.plotly_chart(fig_battery, use_container_width=True)
            
            with col2:
                fig_soc = go.Figure()
                
                fig_soc.add_trace(go.Scatter(
                    x=time_series, y=result['soc_profile'],
                    name="SOC", line=dict(color="red", width=2)
                ))
                
                # SOC制約線
                fig_soc.add_hline(y=10, line_dash="dot", line_color="red", opacity=0.5)
                fig_soc.add_hline(y=90, line_dash="dot", line_color="blue", opacity=0.5)
                
                fig_soc.update_layout(
                    title="SOC変化",
                    xaxis_title="時刻",
                    yaxis_title="SOC (%)",
                    height=350,
                    yaxis=dict(range=[0, 100])
                )
                st.plotly_chart(fig_soc, use_container_width=True)
            
            # 制御詳細情報
            if 'control_info' in result and result['control_info']:
                st.subheader("🔍 制御詳細情報")
                
                with st.expander("制御ロジック詳細", expanded=False):
                    control_info = result['control_info']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        peak_steps = len([x for x in control_info.get('control_types', []) if x == 'peak'])
                        st.metric("ピーク制御ステップ", f"{peak_steps}/96")
                    
                    with col2:
                        bottom_steps = len([x for x in control_info.get('control_types', []) if x == 'bottom'])
                        st.metric("ボトム制御ステップ", f"{bottom_steps}/96")
                    
                    with col3:
                        flat_steps = len([x for x in control_info.get('control_types', []) if x == 'flattening'])
                        st.metric("平準化制御ステップ", f"{flat_steps}/96")
            
            # 結果のダウンロード
            st.subheader("💾 結果データダウンロード")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 制御結果CSVダウンロード", use_container_width=True):
                    # CSVデータ作成
                    result_data = []
                    for i in range(96):
                        result_data.append({
                            'ステップ': i + 1,
                            '時刻': time_series[i].strftime('%H:%M'),
                            '元需要(kW)': st.session_state.original_forecast[i],
                            '制御後需要(kW)': result['demand_after_battery'][i],
                            'バッテリー出力(kWh)': result['battery_output'][i],
                            'SOC(%)': result['soc_profile'][i],
                            '需要削減(kW)': st.session_state.original_forecast[i] - result['demand_after_battery'][i]
                        })
                    
                    result_df = pd.DataFrame(result_data)
                    csv_data = result_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📄 CSVファイルをダウンロード",
                        data=csv_data,
                        file_name=f"battery_control_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col2:
                if st.button("📈 制御パラメータレポートダウンロード", use_container_width=True):
                    # パラメータレポート作成
                    report_data = {
                        "制御パラメータ": [
                            f"ピーク閾値: {peak_percentile}%",
                            f"ボトム閾値: {bottom_percentile}%", 
                            f"ピーク制御比率: {peak_power_ratio:.2f}",
                            f"ボトム制御比率: {bottom_power_ratio:.2f}",
                            f"平準化比率: {flattening_power_ratio:.2f}",
                            f"サイクル目標: {daily_cycle_target:,}kWh"
                        ],
                        "バッテリー設定": [
                            f"容量: {battery_capacity:,}kWh",
                            f"最大出力: {max_power:,}kWh/15分",
                            f"効率: {efficiency:.1%}",
                            f"初期SOC: {initial_soc:.1f}%",
                            "",
                            ""
                        ],
                        "制御結果": [
                            f"ピーク削減: {peak_reduction:.1f}kW",
                            f"総放電: {total_discharge:.0f}kWh",
                            f"総充電: {total_charge:.0f}kWh",
                            f"実際サイクル: {cycle_count:.0f}kWh",
                            f"最終SOC: {final_soc:.1f}%",
                            f"SOC変化: {soc_change:+.1f}%"
                        ]
                    }
                    
                    report_df = pd.DataFrame(report_data)
                    report_csv = report_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📋 パラメータレポートをダウンロード",
                        data=report_csv,
                        file_name=f"battery_control_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        # エラーデータ表示（デバッグ用）
        if ERROR_DATA:
            st.header("⚠️ エラーログ")
            with st.expander("🔍 エラー詳細", expanded=False):
                for i, error in enumerate(ERROR_DATA[-10:]):  # 最新10件
                    st.text(f"{i+1}. {error}")
    
    else:
        st.warning("⚠️ 需要予測データをアップロードしてください")
        st.info("💡 CSVファイルには時刻列と需要列（96ステップ、15分間隔）が必要です")
        
        # サンプルデータの表示
        with st.expander("📋 CSVファイル形式例", expanded=False):
            sample_data = pd.DataFrame({
                '時刻': ['00:00', '00:15', '00:30', '00:45', '01:00'],
                '需要(kW)': [15000, 14800, 14500, 14200, 14000]
            })
            st.dataframe(sample_data)
            st.write("※ 上記のような形式で96行（24時間×4）のデータが必要です")


if __name__ == "__main__":
    main()
