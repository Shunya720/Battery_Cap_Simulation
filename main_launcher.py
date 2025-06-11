"""
バッテリー制御システム メインランチャー
機能別にアプリケーションを選択して実行
"""

import streamlit as st
import subprocess
import sys
import os

def main():
    st.set_page_config(
        page_title="バッテリー制御システム",
        page_icon="🔋",
        layout="wide"
    )
    
    st.title("🔋 バッテリー制御システム - メインメニュー")
    st.write("**用途に応じて適切な機能を選択してください**")
    
    # サイドバーでアプリケーション選択
    st.sidebar.header("🎯 機能選択")
    
    app_choice = st.sidebar.selectbox(
        "使用する機能を選択してください",
        [
            "メインメニュー",
            "バッテリー計画システム",
            "容量シミュレーション比較"
        ],
        help="各機能の詳細は下記説明をご確認ください"
    )
    
    # システム要件表示
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 システム要件")
    
    # 必須ライブラリのチェック
    required_libs = {
        "streamlit": "Webアプリフレームワーク",
        "pandas": "データ処理",
        "numpy": "数値計算",
        "plotly": "グラフ表示"
    }
    
    optional_libs = {
        "optuna": "ベイズ最適化（推奨）",
        "scipy": "科学計算（差分進化）"
    }
    
    st.sidebar.write("**必須ライブラリ:**")
    for lib, desc in required_libs.items():
        try:
            __import__(lib)
            st.sidebar.success(f"✅ {lib}: {desc}")
        except ImportError:
            st.sidebar.error(f"❌ {lib}: {desc}")
    
    st.sidebar.write("**オプションライブラリ:**")
    for lib, desc in optional_libs.items():
        try:
            __import__(lib)
            st.sidebar.success(f"✅ {lib}: {desc}")
        except ImportError:
            st.sidebar.warning(f"⚠️ {lib}: {desc}")
    
    # アプリケーション別の処理
    if app_choice == "メインメニュー":
        show_main_menu()
    elif app_choice == "バッテリー計画システム":
        show_battery_planning_info()
    elif app_choice == "容量シミュレーション比較":
        show_capacity_simulation_info()


def show_main_menu():
    """メインメニュー表示"""
    st.header("🎯 機能選択ガイド")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔋 バッテリー計画システム")
        st.write("""
        **主な機能:**
        - 需要予測データの読み込み・分析
        - 分離制御パラメータの自動最適化
        - リアルタイムシミュレーション
        - 需要滑らかさ（ギザギザ最小化）重視の制御
        - SOC管理とサイクル制約の協調制御
        
        **適用場面:**
        - 既定容量でのバッテリー制御計画策定
        - 制御パラメータの最適化
        - 日常運用での制御効果確認
        - 需要予測精度の影響評価
        """)
        
        if st.button("🚀 バッテリー計画システムを起動", use_container_width=True):
            st.info("💡 サイドバーから「バッテリー計画システム」を選択してください")
    
    with col2:
        st.subheader("📊 容量シミュレーション比較")
        st.write("""
        **主な機能:**
        - 複数バッテリー容量での効果比較
        - 容量別最適パラメータ自動探索
        - 投資効果の定量的評価
        - 推奨容量の自動判定
        - 経済性ガイダンスの提供
        
        **適用場面:**
        - バッテリー導入時の容量検討
        - 既存システムの拡張計画
        - 投資対効果の事前評価
        - 用途別最適容量の選定
        """)
        
        if st.button("📊 容量シミュレーション比較を起動", use_container_width=True):
            st.info("💡 サイドバーから「容量シミュレーション比較」を選択してください")
    
    # 技術的特徴
    st.header("🛠️ 技術的特徴")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("⚡ 分離制御技術")
        st.write("""
        - **独立制御強度**: ピーク・ボトムで異なる制御比率
        - **可変出力制御**: 需要偏差に応じた動的調整
        - **3段階制御**: ピーク/ボトム/平準化の最適組み合わせ
        """)
    
    with col2:
        st.subheader("🤖 自動最適化")
        st.write("""
        - **ベイズ最適化**: Optunaによる高精度探索
        - **差分進化**: 大域的最適化アルゴリズム
        - **多目的最適化**: 滑らかさ+制約満足の協調
        """)
    
    with col3:
        st.subheader("📊 需要滑らかさ重視")
        st.write("""
        - **ギザギザ最小化**: 隣接変動の大幅削減
        - **急変抑制**: 最大変動幅の効果的制御
        - **安定性向上**: 変動パターンの標準化
        """)
    
    # システム構成
    st.header("🏗️ システム構成")
    
    st.write("""
    **モジュール構成:**
    ```
    battery_core_logic.py       # コア計算ロジック（共通）
    ├── PeakBottomOptimizer     # 分離制御エンジン
    ├── BatterySOCManager       # SOC管理
    ├── DemandSmoothnessOptimizer # 滑らかさ最適化
    └── BatteryControlEngine    # 統合制御エンジン
    
    battery_planning_app.py     # バッテリー計画専用UI
    capacity_simulation_app.py  # 容量比較専用UI
    main_launcher.py           # メインランチャー（このファイル）
    ```
    
    **設計思想:**
    - **機能分離**: 計画と容量比較を独立したアプリに分離
    - **ロジック共有**: コア計算ロジックを共通化して冗長性を排除
    - **拡張性**: 新機能追加時の影響範囲を最小化
    - **保守性**: 各機能の独立したテスト・デバッグが可能
    """)


def show_battery_planning_info():
    """バッテリー計画システムの詳細情報"""
    st.header("🔋 バッテリー計画システム")
    
    st.info("**注意**: この機能を使用するには、別途 `battery_planning_app.py` を実行してください。")
    
    st.subheader("📋 使用手順")
    st.write("""
    1. **データ準備**: 需要予測CSV（96ステップ、15分間隔）を用意
    2. **データアップロード**: CSVファイルをアップロードして列選択
    3. **自動最適化**: 分離制御パラメータの最適化実行
    4. **パラメータ適用**: 最適化結果をシステムに適用
    5. **シミュレーション**: ステップ実行または一括実行で結果確認
    6. **結果分析**: グラフとデータテーブルで効果を評価
    7. **結果保存**: CSV形式でデータダウンロード
    """)
    
    st.subheader("⚙️ 設定可能パラメータ")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        **分離制御設定:**
        - ピーク判定閾値（50-100%）
        - ボトム判定閾値（0-50%）
        - ピーク制御強度（0.1-1.0）
        - ボトム制御強度（0.1-1.0）
        - 平準化制御強度（0.1-1.0）
        """)
    
    with col2:
        st.write("""
        **システム設定:**
        - バッテリー容量（kWh）
        - 最大出力（15分エネルギー値）
        - 効率（0.1-1.0）
        - 初期SOC（0-100%）
        - サイクル目標（kWh/day）
        """)
    
    st.subheader("🎯 最適化機能")
    st.write("""
    **自動最適化アルゴリズム:**
    - **Optuna（推奨）**: ベイズ最適化による高精度探索
    - **Differential Evolution**: 差分進化による大域的最適化
    
    **最適化目標:**
    - 需要隣接変動の最小化（40%重み）
    - 2次差分改善（20%重み）
    - 最大変動抑制（15%重み）
    - 変動安定性向上（10%重み）
    - 急変回数削減（10%重み）
    - その他補助指標（5%重み）
    """)
    
    code_example = """
    # バッテリー計画システムの起動方法
    streamlit run battery_planning_app.py
    
    # または
    python -m streamlit run battery_planning_app.py
    """
    
    st.subheader("💻 起動コマンド")
    st.code(code_example, language="bash")


def show_capacity_simulation_info():
    """容量シミュレーション比較の詳細情報"""
    st.header("📊 容量シミュレーション比較")
    
    st.info("**注意**: この機能を使用するには、別途 `capacity_simulation_app.py` を実行してください。")
    
    st.subheader("📋 使用手順")
    st.write("""
    1. **データ準備**: 需要予測CSV（96ステップ、15分間隔）を用意
    2. **容量設定**: 比較したい4つの容量を設定
    3. **比較条件設定**: サイクル比率、許容範囲、最適化試行回数を設定
    4. **一括シミュレーション**: 全容量での自動最適化とシミュレーション実行
    5. **結果比較**: 容量別効果をグラフと表で比較分析
    6. **推奨容量判定**: 用途別推奨容量の自動選定
    7. **ガイダンス**: 経済性を考慮した容量選択指針
    8. **結果保存**: サマリーと詳細データのダウンロード
    """)
    
    st.subheader("📊 比較評価項目")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        **制御効果指標:**
        - ピーク削減量（kW）
        - 需要幅改善量（kW）
        - 隣接変動改善率（%）
        - 最大変動抑制率（%）
        - サイクル制約達成状況
        """)
    
    with col2:
        st.write("""
        **最適化品質:**
        - 容量別最適パラメータ
        - ピーク・ボトム制御比率
        - 制御バランス評価
        - 実際放電量とサイクル目標の整合性
        - 制御効率（理論値に対する実績）
        """)
    
    st.subheader("🏆 推奨容量判定")
    st.write("""
    **判定カテゴリ:**
    - **総合評価1位**: 全指標のバランスが最も優れた容量
    - **ピーク削減効果1位**: 最大需要抑制に最も効果的な容量
    - **滑らかさ改善1位**: 需要カーブが最も滑らかになる容量
    
    **経済性ガイダンス:**
    - 容量増加に伴う投資額の考慮
    - 効果の逓減性（容量2倍≠効果2倍）の説明
    - 運用・保守コストとの兼ね合い指針
    """)
    
    st.subheader("⚙️ 設定可能項目")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        **容量設定:**
        - 容量1〜4（10,000-100,000kWh）
        - 重複チェック機能
        - 容量別最大出力自動計算
        """)
    
    with col2:
        st.write("""
        **最適化設定:**
        - サイクル目標比率（0.5-2.0）
        - サイクル許容範囲（500-5,000kWh）
        - 最適化試行回数（30-100回）
        """)
    
    code_example = """
    # 容量シミュレーション比較の起動方法
    streamlit run capacity_simulation_app.py
    
    # または
    python -m streamlit run capacity_simulation_app.py
    """
    
    st.subheader("💻 起動コマンド")
    st.code(code_example, language="bash")


if __name__ == "__main__":
    main()