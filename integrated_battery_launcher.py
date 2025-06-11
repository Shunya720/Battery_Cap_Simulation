"""
バッテリー制御システム メインランチャー
機能別にアプリケーションを選択して実行
統合版：各アプリを直接起動可能
"""

import streamlit as st
import subprocess
import sys
import os
import threading
import time
from pathlib import Path

def main():
    st.set_page_config(
        page_title="バッテリー制御システム",
        page_icon="🔋",
        layout="wide"
    )
    
    st.title("🔋 バッテリー制御システム - メインメニュー")
    st.write("**用途に応じて適切な機能を選択してください**")
    
    # セッション状態の初期化
    if 'launched_apps' not in st.session_state:
        st.session_state.launched_apps = {}
    
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
    
    # アプリ起動ステータス表示
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 アプリ起動状況")
    
    for app_name, status in st.session_state.launched_apps.items():
        if status['running']:
            st.sidebar.success(f"✅ {app_name}: ポート{status['port']}で実行中")
        else:
            st.sidebar.error(f"❌ {app_name}: 停止")
    
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


def find_available_port(start_port=8502):
    """利用可能なポートを検索"""
    import socket
    port = start_port
    while port < start_port + 10:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            port += 1
    return None


def launch_streamlit_app(app_file, app_name):
    """Streamlitアプリを別ポートで起動"""
    if not os.path.exists(app_file):
        st.error(f"❌ ファイル '{app_file}' が見つかりません")
        return False
    
    # 利用可能なポートを検索
    port = find_available_port()
    if port is None:
        st.error("❌ 利用可能なポートが見つかりません")
        return False
    
    try:
        # Streamlitアプリを別プロセスで起動
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", app_file,
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # セッション状態を更新
        st.session_state.launched_apps[app_name] = {
            'process': process,
            'port': port,
            'running': True,
            'url': f"http://localhost:{port}"
        }
        
        return True
        
    except Exception as e:
        st.error(f"❌ アプリの起動に失敗しました: {str(e)}")
        return False


def stop_app(app_name):
    """アプリを停止"""
    if app_name in st.session_state.launched_apps:
        app_info = st.session_state.launched_apps[app_name]
        if app_info['running'] and 'process' in app_info:
            try:
                app_info['process'].terminate()
                app_info['running'] = False
                st.success(f"✅ {app_name} を停止しました")
            except Exception as e:
                st.error(f"❌ アプリの停止に失敗: {str(e)}")


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
        
        app_name = "バッテリー計画システム"
        app_file = "battery_planning_app.py"
        
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            if st.button("🚀 アプリを起動", key="launch_planning", use_container_width=True):
                if app_name not in st.session_state.launched_apps or not st.session_state.launched_apps[app_name]['running']:
                    with st.spinner(f"{app_name}を起動中..."):
                        if launch_streamlit_app(app_file, app_name):
                            st.success(f"✅ {app_name}を起動しました")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error(f"❌ {app_name}の起動に失敗しました")
                else:
                    st.info("💡 既に起動済みです")
        
        with col1_2:
            if app_name in st.session_state.launched_apps and st.session_state.launched_apps[app_name]['running']:
                if st.button("🌐 ブラウザで開く", key="open_planning", use_container_width=True):
                    url = st.session_state.launched_apps[app_name]['url']
                    st.markdown(f"""
                    <script>window.open('{url}', '_blank');</script>
                    <a href="{url}" target="_blank">🌐 {app_name}を開く (ポート: {st.session_state.launched_apps[app_name]['port']})</a>
                    """, unsafe_allow_html=True)
        
        if app_name in st.session_state.launched_apps and st.session_state.launched_apps[app_name]['running']:
            if st.button("⏹️ アプリを停止", key="stop_planning", use_container_width=True):
                stop_app(app_name)
                st.rerun()
    
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
        
        app_name = "容量シミュレーション比較"
        app_file = "capacity_simulation_app.py"
        
        col2_1, col2_2 = st.columns(2)
        
        with col2_1:
            if st.button("🚀 アプリを起動", key="launch_capacity", use_container_width=True):
                if app_name not in st.session_state.launched_apps or not st.session_state.launched_apps[app_name]['running']:
                    with st.spinner(f"{app_name}を起動中..."):
                        if launch_streamlit_app(app_file, app_name):
                            st.success(f"✅ {app_name}を起動しました")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error(f"❌ {app_name}の起動に失敗しました")
                else:
                    st.info("💡 既に起動済みです")
        
        with col2_2:
            if app_name in st.session_state.launched_apps and st.session_state.launched_apps[app_name]['running']:
                if st.button("🌐 ブラウザで開く", key="open_capacity", use_container_width=True):
                    url = st.session_state.launched_apps[app_name]['url']
                    st.markdown(f"""
                    <script>window.open('{url}', '_blank');</script>
                    <a href="{url}" target="_blank">🌐 {app_name}を開く (ポート: {st.session_state.launched_apps[app_name]['port']})</a>
                    """, unsafe_allow_html=True)
        
        if app_name in st.session_state.launched_apps and st.session_state.launched_apps[app_name]['running']:
            if st.button("⏹️ アプリを停止", key="stop_capacity", use_container_width=True):
                stop_app(app_name)
                st.rerun()
    
    # システム管理セクション
    st.header("🎛️ システム管理")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 ステータス更新", use_container_width=True):
            # プロセスの生存確認
            for app_name, app_info in st.session_state.launched_apps.items():
                if app_info['running'] and 'process' in app_info:
                    if app_info['process'].poll() is not None:
                        app_info['running'] = False
            st.rerun()
    
    with col2:
        if st.button("⏹️ 全アプリ停止", use_container_width=True):
            for app_name in list(st.session_state.launched_apps.keys()):
                stop_app(app_name)
            st.rerun()
    
    with col3:
        if st.button("🗑️ ログクリア", use_container_width=True):
            st.session_state.launched_apps = {}
            st.success("✅ ログをクリアしました")
            st.rerun()
    
    # 技術的特徴（既存のコードを保持）
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
    """)


def show_battery_planning_info():
    """バッテリー計画システムの詳細情報"""
    st.header("🔋 バッテリー計画システム")
    
    app_name = "バッテリー計画システム"
    
    # 起動ボタンを上部に配置
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🚀 アプリを起動", key="info_launch_planning", use_container_width=True):
            if app_name not in st.session_state.launched_apps or not st.session_state.launched_apps[app_name]['running']:
                with st.spinner(f"{app_name}を起動中..."):
                    if launch_streamlit_app("battery_planning_app.py", app_name):
                        st.success(f"✅ {app_name}を起動しました")
                        time.sleep(2)
                        st.rerun()
            else:
                st.info("💡 既に起動済みです")
    
    with col2:
        if app_name in st.session_state.launched_apps and st.session_state.launched_apps[app_name]['running']:
            if st.button("🌐 ブラウザで開く", key="info_open_planning", use_container_width=True):
                url = st.session_state.launched_apps[app_name]['url']
                st.markdown(f"""
                <a href="{url}" target="_blank">🌐 {app_name}を開く</a>
                """, unsafe_allow_html=True)
    
    with col3:
        if app_name in st.session_state.launched_apps and st.session_state.launched_apps[app_name]['running']:
            if st.button("⏹️ アプリを停止", key="info_stop_planning", use_container_width=True):
                stop_app(app_name)
                st.rerun()
    
    # 既存の詳細情報表示
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
    
    # 残りの既存コンテンツ...


def show_capacity_simulation_info():
    """容量シミュレーション比較の詳細情報"""
    st.header("📊 容量シミュレーション比較")
    
    app_name = "容量シミュレーション比較"
    
    # 起動ボタンを上部に配置
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🚀 アプリを起動", key="info_launch_capacity", use_container_width=True):
            if app_name not in st.session_state.launched_apps or not st.session_state.launched_apps[app_name]['running']:
                with st.spinner(f"{app_name}を起動中..."):
                    if launch_streamlit_app("capacity_simulation_app.py", app_name):
                        st.success(f"✅ {app_name}を起動しました")
                        time.sleep(2)
                        st.rerun()
            else:
                st.info("💡 既に起動済みです")
    
    with col2:
        if app_name in st.session_state.launched_apps and st.session_state.launched_apps[app_name]['running']:
            if st.button("🌐 ブラウザで開く", key="info_open_capacity", use_container_width=True):
                url = st.session_state.launched_apps[app_name]['url']
                st.markdown(f"""
                <a href="{url}" target="_blank">🌐 {app_name}を開く</a>
                """, unsafe_allow_html=True)
    
    with col3:
        if app_name in st.session_state.launched_apps and st.session_state.launched_apps[app_name]['running']:
            if st.button("⏹️ アプリを停止", key="info_stop_capacity", use_container_width=True):
                stop_app(app_name)
                st.rerun()
    
    # 既存の詳細情報表示
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
    
    # 残りの既存コンテンツ...


if __name__ == "__main__":
    main()