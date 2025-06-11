import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import time

# ページ設定
st.set_page_config(
    page_title="需要データ変換ツール",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS スタイル
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2c3e50;
        font-size: 2.5rem;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .stat-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.2rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitDemandConverter:
    """Streamlit用需要データ変換クラス"""
    
    def __init__(self):
        if 'converted_data' not in st.session_state:
            st.session_state.converted_data = None
        if 'converted_data_pivot_a' not in st.session_state:
            st.session_state.converted_data_pivot_a = None
        if 'converted_data_pivot_b' not in st.session_state:
            st.session_state.converted_data_pivot_b = None
        if 'original_data' not in st.session_state:
            st.session_state.original_data = None
        if 'conversion_stats' not in st.session_state:
            st.session_state.conversion_stats = None
    
    def parse_date(self, date_value):
        """日付文字列をYYYY/MM/DD形式に変換"""
        try:
            date_str = str(int(date_value))
            if len(date_str) == 8:
                year = date_str[:4]
                month = date_str[4:6]
                day = date_str[6:8]
                return f"{year}/{month}/{day}"
            return None
        except:
            return None
    
    def format_time(self, hour, quarter):
        """時刻をHH:MM形式に変換"""
        minutes = quarter * 15
        return f"{hour:02d}:{minutes:02d}"
    
    def load_and_clean_data(self, uploaded_file):
        """アップロードされたファイルを読み込み、クリーニング"""
        try:
            # ファイル読み込み
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            
            # データ情報
            original_rows = len(df)
            
            if original_rows == 0:
                return {'success': False, 'error': 'CSVファイルが空です'}
            
            # 必要なカラムの存在確認（年月日のみ必須）
            if '年月日' not in df.columns:
                return {
                    'success': False, 
                    'error': '必要なカラム「年月日」が見つかりません'
                }
            
            # 集計ｺｰﾄﾞカラムがない場合はデフォルト値を設定
            if '集計ｺｰﾄﾞ' not in df.columns:
                df['集計ｺｰﾄﾞ'] = 'DEFAULT_CODE'
                st.info("💡 集計ｺｰﾄﾞカラムが見つからないため、デフォルト値を設定しました。")
            
            # null値や無効なデータを含む行を除去
            df = df.dropna(subset=['年月日'])
            
            if len(df) == 0:
                return {'success': False, 'error': '年月日カラムにすべてnull値が含まれています'}
            
            # 年月日が数値型で8桁の行のみ抽出
            try:
                valid_date_mask = (
                    df['年月日'].astype(str).str.len() == 8
                ) & (
                    pd.to_numeric(df['年月日'], errors='coerce').notna()
                )
                
                df = df[valid_date_mask]
                
                if len(df) == 0:
                    return {'success': False, 'error': '有効な日付形式（YYYYMMDD）のデータが見つかりません'}
            
            except Exception as e:
                return {'success': False, 'error': f'日付データの処理中にエラー: {str(e)}'}
            
            # 時間カラムの存在確認
            hour_columns = [f'{h}時' for h in range(24)]
            available_hour_columns = [col for col in hour_columns if col in df.columns]
            
            if len(available_hour_columns) == 0:
                return {
                    'success': False, 
                    'error': '時間データのカラム（0時〜23時）が見つかりません'
                }
            
            # 有効な時間データを持つ行のみ抽出
            valid_rows = []
            for idx, row in df.iterrows():
                has_valid_data = False
                for h in range(24):
                    col_name = f'{h}時'
                    if col_name in df.columns:
                        value = row[col_name]
                        if pd.notna(value) and isinstance(value, (int, float)):
                            has_valid_data = True
                            break
                
                if has_valid_data:
                    valid_rows.append(idx)
            
            if len(valid_rows) == 0:
                return {
                    'success': False, 
                    'error': '有効な時間データを持つ行が見つかりません'
                }
            
            df = df.loc[valid_rows]
            
            st.session_state.original_data = df
            
            return {
                'success': True,
                'original_rows': original_rows,
                'valid_rows': len(df),
                'removed_rows': original_rows - len(df),
                'columns': list(df.columns),
                'available_hour_columns': len(available_hour_columns)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'ファイル処理中にエラーが発生しました: {str(e)}'
            }
    
    def convert_to_15min(self):
        """15分値変換処理"""
        if st.session_state.original_data is None:
            return {'success': False, 'error': '元データがありません'}
        
        df = st.session_state.original_data
        
        # データが空の場合のチェック
        if len(df) == 0:
            return {'success': False, 'error': '有効なデータが0行です。CSVファイルの内容を確認してください。'}
        
        converted_records = []
        
        # 進捗バー
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_days = len(df)
        
        for idx, (_, row) in enumerate(df.iterrows()):
            # 進捗更新
            progress = (idx + 1) / total_days
            progress_bar.progress(progress)
            status_text.text(f'変換中: {idx + 1}/{total_days}日 ({progress*100:.1f}%)')
            
            # 日付と集計コード取得
            date_str = self.parse_date(row['年月日'])
            if not date_str:
                continue
                
            code = row['集計ｺｰﾄﾞ']
            
            # 24時間分のデータを取得
            hourly_data = []
            for hour in range(24):
                col_name = f'{hour}時'
                if col_name in row:
                    value = row[col_name]
                    hourly_data.append(value if pd.notna(value) else 0)
                else:
                    hourly_data.append(0)
            
            # 各時間について15分値を生成
            for hour in range(24):
                current_value = hourly_data[hour]
                
                # 次の時間の値を決定
                if hour == 23:
                    if idx < len(df) - 1:
                        next_day_row = df.iloc[idx + 1]
                        next_value = next_day_row['0時'] if pd.notna(next_day_row['0時']) else current_value
                    else:
                        next_value = current_value
                else:
                    next_value = hourly_data[hour + 1]
                
                # 4つの15分値を線形補間で生成
                for quarter in range(4):
                    interpolation_factor = quarter / 4
                    interpolated_value = current_value + (next_value - current_value) * interpolation_factor
                    
                    converted_records.append({
                        '年月日': date_str,
                        '集計コード': code,
                        '時刻': self.format_time(hour, quarter),
                        '需要値': round(interpolated_value, 2)
                    })
        
        # 変換完了
        progress_bar.progress(1.0)
        status_text.text('変換完了！')
        
        # DataFrameに変換
        converted_df = pd.DataFrame(converted_records)
        
        # 横形式A（日付×時間マトリックス）
        # 日付（縦軸）と時間（横軸）の2次元テーブル作成
        pivot_df_a = converted_df.pivot_table(
            index='年月日',           # 縦軸：日付
            columns='時刻',          # 横軸：時間
            values='需要値',         # 値：需要値
            aggfunc='first'
        ).reset_index()
        pivot_df_a.columns.name = None
        
        # 横形式B（時間×日付マトリックス）
        # 時間（縦軸）と日付（横軸）の2次元テーブル作成
        pivot_df_b = converted_df.pivot_table(
            index='時刻',            # 縦軸：時間
            columns='年月日',        # 横軸：日付
            values='需要値',         # 値：需要値
            aggfunc='first'
        ).reset_index()
        pivot_df_b.columns.name = None
        
        # 元の縦長形式も保持
        st.session_state.converted_data = converted_df
        st.session_state.converted_data_pivot_a = pivot_df_a  # 日付×時間
        st.session_state.converted_data_pivot_b = pivot_df_b  # 時間×日付
        
        # 統計情報保存（ゼロ除算チェック付き）
        if total_days > 0 and len(converted_df) > 0:
            records_per_day = len(converted_df) // total_days
            first_date = converted_df['年月日'].iloc[0] if len(converted_df) > 0 else 'N/A'
            last_date = converted_df['年月日'].iloc[-1] if len(converted_df) > 0 else 'N/A'
        else:
            records_per_day = 0
            first_date = 'N/A'
            last_date = 'N/A'
        
        st.session_state.conversion_stats = {
            'valid_days': total_days,
            'total_records': len(converted_df),
            'records_per_day': records_per_day,
            'first_date': first_date,
            'last_date': last_date
        }
        
        return {'success': True}


def main():
    """メインアプリケーション"""
    
    # ヘッダー
    st.markdown('<h1 class="main-header">🔄 1時間需要データから15分値への変換ツール</h1>', 
                unsafe_allow_html=True)
    
    # サイドバー
    with st.sidebar:
        st.header("📋 操作パネル")
        
        # 情報表示
        st.markdown("""
        <div class="info-box">
        <strong>🎯 機能説明</strong><br>
        • 1時間ごとの需要データを15分間隔に変換<br>
        • 線形補間による精密な値計算<br>
        • null値の自動処理<br>
        • 3つの出力形式に対応：<br>
        　・マトリックス形式A：日付↓×時間→（時系列分析用）<br>
        　・マトリックス形式B：時間↓×日付→（時間帯分析用）<br>
        　・縦形式：従来の行形式（DB用）
        </div>
        """, unsafe_allow_html=True)
        
        # ファイル形式説明
        with st.expander("📄 対応ファイル形式"):
            st.write("""
            **CSVファイル要件:**
            - 年月日列（YYYYMMDD形式）[必須]
            - 0時〜23時の時間別データ列 [必須]
            - 集計コード列 [オプション - ない場合はデフォルト値を設定]
            - UTF-8エンコーディング推奨
            
            **パターン1（集計コード有り）:**
            ```
            年月日,集計ｺｰﾄﾞ,0時,1時,2時,...,23時
            20240401,2100001,27000,25690,25220,...
            ```
            
            **パターン2（集計コード無し）:**
            ```
            年月日,0時,1時,2時,...,23時
            20240401,27000,25690,25220,...
            ```
            """)
            
        # アップロードされたファイルの構造説明
        st.markdown("""
        <div class="info-box">
        <strong>💡 ヒント</strong><br>
        • 添付されたファイルは305行、25カラムの構造に対応<br>
        • 集計コードがない場合は自動でデフォルト値を設定<br>
        • 0時〜23時の全時間データが必要
        </div>
        """, unsafe_allow_html=True)
    
    # メインコンテンツ
    converter = StreamlitDemandConverter()
    
    # ファイルアップロード
    st.header("📁 CSVファイルのアップロード")
    uploaded_file = st.file_uploader(
        "需要データのCSVファイルを選択してください",
        type=['csv'],
        help="1時間ごとの需要データが含まれたCSVファイルをアップロードしてください"
    )
    
    if uploaded_file is not None:
        # ファイル情報表示
        file_details = {
            "ファイル名": uploaded_file.name,
            "ファイルサイズ": f"{uploaded_file.size / 1024:.1f} KB"
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📄 **ファイル名:** {file_details['ファイル名']}")
        with col2:
            st.info(f"📊 **サイズ:** {file_details['ファイルサイズ']}")
        
        # データ読み込み・クリーニング
        if st.button("🔍 データを読み込み・検証", type="primary"):
            with st.spinner("データを読み込み中..."):
                result = converter.load_and_clean_data(uploaded_file)
            
            if result['success']:
                st.markdown(f"""
                <div class="success-message">
                <strong>✅ データ読み込み完了</strong><br>
                • 総行数: {result['original_rows']:,}<br>
                • 有効行数: {result['valid_rows']:,}<br>
                • 除去された行: {result['removed_rows']:,}<br>
                • カラム数: {len(result['columns'])}<br>
                • 利用可能な時間カラム: {result.get('available_hour_columns', '不明')}個
                </div>
                """, unsafe_allow_html=True)
                
                # データプレビュー
                st.subheader("👀 データプレビュー")
                st.dataframe(
                    st.session_state.original_data.head(10),
                    use_container_width=True
                )
                
            else:
                st.markdown(f"""
                <div class="error-message">
                <strong>❌ データ読み込みエラー</strong><br>
                {result['error']}
                </div>
                """, unsafe_allow_html=True)
    
    # 変換処理
    if st.session_state.original_data is not None:
        st.header("🔄 15分値変換")
        
        # 変換前統計
        original_df = st.session_state.original_data
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stat-container">
                <div class="stat-number">{len(original_df):,}</div>
                <div class="stat-label">有効日数</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-container">
                <div class="stat-number">{len(original_df) * 24:,}</div>
                <div class="stat-label">元データ時間数</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-container">
                <div class="stat-number">{len(original_df) * 24 * 4:,}</div>
                <div class="stat-label">予想15分データ数</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stat-container">
                <div class="stat-number">15分</div>
                <div class="stat-label">出力間隔</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 変換実行
        if st.button("🚀 15分値に変換", type="primary"):
            with st.spinner("変換処理中..."):
                result = converter.convert_to_15min()
            
            if result['success']:
                st.balloons()
                st.markdown("""
                <div class="success-message">
                <strong>🎉 変換完了！</strong><br>
                15分値への変換が正常に完了しました。
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="error-message">
                <strong>❌ 変換エラー</strong><br>
                {result['error']}
                </div>
                """, unsafe_allow_html=True)
    
    # 結果表示とダウンロード
    if st.session_state.converted_data is not None and st.session_state.conversion_stats is not None:
        st.header("📊 変換結果")
        
        stats = st.session_state.conversion_stats
        
        # データが空でないことを確認
        if stats['total_records'] == 0:
            st.warning("⚠️ 変換されたデータが0レコードです。入力データを確認してください。")
        else:
            # 結果統計
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="stat-container">
                    <div class="stat-number">{stats['valid_days']:,}</div>
                    <div class="stat-label">変換日数</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-container">
                    <div class="stat-number">{stats['total_records']:,}</div>
                    <div class="stat-label">総レコード数</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stat-container">
                    <div class="stat-number">{stats['records_per_day']:,}</div>
                    <div class="stat-label">1日あたりレコード数</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="stat-container">
                    <div class="stat-number">✅</div>
                    <div class="stat-label">変換完了</div>
                </div>
                """, unsafe_allow_html=True)
            
            # データ期間
            if stats['first_date'] != 'N/A' and stats['last_date'] != 'N/A':
                st.info(f"📅 **データ期間:** {stats['first_date']} ～ {stats['last_date']}")
            
            # データプレビュー
            st.subheader("👀 変換結果プレビュー")
            
            # 出力形式選択
            output_format = st.radio(
                "📊 出力形式を選択:",
                [
                    "マトリックス形式A（日付↓×時間→）", 
                    "マトリックス形式B（時間↓×日付→）", 
                    "縦形式（従来の行形式）"
                ],
                index=0,
                help="A: 日付が縦、時間が横 / B: 時間が縦、日付が横 / 縦形式: 各行に日付・時刻・需要値"
            )
            
            # 表示するデータを選択
            if output_format == "マトリックス形式A（日付↓×時間→）":
                display_data = st.session_state.converted_data_pivot_a
                download_data = st.session_state.converted_data_pivot_a
                format_type = "pivot_a"
            elif output_format == "マトリックス形式B（時間↓×日付→）":
                display_data = st.session_state.converted_data_pivot_b
                download_data = st.session_state.converted_data_pivot_b
                format_type = "pivot_b"
            else:
                display_data = st.session_state.converted_data
                download_data = st.session_state.converted_data
                format_type = "normal"
            
            # タブ表示
            if format_type in ["pivot_a", "pivot_b"]:
                tab1, tab2 = st.tabs(["📋 マトリックス形式プレビュー", "📈 統計情報"])
                
                with tab1:
                    if format_type == "pivot_a":
                        st.write("**日付×時間マトリックス（日付↓、時間→）:**")
                        axis_info = "縦軸: 年月日（上から下に進む）\n横軸: 時刻（左から右に進む：00:00→23:45）"
                        structure_info = f"行数: {len(display_data)} (日付数)\n列数: {len(display_data.columns)} (年月日 + 各時刻)"
                    else:  # pivot_b
                        st.write("**時間×日付マトリックス（時間↓、日付→）:**")
                        axis_info = "縦軸: 時刻（上から下に進む：00:00→23:45）\n横軸: 年月日（左から右に進む）"
                        structure_info = f"行数: {len(display_data)} (時刻数：96個)\n列数: {len(display_data.columns)} (時刻 + 各日付)"
                    
                    # 横スクロール可能な表示
                    st.dataframe(
                        display_data,
                        use_container_width=True,
                        height=400
                    )
                    
                    # データ構造説明
                    st.info(f"""
                    📊 **データ構造:**
                    • {structure_info}
                    • {axis_info}
                    • 各セル: 該当日時の需要値
                    """)
                
                with tab2:
                    # マトリックス形式の統計情報
                    if len(display_data) > 0:
                        st.write(f"**{output_format}データの統計:**")
                        
                        # 数値データのみ抽出して統計計算
                        numeric_cols = display_data.select_dtypes(include=[np.number]).columns
                        
                        if format_type == "pivot_a":
                            # 日付×時間形式：時間列（年月日以外）
                            data_cols = [col for col in numeric_cols if col != '年月日']
                            if len(data_cols) > 0:
                                # 時間別統計
                                stats_data = display_data[data_cols].describe().T
                                st.dataframe(stats_data, use_container_width=True)
                                
                                # 日別統計
                                st.write("**日別需要統計（全時間の平均）:**")
                                daily_stats = display_data[data_cols].mean(axis=1).describe().to_frame('日平均需要')
                                st.dataframe(daily_stats.T, use_container_width=True)
                            else:
                                st.info("時間データが見つかりません。")
                        
                        else:  # pivot_b
                            # 時間×日付形式：日付列（時刻以外）
                            data_cols = [col for col in display_data.columns if col != '時刻']
                            if len(data_cols) > 0:
                                # 日付別統計
                                numeric_data = display_data[data_cols].select_dtypes(include=[np.number])
                                if len(numeric_data.columns) > 0:
                                    stats_data = numeric_data.describe().T
                                    st.dataframe(stats_data, use_container_width=True)
                                    
                                    # 時間別統計
                                    st.write("**時間別需要統計（全日の平均）:**")
                                    time_stats = numeric_data.mean(axis=1)
                                    # 時刻を追加して表示
                                    time_stats_df = pd.DataFrame({
                                        '時刻': display_data['時刻'],
                                        '全日平均需要': time_stats.values
                                    })
                                    st.dataframe(time_stats_df.head(20), use_container_width=True)
                                else:
                                    st.info("数値データが見つかりません。")
                            else:
                                st.info("日付データが見つかりません。")
                    else:
                        st.warning("統計データを表示するデータがありません。")
            
            else:
                tab1, tab2, tab3 = st.tabs(["📋 最初の50行", "📋 最後の50行", "📈 統計情報"])
                
                with tab1:
                    st.dataframe(
                        display_data.head(50),
                        use_container_width=True
                    )
                
                with tab2:
                    st.dataframe(
                        display_data.tail(50),
                        use_container_width=True
                    )
                
                with tab3:
                    # 需要値の基本統計
                    if len(display_data) > 0:
                        st.write("**需要値の統計:**")
                        st.dataframe(
                            display_data['需要値'].describe().to_frame().T,
                            use_container_width=True
                        )
                        
                        # 時間帯別統計
                        st.write("**時間帯別平均需要値:**")
                        df_with_hour = display_data.copy()
                        df_with_hour['時'] = df_with_hour['時刻'].str[:2].astype(int)
                        hourly_stats = df_with_hour.groupby('時')['需要値'].agg(['mean', 'max', 'min'])
                        st.dataframe(hourly_stats, use_container_width=True)
                    else:
                        st.warning("統計データを表示するデータがありません。")
            
            # ダウンロード
            if len(st.session_state.converted_data) > 0:
                st.subheader("📥 結果のダウンロード")
                
                # ダウンロード形式の説明
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    **📊 マトリックス形式A:**
                    - 縦軸: 年月日（上から下に進む）
                    - 横軸: 時刻（左から右に進む）
                    - 各セル: 該当日時の需要値
                    - 時系列分析に最適
                    """)
                
                with col2:
                    st.markdown("""
                    **📊 マトリックス形式B:**
                    - 縦軸: 時刻（上から下に進む）
                    - 横軸: 年月日（左から右に進む）
                    - 各セル: 該当日時の需要値
                    - 時間帯分析に最適
                    """)
                
                with col3:
                    st.markdown("""
                    **📋 縦形式（従来）:**
                    - 各行: 年月日、集計コード、時刻、需要値
                    - 1レコード1行の標準的な形式
                    - データベース取り込みに適している
                    """)
                
                # CSVデータ準備
                csv_buffer = io.StringIO()
                download_data.to_csv(csv_buffer, index=False, encoding='utf-8')
                csv_data = csv_buffer.getvalue()
                
                # BOM付きでエンコード（Excel対応）
                csv_bytes = ('\ufeff' + csv_data).encode('utf-8')
                
                # タイムスタンプ付きファイル名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if format_type == "pivot_a":
                    format_suffix = "matrix_date_time"
                elif format_type == "pivot_b":
                    format_suffix = "matrix_time_date"
                else:
                    format_suffix = "normal"
                
                filename = f"demand_15min_data_{format_suffix}_{timestamp}.csv"
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.download_button(
                        label=f"📥 CSVファイルをダウンロード ({output_format})",
                        data=csv_bytes,
                        file_name=filename,
                        mime="text/csv",
                        type="primary"
                    )
                
                with col2:
                    file_size = len(csv_bytes) / 1024
                    st.metric("ファイルサイズ", f"{file_size:.1f} KB")
                
                st.success(f"💾 ファイル名: `{filename}`")
                
                # プレビュー例の表示
                with st.expander("👀 ダウンロードファイルのプレビュー例"):
                    if format_type == "pivot_a":
                        st.write("**マトリックス形式A CSVの構造例（日付↓×時間→）:**")
                        sample_data = download_data.head(5).iloc[:, :6]  # 最初の5行、6列のみ表示
                        st.dataframe(sample_data, use_container_width=True)
                        st.info("💡 実際のファイルには全ての時刻が含まれます（00:00〜23:45の96列）")
                        st.markdown("""
                        **📊 構造説明:**
                        - 1列目: 年月日
                        - 2列目以降: 各時刻（00:00, 00:15, 00:30, ..., 23:45）
                        - 各行: 1日分のデータ
                        """)
                    
                    elif format_type == "pivot_b":
                        st.write("**マトリックス形式B CSVの構造例（時間↓×日付→）:**")
                        sample_data = download_data.head(10).iloc[:, :6]  # 最初の10行、6列のみ表示
                        st.dataframe(sample_data, use_container_width=True)
                        st.info("💡 実際のファイルには全ての日付が含まれます（305日分の列）")
                        st.markdown("""
                        **📊 構造説明:**
                        - 1列目: 時刻
                        - 2列目以降: 各日付（2024/04/01, 2024/04/02, ...）
                        - 各行: 1つの時刻の全日データ
                        - 96行: 1日の15分間隔データ（00:00〜23:45）
                        """)
                    
                    else:
                        st.write("**縦形式CSVの構造例:**")
                        st.dataframe(download_data.head(10), use_container_width=True)


if __name__ == "__main__":
    main()


# =============================================================================
# 実行方法
# =============================================================================

"""
# ターミナル/コマンドプロンプトで実行:

1. 必要なライブラリをインストール:
   pip install streamlit pandas numpy

2. このファイルを保存 (例: demand_converter.py)

3. Streamlitアプリを起動:
   streamlit run demand_converter.py

4. ブラウザでアプリを使用:
   - 自動でブラウザが開きます (通常 http://localhost:8501)
   - CSVファイルをアップロード
   - 変換処理を実行
   - 結果をダウンロード

# 追加機能:
- リアルタイム進捗表示
- データ統計情報の可視化
- インタラクティブなプレビュー
- 2つの出力形式（横形式・縦形式）選択
- Excel対応CSV出力
- レスポンシブデザイン

# 出力形式:
横形式: 日付時刻が列に展開され、分析やグラフ作成に適した形式
縦形式: 従来の1レコード1行形式で、データベース取り込みに適した形式
"""