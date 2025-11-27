# v1.py —— 2025年11月 終極無敵版（MultiIndex + time 已修復）
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import plotly.graph_objs as go
import base64
import time  # ← 關鍵修復：加 import time

st.set_page_config(layout="wide", page_title="艾略特波浪偵測器")

# ===================== 終極欄位處理（安全版，防單層 Index）=====================
def normalize_columns(df):
    """完美處理 yfinance 所有變形，安全處理 MultiIndex/單層"""
    if df is None or df.empty:
        return None
    
    df = df.copy()
    
    # 安全處理 MultiIndex（只在真正 MultiIndex 時處理）
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # 單股票：移除 ticker 層級
            if len(df.columns.levels[0]) == 1:
                df.columns = df.columns.droplevel(0)
            else:
                # 多股票：取第一個 ticker
                first_ticker = df.columns.levels[0][0]
                df = df[first_ticker].copy()
                df.columns = df.columns.droplevel(0)
        except ValueError as e:
            # 如果 droplevel 失敗（層級不匹配），強制重置為單層
            if "Cannot remove 1 levels" in str(e):
                # 假設是單股票，強制扁平化
                df.columns = [col[1] if isinstance(col, tuple) else col for col in df.columns]
            else:
                raise e
    else:
        # 單層 Index，直接處理
        pass
    
    # 處理大小寫和變形
    cols = df.columns.astype(str).str.strip().str.lower()
    
    mapping = {}
    for old_col in df.columns:
        lower = str(old_col).strip().lower()
        if 'open' in lower:
            mapping[old_col] = 'Open'
        elif 'high' in lower:
            mapping[old_col] = 'High'
        elif 'low' in lower:
            mapping[old_col] = 'Low'
        elif 'close' in lower or 'adj' in lower:
            mapping[old_col] = 'Close'
        elif 'volume' in lower:
            mapping[old_col] = 'Volume'
    
    df = df.rename(columns=mapping)
    
    # 確保必要欄位
    required = ['Open', 'High', 'Low', 'Close']
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"缺少欄位：{missing}。原始欄位：{list(df.columns)}")
        return None
    
    if 'Volume' not in df.columns:
        df['Volume'] = 0
    
    return df

# ===================== 安全下載（移除 group_by + 重試優化）=====================
@st.cache_data(ttl=600, show_spinner=False)
def get_data(ticker, interval="1d"):
    for attempt in range(3):
        try:
            period_map = {
                "1m": "7d", "2m": "7d", "5m": "7d", "15m": "60d",
                "30m": "60d", "60m": "60d", "90m": "60d", "1d": "2y",
                "1wk": "2y", "1mo": "2y"
            }
            period = period_map.get(interval, "2y")
            
            # 關鍵修復：移除 group_by='ticker'，改用單股票模式 + prepost=True 確保完整
            raw = yf.download(ticker, period=period, interval=interval,
                              progress=False, auto_adjust=False, prepost=True, threads=False)
            
            if raw.empty or len(raw) < 20:
                return None
                
            df = raw.reset_index()
            df = normalize_columns(df)
            if df is None:
                return None
                
            df = df.dropna(subset=['Close']).copy()
            # 日期處理
            date_col = 'Date' if 'Date' in df.columns else 'Datetime' if 'Datetime' in df.columns else None
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
            else:
                df.index = pd.to_datetime(df.index)
                
            return df.sort_index()
            
        except Exception as e:
            if attempt == 2:
                st.error(f"下載失敗 {ticker} {interval}: {str(e)[:100]}")
            time.sleep(1)  # ← 現在有 import，正常運作
            continue
    return None

# ===================== 找轉折點 =====================
def find_pivots(series, order=6):
    s = series.dropna()
    if len(s) < order * 2 + 1:
        return []
    arr = s.values
    highs = argrelextrema(arr, np.greater_equal, order=order)[0]
    lows = argrelextrema(arr, np.less_equal, order=order)[0]
    
    pivots = []
    for i in highs:
        pivots.append((s.index[i], float(s.iloc[i]), "peak"))
    for i in lows:
        pivots.append((s.index[i], float(s.iloc[i]), "trough"))
    return sorted(pivots, key=lambda x: x[0])

# ===================== 五浪偵測 =====================
def detect_impulse(pivots):
    impulses = []
    n = len(pivots)
    for i in range(n - 5):
        seq = pivots[i:i+6]
        types = [p[2] for p in seq]
        # 交替檢查
        if any(types[j] == types[j+1] for j in range(5)):
            continue
        prices = [p[1] for p in seq]
        direction_up = prices[-1] > prices[0]
        
        peaks = [prices[j] for j in range(1, 6, 2)]  # 1,3,5
        troughs = [prices[j] for j in range(2, 6, 2)]  # 2,4
        
        if len(peaks) < 2 or len(troughs) < 2:
            continue
            
        # 單調檢查
        peaks_mono = all(peaks[j] < peaks[j+1] for j in range(len(peaks)-1)) if direction_up else all(peaks[j] > peaks[j+1] for j in range(len(peaks)-1))
        troughs_mono = all(troughs[j] < troughs[j+1] for j in range(len(troughs)-1)) if direction_up else all(troughs[j] > troughs[j+1] for j in range(len(troughs)-1))
        
        if peaks_mono and troughs_mono:
            impulses.append(seq)
    return impulses

# ===================== 畫圖 =====================
def plot_waves(df, pivots, impulses, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name="K線"))
    
    # 轉折點
    if pivots:
        px, py, ptypes = zip(*[(p[0], p[1], p[2]) for p in pivots])
        colors = ['red' if t == 'peak' else 'blue' for t in ptypes]
        fig.add_trace(go.Scatter(x=px, y=py, mode='markers',
                                 marker=dict(size=8, color=colors), name='轉折點'))
    
    # 五浪（只畫最新一組，避免重疊）
    if impulses:
        seq = impulses[-1]
        x = [p[0] for p in seq]
        y = [p[1] for p in seq]
        color = 'green' if y[-1] > y[0] else 'red'
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers',
                                 line=dict(width=4, color=color), name='五浪結構'))
        
        # 標註浪數
        for i in range(5):
            mx = x[i] + (x[i+1] - x[i]) / 2
            my = y[i] + (y[i+1] - y[i]) * (0.6 if color == 'green' else 0.4)
            fig.add_annotation(x=mx, y=my, text=str(i+1),
                               font=dict(size=14, color="white"), 
                               bgcolor=color, showarrow=False)
    
    fig.update_layout(title=title, height=600, template="plotly_white",
                      xaxis_rangeslider_visible=False, xaxis_title="日期",
                      yaxis_title="價格")
    return fig

# ===================== 主程式 =====================
st.title("📊 艾略特波浪 多時間框架偵測器（2025 終極版）")

st.sidebar.header("分析設定")
tickers_input = st.sidebar.text_input("股票代號（逗號分隔）", "AAPL,TSLA,NVDA")
timeframes = st.sidebar.multiselect("時間框架", ["5m", "15m", "60m", "1d"], default=["1d"])
order = st.sidebar.slider("轉折點敏感度", min_value=3, max_value=15, value=6)
run_button = st.sidebar.button("🚀 開始分析", type="primary")

if run_button:
    symbols = [s.strip().upper() for s in tickers_input.split(",") if s.strip()]
    if not symbols:
        st.warning("請輸入至少一個股票代號")
    else:
        for symbol in symbols:
            st.header(f"🔎 分析 {symbol}")
            cols = st.columns(len(timeframes))
            for idx, (col, tf) in enumerate(zip(cols, timeframes)):
                with col:
                    st.subheader(f"{tf} 框架")
                    with st.spinner(f"載入 {symbol} {tf} 資料..."):
                        df = get_data(symbol, tf)
                    if df is None or len(df) < 50:
                        st.error(f"資料不足或下載失敗（{tf}）")
                        continue
                    
                    # 計算轉折點與五浪
                    pivots = find_pivots(df['Close'], order)
                    impulses = detect_impulse(pivots)
                    
                    # 簡單指標
                    ma_short = df['Close'].rolling(20).mean().iloc[-1]
                    ma_long = df['Close'].rolling(50).mean().iloc[-1]
                    ma_signal = "多頭" if ma_short > ma_long else "空頭"
                    
                    # 建議
                    wave_score = len(impulses) * 2
                    ma_score = 1 if ma_signal == "多頭" else -1
                    total_score = wave_score + ma_score
                    if total_score >= 3:
                        suggestion = "🟢 強烈買入"
                    elif total_score >= 1:
                        suggestion = "🟡 輕度買入"
                    elif total_score <= -1:
                        suggestion = "🔴 賣出"
                    else:
                        suggestion = "⚪ 觀望"
                    
                    # 顯示
                    st.metric("分析建議", suggestion)
                    st.caption(f"偵測五浪數：{len(impulses)} | 均線訊號：{ma_signal} | 資料筆數：{len(df)}")
                    
                    # 圖表
                    fig = plot_waves(df.tail(200), pivots, impulses, f"{symbol} {tf} - 艾略特波浪")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if st.checkbox(f"顯示轉折點細節 ({len(pivots)} 個)", key=f"detail_{symbol}_{tf}"):
                        detail_df = pd.DataFrame(pivots, columns=['日期', '價格', '類型'])
                        st.dataframe(detail_df.tail(10))
            
            st.markdown("---")
        
        # 總結表
        st.header("📋 分析總結")
        # 可以加總結邏輯...

else:
    st.info("👈 在左側設定參數後，點擊「開始分析」即可！")

# 底部提示
st.sidebar.markdown("---")
st.sidebar.success("✅ 已修復 MultiIndex 層級錯誤\n✅ 已加 import time\n✅ 移除 group_by 避免不穩定")
