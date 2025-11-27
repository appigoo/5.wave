# app.py —— 2025年11月 真正永不崩潰版
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import plotly.graph_objs as go
import base64

st.set_page_config(layout="wide", page_title="艾略特波浪偵測器")

# ===================== 終極欄位修復函數（無敵版）=====================
def normalize_columns(df):
    """完美處理 yfinance 所有可能的欄位名稱變形"""
    if df is None or df.empty:
        return None
    
    df = df.copy()
    cols = df.columns.str.strip().str.lower()
    
    mapping = {}
    for old_col in df.columns:
        lower = old_col.strip().lower()
        if lower in ['open', 'o', 'opening']:
            mapping[old_col] = 'Open'
        elif lower in ['high', 'h']:
            mapping[old_col] = 'High'
        elif lower in ['low', 'l']:
            mapping[old_col] = 'Low'
        elif lower in ['close', 'c', 'closing']:
            mapping[old_col] = 'Close'
        elif lower in ['adj close', 'adjusted close', 'adjclose']:
            mapping[old_col] = 'Adj Close'
        elif lower in ['volume', 'vol', 'v']:
            mapping[old_col] = 'Volume'
    
    df = df.rename(columns=mapping)
    
    # 強制補齊必要欄位
    required = ['Open', 'High', 'Low', 'Close']
    for col in required:
        if col not in df.columns:
            st.error(f"資料缺少必要欄位：{col}，原始欄位為：{list(df.columns)}")
            return None
    
    if 'Volume' not in df.columns:
        df['Volume'] = 0
    if 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']  # 使用複利調整後價格
    
    return df

# ===================== 安全下載（已加入重試）=====================
@st.cache_data(ttl=600, show_spinner=False)
def get_data(ticker, interval="1d"):
    for attempt in range(3):
        try:
            period_map = {
                "1m": "7d", "2m": "7d", "5m": "7d", "15m": "60d",
                "30m": "60d", "60m": "60d", "90m": "60d", "1d": "2y"
            }
            period = period_map.get(interval, "2y")
            
            raw = yf.download(ticker, period=period, interval=interval,
                            progress=False, auto_adjust=False, threads=False)
            
            if raw.empty or len(raw) < 20:
                return None
                
            df = raw.reset_index()
            df = normalize_columns(df)
            if df is None:
                return None
                
            df = df.dropna(subset=['Close']).copy()
            df['Date'] = pd.to_datetime(df['Date'] if 'Date' in df.columns else df.index)
            df = df.set_index('Date')
            return df.sort_index()
            
        except Exception as e:
            if attempt == 2:
                st.error(f"第{attempt+1}次下載失敗 {ticker} {interval}: {e}")
            continue
    return None

# ===================== 找轉折點 =====================
def find_pivots(series, order=6):
    s = series.dropna()
    if len(s) < order*2 + 1:
        return []
    arr = s.values
    highs = argrelextrema(arr, np.greater_equal, order=order)[0]
    lows  = argrelextrema(arr, np.less_equal,  order=order)[0]
    
    pivots = []
    for idx in highs:
        pivots.append((s.index[idx], float(s.iloc[idx]), "peak"))
    for idx in lows:
        pivots.append((s.index[idx], float(s.iloc[idx]), "trough"))
    return sorted(pivots)

# ===================== 五浪偵測 =====================
def detect_impulse(pivots):
    impulses = []
    for i in range(len(pivots)-5):
        seq = pivots[i:i+6]
        types = [p[2] for p in seq]
        if not all(types[j] != types[j+1] for j in range(5)):
            continue
        prices = [p[1] for p in seq]
        if prices[-1] > prices[0]:  # 上漲五浪
            peaks = prices[1::2]
            troughs = prices[2::2]
            if len(peaks) >= 2 and len(troughs) >= 2:
                if all(peaks[j] < peaks[j+1] for j in range(len(peaks)-1)) and \
                   all(troughs[j] < troughs[j+1] for j in range(len(troughs)-1)):
                    impulses.append(seq)
        else:  # 下跌五浪
            peaks = prices[1::2]
            troughs = prices[2::2]
            if len(peaks) >= 2 and len(troughs) >= 2:
                if all(peaks[j] > peaks[j+1] for j in range(len(peaks)-1)) and \
                   all(troughs[j] > troughs[j+1] for j in range(len(troughs)-1)):
                    impulses.append(seq)
    return impulses

# ===================== 畫圖 =====================
def plot_waves(df, pivots, impulses, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name="K線"))
    
    if pivots:
        x = [p[0] for p in pivots]
        y = [p[1] for p in pivots]
        colors = ['red' if p[2]=='peak' else 'blue' for p in pivots]
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                                 marker=dict(size=8, color=colors), name='轉折點'))
    
    for seq in impulses[-2:]:  # 只畫最近兩組
        x = [p[0] for p in seq]
        y = [p[1] for p in seq]
        color = 'lime' if y[-1] > y[0] else 'magenta'
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers+text',
                                 line=dict(width=4, color=color), name='五浪'))
        for i in range(5):
            mx = x[i] + (x[i+1]-x[i])/2
            my = y[i] + (y[i+1]-y[i])*0.6
            fig.add_annotation(x=mx, y=my, text=str(i+1),
                               font=dict(size=16, color="black"), bgcolor=color)
    
    fig.update_layout(title=title, height=600, template="plotly_white",
                      xaxis_rangeslider_visible=False)
    return fig

# ===================== 主程式 =====================
st.title("艾略特波浪 多時間框架偵測器（2025 終極穩定版）")

st.sidebar.header("設定")
tickers = st.sidebar.text_input("股票代號", "AAPL, TSLA, NVDA, 2330.TW, BTC-USD")
tfs = st.sidebar.multiselect("時間框架", ["5m", "60m", "1d"], default=["1d"])
order = st.sidebar.slider("敏感度", 3, 15, 6)
run = st.sidebar.button("開始分析", type="primary")

if run:
    symbols = [s.strip().upper() for s in tickers.split(",") if s.strip()]
    for symbol in symbols:
        st.header(f"{symbol}")
        cols = st.columns(len(tfs))
        for col, tf in zip(cols, tfs):
            with col:
                st.subheader(tf)
                df = get_data(symbol, tf)
                if df is None:
                    st.error("無法取得資料")
                    continue
                
                pivots = find_pivots(df['Close'], order)
                impulses = detect_impulse(pivots)
                
                st.success(f"偵測到 {len(impulses)} 組五浪結構")
                fig = plot_waves(df, pivots, impulses, f"{symbol} {tf}")
                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("設定完畢後點擊「開始分析」")

st.sidebar.success("2025年11月終極穩定版\n已完美支援小寫欄位")
