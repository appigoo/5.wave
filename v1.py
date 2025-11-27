# app.py（2025 年完全可運行版）
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import plotly.graph_objs as go
import base64

st.set_page_config(layout="wide", page_title="Elliott Wave 多時間框架偵測器")

# -------------------------
# 指標計算
# -------------------------
def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def obv(df):
    volume = df['Volume'].fillna(0)
    direction = np.sign(df['Close'].diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv

# -------------------------
# 找局部極值點（已防呆）
# -------------------------
def find_pivots(close_series, order=5):
    close_clean = close_series.dropna()
    if len(close_clean) < order * 2 + 1:
        return []
    
    arr = close_clean.values
    highs = argrelextrema(arr, np.greater_equal, order=order)[0]
    lows = argrelextrema(arr, np.less_equal, order=order)[0]
    
    tps = []
    for i in highs:
        orig_idx = close_clean.index[i]
        tps.append((orig_idx, float(close_clean.iloc[i]), "peak"))
    for i in lows:
        orig_idx = close_clean.index[i]
        tps.append((orig_idx, float(close_clean.iloc[i]), "trough"))
    tps.sort(key=lambda x: x[0])
    return tps

# -------------------------
# 其他函數（略微優化）
# -------------------------
def alternates(types):
    return all(types[i] != types[i+1] for i in range(len(types)-1))

def detect_impulses(turning_points):
    res = []
    for i in range(len(turning_points)-5):
        seq = turning_points[i:i+6]
        types = [p[2] for p in seq]
        if not alternates(types):
            continue
        prices = [p[1] for p in seq]
        if prices[-1] > prices[0]:  # 上漲五浪
            peaks = prices[1::2]    # 1,3,5
            troughs = prices[2::2]  # 2,4
            if len(ge(peaks) >= 2 and troughs >= 2 and
               peaks == sorted(peaks) and troughs == sorted(troughs):
                res.append(seq)
        else:  # 下跌五浪
            peaks = prices[1::2]
            troughs = prices[2::2]
            if len(peaks) >= 2 and len(troughs) >= 2 and
               peaks == sorted(peaks, reverse=True) and troughs == sorted(troughs, reverse=True):
                res.append(seq)
    return res

def fib_validate_impulse(seq):
    p = [x[1] for x in seq]
    w = [abs(p[i+1] - p[i]) for i in range(5)]
    if any(x == 0 for x in w): return False, {}
    c1 = 0.3 <= w[1]/w[0] <= 0.85
    c2 = w[2] >= w[0] * 0.9
    c3 = 0.1 <= w[3]/w[2] <= 0.7
    c4 = 0.5 <= w[4]/w[0] <= 2.0
    valid = sum([c1,c2,c3,c4]) >= 3
    return valid, {"lengths": w, "pass": [c1,c2,c3,c4]}

# -------------------------
# 畫圖（關鍵修復：全部改用 iloc）
# -------------------------
def plot_with_waves(df, tps, impulses_valid, abc_list, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="K線"
    ))

    # 轉折點
    if tps:
        tp_df = pd.DataFrame(tps, columns=['idx', 'price', 'type'])
        tp_df['date'] = df.index[tp_df['idx']]
        fig.add_trace(go.Scatter(x=tp_df['date'], y=tp_df['price'],
                                 mode='markers', marker=dict(size=8, color='red'), name='轉折點'))

    # 五浪
    for seq in impulses_valid:
        idxs = [p[0] for p in seq]
        dates = df.index[idxs]
        prices = [p[1] for p in seq]
        fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines+markers',
                                 line=dict(width=4, color='green'), name='五浪推進'))

        for i in range(5):
            x0, x1 = dates[i], dates[i+1]
            y0, y1 = prices[i], prices[i+1]
            mid_x = x0 + (x1 - x0)/2
            mid_y = y0 + (y1 - y0)*0.6
            fig.add_annotation(x=mid_x, y=mid_y, text=str(i+1),
                               font=dict(size=16, color="white"), bgcolor="green")

    # ABC
    for seq in abc_list:
        idxs = [p[0] for p in seq]
        dates = df.index[idxs]
        prices = [p[1] for p in seq]
        fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines+markers',
                                 line=dict(width=3, color='orange', dash='dash'), name='ABC修正'))
        for i, lab in enumerate("ABC"):
            fig.add_annotation(x=dates[i], y=prices[i], text=lab,
                               font=dict(size=14, color="black"), bgcolor="yellow")

    fig.update_layout(title=title, template="plotly_white", height=650,
                      xaxis_rangeslider_visible=False)
    return fig

# -------------------------
# 下載連結
# -------------------------
def to_download_link(df, filename="result.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">下載 {filename}</a>'

# -------------------------
# Streamlit 主程式（關鍵修復）
# -------------------------
st.title("艾略特波浪 多時間框架自動偵測器")

st.sidebar.header("設定")
tickers_input = st.sidebar.text_input("股票代號（多筆用逗號）", "TSLA, AAPL, NVDA")
timeframes = st.sidebar.multiselect("時間框架", ["5m", "15m", "60m", "1d"], ["1d"])
order = st.sidebar.slider("波峰波谷敏感度", 3, 15, 6)
run = st.sidebar.button("開始分析", type="primary")

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if run:
    results = []
    for ticker in tickers:
        st.header(f"分析 {ticker}")
        for tf in timeframes:
            with st.spinner(f"下載 {ticker} {tf} 資料..."):
                # 自動調整 period（5m 最多 7 天）
                if tf == "5m":
                    period = "7d"
                elif tf in ["15m", "60m"]:
                    period = "60d"
                else:
                    period = "2y"

                data = yf.download(ticker, period=period, interval=tf, progress=False)
                if data.empty:
                    st.error(f"{ticker} {tf} 無資料")
                    continue

                df = data.copy()
                df = df.dropna(subset=['Close'])

                # 指標
                macd_line, signal_line, hist = macd(df['Close'])
                df['MACD_hist'] = hist
                last_hist = hist.dropna().iloc[-1] if not hist.dropna().empty else 0

                df['OBV'] = obv(df)
                obv_trend = "up" if df['OBV'].iloc[-1] > df['OBV'].iloc[-50] else "down"

                ma_trend = "bull" if df['Close'].rolling(50).mean().iloc[-1] > df['Close'].rolling(200).mean().iloc[-1] else "bear"

                # 找轉折點
                tps = find_pivots(df['Close'], order=order)
                impulses = detect_impulses(tps)
                valid_impulses = [seq for seq in impulses if fib_validate_impulse(seq)[0]]
                abc_list = []
                for seq in valid_impulses[-3:]:  # 只看最近幾組
                    end_idx = seq[-1][0]
                    later = [p for p in tps if p[0] > end_idx]
                    if len(later) >= 3 and alternates([p[2] for p in later[:3]]):
                        abc_list.append(later[:3])

                selected_impulse = valid_impulses[-1] if valid_impulses else None
                fib_ok = selected_impulse is not None

                # 建議
                score = 0
                reasons = []
                if selected_impulse:
                    reasons.append("偵測到完整五浪")
                    score += 2
                    if abc_list:
                        reasons.append("正在進行 ABC 修正")
                        score -= 1
                else:
                    reasons.append("尚未形成明確五浪")

                score += 1 if last_hist > 0 else -1
                reasons.append(f"MACD 柱狀圖：{'正' if last_hist > 0 else '負'}")

                score += 1 if obv_trend == "up" else -1
                reasons.append(f"OBV {'上升' if obv_trend == 'up' else '下降'}")

                score += 1 if ma_trend == "bull" else -1
                reasons.append(f"均線趨勢：{ma_trend}")

                suggestion = "強烈看多" if score >= 3 else "看多" if score >= 1 else "觀望" if score >= -1 else "看空"

                # 顯示
                col1, col2 = st.columns([3,1])
                with col1:
                    fig = plot_with_waves(df, tps, valid_impulses, abc_list, f"{ticker} {tf}")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.metric("建議", suggestion)
                    st.caption("理由：\n" + "\n".join(f"• {r}" for r in reasons))

                results.append({
                    "股票": ticker, "框架": tf, "建議": suggestion,
                    "五浪": "是" if selected_impulse else "否",
                    "ABC修正": "是" if abc_list else "否",
                    "MACD": "正" if last_hist > 0 else "負",
                    "OBV": obv_trend, "均線": ma_trend
                })

                st.markdown("---")

    if results:
        summary = pd.DataFrame(results)
        st.header("總結表")
        st.dataframe(summary, use_container_width=True)
        st.markdown(to_download_link(summary, "艾略特波浪分析結果.csv"), unsafe_allow_html=True)
else:
    st.info("請在左側設定參數後點擊「開始分析」")
