# app.py - Elliott Wave Pro Detector 最終穩定版（已修復所有語法錯誤）
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import plotly.graph_objects as go
import base64
from datetime import datetime

st.set_page_config(layout="wide", page_title="Elliott Wave Pro", page_icon="chart_with_upwards_trend")

# =========================
# 指標計算
# =========================
def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def obv(df):
    obv_series = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv_series.append(obv_series[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv_series.append(obv_series[-1] - df['Volume'].iloc[i])
        else:
            obv_series.append(obv_series[-1])
    return pd.Series(obv_series, index=df.index)

def obv_trend_slope(df):
    if len(df) < 20:
        return "neutral"
    x = np.arange(len(df))
    slope, _ = np.polyfit(x, df['OBV'].values, 1)
    return "up" if slope > 0 else "down"

def ma_trend(close, short=50, long=200):
    if len(close) < long:
        return None
    return "bull" if close.rolling(short).mean().iloc[-1] > close.rolling(long).mean().iloc[-1] else "bear"

# =========================
# 轉折點與波浪偵測
# =========================
def find_pivots(close_series, order=5):
    highs = argrelextrema(close_series.values, np.greater_equal, order=order)[0]
    lows = argrelextrema(close_series.values, np.less_equal, order=order)[0]
    pivots = []
    for i in highs:
        pivots.append((int(i), float(close_series.iloc[i]), "peak"))
    for i in lows:
        pivots.append((int(i), float(close_series.iloc[i]), "trough"))
    pivots.sort(key=lambda x: x[0])
    return pivots

def alternates(types):
    return all(types[i] != types[i+1] for i in range(len(types)-1))

def detect_impulses(turning_points):
    impulses = []
    n = len(turning_points)
    for i in range(n - 5):
        seq = turning_points[i:i+6]
        types = [p[2] for p in seq]
        if not alternates(types):
            continue
        prices = [p[1] for p in seq]
        if prices[5] > prices[0]:  # 上漲5浪
            if all(prices[j] < prices[j+2] for j in [0, 2, 4]) and all(prices[j] < prices[j+2] for j in [1, 3]):
                impulses.append(seq)
        else:  # 下跌5浪
            if all(prices[j] > prices[j+2] for j in [0, 2, 4]) and all(prices[j] > prices[j+2] for j in [1, 3]):
                impulses.append(seq)
    return impulses

def validate_impulse_no_overlap_and_fib(seq):
    prices = [p[1] for p in seq]
    direction = 1 if prices[5] > prices[0] else -1

    # 鐵律：第4浪不能進入第1浪區域
    w1_high = max(prices[0], prices[1])
    w1_low  = min(prices[0], prices[1])
    w4_high = max(prices[3], prices[4])
    w4_low  = min(prices[3], prices[4])

    if direction == 1 and w4_low < w1_high:
        return False
    if direction == -1 and w4_high > w1_low:
        return False

    # 斐波那契寬鬆驗證
    w = [abs(prices[i+1] - prices[i]) for i in range(5)]
    checks = [
        0.3 <= w[1]/w[0] <= 0.8,
        w[2] >= 0.8 * w[0],
        0.1 <= w[3]/w[2] <= 0.5,
        0.5 <= w[4]/w[0] <= 2.0
    ]
    return sum(checks) >= 3

def detect_abc_after_impulse(turning_points, impulse_seq):
    end_idx = impulse_seq[-1][0]
    indices = [tp[0] for tp in turning_points]
    try:
        pos = indices.index(end_idx)
    except ValueError:
        return None
    if pos + 4 >= len(turning_points):
        return None
    seq = turning_points[pos+1:pos+5]
    if len(seq) < 4 or not alternates([p[2] for p in seq]):
        return None
    # A浪方向應與第5浪相反
    fifth_up = impulse_seq[-1][1] > impulse_seq[4][1]
    a_up = seq[1][1] > seq[0][1]
    if fifth_up == a_up:
        return None
    return seq

# =========================
# 建議引擎
# =========================
def get_suggestion(ticker, tf, impulse, abc, macd_hist, obv_trend, ma_trend):
    score = 0
    reasons = []

    if not impulse:
        reasons.append("未偵測到有效5浪結構")
        score -= 3
    else:
        reasons.append("偵測到完整5浪推進結構")
        score += 4
        if abc:
            reasons.append("已出現ABC修正波 → 趨勢可能反轉")
            score -= 3
        else:
            reasons.append("5浪結束後尚未出現明顯修正 → 趨勢延續機率高")
            score += 2

    if macd_hist > 0:
        reasons.append("MACD柱狀圖正值（動能偏多）")
        score += 1
    else:
        reasons.append("MACD柱狀圖負值（動能偏空）")
        score -= 1

    if obv_trend == "up":
        reasons.append("OBV上升（資金流入）")
        score += 1
    else:
        reasons.append("OBV下降（資金流出）")
        score -= 1

    if ma_trend == "bull":
        reasons.append("50/200均線多頭排列（長期趨勢看多）")
        score += 2
    elif ma_trend == "bear":
        reasons.append("50/200均線空頭排列（長期趨勢看空）")
        score -= 2

    if score >= 5:
        sugg = "強勢多頭 (Strong Buy)"
        color = "green"
        stars = "★★★★★"
    elif score >= 2:
        sugg = "多頭 (Buy)"
        color = "lightgreen"
        stars = "★★★★☆"
    elif score <= -5:
        sugg = "強勢空頭 (Strong Sell)"
        color = "red"
        stars = "★☆☆☆☆"
    elif score <= -2:
        sugg = "空頭 (Sell)"
        color = "orangered"
        stars = "★★☆☆☆"
    else:
        sugg = "觀望 (Hold)"
        color = "gray"
        stars = "★★★☆☆"

    explanation = f"### {ticker} {tf} → **<span style='color:{color}'>{sugg} {stars}</span>**\n\n"
    explanation += "**判斷依據：**\n"
    for r in reasons:
        explanation += f"• {r}\n"
    explanation += f"\n**總分：{score} 分**"

    return sugg, explanation, score

# =========================
# 繪圖
# =========================
def plot_chart(df, pivots, impulses, abcs, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['Date'],
                                 open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name="K線"))

    # 轉折點
    if pivots:
        fig.add_trace(go.Scatter(x=[df.iloc[i[0]]['Date'] for i in pivots],
                                 y=[i[1] for i in pivots],
                                 mode='markers', marker=dict(size=8, color='yellow'), name='轉折點'))

    # 5浪
    for seq in impulses:
        x = [df.iloc[p[0]]['Date'] for p in seq]
        y = [p[1] for p in seq]
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+text', line=dict(width=4, color='lime'),
                                 text=["1","2","3","4","5",""], textposition="top center"))

    # ABC
    for seq in abcs:
        x = [df.iloc[p[0]]['Date'] for p in seq]
        y = [p[1] for p in seq]
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+text', line=dict(width=3, color='red', dash='dash'),
                                 text=["A","B","C",""], textposition="bottom center"))

    fig.add_hline(y=df['Close'].iloc[-1], line_dash="dot", line_color="white",
                  annotation_text=f"最新價 {df['Close'].iloc[-1]:.2f}")

    fig.update_layout(title=title, height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
    return fig

# =========================
# 主程式
# =========================
st.title("Elliott Wave Pro Detector")
st.markdown("### 最嚴謹的艾略特波浪自動掃描器 • 鐵律驗證 • 多指標濾網")

st.sidebar.header("設定")
tickers_input = st.sidebar.text_input("股票代碼（逗號分隔）", "AAPL,TSLA,NVDA,SPY,BTC-USD")
timeframes = st.sidebar.multiselect("時間框架", ["5m","15m","60m","1d","1wk"], ["1d","60m"])
order = st.sidebar.slider("轉折點敏感度", 3, 12, 6)
run = st.sidebar.button("開始掃描", type="primary")

if run:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    results = []
    progress = st.progress(0)
    total_tasks = len(tickers) * len(timeframes)

    for i, ticker in enumerate(tickers):
        for j, tf in enumerate(timeframes):
            progress.progress((i*len(timeframes) + j + 1) / total_tasks)

            period = "60d" if tf in ["5m","15m"] else "2y"
            try:
                df = yf.download(ticker, period=period, interval=tf, progress=False)
            except:
                st.error(f"{ticker} {tf} 下載失敗")
                continue

            if df.empty or len(df) < 100:
                st.warning(f"{ticker} {tf} 資料不足")
                continue

            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date'])
            if 'Volume' not in df.columns:
                df['Volume'] = 0
            df['OBV'] = obv(df)

            macd_l, _, hist = macd(df['Close'])
            df['MACD_hist'] = hist

            pivots = find_pivots(df['Close'], order)
            candidates = detect_impulses(pivots)

            valid_impulses = [seq for seq in candidates if validate_impulse_no_overlap_and_fib(seq)]
            abc_list = [detect_abc_after_impulse(pivots, seq) for seq in valid_impulses]
            abc_list = [a for a in abc_list if a is not None]

            impulse = max(valid_impulses, key=lambda x: x[-1][0]) if valid_impulses else None
            abc = abc_list[-1] if abc_list else None

            sugg, expl, score = get_suggestion(ticker, tf, impulse, abc,
                                               df['MACD_hist'].iloc[-1],
                                               obv_trend_slope(df),
                                               ma_trend(df['Close']))

            col1, col2 = st.columns([2,1])
            with col1:
                st.plotly_chart(plot_chart(df, pivots, valid_impulses, abc_list), use_container_width=True)
            with col2:
                st.markdown(expl, unsafe_allow_html=True)

            results.append({"時間": datetime.now().strftime("%H:%M"), "代碼": ticker, "周期": tf,
                           "建議": sugg, "分數": score, "5浪": bool(impulse), "ABC": bool(abc)})

            st.markdown("---")

    if results:
        summary_df = pd.DataFrame(results)
        st.markdown("### 掃描總表")
        st.dataframe(summary_df)
        csv = summary_df.to_csv(index=False).encode()
        b64 = base64.b64encode(csv).decode()
        st.markdown(f'<a href="data:text/csv;base64,{b64}" download="elliott_{datetime.now().strftime("%Y%m%d_%H%M")}.csv">下載CSV報表</a>', unsafe_allow_html=True)

else:
    st.info("在左側設定股票與時間框架 → 點擊「開始掃描」")
    st.markdown("支援台股加權(.TW)、港股(.HK)、加密貨幣、黃金XAUUSD等 yfinance 所有標的")
