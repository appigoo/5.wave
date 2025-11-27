# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import plotly.graph_objs as go
from io import StringIO
import base64

st.set_page_config(layout="wide", page_title="Elliott Multi-Timeframe Detector")

# -------------------------
# Helper indicators
# -------------------------
def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def obv(df):
    # On-balance volume
    obv_series = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv_series.append(obv_series[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv_series.append(obv_series[-1] - df['Volume'].iloc[i])
        else:
            obv_series.append(obv_series[-1])
    return pd.Series(obv_series, index=df.index)

def moving_average_trend(series, short=50, long=200):
    if len(series) < long:
        return None  # not enough data
    ma_short = series.rolling(short).mean().iloc[-1]
    ma_long = series.rolling(long).mean().iloc[-1]
    return "bull" if ma_short > ma_long else "bear"

# -------------------------
# Pivot detection (local peaks/troughs)
# -------------------------
def find_pivots(close_series, order=5):
    # returns list of tuples (idx, price, type)
    n = len(close_series)
    highs = argrelextrema(close_series.values, np.greater_equal, order=order)[0]
    lows = argrelextrema(close_series.values, np.less_equal, order=order)[0]
    tps = []
    for i in highs:
        tps.append((int(i), float(close_series.iloc[i]), "peak"))
    for i in lows:
        tps.append((int(i), float(close_series.iloc[i]), "trough"))
    tps.sort(key=lambda x: x[0])
    return tps

# -------------------------
# Detect 5-wave impulse + ABC corrective heuristics
# -------------------------
def alternates(types):
    return all(types[i] != types[i+1] for i in range(len(types)-1))

def detect_impulses(turning_points):
    # turning_points: [(idx, price, type), ...]
    res = []
    n = len(turning_points)
    for i in range(n-5):
        seq = turning_points[i:i+6]
        types = [s[2] for s in seq]
        if not alternates(types):
            continue
        start_price = seq[0][1]; end_price = seq[-1][1]
        # require net move (direction) and monotonic peaks/troughs
        if end_price > start_price:
            peaks = [s[1] for s in seq if s[2]=="peak"]
            troughs = [s[1] for s in seq if s[2]=="trough"]
            if len(peaks)>=2 and len(troughs)>=2:
                if all(peaks[i] < peaks[i+1] for i in range(len(peaks)-1)) and all(troughs[i] < troughs[i+1] for i in range(len(troughs)-1)):
                    res.append(seq)
        elif end_price < start_price:
            peaks = [s[1] for s in seq if s[2]=="peak"]
            troughs = [s[1] for s in seq if s[2]=="trough"]
            if len(peaks)>=2 and len(troughs)>=2:
                if all(peaks[i] > peaks[i+1] for i in range(len(peaks)-1)) and all(troughs[i] > troughs[i+1] for i in range(len(troughs)-1)):
                    res.append(seq)
    return res

def detect_abc_after_impulse(turning_points, impulse_seq):
    # find next 3 turning points after impulse end to form A-B-C (4 pts -> A B C moves)
    end_idx = impulse_seq[-1][0]
    # find index in turning_points list
    indices = [tp[0] for tp in turning_points]
    try:
        pos = indices.index(end_idx)
    except ValueError:
        return None
    if pos + 3 >= len(turning_points):
        return None
    abc_seq = turning_points[pos+1:pos+4]  # three moves -> 4 pts? but we take 3 pts as A B C peaks/troughs
    # ensure alternation
    if not alternates([s[2] for s in abc_seq]):
        return None
    return abc_seq

# -------------------------
# Fibonacci checks (tolerant)
# -------------------------
def fib_validate_impulse(seq):
    # seq: six turning points t0..t5 (idx, price, type)
    # compute wave magnitudes (absolute)
    p = [s[1] for s in seq]
    w1 = abs(p[1]-p[0])
    w2 = abs(p[2]-p[1])
    w3 = abs(p[3]-p[2])
    w4 = abs(p[4]-p[3])
    w5 = abs(p[5]-p[4])
    # tolerant checks
    c1 = (0.3 <= w2 / (w1+1e-9) <= 0.8)   # wave2 retrace 38-78%
    c2 = (w3 >= 1.3 * w1)                 # wave3 larger than wave1 (tolerance)
    c3 = (0.18 <= w4 / (w3+1e-9) <= 0.6)  # wave4 retrace of wave3
    c4 = (0.3 <= w5 / (w1+1e-9) <= 1.5)   # wave5 relative to wave1 (tolerant)
    valid = sum([c1,c2,c3,c4]) >= 3  # require at least 3/4 checks pass
    details = {"wave_lengths":[w1,w2,w3,w4,w5], "checks":[c1,c2,c3,c4]}
    return valid, details

# -------------------------
# Investment suggestion logic
# -------------------------
def compose_suggestion(ticker, timeframe, impulse, abc, fib_ok, macd_hist, obv_trend, ma_trend):
    reasons = []
    score = 0

    if impulse is None:
        reasons.append("未偵測到明確 5 浪結構")
    else:
        # determine where price stands relative to impulse seq
        last_wave_end_price = impulse[-1][1]  # p5 endpoint price
        # if abc exists -> likely correction in progress
        if abc is not None:
            reasons.append("偵測到 A-B-C 修正（修正中）")
            score -= 2
        else:
            reasons.append("5 浪結構可能尚未完成或剛完成")
            score += 0

        if fib_ok:
            reasons.append("通過斐波那契檢驗（增加信心）")
            score += 1
        else:
            reasons.append("未通過斐波那契檢驗（降低信心）")
            score -= 1

    # MACD: positive histogram -> momentum up
    if macd_hist is not None:
        if macd_hist[-1] > 0:
            reasons.append("MACD histogram 正值（動能偏多）")
            score += 1
        else:
            reasons.append("MACD histogram 負值（動能偏空）")
            score -= 1

    # OBV trend: last vs first
    if obv_trend is not None:
        if obv_trend == "up":
            reasons.append("OBV 上升（資金流入）")
            score += 1
        else:
            reasons.append("OBV 下降（資金流出）")
            score -= 1

    # MA trend filter
    if ma_trend == "bull":
        reasons.append("短中期均線看多（趨勢濾網：多頭）")
        score += 1
    elif ma_trend == "bear":
        reasons.append("短中期均線看空（趨勢濾網：空頭）")
        score -= 1

    # derive suggestion
    if score >= 2:
        suggestion = "多頭 (Buy/趨勢向上)"
    elif score <= -2:
        suggestion = "空頭 (Sell/趨勢向下)"
    else:
        suggestion = "觀望 (Hold/Wait)"

    # compose text explanation
    explanation = f"{ticker} {timeframe} => 建議：**{suggestion}**\n\n理由：\n"
    for r in reasons:
        explanation += f"- {r}\n"
    explanation += f"\n(綜合分數 {score})"
    return suggestion, explanation

# -------------------------
# Plot helper to draw waves and labels
# -------------------------
def plot_with_waves(df, turning_points, impulses_valid, abc_list, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    # mark turning points
    if turning_points:
        fig.add_trace(go.Scatter(
            x=[df.loc[p[0],'Date'] for p in turning_points],
            y=[p[1] for p in turning_points],
            mode='markers', marker=dict(size=7), name='Turning Points'
        ))
    # draw valid impulses
    for seq in impulses_valid:
        xs = [df.loc[p[0],'Date'] for p in seq]
        ys = [p[1] for p in seq]
        # connect successive points (t0..t5)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', line=dict(width=3), name='Impulse (valid)'))
        # annotate wave numbers 1..5 on segments
        for w in range(5):
            x0 = xs[w]; x1 = xs[w+1]
            y0 = ys[w]; y1 = ys[w+1]
            midx = x0 + (x1 - x0)/2
            midy = (y0 + y1)/2
            fig.add_annotation(x=midx, y=midy, text=str(w+1), showarrow=False, font=dict(size=12, color='black'))

    # draw ABC sequences
    for seq in abc_list:
        xs = [df.loc[p[0],'Date'] for p in seq]
        ys = [p[1] for p in seq]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', line=dict(dash='dash'), name='Corrective A-B-C'))
        # label A,B,C
        for i, lab in enumerate(['A','B','C']):
            fig.add_annotation(x=xs[i], y=ys[i], text=lab, showarrow=True, arrowhead=2)
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, template='plotly_white', height=600)
    return fig

# -------------------------
# Utility CSV download
# -------------------------
def to_download_link(df, filename="result.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">下載 {filename}</a>'
    return href

# -------------------------
# Streamlit UI
# -------------------------
st.title("📊 Multi-Timeframe Elliott Wave + MACD/OBV Filter")

st.sidebar.header("設定")
tickers_input = st.sidebar.text_input("輸入多支股票（逗號分隔）", value="TSLA, NVDA, AAPL")
timeframes = st.sidebar.multiselect("選擇時間框架（可多選）", ["5m", "60m", "1d"], default=["1d"])
order = st.sidebar.slider("局部極值 sensitivity (order)", 2, 12, 5)
ma_short = st.sidebar.number_input("短期 MA (天)", value=50)
ma_long = st.sidebar.number_input("長期 MA (天)", value=200)
run = st.sidebar.button("執行偵測")

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# -------------------------
# Main processing
# -------------------------
if run:
    overall_results = []  # collect per stock/timeframe summary rows
    for ticker in tickers:
        st.header(f"🔎 {ticker}")
        for tf in timeframes:
            st.subheader(f"Timeframe: {tf}")
            # download data
            try:
                df = yf.download(ticker, period="60d" if tf=="5m" else "180d", interval=tf, progress=False, auto_adjust=True)
            except Exception as e:
                st.error(f"下載 {ticker} {tf} 失敗：{e}")
                continue
            if df.empty:
                st.warning(f"{ticker} {tf} 無資料")
                continue
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date'])
            # ensure volume exists (for intraday yfinance may include volume)
            if 'Volume' not in df.columns:
                df['Volume'] = 0

            # indicators
            df['MA_short'] = df['Close'].rolling(ma_short).mean()
            df['MA_long'] = df['Close'].rolling(ma_long).mean()
            ma_trend = None
            if len(df) >= ma_long:
                ma_trend = "bull" if df['MA_short'].iloc[-1] > df['MA_long'].iloc[-1] else "bear"

            macd_line, signal_line, macd_hist = macd(df['Close'])
            df['MACD'] = macd_line
            df['MACD_signal'] = signal_line
            df['MACD_hist'] = macd_hist

            obv_series = obv(df)
            df['OBV'] = obv_series
            obv_trend = "up" if df['OBV'].iloc[-1] > df['OBV'].iloc[max(0, -int(len(df)/3))] else "down"

            # pivots & waves
            tps = find_pivots(df['Close'], order=order)
            impulses = detect_impulses(tps)
            impulses_valid = []
            impulses_info = []
            abc_list = []
            for seq in impulses:
                valid, details = fib_validate_impulse(seq)
                if valid:
                    impulses_valid.append(seq)
                    impulses_info.append(details)
                    # find ABC after impulse
                    abc = detect_abc_after_impulse(tps, seq)
                    if abc:
                        abc_list.append(abc)

            # pick the most recent valid impulse (if any)
            selected_impulse = impulses_valid[-1] if impulses_valid else None
            selected_abc = abc_list[-1] if abc_list else None
            fib_ok = bool(selected_impulse is not None)

            # compose suggestion
            suggestion, explanation = compose_suggestion(ticker, tf, selected_impulse, selected_abc, fib_ok,
                                                        df['MACD_hist'].values if 'MACD_hist' in df else None,
                                                        obv_trend, ma_trend)
            # display chart
            fig = plot_with_waves(df, tps, impulses_valid, abc_list, f"{ticker} {tf}")
            st.plotly_chart(fig, use_container_width=True)

            # show metrics and suggestion
            st.markdown(f"**建議： {suggestion}**")
            st.markdown("**判斷理由：**")
            st.write(explanation)

            # show small table of impulse details
            if selected_impulse:
                rows = []
                for i, p in enumerate(selected_impulse):
                    rows.append({"point_idx": p[0], "date": df.loc[p[0],'Date'], "price": p[1], "type": p[2]})
                st.table(pd.DataFrame(rows))

            # collect result
            overall_results.append({
                "ticker": ticker,
                "timeframe": tf,
                "suggestion": suggestion,
                "fib_ok": fib_ok,
                "ma_trend": ma_trend,
                "obv_trend": obv_trend
            })

            # small pause for readability
            st.markdown("---")

    # show summary and download
    if overall_results:
        summary_df = pd.DataFrame(overall_results)
        st.header("📋 分析總表")
        st.dataframe(summary_df)
        st.markdown(to_download_link(summary_df, filename="elliott_summary.csv"), unsafe_allow_html=True)

else:
    st.info("設定參數後按左側「執行偵測」開始。")
