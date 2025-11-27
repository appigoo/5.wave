# app.py - 專業版 Elliott Wave 多時間框架自動偵測器（2025 最新修訂版）
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import plotly.graph_objects as go
import base64
from datetime import datetime

st.set_page_config(layout="wide", page_title="Elliott Wave Pro Detector", page_icon="Chart Increasing")

# =========================
# 核心指標函數
# =========================
def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def obv_trend_slope(df):
    if len(df) < 10:
        return "neutral"
    x = np.arange(len(df))
    slope = np.polyfit(x, df['OBV'].values, 1)[0]
    return "up" if slope > 0 else "down"

def ma_trend(close, short=50, long=200):
    if len(close) < long:
        return None
    return "bull" if close.rolling(short).mean().iloc[-1] > close.rolling(long).mean().iloc[-1] else "bear"

# =========================
# 轉折點偵測
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

# =========================
# 5浪推進結構偵測（加強版）
# =========================
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
            peaks = prices[1::2]    # wave 2,4 低點
            troughs = prices[::2]   # wave 1,3,5 高點
            if len(peaks) >= 2 and len(troughs) >= 3:
                if all(peaks[j] < peaks[j+1] for j in range(len(peaks)-1)) and \
                   all(troughs[j] < troughs[j+1] for j in range(len(troughs)-1)):
                    impulses.append(seq)
        else:  # 下跌5浪
            peaks = prices[1::2]
            troughs = prices[::2]
            if len(peaks) >= 2 and len(troughs) >= 3:
                if all(peaks[j] > peaks[j+1] for j in range(len(peaks)-1)) and \
                   all(troughs[j] > troughs[j+1] for j in range(len(troughs)-1)):
                    impulses.append(seq)
    return impulses

# =========================
# 關鍵鐵律：波浪4不可重疊波浪1 + 斐波那契驗證
# =========================
def validate_impulse_structure(seq):
    if len(seq) != 6:
        return False, "點數不足"
    prices = [p[1] for p in seq]
    w1_start, w1_end = prices[0], prices[1]
    w4_start, w4_end = prices[3], prices[4]
    direction = 1 if prices[5] > prices[0] else -1

    # 鐵律：波浪4不能進入波浪1價格區域
    wave1_zone = (min(w1_start, w1_end), max(w1_start, w1_end))
    wave4_zone = (min(w4_start, w4_end), max(w4_start, w4_end))
    if direction == 1:  # 上漲
        if wave4_zone[0] < wave1_zone[1]:
            return False, "違反規則：第4浪進入第1浪區域"
    else:  # 下跌
        if wave4_zone[1] > wave1_zone[0]:
            return False, "違反規則：第4浪進入第1浪區域"

    # 斐波那契比例（寬鬆版，至少3項通過）
    w1 = abs(prices[1] - prices[0])
    w2 = abs(prices[2] - prices[1])
    w3 = abs(prices[3] - prices[2])
    w4 = abs(prices[4] - prices[3])
    w5 = abs(prices[5] - prices[4])

    checks = [
        0.30 <= w2 / (w1 + 1e-8) <= 0.80,   # 2浪回吐38%-78%
        w3 >= 0.9 * w1,                     # 3浪通常最長（放寬）
        0.15 <= w4 / (w3 + 1e-8) <= 0.50,   # 4浪回吐不超過50%
        0.5 <= w5 / (w1 + 1e-8) <= 1.8      # 5浪常與1浪接近或延伸
    ]
    passed = sum(checks)
    return passed >= 3, f"斐波那契通過 {passed}/4"

# =========================
# ABC修正波偵測（修正後取4點）
# =========================
def detect_abc_after_impulse(turning_points, impulse_seq):
    end_idx = impulse_seq[-1][0]
    indices = [tp[0] for tp in turning_points]
    try:
        pos = indices.index(end_idx)
    except ValueError:
        return None
    if pos + 4 >= len(turning_points):
        return None
    abc_seq = turning_points[pos+1:pos+5]  # 4 points → A→B→C
    types = [p[2] for p in abc_seq]
    if len(types) < 4 or not alternates(types):
        return None
    # A浪應與第5浪反向
    impulse_dir = "up" if impulse_seq[-1][1] > impulse_seq[0][1] else "down"
    a_move_up = abc_seq[1][1] > abc_seq[0][1]
    if (impulse_dir == "up" and a_move_up) or (impulse_dir == "down" and not a_move_up):
        return None
    return abc_seq

# =========================
# 投資建議引擎（加分數 + 星級）
# =========================
def get_suggestion(ticker, tf, impulse, abc, fib_ok, macd_hist_last, obv_trend, ma_trend):
    score = 0
    reasons = []

    if impulse is None:
        reasons.append("未偵測到完整5浪結構")
        score -= 2
    else:
        if abc is not None:
            reasons.append("已出現A-B-C修正波（趨勢可能轉折）")
            score -= 2
        # 修正中偏保守
        else:
            reasons.append("5浪結構已完成且無明顯修正（趨勢延續機率高）")
            score += 3

        reasons.append("通過波浪4不可重疊鐵律 + 斐波那契驗證" if fib_ok else "斐波那契比例未達標")
        score += 2 if fib_ok else -1

    # 技術指標加分
    if macd_hist_last > 0:
        reasons.append("MACD柱狀體為正（動能偏多）")
        score += 1
    else:
        reasons.append("MACD柱狀體為負（動能偏空）")
        score -= 1

    if obv_trend == "up":
        reasons.append("OBV上升（資金流入）")
        score += 1
    elif obv_trend == "down":
        reasons.append("OBV下降（資金流出）")
        score -= 1

    if ma_trend == "bull":
        reasons.append("50/200均線多頭排列（長期趨勢看多）")
        score += 2
    elif ma_trend == "bear":
        reasons.append("50/200均線空頭排列（長期趨勢看空）")
        score -= 2

    # 最終判斷
    if score >= 5:
        sugg = "強力多頭 (Strong Buy)"
        stars = "★★★★★"
        color = "green"
    elif score >= 2:
        sugg = "多頭 (Buy)"
        stars = "★★★★☆"
        color = "lightgreen"
    elif score <= -5:
        sugg = "強力空頭 (Strong Sell)"
        stars = "★☆☆☆☆"
        color = "red"
    elif score <= -2:
        sugg = "空頭 (Sell)"
        stars = "★★☆☆☆"
        color = "orangered"
    else:
        sugg = "觀望 (Hold)"
        stars = "★★★☆☆"
        color = "gray"

    "

    explanation = f"### {ticker} {tf} → **<span style='color:{color}'>{sugg} {stars}</span>**\n\n"
    explanation += "**判斷依據：**\n"
    for r in reasons:
        explanation += f"• {r}\n"
    explanation += f"\n**綜合信心分數：{score} 分**"

    return sugg, explanation, score, stars

# =========================
# 繪圖函數（標記最新價 + 美化）
# =========================
def plot_with_waves(df, pivots, valid_impulses, abc_seqs, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name="K線"))

    # 轉折點
    if pivots:
        px = [df.iloc[p[0]]['Date'] for p in pivots]
        py = [p[1] for p in pivots]
        fig.add_trace(go.Scatter(x=px, y=py, mode='markers', marker=dict(size=8, color='yellow', symbol='circle'), name='轉折點'))

    # 標記有效5浪
    for seq in valid_impulses:
        xs = [df.iloc[p[0]]['Date'] for p in seq]
        ys = [p[1] for p in seq]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers+text',
                                 line=dict(width=4, color='royalblue'),
                                 text=[f"W{i+1}" for i in range(6)],
                                 textposition="top center",
                                 name="5浪推進"))

    # ABC修正波
    for seq in abc_seqs:
        xs = [df.iloc[p[0]]['Date'] for p in seq]
        ys = [p[1] for p in seq]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers',
                                 line=dict(width=3, color='crimson', dash='dot'),
                                 text=['A','B','C','?'], textposition="bottom center",
                                 name="ABC修正"))

    # 最新價格標記
    latest_price = df['Close'].iloc[-1]
    fig.add_hline(y=latest_price, line_dash="dot", line_color="white",
                  annotation_text=f"最新價 {latest_price:.2f}", annotation_position="top right")

    fig.update_layout(title=title, template="plotly_dark", height=650,
                      xaxis_rangeslider_visible=False, showlegend=False)
    return fig

# =========================
# 下載連結
# =========================
def download_link(df, filename="elliott_report.csv"):
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:text/csv;base64,{b64}" download="{filename}">下載 {filename}</a>'

# =========================
# Streamlit UI
# =========================
st.title("Elliott Wave Pro Detector - 專業艾略特波浪多時間框架掃描器")
st.markdown("### 自動偵測 5浪推進 + ABC修正 + 嚴格鐵律驗證 + 多指標濾網")

st.sidebar.header("設定參數")
tickers_input = st.sidebar.text_input("股票代碼（多筆用逗號分隔）", value="AAPL, TSLA, NVDA, SPY, QQQ, BTC-USD")
timeframes = st.sidebar.multiselect("時間框架", ["5m", "15m", "60m", "1d", "1wk"], default=["1d", "60m"])
order = st.sidebar.slider("轉折點敏感度 (order)", 3, 15, 6)
run = st.sidebar.button("開始掃描", type="primary")

if run:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    results = []
    progress = st.progress(0)
    total = len(tickers) * len(timeframes)
    step = 0

    for ticker in tickers:
        st.markdown(f"## {ticker}")
        cols = st.columns(len(timeframes))
        for idx, tf in enumerate(timeframes):
            with cols[idx]:
                st.subheader(tf)

                period = "60d" if tf in ["5m", "15m"] else "2y" if tf == "1wk" else "1y"
                try:
                    df = yf.download(ticker, period=period, interval=tf, progress=False, auto_adjust=True)
                except:
                    st.error("下載失敗")
                    continue
                if df.empty or len(df) < 100:
                    st.warning("資料不足")
                    continue

                df = df.reset_index()
                df['Date'] = pd.to_datetime(df['Date'])
                if 'Volume' not in df.columns:
                    df['Volume'] = 0
               

                # 指標
                df['OBV'] = obv(df) if df['Volume'].sum() > 0 else pd.Series(np.zeros(len(df)))
                macd_line, signal_line, hist = macd(df['Close'])
                df['MACD_hist'] = hist

                # 偵測
                pivots = find_pivots(df['Close'], order=order)
                candidates = detect_impulses(pivots)
                valid_impulses = []
                abc_list = []

                for seq in candidates:
                    ok, msg = validate_impulse_structure(seq)
                    if ok:
                        valid_impulses.append(seq)
                        abc = detect_abc_after_impulse(pivots, seq)
                        if abc:
                            abc_list.append(abc)

                # 選最近完成的結構
                impulse = max(valid_impulses, key=lambda x: x[-1][0]) if valid_impulses else None
                abc = abc_list[-1] if abc_list else None
                fib_ok = impulse is not None

                # 建議
                sugg, expl, score, stars = get_suggestion(
                    ticker, tf, impulse, abc, fib_ok,
                    df['MACD_hist'].iloc[-1],
                    obv_trend_slope(df),
                    ma_trend(df['Close'])
                )

                st.markdown(expl, unsafe_allow_html=True)
                fig = plot_with_waves(df, pivots, valid_impulses, abc_list if abc else [], f"{ticker} {tf}")
                st.plotly_chart(fig, use_container_width=True)

                results.append({
                    "時間": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "代碼": ticker,
                    "週期": tf,
                    "建議": sugg,
                    "信心分數": score,
                    "星級": stars,
                    "5浪完成": "是" if impulse else "否",
                    "ABC修正": "是" if abc else "否"
                })

            step += 1
            progress.progress(step / total)

    # 總表
    if results:
        summary = pd.DataFrame(results)
        st.markdown("## 掃描總表")
        st.dataframe(summary, use_container_width=True)
        st.markdown(download_link(summary, f"elliott_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"),
                    unsafe_allow_html=True)
else:
    st.info("在左側設定股票與時間框架後，點擊「開始掃描」即可使用")
    st.markdown("支援：美股、港股、台股加權、比特幣、黃金等 yfinance 涵蓋的所有標的")
