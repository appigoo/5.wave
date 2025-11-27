# app.py
# Elliott Wave Pro Detector — 2025 終極穩定版
# 支援：美股、台股、港股、加密貨幣、黃金、外匯等所有 yfinance 標的

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import plotly.graph_objects as go
import base64
from datetime import datetime

st.set_page_config(
    page_title="Elliott Wave Pro",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)

# =========================
# 安全下載資料（解決 yfinance MultiIndex 問題）
# =========================
@st.cache_data(ttl=600, show_spinner=False)  # 10分鐘快取
def get_data(ticker: str, period: str = "2y", interval: str = "1d"):
    try:
        df = yf.download(tickers=ticker,
                         period=period,
                         interval=interval,
                         progress=False,
                         auto_adjust=True,
                         threads=True)

        # 關鍵修復：處理單股時出現的 MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker.split('.')[0].upper(), axis=1, level=0, drop_level=True)

        if df.empty or len(df) < 80:
            return None

        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])

        if 'Volume' not in df.columns:
            df['Volume'] = 0

        return df

    except Exception as e:
        st.error(f"下載失敗 {ticker} {interval}: {e}")
        return None

# =========================
# 指標計算
# =========================
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

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def ma_trend(close, short=50, long=200):
    if len(close) < long:
        return None
    return "bull" if close.rolling(short).mean().iloc[-1] > close.rolling(long).mean().iloc[-1] else "bear"

def obv_trend_slope(df):
    if len(df) < 20:
        return "neutral"
    x = np.arange(len(df))
    slope = np.polyfit(x, df['OBV'], 1)[0]
    return "up" if slope > 0 else "down"

# =========================
# 波浪偵測核心
# =========================
def find_pivots(close, order=6):
    pivots = []
    highs = argrelextrema(close.values, np.greater_equal, order=order)[0]
    lows = argrelextrema(close.values, np.less_equal, order=order)[0]
    for i in highs:
        pivots.append((int(i), float(close.iloc[i]), "peak"))
    for i in lows:
        pivots.append((int(i), float(close.iloc[i]), "trough"))
    pivots.sort(key=lambda x: x[0])
    return pivots

def detect_impulses(pivots):
    impulses = []
    for i in range(len(pivots) - 5):
        seq = pivots[i:i+6]
        types = [p[2] for p in seq]
        if not all(types[j] != types[j+1] for j in range(5)):
            continue
        prices = [p[1] for p in seq]
        up_trend = prices[5] > prices[0]
        if up_trend:
            if (prices[1] > prices[0] and prices[3] > prices[1] and prices[5] > prices[3] and
                prices[2] < prices[1] and prices[4] < prices[3]):
                impulses.append(seq)
        else:
            if (prices[1] < prices[0] and prices[3] < prices[1] and prices[5] < prices[3] and
                prices[2] > prices[1] and prices[4] > prices[3]):
                impulses.append(seq)
    return impulses

def validate_impulse(seq):
    p = [x[1] for x in seq]
    direction = 1 if p[5] > p[0] else -1

    # 鐵律：第4浪不能進入第1浪區域
    w1_high = max(p[0], p[1])
    w1_low  = min(p[0], p[1])
    w4_high = max(p[3], p[4])
    w4_low  = min(p[3], p[4])
    if direction == 1 and w4_low < w1_high:
        return False
    if direction == -1 and w4_high > w1_low:
        return False

    # 斐波那契寬鬆驗證（至少3條通過）
    w = [abs(p[i+1] - p[i]) for i in range(5)]
    checks = [
        0.30 <= w[1]/w[0] <= 0.85,
        w[2] >= 0.8 * w[0],
        0.10 <= w[3]/w[2] <= 0.50,
        0.5 <= w[4]/w[0] <= 2.0
    ]
    return sum(checks) >= 3

def detect_abc(pivots, impulse):
    if not impulse:
        return None
    end_idx = impulse[-1][0]
    idx_list = [p[0] for p in pivots]
    try:
        pos = idx_list.index(end_idx)
    except ValueError:
        return None
    if pos + 4 >= len(pivots):
        return None
    seq = pivots[pos+1:pos+5]
    if len(seq) < 4:
        return None
    types = [p[2] for p in seq]
    if not all(types[j] != types[j+1] for j in range(3)):
        return None
    # A浪應與第5浪反向
    fifth_up = impulse[-1][1] > impulse[-2][1]
    a_up = seq[1][1] > seq[0][1]
    if fifth_up == a_up:
        return None
    return seq

# =========================
# 建議引擎
# =========================
def make_suggestion(ticker, tf, impulse, abc, macd_hist, obv_trend, ma_trend):
    score = 0
    reasons = []

    if impulse:
        score += 4
        reasons.append("偵測到完整5浪推進結構")
        if abc:
            score -= 3
            reasons.append("已出現ABC修正波 → 警惕反轉")
        else:
            score += 2
            reasons.append("5浪結束後尚未修正 → 趨勢延續機率高")
    else:
        score -= 3
        reasons.append("未發現有效5浪結構")

    if macd_hist > 0:
        score += 1
        reasons.append("MACD柱狀圖正值（動能偏多）")
    else:
        score -= 1
        reasons.append("MACD柱狀圖負值（動能偏空）")

    if obv_trend == "up":
        score += 1
        reasons.append("OBV上升（資金流入）")
    else:
        score -= 1
        reasons.append("OBV下降（資金流出）")

    if ma_trend == "bull":
        score += 2
        reasons.append("50/200均線多頭排列")
    elif ma_trend == "bear":
        score -= 2
        reasons.append("50/200均線空頭排列")

    if score >= 6:
        sugg, color, stars = "強勢多頭 (Strong Buy)", "green", "★★★★★"
    elif score >= 3:
        sugg, color, stars = "多頭 (Buy)", "lightgreen", "★★★★☆"
    elif score <= -6:
        sugg, color, stars = "強勢空頭 (Strong Sell)", "red", "★☆☆☆☆"
    elif score <= -3:
        sugg, color, stars = "空頭 (Sell)", "orangered", "★★☆☆☆"
    else:
        sugg, color, stars = "觀望 (Hold)", "gray", "★★★☆☆"

    expl = f"### {ticker} {tf} → **<span style='color:{color}'>{sugg} {stars}</span>**\n\n"
    expl += "**判斷依據：**\n"
    for r in reasons:
        expl += f"• {r}\n"
    expl += f"\n**總分：{score} 分**"
    return sugg, expl, score

# =========================
# 繪圖
# =========================
def plot_chart(df, pivots, impulses, abcs, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['Date'],
                                 open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'],
                                 name="K線"))

    # 轉折點
    if pivots:
        px = [df.iloc[p[0]]['Date'] for p in pivots]
        py = [p[1] for p in pivots]
        fig.add_trace(go.Scatter(x=px, y=py, mode='markers',
                                 marker=dict(size=8, color='yellow', symbol='circle'),
                                 name='轉折點'))

    # 5浪
    for seq in impulses:
        x = [df.iloc[p[0]]['Date'] for p in seq]
        y = [p[1] for p in seq]
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+text',
                                 line=dict(width=4, color='lime'),
                                 text=["1","2","3","4","5",""],
                                 textposition="top center", name="5浪"))

    # ABC
    for seq in abcs:
        if seq:
            x = [df.iloc[p[0]]['Date'] for p in seq]
            y = [p[1] for p in seq]
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines+text',
                                     line=dict(width=3, color='red', dash='dash'),
                                     text=["A","B","C",""], textposition="bottom center"))

    latest = df['Close'].iloc[-1]
    fig.add_hline(y=latest, line_dash="dot", line_color="white",
                  annotation_text=f"最新價 {latest:.2f}")

    fig.update_layout(title=title, height=600, template="plotly_dark",
                      xaxis_rangeslider_visible=False, showlegend=False)
    return fig

# =========================
# Streamlit UI
# =========================
st.title("Elliott Wave Pro Detector")
st.markdown("### 最嚴謹的艾略特波浪自動掃描器 • 鐵律驗證 • 多指標濾網 • 2025最新版")

st.sidebar.header("設定")
tickers_input = st.sidebar.text_input("輸入股票代碼（多筆用逗號分隔）",
                                    value="AAPL, TSLA, NVDA, 2330.TW, BTC-USD, ^IXIC")
timeframes = st.sidebar.multiselect("時間框架", ["5m","15m","60m","1d","1wk"], ["1d","60m"])
order = st.sidebar.slider("轉折點敏感度 (order)", 3, 15, 6)
run = st.sidebar.button("開始掃描", type="primary", use_container_width=True)

if run:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    results = []
    progress_bar = st.progress(0)
    total = len(tickers) * len(timeframes)

    for i, ticker in enumerate(tickers):
        for j, tf in enumerate(timeframes):
            progress_bar.progress((i * len(timeframes) + j + 1) / total)

            period = "60d" if tf in ["5m","15m"] else "5y" if tf=="1wk" else "2y"

            df = get_data(ticker, period=period, interval=tf)
            if df is None or len(df) < 100:
                st.warning(f"{ticker} {tf} 資料不足")
                continue

            df['OBV'] = obv(df)
            _, _, hist = macd(df['Close'])
            df['MACD_hist'] = hist

            pivots = find_pivots(df['Close'], order)
            candidates = detect_impulses(pivots)
            valid_impulses = [s for s in candidates if validate_impulse(s)]
            abc_list = [detect_abc(pivots, s) for s in valid_impulses]
            abc_list = [a for a in abc_list if a]

            impulse = max(valid_impulses, key=lambda x: x[-1][0]) if valid_impulses else None
            abc = abc_list[-1] if abc_list else None

            sugg, expl, score = make_suggestion(
                ticker, tf, impulse, abc,
                df['MACD_hist'].iloc[-1],
                obv_trend_slope(df),
                ma_trend(df['Close'])
            )

            col1, col2 = st.columns([3,1])
            with col1:
                st.plotly_chart(plot_chart(df, pivots, valid_impulses, abc_list),
                                use_container_width=True)
            with col2:
                st.markdown(expl, unsafe_allow_html=True)

            results.append({
                "時間": datetime.now().strftime("%H:%M"),
                "代碼": ticker,
                "週期": tf,
                "建議": sugg,
                "總分": score,
                "5浪": "是" if impulse else "否",
                "ABC修正": "是" if abc else "否"
            })

        st.markdown("---")

    if results:
        summary = pd.DataFrame(results)
        st.success(f"掃描完成！共分析 {len(results)} 個項目")
        st.dataframe(summary, use_container_width=True)
        csv = summary.to_csv(index=False).encode()
        b64 = base64.b64encode(csv).decode()
        st.markdown(
            f'<a href="data:text/csv;base64,{b64}" download="elliott_scan_{datetime.now().strftime("%Y%m%d_%H%M")}.csv">'
            f'下載報表.csv</a>',
            unsafe_allow_html=True
        )
else:
    st.info("在左側輸入股票代碼 → 選擇時間框架 → 點擊「開始掃描」")
    st.markdown("支援：台股加權 `.TW`、港股 `.HK`、比特幣 `BTC-USD`、黃金 `GC=F` 等")

st.markdown("---")
st.caption("Elliott Wave Pro Detector © 2025 | 僅供參考，非投資建議")
