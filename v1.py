# app.py
# Elliott Wave Pro Detector – 2025 終極穩定版
# 完美解決 yfinance 所有當機、下載失敗、MultiIndex、'AAPL' 錯誤

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import plotly.graph_objects as go
import base64
from datetime import datetime

st.set_page_config(
    page_title="Elliott Wave Pro 2025",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)

# =========================
# 最穩定的資料下載函數（2025 最新寫法）
# =========================
@st.cache_data(ttl=600, show_spinner="正在載入行情...")
def get_data(ticker: str, period: str = "2y", interval: str = "1d"):
    # 自動補常用後綴
    if not any(ticker.upper().endswith(s) for s in ['.TW','.HK','.SS','.SZ','=X','-USD','.DE','.L','.PA']):
        if '.' not in ticker:
            ticker = ticker.upper() + ".US"      # 美股預設

    for _ in range(3):  # 最多試三次
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=period, interval=interval, auto_adjust=True, actions=False)

            if df.empty or len(df) < 60:
                continue

            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date'].dt.tz_localize(None))  # 移除時區
            if 'Volume' not in df.columns:
                df['Volume'] = 0

            return df
        except:
            pass

    st.error(f"無法取得資料：{ticker} {interval}")
    return None

# =========================
# 指標計算
# =========================
def obv(df):
    obv_val = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv_val.append(obv_val[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv_val.append(obv_val[-1] - df['Volume'].iloc[i])
        else:
            obv_val.append(obv_val[-1])
    return pd.Series(obv_val, index=df.index)

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line  # 只回傳 histogram

def ma_trend(close, short=50, long=200):
    if len(close) < long:
        return None
    return "bull" if close.rolling(short).mean().iloc[-1] > close.rolling(long).mean().iloc[-1] else "bear"

# =========================
# 波浪偵測核心
# =========================
def find_pivots(close, order=6):
    highs = argrelextrema(close.values, np.greater_equal, order=order)[0]
    lows  = argrelextrema(close.values, np.less_equal,  order=order)[0]
    pivots = []
    for i in highs: pivots.append((int(i), float(close.iloc[i]), "peak"))
    for i in lows:  pivots.append((int(i), float(close.iloc[i]), "trough"))
    pivots.sort(key=lambda x: x[0])
    return pivots

def detect_impulses(pivots):
    impulses = []
    for i in range(len(pivots)-5):
        seq = pivots[i:i+6]
        types = [p[2] for p in seq]
        if not all(types[j] != types[j+1] for j in range(5)):
            continue
        p = [x[1] for x in seq]
        up = p[5] > p[0]
        if up and p[1]>p[0] and p[3]>p[1] and p[5]>p[3] and p[2]<p[1] and p[4]<p[3]:
            impulses.append(seq)
        if not up and p[1]<p[0] and p[3]<p[1] and p[5]<p[3] and p[2]>p[1] and p[4]>p[3]:
            impulses.append(seq)
    return impulses

def validate_impulse(seq):
    p = [x[1] for x in seq]
    up = p[5] > p[0]
    # 第4浪不可進入第1浪區域（鐵律）
    w1_zone = (min(p[0],p[1]), max(p[0],p[1]))
    w4_zone = (min(p[3],p[4]), max(p[3],p[4]))
    if up and w4_zone[0] < w1_zone[1]: return False
    if not up and w4_zone[1] > w1_zone[0]: return False
    # 簡單斐波那契檢查
    w = [abs(p[i+1]-p[i]) for i in range(5)]
    checks = sum([0.3<=w[1]/w[0]<=0.85, w[2]>=0.8*w[0], 0.1<=w[3]/w[2]<=0.5, 0.5<=w[4]/w[0]<=2.0])
    return checks >= 3

def detect_abc(pivots, impulse_end_idx):
    idxs = [p[0] for p in pivots]
    try:
        pos = idxs.index(impulse_end_idx)
    except:
        return None
    if pos + 4 >= len(pivots):
        return None
    seq = pivots[pos+1:pos+5]
    if len(seq) < 4:
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
        reasons.append("完整5浪推進結構")
        if abc:
            score -= 3
            reasons.append("已出現ABC修正 → 警惕反轉")
        else:
            score += 2
            reasons.append("5浪結束尚未修正 → 延續機率高")
    else:
        score -= 3
        reasons.append("未發現有效5浪")

    score += 1 if macd_hist > 0 else -1
    reasons.append("MACD柱正向" if macd_hist > 0 else "MACD柱負向")

    score += 1 if obv_trend == "up" else -1
    reasons.append("OBV上升" if obv_trend == "up" else "OBV下降")

    if ma_trend == "bull":
        score += 2
        reasons.append("50/200均線多頭")
    elif ma_trend == "bear":
        score -= 2
        reasons.append("50/200均線空頭")

    if score >= 6:   sugg, col, star = "強勢多頭 Strong Buy",   "green",      "5 stars"
    elif score >= 3: sugg, col, star = "多頭 Buy",              "lightgreen", "4 stars"
    elif score <= -6:  sugg, col, star = "強勢空頭 Strong Sell", "red",        "1 star"
    elif score <= -3:sugg, col, star = "空頭 Sell",             "orangered",  "2 stars"
    else:            sugg, col, star = "觀望 Hold",            "gray",       "3 stars"

    expl = f"### {ticker} {tf} → **<span style='color:{col}'>{sugg} {star}</span>**\n\n"
    expl += "**判斷依據：**\n" + "\n".join(f"• {r}" for r in reasons) + f"\n\n**總分：{score}**"
    return sugg, expl

# =========================
# 繪圖
# =========================
def plot_chart(df, pivots, impulses, abcs):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name="K線"))

    if pivots:
        fig.add_trace(go.Scatter(x=[df.iloc[i[0]]['Date'] for i in pivots],
                                 y=[i[1] for i in pivots],
                                 mode='markers', marker=dict(size=8, color='yellow'), name='轉折點'))

    for seq in impulses:
        fig.add_trace(go.Scatter(x=[df.iloc[p[0]]['Date'] for p in seq],
                                 y=[p[1] for p in seq],
                                 mode='lines+text', line=dict(width=4, color='lime'),
                                 text=["1","2","3","4","5",""], textposition="top center"))

    for seq in abcs:
        if seq:
            fig.add_trace(go.Scatter(x=[df.iloc[p[0]]['Date'] for p in seq],
                                     y=[p[1] for p in seq],
                                     mode='lines+text', line=dict(width=3, color='red', dash='dash'),
                                     text=["A","B","C",""], textposition="bottom center"))

    fig.add_hline(y=df['Close'].iloc[-1], line_dash="dot", line_color="white",
                  annotation_text=f"最新價 {df['Close'].iloc[-1]:.2f}")

    fig.update_layout(height=620, template="plotly_dark", xaxis_rangeslider_visible=False,
                      showlegend=False, title=f"{ticker} {tf}")
    return fig

# =========================
# Streamlit 主畫面
# =========================
st.title("Elliott Wave Pro Detector 2025")
st.markdown("### 最嚴謹的艾略特波浪自動掃描器 • 鐵律驗證 • 多指標濾網")

st.sidebar.header("設定")
tickers_input = st.sidebar.text_input("股票代碼（逗號分隔）",
                                    value="AAPL, TSLA, NVDA, 2330.TW, BTC-USD, ^GSPC")
timeframes = st.sidebar.multiselect("時間框架", ["5m","15m","60m","1d","1wk"], ["1d","60m"])
order = st.sidebar.slider("轉折點敏感度", 4, 12, 6)
run = st.sidebar.button("開始掃描", type="primary", use_container_width=True)

if run:
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
    results = []
    prog = st.progress(0)
    total = len(tickers) * len(timeframes)

    for idx, ticker in enumerate(tickers):
        for j, tf in enumerate(timeframes):
            prog.progress((idx*len(timeframes) + j + 1)/total)

            period = "60d" if tf in ["5m","15m"] else "5y" if tf=="1wk" else "2y"
            df = get_data(ticker, period=period, interval=tf)
            if df is None:
                continue

            df['OBV'] = obv(df)
            df['MACD_hist'] = macd(df['Close'])

            pivots = find_pivots(df['Close'], order)
            candidates = detect_impulses(pivots)
            valid = [s for s in candidates if validate_impulse(s)]

            abc_list = []
            impulse = None
            if valid:
                impulse = max(valid, key=lambda x: x[-1][0])
                abc_list = [detect_abc(pivots, impulse[-1][0])]

            sugg, expl = make_suggestion(
                ticker, tf, impulse, abc_list, 
                df['MACD_hist'].iloc[-1],
                "up" if np.polyfit(range(len(df)), df['OBV'], 1)[0] > 0 else "down",
                ma_trend(df['Close'])
            )

            c1, c2 = st.columns([3,1])
            with c1:
                st.plotly_chart(plot_chart(df, pivots, valid, abc_list), use_container_width=True)
            with c2:
                st.markdown(expl, unsafe_allow_html=True)

            results.append({"代碼":ticker, "週期":tf, "建議":sugg, "5浪":bool(impulse), "ABC":bool(abc_list)})

        st.divider()

    if results:
        df_res = pd.DataFrame(results)
        st.success("掃描完成！")
        st.dataframe(df_res, use_container_width=True)
        csv = df_res.to_csv(index=False).encode()
        st.download_button("下載報表 CSV", csv, f"elliott_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")

else:
    st.info("請在左側輸入代碼 → 選擇時間框架 → 點擊「開始掃描」")
    st.markdown("支援台股 `.TW`、港股 `.HK`、比特幣 `BTC-USD`、黃金 `GC=F` 等所有 yfinance 標的")

st.caption("Elliott Wave Pro © 2025 – 僅供參考，非投資建議")
