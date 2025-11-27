# elliott_pro_max_fixed.py
# 終極版：波浪 + MACD + OBV + 均線 多空濾網 → 極致準確！

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

st.set_page_config(page_title="艾略特波浪 PRO MAX", layout="wide")
st.title("艾略特波浪 PRO MAX 四重共振選股系統")
st.markdown("### 波浪結構 + MACD + OBV + 均線趨勢 → 買賣訊號精準到爆！")

# ==================== 技術指標 ====================
def add_indicators(df):
    df = df.copy()
    # MACD
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp12 - exp26
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['signal']

    # OBV
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    df['obv'] = df['obv'].fillna(0)

    # 均線
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    df['ma120'] = df['close'].rolling(120).mean()

    return df

def get_trend_score(df):
    """回傳 0~3 分的多頭強度"""
    last = df.iloc[-1]
    ma_bull = last['ma20'] > last['ma60'] > last['ma120']
    macd_golden = df['macd'].iloc[-2] < df['signal'].iloc[-2] and df['macd'].iloc[-1] > df['signal'].iloc[-1]
    obv_up = last['obv'] > df['obv'].rolling(20).mean().iloc[-1]
    return sum([ma_bull, macd_golden, obv_up])

# ==================== ZigZag ====================
def zigzag(df, deviation=5.0):
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    dates = df.index
    pivots = []
    last_price = close[0]
    last_idx = 0
    up = None

    for i in range(1, len(df)):
        if up is None:
            up = high[i] > last_price

        if up:
            if high[i] > last_price:
                last_price = high[i]
                last_idx = i
            if (last_price - low[i]) / last_price * 100 >= deviation:
                pivots.append((dates[last_idx], last_price, 'high'))
                last_price = low[i]
                last_idx = i
                up = False
        else:
            if low[i] < last_price:
                last_price = low[i]
                last_idx = i
            if (high[i] - last_price) / last_price * 100 >= deviation:
                pivots.append((dates[last_idx], last_price, 'low'))
                last_price = high[i]
                last_idx = i
                up = True

    # 加最後一根
    final_type = 'high' if up else 'low'
    if not pivots or pivots[-1][0] != dates[-1]:
        pivots.append((dates[-1], close[-1], final_type))

    return pd.DataFrame(pivots, columns=['date', 'price', 'type'])

# ==================== 彈性波浪辨識（超靈敏） ====================
def find_best_elliott(pivot_df):
    if len(pivot_df) < 5:
        pivot_df = pivot_df.copy()
        pivot_df['label'] = ""
        return pivot_df, False

    types = pivot_df['type'].tolist()
    # 已轉 list，安全！
    prices = pivot_df['price'].values
    labels = [""] * len(pivot_df)

    pattern_up   = ['high','low','high','low','high','low','high','low','high']
    pattern_down = ['low','high','low','high','low','high','low','high','low']

    best_score = 0
    best_i = -1
    direction = None

    for i in range(len(types)-4):
        seg = types[i:i+9]
        up_score = sum(a == b for a, b in zip(seg, pattern_up))
        down_score = sum(a == b for a, b in zip(seg, pattern_down))
        score = max(up_score, down_score)

        if score >= 5 and score > best_score:
            best_score = score
            best_i = i
            direction = "up" if up_score > down_score else "down"

    if best_i >= 0:
        wave_labels = ["", "1","2","3","4","5","A","B","C"] if direction == "up" else ["", "1","2","3","4","5","a","b","c"]
        pattern = pattern_up if direction == "up" else pattern_down
        label_idx = 1
        for j in range(best_i, min(best_i + 9, len(types))):
            if types[j] == pattern[j - best_i]:
                labels[j] = wave_labels[label_idx]
                label_idx += 1
                if label_idx >= len(wave_labels):
                    break

    pivot_df = pivot_df.copy()
    pivot_df['label'] = labels
    return pivot_df, best_score >= 5

# ==================== 四重共振訊號 ====================
def get_signal(pivot_df, df):
    labeled = pivot_df[pivot_df['label'] != ""]
    if labeled.empty:
        return "觀望", "無明確波浪"

    last_wave = labeled.iloc[-1]['label']
    score = get_trend_score(df)

    if last_wave == "3" and score >= 2:
        return "超強買入", "第三浪起飛 + 多頭共振"
    elif last_wave == "5" and score <= 1:
        return "強烈賣出", "第五浪末期 + 趨勢轉弱"
    elif last_wave == "C" and df['macd_hist'].iloc[-1] > 0:
        return "強力買入", "C浪落底 + MACD翻紅"
    elif last_wave in ["1","2"] and score >= 2:
        return "提前布局", "準備迎接第三浪"
    elif last_wave == "4":
        return "減碼等待", "第四浪整理中"
    else:
        return ["觀望", "輕度關注", "中度關注", "高度關注"][score], f"趨勢分數 {score}/3"

# ==================== 主介面 ====================
col1, col2 = st.columns([1, 4])

with col1:
    st.subheader("設定")
    ticker_input = st.text_area("股票代號（每行一檔）",
        value="""2330.TW
AAPL
TSLA
NVDA
BTC-USD
SMCI
AMD
0700.HK""", height=250)
    tickers = [t.strip() for t in ticker_input.split("\n") if t.strip()]

    period = st.selectbox("資料期間", ["1y", "2y", "3y", "5y", "max"], index=1)
    deviation = st.slider("波浪靈敏度 (%)", 3.0, 10.0, 4.8, 0.2)
    run = st.button("啟動 PRO MAX 掃盤", type="primary", use_container_width=True)

if run:
    if not tickers:
        st.error("請輸入股票代號")
        st.stop()

    results = []
    progress = st.progress(0)

    for idx, ticker in enumerate(tickers):
        progress.progress((idx + 1) / len(tickers))

        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if len(df) < 100:
            continue

        df = df[['Open','High','Low','Close','Volume']].copy()
        df.columns = ['open','high','low','close','volume']
        df = add_indicators(df)
        pivot_df = zigzag(df, deviation)
        pivot_df, has_wave = find_best_elliott(pivot_df)
        signal, reason = get_signal(pivot_df, df)

        # 畫三格圖
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=(f"{ticker} → {signal}", "MACD", "OBV"),
                            row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.05)

        # K線 + 波浪
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'], name="K線"), row=1, col=1)
        if not pivot_df[pivot_df['label'] != ""].empty:
            fig.add_trace(go.Scatter(x=pivot_df['date'], y=pivot_df['price'],
                                     mode='lines+markers+text', text=pivot_df['label'],
                                     textposition="top center", textfont=dict(size=20, color="yellow"),
                                     line=dict(color="orange", width=3), name="波浪"), row=1, col=1)

        # 均線
        for ma, color in zip(['ma20','ma60','ma120'], ['yellow','purple','white']):
            fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(color=color)), row=1, col=1)

        # MACD
        fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD', line=dict(color='cyan')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['signal'], name='Signal', line=dict(color='magenta')), row=2, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], name='Histogram'), row=2, col=1)

        # OBV
        fig.add_trace(go.Scatter(x=df.index, y=df['obv'], name='OBV', line=dict(color='purple')), row=3, col=1)

        fig.update_layout(height=900, template="plotly_dark", title_text=f"{ticker} | {signal} | {reason}")
        st.plotly_chart(fig, use_container_width=True)

        results.append({
            "代號": ticker,
            "最新價": f"{df['close'].iloc[-1]:.2f}",
            "波浪": "".join(pivot_df['label'].tolist()[-8:]),
            "趨勢分": get_trend_score(df),
            "訊號": signal,
            "原因": reason
        })

    # 總表
    if results:
        df_res = pd.DataFrame(results)
        def highlight(val):
            if "超強" in val or "強力" in val: return "background:lime; color:black; font-weight:bold"
            if "賣出" in val: return "background:red; color:white; font-weight:bold"
            return ""
        styled = df_res.style.applymap(highlight, subset=["訊號"])
        st.dataframe(styled, use_container_width=True)

        csv = df_res.to_csv(index=False).encode('utf-8-sig')
        st.download_button("下載報告", csv, "波浪四重共振選股.csv", "text/csv")

else:
    st.info("貼上股票代號 → 點擊按鈕 → 幾秒後看到專業級四重共振圖表與訊號！")
    st.balloons()
