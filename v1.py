# elliott_pro_max.py
# 終極版：艾略特波浪 + MACD + OBV + 趨勢濾網 四重共振選股神器

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

st.set_page_config(page_title="艾略特波浪 PRO MAX", layout="wide")
st.title("艾略特波浪 PRO MAX 四重共振版")
st.markdown("### 波浪結構 + MACD金叉死叉 + OBV趨勢 + 均線多頭排列 → 極致準確買賣訊號")

# ==================== 技術指標計算 ====================
def add_indicators(df):
    """加入 MACD、OBV、20/60/120均線"""
    # MACD
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    # MACD線
    df['signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['signal']

    # OBV
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    df['obv'] = df['obv'].fillna(0)

    # 均線
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    df['ma120'] = df['close'].rolling(120).mean()

    return df

def get_trend_filter(df):
    """趨勢濾網：多頭排列 + MACD金叉 + OBV上升"""
    last = df.iloc[-1]
    prev = df.iloc[-2]

    ma_bull = last['ma20'] > last['ma60'] > last['ma120']
    macd_bull = prev['macd'] < prev['signal'] and last['macd'] > last['signal']  # 金叉
    obv_up = last['obv'] > df['obv'].iloc[-20:].mean()

    score = sum([ma_bull, macd_bull, obv_up])
    return score  # 0~3分

# ==================== 改良版彈性波浪辨識（超級靈敏）===================
def find_best_elliott_flexible(pivot_df):
    if len(pivot_df) < 5:
        pivot_df = pivot_df.copy()
        pivot_df['label'] = ""
        return pivot_df, False

    types = pivot_df['type'].tolist()
    prices = pivot_df['price'].values
    labels = [""] * len(pivot_df)

    pattern_up   = ['high','low','high','low','high','low','high','low','high']
    pattern_down = ['low','high','low','high','low','high','low','high','low']

    best_match = 0
    best_i = -1
    direction = None

    for i in range(len(types)-4):
        seg = types[i:i+9]
        up_score = sum(a==b for a,b in zip(seg, pattern_up))
        down_score = sum(a==b for a,b in zip(seg, pattern_down))
        score = max(up_score, down_score)

        if score >= 5 and score > best_match:
            best_match = score
            best_i = i
            direction = "up" if up_score > down_score else "down"

    if best_i >= 0:
        wave_labels = ["", "1","2","3","4","5","A","B","C"] if direction == "up" else ["", "1","2","3","4","5","a","b","c"]
        matched_pattern = pattern_up if direction == "up" else pattern_down
        for j = 0
        for k in range(best_i, min(best_i+9, len(types))):
            if types[k] == matched_pattern[k - best_i]:
                labels[k] = wave_labels[j]
                j += 1

    pivot_df = pivot_df.copy()
    pivot_df['label'] = labels
    return pivot_df, best_match >= 5

# ==================== 四重共振訊號引擎 ====================
def get_pro_signal(pivot_df, df):
    labeled = pivot_df[pivot_df['label'] != ""]
    if labeled.empty:
        return "觀望", "無波浪結構", "gray"

    last_label = labeled.iloc[-1]['label']
    trend_score = get_trend_filter(df)

    # 核心邏輯：波浪位置 + 趨勢濾網共振
    if last_label == "3" and trend_score >= 2:
        return "超強買入", "第三浪 + 多頭共振 → 主升段啟動！", "lime"
    elif last_label == "5" and trend_score <= 1:
        return "強烈賣出", "第五浪末期 + 趨勢轉弱 → 逃頂訊號", "red"
    elif last_label == "C" and df.iloc[-1]['macd_hist'] > 0:
        return "強力買入", "C浪落底 + MACD翻紅 → 反轉起漲", "green"
    elif last_label in ["1", "2"] and trend_score >= 2:
        return "進場布局", "第一/二浪 + 多頭排列 → 準備第三浪", "lightgreen"
    elif last_label == "4":
        return "減碼等待", "第四浪整理，準備第五浪", "orange"
    else:
        strength = ["觀望", "輕度關注", "中度關注", "高度關注"][trend_score]
        return strength, f"波浪{last_label} + 趨勢分數 {trend_score}/3", "yellow"

# ==================== 主程式 ====================
col1, col2 = st.columns([1,4])

with col1:
    st.subheader("設定")
    tickers = st.text_area("股票代號（每行一檔）",
        value="""2330.TW
AAPL
TSLA
NVDA
BTC-USD
SMCI
AMD""", height=250).split("\n")
    tickers = [t.strip() for t in tickers if t.strip()]

    period = st.selectbox("期間", ["1y","2y","3y","max"], index=1)
    deviation = st.slider("波浪靈敏度", 3.0, 10.0, 4.8, 0.2)

    run = st.button("啟動 PRO MAX 掃盤", type="primary", use_container_width=True)

if run:
    results = []
    for ticker in tickers:
        with st.spinner(f"分析 {ticker}..."):
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if len(df) < 100:
                continue

            df = df[['Open','High','Low','Close','Volume']].copy()
            df.columns = ['open','high','low','close','volume']
            df = add_indicators(df)

            # ZigZag
            pivots = []
            last_price = df['close'].iloc[0]
            last_idx = 0
            up = None
            for i in range(1, len(df)):
                if up is None:
                    up = df['high'].iloc[i] > last_price
                if up:
                    if df['high'].iloc[i] > last_price:
                        last_price = df['high'].iloc[i]
                        last_idx = i
                    if (last_price - df['low'].iloc[i])/last_price > 0.01*deviation:
                        pivots.append([df.index[last_idx], last_price, 'high'])
                        last_price = df['low'].iloc[i]
                        last_idx = i
                        up = False
                else:
                    if df['low'].iloc[i] < last_price:
                        last_price = df['low'].iloc[i]
                        last_idx = i
                    if (df['high'].iloc[i] - last_price)/last_price > 0.01*deviation:
                        pivots.append([df.index[last_idx], last_price, 'low'])
                        last_price = df['high'].iloc[i]
                        last_idx = i
                        up = True
            if pivots:
                pivots.append([df.index[-1], df['close'].iloc[-1], 'high' if up else 'low'])
            pivot_df = pd.DataFrame(pivots, columns=['date','price','type'])
            pivot_df, has_wave = find_best_elliott_flexible(pivot_df)
            signal, reason, color = get_pro_signal(pivot_df, df)

            # 畫圖（K線 + 波浪 + MACD + OBV）
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                subplot_titles=(f"{ticker} - {signal}", "MACD", "OBV"),
                                row_heights=[0.6, 0.2, 0.2])

            # K線 + 波浪
            fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                         low=df['low'], close=df['close'], name="K線"), row=1, col=1)
            if not pivot_df[pivot_df['label']!=''].empty:
                fig.add_trace(go.Scatter(x=pivot_df['date'], y=pivot_df['price'],
                                         mode='lines+markers+text', text=pivot_df['label'],
                                         textposition="top center", line=dict(color='orange', width=3),
                                         textfont=dict(size=20, color="yellow"), name="波浪"), row=1, col=1)

            # 均線
            for ma in ['ma20','ma60','ma120']:
                fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma), row=1, col=1)

            # MACD
            fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['signal'], name='Signal'), row=2, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], name='Hist'), row=2, col=1)

            # OBV
            fig.add_trace(go.Scatter(x=df.index, y=df['obv'], name='OBV', line=dict(color='purple')), row=3, col=1)

            fig.update_layout(height=f"{ticker}｜{signal}｜{reason}", height=900, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            results.append({
                "代號": ticker,
                "波浪": ''.join(pivot_df['label'].tolist()[-6:]),
                "趨勢分": get_trend_filter(df),
                "訊號": signal,
                "原因": reason
            })

    # 總表
    if results:
        df_res = pd.DataFrame(results)
        def color_signal(val):
            if "超強" in val or "強力" in val: return "background:lime; color:black"
            if "賣出" in val: return "background:red; color:white"
            return ""
        styled = df_res.style.applymap(color_signal, subset=["訊號"])
        st.dataframe(styled, use_container_width=True)

else:
    st.info("這是目前最強的艾略特波浪選股工具：四重共振，極致準確！")
    st.balloons()
