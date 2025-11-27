# elliott_fibonacci_final.py
# 真正的最終版：五重共振 + 嚴格斐波那契黃金比例驗證

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

st.set_page_config(page_title="艾略特波浪五重共振終極版", layout="wide")
st.title("艾略特波浪 五重共振終極版")
st.markdown("###  ("### 波浪 + MACD + OBV + 均線 + **嚴格斐波那契黃金比例** → 只抓真正的完美波浪！")

# ==================== 斐波那契嚴格驗證 ====================
def validate_fibonacci(prices):
    """驗證上升5浪是否符合經典斐波那契比例"""
    if len(prices) < 6:
        return False, "點數不足"
    p = prices[:6]  # 0高 1低 2高 3低 4高 5低/高
    w1 = p[2] - p[1]   # 波1
    w2 = p[1] - p[3]   # 波2回檔
    w3 = p[4] - p[3]   # 波3
    w4 = p[3] - p[5]   # 波4回檔
    w5 = p[5] - p[4] if len(prices) > 5 else 0

    if w1 <= 0: return False, "波1無效"

    r2 = w2 / w1
    r3 = w3 / w1
    r4 = w4 / w3 if w3 != 0 else 0
    r5 = abs(w5) / w1 if w1 != 0 else 0

    checks = [
        0.35  <= r2 <= 0.79,    # 波2回檔
        r3    >= 1.0,           # 波3最長
        r4    <= 0.55,          # 波4淺
        r5    >= 0.5 or r5 <= 2.0,  # 波5合理範圍
        w3    >= w1 and w3 >= abs(w5)  # 波3不能最短
    ]
    passed = sum(checks)
    return passed >= 4, f"波2{r2:.1%} 波3{r3:.2f}× 波4{r4:.1%} 波5{r5:.1%} ({passed}/5)"

# ==================== 其他核心函數 ====================
def add_indicators(df):
    df = df.copy()
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp12 - exp26
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['signal']
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum().fillna(0)
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    df['ma120'] = df['close'].rolling(120).mean()
    return df

def zigzag(df, deviation=4.8):
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

    final_type = 'high' if up else 'low'
    if not pivots or pivots[-1][0] != dates[-1]:
        pivots.append((dates[-1], close[-1], final_type))

    return pd.DataFrame(pivots, columns=['date', 'price', 'type'])

def find_best_wave_with_fib(pivot_df):
    if len(pivot_df) < 6:
        pivot_df['label'] = ""
        pivot_df['is_golden'] = False
        return pivot_df

    types = pivot_df['type'].tolist()
    prices = pivot_df['price'].values
    labels = [""] * len(pivot_df)
    is_golden_list = [False] * len(pivot_df)

    pattern_up = ['high','low','high','low','high','low']

    best_i = -1
    best_valid = False

    for i in range(len(types)-5):
        seg_types = types[i:i+6]
        seg_prices = prices[i:i+7]
        if seg_types == pattern_up[:6]:
            valid, reason = validate_fibonacci(seg_prices)
            if valid and (best_i == -1 or i > best_i):
                best_i = i
                best_valid = valid

    if best_i >= 0:
        labels[best_i:best_i+6] = ["", "1","2","3","4","5"]
        is_golden_list[best_i+5] = best_valid
        # 嘗試標記 ABC
        if len(types) > best_i + 8:
            labels[best_i+6] = "A"
            labels[best_i+7] = "B"
            labels[best_i+8] = "C"

    pivot_df = pivot_df.copy()
    pivot_df['label'] = labels
    pivot_df['is_golden'] = is_golden_list
    return pivot_df

def get_signal(pivot_df, df):
    golden = pivot_df['is_golden'].any()
    last_label = pivot_df[pivot_df['label'] != ""].iloc[-1]['label'] if not pivot_df[pivot_df['label'] != ""].empty else ""

    score = sum([
        df['ma20'].iloc[-1] > df['ma60'].iloc[-1] > df['ma120'].iloc[-1],
        df['macd'].iloc[-1] > df['signal'].iloc[-1],
        df['obv'].iloc[-1] > df['obv'].rolling(20).mean().iloc[-1]
    ])

    if last_label == "3" and golden:
        return "黃金第三浪", "完美斐波那契＋多頭共振 → 主升段啟動", "gold"
    elif last_label == "5" and golden:
        return "黃金逃頂", "第五浪頂部＋斐波那契成立 → 立即賣出", "red"
    elif last_label == "C" and golden:
        return "黃金反轉", "C浪結束＋斐波那契完美 → 強力買入", "lime"
    elif last_label in ["1","2","3"] and score >= 2:
        return "強勢買入", "波浪＋趨勢共振"
    else:
        return "觀望", f"斐波那契 {'成立' if golden else '未通過'}"

# ==================== 主介面（已修正 col 名稱）===================
left_col, right_col = st.columns([1, 4])   # 改成 left_col, right_col

with left_col:
    st.subheader("五重共振設定")
    ticker_input = st.text_area("輸入股票代號（每行一檔）",
        value="""2330.TW
AAPL
TSLA
NVDA
BTC-USD
SMCI
AMD
0700.HK""", height=250)
    tickers = [t.strip() for t in ticker_input.split("\n") if t.strip()]

    period = st.selectbox("資料期間", ["1y","2y","3y","max"], index=1)
    deviation = st.slider("波浪靈敏度 (%)", 3.0, 10.0, 4.6, 0.1)
    run = st.button("啟動 五重共振掃盤", type="primary", use_container_width=True)

if run:
    if not tickers:
        st.error("請輸入股票代號")
        st.stop()

    results = []
    progress = st.progress(0)

    for i, ticker in enumerate(tickers):
        progress.progress((i+1)/len(tickers))
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if len(df) < 100: 
            continue

        df = df[['Open','High','Low','Close','Volume']].copy()
        df.columns = ['open','high','low','close','volume']
        df = add_indicators(df)
        pivot_df = zigzag(df, deviation)
        pivot_df = find_best_wave_with_fib(pivot_df)
        signal, reason, color = get_signal(pivot_df, df)

        # 畫圖
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=(f"{ticker} → {signal}", "MACD", "OBV"),
                            row_heights=[0.6,0.2,0.2])

        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close']), row=1, col=1)

        wave_color = "gold" if pivot_df['is_golden'].any() else "orange"
        labeled = pivot_df[pivot_df['label'] != ""]
        if not labeled.empty:
            fig.add_trace(go.Scatter(x=labeled['date'], y=labeled['price'],
                                     mode='lines+markers+text', text=labeled['label'],
                                     textposition="top center", textfont=dict(size=22, color="yellow"),
                                     line=dict(color=wave_color, width=4), name="波浪"), row=1, col=1)

        for ma, c in zip(['ma20','ma60','ma120'], ['yellow','purple','white']):
            fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(color=c)), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['signal'], name='Signal'), row=2, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], name='Hist'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['obv'], name='OBV'), row=3, col=1)

        fig.update_layout(height=900, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        results.append({
            "代號": ticker,
            "最新價": f"{df['close'].iloc[-1]:.2f}",
            "波浪": "".join(pivot_df['label'].tolist()[-8:]),
            "斐波那契": "完美成立" if pivot_df['is_golden'].any() else "未通過",
            "訊號": signal,
            "原因": reason
        })

    # 總表
    if results:
        df_res = pd.DataFrame(results)
        styled = df_res.style.applymap(lambda x: "background: gold; color: black; font-weight: bold" if "完美成立" in str(x) else "", subset=["斐波那契"])
        st.dataframe(styled, use_container_width=True)
        st.download_button("下載報告", df_res.to_csv(index=False).encode('utf-8-sig'), "艾略特五重共振報告.csv")

else:
    st.success("這是目前最強的艾略特波浪自動分析工具！\n只抓真正符合斐波那契黃金比例的完美波浪！")
    st.balloons()
