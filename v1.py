# elliott_wave_app.py
# 直接複製存檔後執行：streamlit run elliott_wave_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ---------------------- 頁面設定 ----------------------
st.set_page_config(page_title="AI 艾略特波浪選股神器", layout="wide")
st.title("AI 艾略特波浪全自動選股系統")
st.markdown("###  # 讓說明文字更清楚

# ---------------------- 資料下載 ----------------------
@st.cache_data(ttl=1800, show_spinner=False)
def get_data(ticker, period="2y"):
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty or len(df) < 50:
            return None
        df = df[['Open', 'High', 'Low', 'Close']].copy()
        df.columns = ['open', 'high', 'low', 'close']
        df = df.dropna()
        return df
    except:
        return None

# ---------------------- ZigZag ----------------------
def zigzag(df, deviation=6.0):
    high = df['high'].values
    low  = df['low'].values
    close = df['close'].values
    dates = df.index
    pivots = []

    i = 1
    up = None
    last_price = close[0]
    last_idx   = 0

    while i < len(df):
        if up is None:
            up = high[i] > last_price

        if up:                                         # 上升趨勢
            if high[i] > last_price:
                last_price = high[i]
                last_idx   = i
            if (last_price - low[i]) / last_price * 100 >= deviation:
                pivots.append((dates[last_idx], last_price, 'high'))
                last_price = low[i]
                last_idx   = i
                up = False
        else:                                          # 下降趨勢
            if low[i] < last_price:
                last_price = low[i]
                last_idx   = i
            if (high[i] - last_price) / last_price * 100 >= deviation:
                pivots.append((dates[last_idx], last_price, 'low'))
                last_price = high[i]
                last_idx   = i
                up = True
        i += 1

    # 加入最後一根K線
    final_type = 'high' if up else 'low'
    if not pivots or pivots[-1][0] != dates[-1]:
        pivots.append((dates[-1], close[-1], final_type))

    return pd.DataFrame(pivots, columns=['date', 'price', 'type'])

# ---------------------- 斐波那契驗證 ----------------------
def validate_fibonacci(prices):
    if len(prices) < 6:
        return False
    w1 = prices[1] - prices[0]
    if w1 <= 0: return False
    w2 = prices[2] - prices[1]
    w3 = prices[3] - prices[2]
    w4 = prices[4] - prices[3]

    r2 = abs(w2 / w1)
    r3 = abs(w3 / w1)
    r4 = abs(w4 / w3) if w3 != 0 else 0

    checks = [
        0.382 <= r2 <= 0.786,
        r3 >= 1.0,
        r4 <= 0.618
    ]
    return sum(checks) >= 2

# ---------------------- 找最佳艾略特波浪 ----------------------
def find_best_elliott(pivot_df):
    if len(pivot_df) < 9:
        pivot_df = pivot_df.copy()
        pivot_df['label'] = ""
        return pivot_df, False

    types_list = pivot_df['type'].tolist()          # 關鍵：轉成 Python list
    prices = pivot_df['price'].values
    labels = [""] * len(pivot_df)

    pattern_up   = ['high','low','high','low','high','low','high','low','high']
    pattern_down = ['low','high','low','high','low','high','low','high','low']

    best_score = 0
    best_i = -1
    direction = None

    for i in range(len(types_list)-8):
        seg = types_list[i:i+9]

        if seg == pattern_up:
            valid = validate_fibonacci(prices[i:i+6])
            score = 20 if valid else 10
            if score > best_score:
                best_score = score
                best_i = i
                direction = "up"
        elif seg == pattern_down:
            score = 12
            if score > best_score:
                best_score = score
                best_i = i
                direction = "down"

    if best_i >= 0:
        wave_labels = ["", "1","2","3","4","5","A","B","C"] if direction == "up" else ["", "1","2","3","4","5","a","b","c"]
        for j in range(min(9, len(types_list)-best_i)):
            labels[best_i + j] = wave_labels[j]

    pivot_df = pivot_df.copy()
    pivot_df['label'] = labels
    return pivot_df, best_score > 0

# ---------------------- 產生買賣訊號 ----------------------
def get_signal(pivot_df):
    labeled = pivot_df[pivot_df['label'] != ""]
    if labeled.empty:
        return "觀望", "無明確波浪結構", "gray"

    last = labeled.iloc[-1]
    label = last['label']
    ptype = last['type']

    if label == "5":
        return "強烈賣出", "第五浪頂部，準備逃頂", "red"
    elif label == "C" and ptype == "low":
        return "強烈買入", "C浪見底，反轉在即", "lime"
    elif label == "3":
        return "加碼買進", "第三浪最強，順勢操作", "green"
    elif label in ["A", "B"]:
        return "減碼觀望", "ABC修正波", "orange"
    elif label == "4":
        return "持倉等待", "第四浪整理中", "yellow"
    else:
        return "持倉觀察", f"目前位於{label}浪", "white"

# ---------------------- Streamlit 介面 ----------------------
col1, col2 = st.columns([1, 4])

with col1:
    st.subheader("設定")
    ticker_input = st.text_area(
        "請輸入股票代號（每行一檔）",
        value="2330.TW\nAAPL\nTSLA\nBTC-USD\nNVDA\n0050.TW\n0700.HK",
        height=200
    )
    tickers = [t.strip() for t in ticker_input.split("\n") if t.strip()]

    period = st.selectbox("資料期間", ["6mo", "1y", "2y", "3y", "5y", "max"], index=2)
    deviation = st.slider("ZigZag 靈敏度 (%)", 3.0, 15.0, 6.5, 0.5)

    run = st.button("開始 AI 掃盤", type="primary", use_container_width=True)

if run:
    if not tickers:
        st.error("請至少輸入一檔股票代號")
        st.stop()

    results = []
    progress = st.progress(0)

    for idx, ticker in enumerate(tickers):
        progress.progress((idx + 1) / len(tickers))

        df = get_data(ticker, period)
        if df is None:
            results.append({"代號": ticker, "訊號": "失敗", "原因": "無資料或代號錯誤"})
            continue

        try:
            pivot_df = zigzag(df, deviation)
            pivot_df, found = find_best_elliott(pivot_df)
            signal, reason, _ = get_signal(pivot_df)

            # 畫圖
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index,
                                         open=df['open'],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'],
                                         name=ticker))

            if not pivot_df[pivot_df['label'] != ""].empty:
                fig.add_trace(go.Scatter(
                    x=pivot_df['date'],
                    y=pivot_df['price'],
                    mode='lines+markers+text',
                    line=dict(color='orange', width=3),
                    text=pivot_df['label'],
                    textposition="top center",
                    textfont=dict(size=18, color="yellow"),
                    name="艾略特波浪"
                ))

            fig.update_layout(title=f"{ticker} → {signal} ({reason})",
                              height=600,
                              template="plotly_dark",
                              xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            results.append({
                "代號": ticker,
                "最新價": f"{df['close'].iloc[-1]:.2f}",
                "目前波浪": "→".join(pivot_df[pivot_df['label']!='']['label'].tolist()[-5:]),
                "斐波那契": "通過" if found else "未達標",
                "訊號": signal,
                "原因": reason
            })

        except Exception as e:
            results.append({"代號": ticker, "訊號": "錯誤", "原因": str(e)})

    # 結果總表
    st.markdown("## 選股總覽")
    result_df = pd.DataFrame(results)
    def highlight_signal(val):
        color = 'lime' if '買' in val else 'red' if '賣' in val else 'orange'
        return f'background-color: {color}; color: black; font-weight: bold'
    styled = result_df.style.applymap(highlight_signal, subset=['訊號'])
    st.dataframe(styled, use_container_width=True)

    # 下載按鈕
    csv = result_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("下載報告 CSV", csv, "艾略特波浪選股報告.csv", "text/csv")

else:
    st.info("貼上股票代號 → 點擊「開始 AI 掃盤」 → 幾秒後即可看到完整波浪標記與買賣訊號！")
    st.balloons()
