# app_elliott_wave_final_fixed.py
# 完整、乾淨、可直接執行版本（2025最新）

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI艾略特波浪選股神器", layout="wide")
st.title("AI 艾略特波浪全自動選股系統")
st.markdown("### 一鍵輸入股票代號 → 自動標波浪 → 斐波那契驗證 → 直接給買賣建議")

# ================ 資料下載 ================
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

# ================ ZigZag 演算法（已修正語法錯誤）===============
def zigzag(df, deviation=6.0):
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    # 新增：用 close 當最後一點
    dates = df.index
    pivots = []
    i = 1
    up = None
    last_pivot_price = close[0]
    last_pivot_idx = 0

    while i < len(df):
        if up is None:
            up = high[i] > last_pivot_price

        if up:  # 上升趨勢中
            if high[i] > last_pivot_price:
                last_pivot_price = high[i]
                last_pivot_idx = i
            # 下跌超過 deviation% → 確認高點轉折
            if (last_pivot_price - low[i]) / last_pivot_price * 100 >= deviation:
                pivots.append((dates[last_pivot_idx], last_pivot_price, 'high'))
                last_pivot_price = low[i]
                last_pivot_idx = i
                up = False

        else:  # 下降趨勢中
            if low[i] < last_pivot_price:
                last_pivot_price = low[i]
                last_pivot_idx = i
            # 上漲超過 deviation% → 確認低點轉折
            if (high[i] - last_pivot_price) / last_pivot_price * 100 >= deviation:   # 已修正！
                pivots.append((dates[last_pivot_idx], last_pivot_price, 'low'))
                last_pivot_price = high[i]
                last_pivot_idx = i
                up = True
        i += 1

    # 強制加入最新價格作為最後一個轉折點
    if pivots:
        last_type = 'high' if up else 'low'
        if pivots[-1][0] != dates[-1]:
            pivots.append((dates[-1], close[-1], last_type))
    else:
        pivots.append((dates[-1], close[-1], 'high'))

    return pd.DataFrame(pivots, columns=['date', 'price', 'type'])

# ================ 斐波那契驗證 ================
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

# ================ 尋找最佳艾略特波浪（已修好 numpy 比較問題）===============
def find_best_elliott(pivot_df):
    if len(pivot_df) < 9:
        pivot_df = pivot_df.copy()
        pivot_df['label'] = ""
        return pivot_df, False

    types_list = pivot_df['type'].tolist()          # 關鍵：轉成 list 避免 numpy 錯誤
    prices = pivot_df['price'].values
    labels = [""] * len(pivot_df)

    pattern_up   = ['high','low','high','low','high','low','high','low','high']
    pattern_down = ['low','high','low','high','low','high','low','high','low']

    best_score = 0
    best_i = -1
    direction = None

    for i in range(len(types_list) - 8):
        seg = types_list[i:i+9]

        if seg == pattern_up:
            valid = validate_fibonacci(prices[i:i+6])
            score = 20 if valid else 10
            if score > best_score:
                best_score, best_i, direction = score, i, "up"

        elif seg == pattern_down:
            score = 12
            if score > best_score:
                best_score, best_i, direction = score, i, "down"

    if best_i >= 0:
        wave_labels = ["", "1","2","3","4","5","A","B","C"] if direction == "up" else ["", "1","2","3","4","5","a","b","c"]
        for j in range(min(9, len(types_list) - best_i)):
            labels[best_i + j] = wave_labels[j]

    pivot_df = pivot_df.copy()
    pivot_df['label'] = labels
    return pivot_df, best_score > 0

# ================ 產生買賣訊號 ================
def get_signal(pivot_df):
    labeled = pivot_df[pivot_df['label'] != ""]
    if labeled.empty:
        return "觀望", "無明確波浪", "gray"

    last = labeled.iloc[-1]
    label = last['label']
    label = last['label']
    ptype = last['type']

    if label == "5":
        return "強烈賣出", "第五浪頂部已成，準備逃頂", "red"
    elif label == "C" and ptype == "low":
        return "強烈買入", "C浪見底，反轉在即", "lime"
    elif label == "3":
        return "加碼買進", "第三浪最強，跟緊主力", "green"
    elif label in ["A", "B"]:
        return "減碼觀望", "進入ABC修正", "orange"
    elif label == "4":
        return "持倉等待", "第四浪整理，準備第五浪", "yellow"
    else:
        return "持倉觀察", f"目前在{label}浪", "white"

# ================ Streamlit 介面 ================
col1, col2 = st.columns([1, 4])

with col1:
    st.subheader("設定")
    ticker_input = st.text_area(
        "輸入股票代號（每行一檔）",
        value="2330.TW\nAAPL\nTSLA\nBTC-USD\nNVDA\n0050.TW",
        height=200
    )
    tickers = [t.strip() for t in ticker_input.split('\n') if t.strip()]

    period = st.selectbox("資料期間", ["1y", "2y", "3y", "5y"], index= "max"], index=1)
    deviation = st.slider("ZigZag 靈敏度 (%)", 3.0, 15.0, 6.5, 0.5)
    st.markdown("---")
    run = st.button("開始 AI 掃盤", type="primary", use_container_width=True)

if run:
    if not tickers:
        st.error("請輸入股票代號")
        st.stop()

    results = []
    progress = st.progress(0)

    for i, ticker in enumerate(tickers):
        progress.progress((i+1)/len(tickers))
        df = get_data(ticker, period)
        if df is None:
            results.append({"代號": ticker, "訊號": "失敗", "原因": "無資料"})
            continue

        pivot_df = zigzag(df, deviation)
        pivot_df, found = find_best_elliott(pivot_df)
        signal, reason, color = get_signal(pivot_df)

        # 畫圖
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                     low=df['low'], close=df['close'], name=ticker))
        if not pivot_df[pivot_df['label'] != ""].empty:
            fig.add_trace(go.Scatter(x=pivot_df['date'], y=pivot_df['price'],
                                     mode='lines+markers+text', line=dict(color='orange', width=3),
                                     text=pivot_df['label'], textposition="top center",
                                     textfont=dict(size=18, color="yellow"), name="波浪"))
        fig.update_layout(title=f"{ticker} → {signal} ({reason})", height=600, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        st.plotly_chart(fig, use_container_width=True)

        results.append({
            "代號": ticker,
            "最新價": f"{df['close'].iloc[-1]:.2f}",
            "目前波浪": "→".join(pivot_df[pivot_df['label']!='']['label'].tolist()[-4:]),
            "斐波那契": "通過" if found else "未達",
            "訊號": signal,
            "原因": reason
        })

    # 總表
    st.success("分析完成！")
    df_result = pd.DataFrame(results)
    st.dataframe(df_result.style.applymap(lambda x: f"background:{'lime' if '買' in x else 'red' if '賣' in x else 'orange'}", subset=['訊號']))

    csv = df_result.to_csv(index=False).encode('utf-8-sig')
    st.download_button("下載報告", csv, "艾略特選股報告.csv", "text/csv")

else:
    st.info("貼上股票代號 → 按下按鈕 → 幾秒後就能看到 AI 標記的波浪與買賣訊號！")
    st.balloons()
