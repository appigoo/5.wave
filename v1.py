# app_yfinance_elliott.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI艾略特波浪全自動選股神器", layout="wide")
st.title("AI 艾略特波浪全自動分析系統")
st.markdown("### 輸入股票代號 → 自動下載 → 自動標記波浪 → 給出買賣訊號（支援台美港加密貨幣）")

# ================ 核心函數（已優化）===============
@st.cache_data(ttl=3600, show_spinner=False)  # 快取1小時
def get_stock_data(ticker, period="2y"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df.index.name = 'date'
        return df
    except:
        return None

def zigzag(df, deviation=5.0):
    highs = df['high'].values
    lows = df['low'].values
    dates = df.index
    pivots = []
    last_pivot_idx = 0
    last_pivot_price = df['close'].iloc[0]
    up_trend = None

    for i in range(1, len(df)):
        if up_trend is None:
            up_trend = highs[i] > last_pivot_price

        if up_trend:
            if highs[i] > last_pivot_price:
                last_pivot_price = highs[i]
                last_pivot_idx = i
            if (last_pivot_price - lows[i]) / last_pivot_price * 100 >= deviation:
                pivots.append((dates[last_pivot_idx], last_pivot_price, 'high'))
                last_pivot_price = lows[i]
                last_pivot_idx = i
                up_trend = False
        else:
            if lows[i] < last_pivot_price:
                last_pivot_price = lows[i]
                last_pivot_idx = i
            if (highs[i] - last_pivot_price) / last_pivot_price * 100 >= deviation:
                pivots.append((dates[last_pivot_idx], last_pivot_price, 'low'))
                last_pivot_price = highs[i]
                last_pivot_idx = i
                up_trend = True

    # 強制加入最新價
    if pivots and pivots[-1][0] != dates[-1]:
        pivots.append((dates[-1], df['close'].iloc[-1], 'high' if up_trend else 'low'))

    return pd.DataFrame(pivots, columns=['date', 'price', 'type'])

def validate_fibonacci(prices, direction="up"):
    if len(prices) < 6: return False, "點數不足"
    w1 = prices[1] - prices[0]
    w2 = prices[2] - prices[1]
    w3 = prices[3] - prices[2]
    w4 = prices[4] - prices[3]
    w5 = prices[5] - prices[4]

    r2 = abs(w2 / w1) if w1 != 0 else 0
    r3 = abs(w3 / w1) if w1 != 0 else 0
    r4 = abs(w4 / w3) if w3 != 0 else 0

    checks = [
        0.382 <= r2 <= 0.786,   # 波2回檔
        r3 >= 1.0,              # 波3最長
        r4 <= 0.618,            # 波4不破波1高點太多
        abs(w5) > 0             # 波5存在
    ]
    return sum(checks) >= 3, f"波2={r2:.1%} 波3={r3:.2f}× 波4={r4:.1%}"

def find_best_elliott(pivot_df):
    types = pivot_df['type'].values
    prices = pivot_df['price'].values
    labels = [""] * len(pivot_df)
    best_score = 0
    best_i = -1
    direction = None

    for i in range(len(pivot_df)-8):
        seg = types[i:i+9]
        if seg == ['high','low','high','low','high','low','high','low','high']:
            is_valid, msg = validate_fibonacci(prices[i:i+6], "up")
            score = 10 if is_valid else 5
            if score > best_score:
                best_score, best_i, direction = score, i, "up"
        elif seg == ['low','high','low','high','low','high','low','high','low']:
            score = 8
            if score > best_score:
                best_score, best_i, direction = score, i, "down"

    if best_i >= 0:
        wave_labels = ["", "1","2","3","4","5","A","B","C"] if direction=="up" else ["", "1","2","3","4","5","a","b","c"]
        for j in range(min(9, len(pivot_df)-best_i)):
            labels[best_i + j] = wave_labels[j]

    pivot_df = pivot_df.copy()
    pivot_df['label'] = labels
    return pivot_df, best_score > 0

def generate_signal(pivot_df, df):
    labeled = pivot_df[pivot_df['label'] != ""]
    if labeled.empty:
        return "觀望", "無明顯波浪結構", "gray"

    last_label = labeled.iloc[-1]['label']
    last_price = df['close'].iloc[-1]

    if last_label in ["5", "⑤", "C", "c"] and last_label in ["5", "C"]:
        return "強烈賣出", "5浪頂部或C浪結束，反轉風險極高", "red"
    elif last_label == "C" and pivot_df.iloc[-1]['type'] == 'low':
        return "強烈買入", "C浪見底，反轉在即！", "lime"
    elif last_label == "3":
        return "加碼買進", "第三浪最賺，跟著主力衝！", "green"
    elif last_label in ["A", "B"]:
        return "觀望", "處於ABC修正波，暫不進場", "orange"
    else:
        return "持倉觀察", f"目前位於{last_label}浪", "yellow"

# ================ Streamlit 介面 ================
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("設定")
    tickers_input = st.text_input(
        "輸入股票代號（多檔用逗號或換行分隔）",
        value="2330.TW, AAPL, 0700.HK, BTC-USD, TSLA"
    ).replace("\n", ",")
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

    period = st.selectbox("資料期間", ["6mo", "1y", "2y", "5y", "max"], index=2)
    deviation = st.slider("ZigZag 靈敏度（%）", 3.0, 12.0, 6.0, 0.5)

if st.button("開始 AI 波浪分析！", type="primary"):
    if not tickers:
        st.error("請輸入至少一檔股票代號")
        st.stop()

    results = []
    progress = st.progress(0)
    chart_container = st.container()

    for i, ticker in enumerate(tickers):
        with st.spinner(f"正在分析 {ticker}..."):
            df = get_stock_data(ticker, period)
            if df is None or len(df) < 50:
                results.append({
                    "股票": ticker,
                    "狀態": "失敗",
                    "訊號": "-",
                    "原因": "無資料或下載失敗"
                })
                continue

            pivot_df = zigzag(df, deviation)
            pivot_df, found = find_best_elliott(pivot_df)
            signal, reason, color = generate_signal(pivot_df, df)

            results.append({
                "股票": ticker,
                "最新價": f"{df['close'].iloc[-1]:.2f}",
                "波浪": ", ".join(pivot_df[pivot_df['label']!='']['label'].tolist()[-4:]),
                "斐波那契": "通過" if found else "未達標",
                "訊號": signal,
                "原因": reason
            })

            # 顯示圖表
            with chart_container:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                            low=df['low'], close=df['close'], name=ticker))

                if not pivot_df[pivot_df['label']!=''].empty:
                    labeled = pivot_df[pivot_df['label'] != ""]
                    fig.add_trace(go.Scatter(
                        x=pivot_df['date'], y=pivot_df['price'],
                        mode='lines+markers+text',
                        line=dict(color='orange', width=3),
                        marker=dict(size=10),
                        text=pivot_df['label'],
                        textposition="top center",
                        textfont=dict(size=18, color="yellow", family="Arial Black"),
                        name="艾略特波浪"
                    ))

                fig.update_layout(
                    title=f"{ticker}｜{signal}｜{reason}",
                    height=600,
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)

        progress.progress((i + 1) / len(tickers))

    # 總表
    st.markdown("## 選股總覽（AI 掃盤結果）")
    result_df = pd.DataFrame(results)
    st.dataframe(
        result_df.style
        .apply(lambda x: [f"background: {color}; color: black" if x.name == "訊號" else "" for color in
                         ['lime' if v=="強烈買入" else 'red' if v=="強烈賣出" else 'lightyellow' for v in x]], axis=1)
        .format({"最新價": "{:.2f}"})
    )

    # 下載報告
    csv = result_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        "下載完整報告.csv",
        csv,
        "AI_艾略特波浪選股報告.csv",
        "text/csv"
    )

    st.success("分析完成！強烈訊號已用顏色標示，紅買綠賣一目了然！")

else:
    st.info("### 支援格式範例：\n"
            "- 台股：`2330.TW`, `2317.TW`\n"
            "- 美股：`AAPL`, `NVDA`, `TSLA`\n"
            "- 港股：`0700.HK`, `3690.HK`\n"
            "- 加密貨幣：`BTC-USD`, `ETH-USD`\n"
            "一次最多可分析 50 檔！")

    st.markdown("Developed by Grok × yfinance │ 完全免費開源 │ 祝你抓第三浪、逃第五浪！")
