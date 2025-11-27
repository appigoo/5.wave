# app_elliott_wave_final.py
# 直接複製存成這個檔名，然後執行：streamlit run app_elliott_wave_final.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ================ 頁面設定 ================
st.set_page_config(page_title="AI艾略特波浪選股神器", layout="wide")
st.title("AI 艾略特波浪全自動選股系統（2025終極版）")
st.markdown("### 輸入股票代號 → 自動下載 → 自動標記波浪 → 斐波那契驗證 → 直接給買賣訊號")

# ================ 核心函數 ================
@st.cache_data(ttl=1800, show_spinner=False)  # 快取30分鐘
def get_data(ticker, period="2y"):
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty or len(df) < 50:
            return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df.index.name = 'date'
        df = df.dropna()
        return df
    except:
        return None

def zigzag(df, deviation=6.0):
    """穩定版 ZigZag"""
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    dates = df.index
    pivots = []
    i = 1
    up = None
    last_pivot_price = close[0]
    last_pivot_idx = 0

    while i < len(df):
        if up is None:
            up = high[i] > last_pivot_price

        if up:
            if high[i] > last_pivot_price:
                last_pivot_price = high[i]
                last_pivot_idx = i
            if (last_pivot_price - low[i]) / last_pivot_price * 100 >= deviation:
                pivots.append((dates[last_pivot_idx], last_pivot_price, 'high'))
                last_pivot_price = low[i]
                last_pivot_idx = i
                up = False
        else:
            if low[i] < last_pivot_price:
                last_pivot_price = low[i]
                last_pivot_idx = i
            if (high[i] - last_pivot_price) / last_pivot_price *  * 100 >= deviation:
                pivots.append((dates[last_pivot_idx], last_pivot_price, 'low'))
                last_pivot_price = high[i]
                last_pivot_idx = i
                up = True
        i += 1

    # 強制加入最新一根
    last_price = close[-1]
    last_date = dates[-1]
    last_type = 'high' if up else 'low'
    if not pivots or pivots[-1][0] != last_date:
        pivots.append((last_date, last_price, last_type))

    return pd.DataFrame(pivots, columns=['date', 'price', 'type'])

def validate_fibonacci(prices):
    """驗證上升5浪是否符合斐波那契黃金比例"""
    if len(prices) < 6:
        return False
    w1 = prices[1] - prices[0]
    w2 = prices[2] - prices[1]
    w3 = prices[3] - prices[2]
    w4 = prices[4] - prices[3]
    w5 = prices[5] - prices[4]

    if w1 <= 0: return False

    r2 = abs(w2 / w1)
    r3 = abs(w3 / w1)
    r4 = abs(w4 / w3) if w3 != 0 else 0

    checks = [
        0.382 <= r2 <= 0.786,    # 波2回檔
        r3 >= 1.0,               # 波3最長
        r4 <= 0.5,               # 波4淺
        abs(w5) > 0.3 * abs(w1)  # 波5有延展性
    ]
    return sum(checks) >= 3

def find_best_elliott(pivot_df):
    """完全修正版：避免 numpy array 比較錯誤"""
    if len(pivot_df) < 9:
        pivot_df = pivot_df.copy()
        pivot_df['label'] = ""
        return pivot_df, False

    types = pivot_df['type'].tolist()          # 關鍵！轉成 list
    prices = pivot_df['price'].values
    labels = [""] * len(pivot_df)

    pattern_up   = ['high','low','high','low','high','low','high','low','high']
    pattern_down = ['low','high','low','high','low','high','low','high','low']

    best_score = 0
    best_start = -1
    direction = None

    for i in range(len(types) - 8):
        seg = types[i:i+9]

        if seg == pattern_up:
            valid = validate_fibonacci(prices[i:i+6])
            score = 15 if valid else 8
            if score > best_score:
                best_score = score
                best_start = i
                direction = "up"

        elif seg == pattern_down:
            score = 10
            if score > best_score:
                best_score = score
                best_start = i
                direction = "down"

    # 標記波浪
    if best_start >= 0:
        wave_labels = ["", "1","2","3","4","5","A","B","C"] if direction == "up" else ["", "1","2","3","4","5","a","b","c"]
        for j in range(min(9, len(types) - best_start)):
            labels[best_start + j] = wave_labels[j]

    pivot_df = pivot_df.copy()
    pivot_df['label'] = labels
    return pivot_df, best_score > 0

def get_signal(pivot_df, current_price):
    """根據最新標記給出投資建議"""
    labeled = pivot_df[pivot_df['label'] != ""]
    if labeled.empty:
        return "觀望", "無明確波浪結構", "#CCCCCC"

    last_label = labeled.iloc[-1]['label']
    last_type = labeled.iloc[-1]['type']

    if last_label == "5":
        return "強烈賣出", "第五浪頂部已現，準備逃頂！", "#FF3333"
    elif last_label == "C" and last_type == "low":
        return "強烈買入", "C浪見底，反轉在即！", "#00FF00"
    elif last_label == "3":
        return "加碼買進", "第三浪最強，順勢做多！", "#33CC33"
    elif last_label in ["A", "B"]:
        return "減碼觀望", "進入ABC修正波", "#FFAA00"
    elif last_label == "4":
        return "持倉等待", "第四浪整理中，準備第五浪", "#AAAA00"
    else:
        return "持倉觀察", f"目前位於{last_label}浪", "#888888"

# ================ Streamlit 介面 ================
col1, col2 = st.columns([1, 4])

with col1:
    st.subheader("設定")
    ticker_input = st.text_area(
        "輸入股票代號（每行一檔或用逗號分隔）",
        value="2330.TW\nAAPL\nTSLA\nBTC-USD\n0700.HK\nNVDA\n0050.TW",
        height=200
    )
    tickers = [t.strip() for t in ticker_input.replace(",", "\n").split("\n") if t.strip()]

    period = st.selectbox("資料期間", ["1y", "2y", "3y", "5y", "max"], index=1)
    deviation = st.slider("ZigZag 靈敏度 (%)", 3.0, 15.0, 6.5, 0.5)
    st.markdown("---")
    start_btn = st.button("開始 AI 掃盤！", type="primary", use_container_width=True)

if start_btn:
    if not tickers:
        st.error("請輸入至少一檔股票代號")
        st.stop()

    results = []
    progress_bar = st.progress(0)
    chart_cols = st.columns(1)

    for idx, ticker in enumerate(tickers):
        progress_bar.progress((idx + 1) / len(tickers))

        df = get_data(ticker, period)
        if df is None:
            results.append({
                "代號": ticker,
                "名稱": "下載失敗",
                "最新價": "-",
                "波浪": "-",
                "訊號": "失敗",
                "原因": "無資料或代號錯誤"
            })
            continue

        try:
            pivot_df = zigzag(df, deviation)
            pivot_df, found_wave = find_best_elliott(pivot_df)
            signal, reason, color = get_signal(pivot_df, df['close'].iloc[-1])

            # 顯示圖表（每檔一張）
            with st.container():
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df['open'], high=df['high'],
                    low=df['low'], close=df['close'], name=ticker
                ))

                if not pivot_df[pivot_df['label'] != ""].empty:
                    fig.add_trace(go.Scatter(
                        x=pivot_df['date'], y=pivot_df['price'],
                        mode='lines+markers+text',
                        line=dict(color='orange', width=3),
                        marker=dict(size=12),
                        text=pivot_df['label'],
                        textposition="top center",
                        textfont=dict(size=18, color="yellow", family="Arial Black"),
                        name="艾略特波浪"
                    ))

                fig.update_layout(
                    title=f"{ticker} ｜ {signal} ｜ {reason}",
                    height=600, template="plotly_dark",
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)

            results.append({
                "代號": ticker,
                "名稱": yf.Ticker(ticker).info.get('longName', ticker),
                "最新價": f"{df['close'].iloc[-1]:.2f}",
                "波浪": "→".join(pivot_df[pivot_df['label']!='']['label'].tolist()[-5:]),
                "斐波那契": "通過" if found_wave else "未達標",
                "訊號": signal,
                "原因": reason
            })

        except Exception as e:
            results.append({"代號": ticker, "訊號": "錯誤", "原因": str(e)})

    # 總表
    st.markdown("## AI 選股總覽（紅買綠賣一目了然）")
    result_df = pd.DataFrame(results)
    st.dataframe(
        result_df.style
        .applymap(lambda x: f"background-color: {color}; color: black; font-weight: bold"
                  if x in ["強烈買入", "強烈賣出"] else "", subset=["訊號"])
        .applymap(lambda x: "color: lime" if "強烈買入" in x else "color: red" if "強烈賣" in x else "", subset=["訊號"])
    )

    # 匯出
    csv = result_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("下載完整報告.csv", csv, "艾略特波浪選股報告.csv", "text/csv")

    st.success(f"分析完成！共處理 {len(tickers)} 檔股票")

else:
    st.info("""
    ### 支援所有市場：
    - 台股：`2330.TW`, `2317.TW`, `0050.TW`
    - 美股：`AAPL`, `NVDA`, `TSLA`
    - 港股：`0700.HK`, `3690.HK`
    - 加密貨幣：`BTC-USD`, `ETH-USD`
    - 指數：`^GSPC`, `^IXIC`, `^DJI`

    直接貼上代號，一鍵掃完全市場！
    """)
    st.balloons()
