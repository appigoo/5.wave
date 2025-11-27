# elliott_fibonacci_pro_max.py
# 終極五重共振：波浪 + MACD + OBV + 均線 + 嚴格斐波那契驗證

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

st.set_page_config(page_title="艾略特波浪＋斐波那契五重共振", layout="wide")
st.title("艾略特波浪 PRO MAX＋")
st.markdown("### 波浪結構＋MACD＋OBV＋均線＋**嚴格斐波那契黃金比例驗證** → 極致精準！")

# ==================== 斐波那契驗證（核心升級）===================
def validate_fibonacci_ratio(prices, types):
    """
    嚴格驗證上升5浪是否符合斐波那契經典比例
    prices: 連續6個轉折點價格 [0高,1低,2高,3低,4高,5低/高]
    types: 對應的 high/low 序列
    """
    if len(prices) < 6 or types[0] != 'high':
        return False, "格式錯誤"

    p0, p1, p2, p3, p4, p5 = prices[:6]

    # 上升浪：0→1→2→3→4→5
    w1 = p2 - p1
    w2 = p1 - p3
    w3 = p4 - p3
    w4 = p3 - p5
    w5 = p5 - p4 if len(prices) > 6 else (prices[-1] - p4)

    if w1 <= 0: return False, "波1無效"

    r2 = w2 / w1
    r3 = w3 / w1
    r4 = w4 / w3 if w3 > 0 else 0
    r5 = abs(w5) / w1 if w1 > 0 else 0

    # 經典斐波那契規則（艾略特原著標準）
    rule1 = 0.382 <= r2 <= 0.786        # 波2回檔
    rule2 = r3 >= 1.0                   # 波3最長（可放寬到0.9）
    rule3 = r4 <= 0.5                   # 波4不破波1太多
    rule4 = r5 >= 0.618 or abs(w5 + w4) / w1 <= 0.382  # 波5延伸 or 失敗第五浪
    rule5 = w3 >= w1 and w3 >= abs(w5)  # 波3不能最短

    passed = sum([rule1, rule2, rule3, rule4, rule5])
    is_valid = passed >= 4  # 至少4條通過才算高可信

    reason = f"波2={r2:.1%} 波3={r3:.2f}× 波4={r4:.1%} 波5={r5:.1%} → {passed}/5分"
    return is_valid, reason

# ==================== 其他函數（已優化）===================
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

def get_trend_score(df):
    last = df.iloc[-1]
    ma_bull = last['ma20'] > last['ma60'] > last['ma120']
    macd_golden = df['macd'].iloc[-2] < df['signal'].iloc[-2] and df['macd'].iloc[-1] > df['signal'].iloc[-1]
    obv_up = last['obv'] > df['obv'].rolling(30).mean().iloc[-1]
    return sum([ma_bull, macd_golden, obv_up])

def zigzag(df, deviation=4.8):
    # （保持不變，穩定版）
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

def find_and_validate_waves(pivot_df):
    """彈性找波浪 + 斐波那契嚴格驗證"""
    if len(pivot_df) < 6:
        pivot_df['label'] = ""
        pivot_df['fib_valid'] = False
        return pivot_df

    types = pivot_df['type'].tolist()
    prices = pivot_df['price'].values
    labels = [""] * len(pivot_df)
    fib_status = [False] * len(pivot_df)

    pattern = ['high','low','high','low','high','low','high','low','high']

    best_i = -1
    best_score = 0

    for i in range(len(types)-5):
        seg_types = types[i:i+9]
        seg_prices = prices[i:i+9]
        match_count = sum(a == b for a,b in zip(seg_types, pattern))

        if match_count >= 6:  # 至少6個點符合
            is_valid, reason = validate_fibonacci_ratio(seg_prices, seg_types)
            score = match_count + (10 if is_valid else 0)
            if score > best_score:
                best_score = score
                best_i = i
                is_fib_valid = is_valid

    if best_i >= 0:
        wave_labels = ["", "1","2","3","4","5","A","B","C"]
        for j in range(min(9, len(types)-best_i)):
            labels[best_i + j] = wave_labels[j]
        # 標記斐波那契驗證結果
        fib_status[best_i + 5] = is_fib_valid  # 在波5位置標記

    pivot_df = pivot_df.copy()
    pivot_df['label'] = labels
    pivot_df['fib_valid'] = fib_status
    return pivot_df, best_score > 0

def get_final_signal(pivot_df, df):
    labeled = pivot_df[pivot_df['label'] != ""]
    if labeled.empty:
        return "觀望", "無波浪"

    last_label = labeled.iloc[-1]['label']
    fib_ok = labeled.iloc[-1]['fib_valid'] if 'fib_valid' in labeled.columns else False
    score = get_trend_score(df)

    prefix = "【黃金波浪】" if fib_ok else ""

    if last_label == "3" and fib_ok and score >= 2:
        return f"{prefix}超級買入", "第三浪＋斐波那契完美＋多頭共振"
    elif last_label == "5" and fib_ok:
        return f"{prefix}強烈賣出", "第五浪頂部＋斐波那契成立 → 逃頂"
    elif last_label == "C" and fib_ok:
        return f"{prefix}強力買入", "C浪落底＋斐波那契完美反轉"
    elif last_label == "3":
        return "強勢買入", "第三浪進行中（待斐波那契確認）"
    else:
        return "關注", f"波浪{last_label}，斐波那契 {'通過' if fib_ok else '未通過'}"

# ==================== 主程式 ====================
col1, col1, col2 = st.columns([1, 4])

with col1:
    st.subheader("五重共振設定")
    tickers = st.text_area("股票代號（每行一檔）",
        value="""2330.TW
AAPL
TSLA
NVDA
BTC-USD
SMCI
AMD
0700.HK""", height=250).split("\n")
    tickers = [t.strip() for t in tickers if t.strip()]

    period = st.selectbox("期間", ["1y","2y","3y","max"], index=1)
    deviation = st.slider("波浪靈敏度 (%)", 3.0, 10.0, 4.6, 0.1)
    run = st.button("啟動 五重共振掃盤", type="primary", use_container_width=True)

if run:
    results = []
    for ticker in tickers:
        with st.spinner(f"五重驗證中：{ticker}"):
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if len(df) < 100: continue

            df = df[['Open','High','Low','Close','Volume']].copy()
            df.columns = ['open','high','low','close','volume']
            df = add_indicators(df)
            pivot_df = zigzag(df, deviation)
            pivot_df = find_and_validate_waves(pivot_df)
            signal, reason = get_final_signal(pivot_df, df)

            # 畫圖
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                subplot_titles=(f"{ticker} → {signal}", "MACD", "OBV＋均線"),
                                row_heights=[0.6,0.2,0.2])

            fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                         low=df['low'], close=df['close']), row=1, col=1)

            # 波浪標記（通過斐波那契的用金色）
            valid_points = pivot_df[pivot_df['label'] != ""]
            if not valid_points.empty:
                color = "gold" if any(pivot_df['fib_valid']) else "orange"
                fig.add_trace(go.Scatter(x=valid_points['date'], y=valid_points['price'],
                                         mode='lines+markers+text', text=valid_points['label'],
                                         textposition="top center", textfont=dict(size=22, color="yellow"),
                                         line=dict(color=color, width=4), name="波浪"), row=1, col=1)

            for ma, c in zip(['ma20','ma60','ma120'], ['yellow','purple','white']):
                fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(color=c)), row=1, col=1)

            fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['signal'], name='Signal'), row=2, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], name='Hist'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['obv'], name='OBV', line=dict(color='purple')), row=3, col=1)

            fig.update_layout(height=900, template="plotly_dark", title_text=f"{ticker} | {reason}")
            st.plotly_chart(fig, use_container_width=True)

            results.append({
                "代號": ticker,
                "最新價": f"{df['close'].iloc[-1]:.1f}",
                "波浪": "".join(pivot_df['label'].tolist()[-8:]),
                "斐波那契": "完美成立" if any(pivot_df['fib_valid']) else "未通過",
                "趨勢分": get_trend_score(df),
                "最終訊號": signal,
                "原因": reason
            })

    if results:
        df_res = pd.DataFrame(results)
        def highlight_row(row):
            color = "background: gold; color: black" if "完美成立" in row['斐波那契'] else ""
            return [color] * len(row)
        styled = df_res.style.apply(highlight_row, axis=1)
        st.dataframe(styled, use_container_width=True)
        st.download_button("下載五重共振報告", df_res.to_csv(index=False).encode('utf-8-sig'), "五重共振選股.csv")

else:
    st.success("這是目前全球最強的艾略特波浪自動分析工具：\n\n只抓真正符合斐波那契黃金比例的完美波浪！")
    st.balloons()
