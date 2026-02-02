import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import itertools

# ================= LOAD NAV =================
nav_df = pd.read_csv("fund_nav_5y.csv")
nav_df["date"] = pd.to_datetime(nav_df["date"], errors="coerce")
nav_df = nav_df.sort_values(["fund","date"])

# ================= FUNCTIONS =================
def calc_metrics(nav, rf=0.02):
    returns = nav.pct_change().dropna()
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (252/len(nav)) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = (cagr - rf) / vol if vol > 0 else 0
    peak = nav.cummax()
    dd = (nav - peak) / peak
    maxdd = dd.min()
    current = 0
    durations = []
    for v in dd:
        if v < 0:
            current += 1
        else:
            if current > 0:
                durations.append(current)
            current = 0
    dd_duration = max(durations) if durations else 0
    window = min(252, len(returns))
    roll = (1+returns).rolling(window).apply(np.prod, raw=True) - 1
    return {
        "Return_%": total_return*100,
        "CAGR_%": cagr*100,
        "Volatility_%": vol*100,
        "Sharpe": sharpe,
        "MaxDD_%": maxdd*100,
        "Worst_Rolling_%": roll.min()*100,
        "Best_Rolling_%": roll.max()*100,
        "DD_Duration_days": dd_duration
    }

def build_equal_weight_nav(nav_df, funds):
    df = nav_df[nav_df["fund"].isin(funds)].pivot(index="date", columns="fund", values="nav").sort_index()
    start_date = df.dropna().index.min()
    df = df[df.index >= start_date].ffill()
    returns = df.pct_change().fillna(0)
    weights = np.array([1/len(funds)] * len(funds))
    port_ret = (returns * weights).sum(axis=1)
    port_nav = (1 + port_ret).cumprod() * 100
    df_norm = (1 + returns).cumprod() * 100
    return df_norm, port_nav

def filter_by_timeframe(nav_series, tf):
    end = nav_series.index.max()
    if tf == "MTD": 
        start = end.replace(day=1)
    elif tf == "YTD":
        start = end.replace(month=1, day=1)
    elif tf == "1M":
        start = end - pd.DateOffset(months=1)
    elif tf == "3M":
        start = end - pd.DateOffset(months=3)
    elif tf == "6M":
        start = end - pd.DateOffset(months=6)
    elif tf == "1Y":
        start = end - pd.DateOffset(years=1)
    elif tf == "3Y":
        start = end - pd.DateOffset(years=3)
    elif tf == "5Y":
        start = end - pd.DateOffset(years=5)
    elif tf == "MAX":
        start = nav_series.index.min()
    else:
        return nav_series
    return nav_series[nav_series.index >= start]

def recommend(row, tf):
    try:
        cagr = row[f"{tf}_CAGR_%"]
        sharpe = row[f"{tf}_Sharpe"]
        dd = abs(row[f"{tf}_MaxDD_%"])
    except:
        return "ข้อมูลไม่พอ"
    if pd.isna(cagr) or pd.isna(sharpe):
        return "ข้อมูลไม่พอ"
    if cagr > 8 and sharpe > 1 and dd < 25:
        return "🟢 ควรเพิ่ม"
    elif cagr > 5 and sharpe > 0.5:
        return "🟡 ถือรอ"
    elif sharpe < 0.3 or dd > 40:
        return "🟠 ควรลด"
    else:
        return "🔴 ควรขาย/สับ"

def multi_vote(row):
    frames = ["3M","6M","1Y","3Y"]
    votes = []
    for tf in frames:
        try:
            v = recommend(row, tf)
            if v != "ข้อมูลไม่พอ":
                votes.append(v)
        except:
            pass
    if len(votes) == 0:
        return "ข้อมูลไม่พอ"
    final = max(set(votes), key=votes.count)
    return final

# ================= PRE-CALC =================
timeframes = ["MTD","YTD","1M","3M","6M","1Y","3Y","5Y","MAX"]
METRICS = [
    "Return_%","CAGR_%","Volatility_%","Sharpe",
    "MaxDD_%","Worst_Rolling_%","Best_Rolling_%","DD_Duration_days"
]

rows = []
for fund, g in nav_df.groupby("fund"):
    g = g.sort_values("date").set_index("date")
    full = g["nav"]
    data = {"fund": fund}
    for tf in timeframes:
        sub = filter_by_timeframe(full, tf)
        if len(sub) >= 20:
            m = calc_metrics(sub)
            for k in METRICS:
                if tf == "YTD" and k in ["CAGR_%","Sharpe"]:
                    data[f"{tf}_{k}"] = np.nan
                else:
                    data[f"{tf}_{k}"] = m[k]
        else:
            for k in METRICS:
                data[f"{tf}_{k}"] = np.nan
    rows.append(data)

df = pd.DataFrame(rows)

# ================= UI =================
st.set_page_config(page_title="Fund Dashboard", layout="centered")
st.title("📊 Fund Performance Dashboard (Investor View)")

# ================= MOBILE-FRIENDLY SETTINGS =================
with st.expander("🔧 ตัวเลือกกองทุน / Timeframe", expanded=True):
    tf = st.radio(
        "📅 Timeframe",
        ["MTD","YTD","1M","3M","6M","1Y","3Y","5Y","MAX"],
        index=5,
        horizontal=True
    )
    funds = st.multiselect(
        "เลือกกองทุน",
        df["fund"].unique(),
        default=list(df["fund"].unique())
    )

dff = df[df["fund"].isin(funds)]

# ================= OVERVIEW =================
with st.expander(f"📊 Overview ({tf})", expanded=True):
    st.subheader("🧭 Decision (Multi-Timeframe Voting)")
    dfp = dff.dropna(subset=[f"{tf}_Return_%" if tf in ["MTD","YTD"] else f"{tf}_CAGR_%"]).copy()
    df_engine = dfp.copy()
    df_engine["3M"] = df_engine.apply(lambda r: recommend(r, "3M"), axis=1)
    df_engine["6M"] = df_engine.apply(lambda r: recommend(r, "6M"), axis=1)
    df_engine["1Y"] = df_engine.apply(lambda r: recommend(r, "1Y"), axis=1)
    df_engine["3Y"] = df_engine.apply(lambda r: recommend(r, "3Y"), axis=1)
    df_engine["Final Action"] = df_engine.apply(multi_vote, axis=1)
    decision_cols = ["fund","3M","6M","1Y","3Y","Final Action"]
    st.dataframe(df_engine[decision_cols], use_container_width=True)

    # Risk vs Return
    ycol = f"{tf}_Return_%" if tf in ["MTD","YTD"] else f"{tf}_CAGR_%"
    fig = px.scatter(
        dfp, x=f"{tf}_Volatility_%", y=ycol, size=dfp[f"{tf}_MaxDD_%"].abs(),
        text="fund", title="Risk vs Return"
    )
    fig.update_layout(font=dict(size=10))
    st.plotly_chart(fig, use_container_width=True, height=400)

    # Metrics table
    metric_cols = ["fund", f"{tf}_Return_%" if tf in ["MTD","YTD"] else f"{tf}_CAGR_%",
                   f"{tf}_Sharpe", f"{tf}_Volatility_%", f"{tf}_MaxDD_%"]
    st.subheader("📊 Metrics")
    st.dataframe(df_engine[metric_cols].round(2), use_container_width=True)

    # NAV Curve
    st.subheader("📈 Fund NAV Curve")
    df_plot = nav_df[nav_df["fund"].isin(dff["fund"])]
    fig = px.line(df_plot, x="date", y="nav", color="fund", title="Fund NAV Curve")
    fig.update_layout(yaxis_title="NAV (หน่วย)", xaxis_title="วันที่", font=dict(size=10))
    st.plotly_chart(fig, use_container_width=True, height=400)

# ================= MENTAL PAIN =================
with st.expander(f"😈 Mental Pain ({tf})"):
    dfp = dff.dropna(subset=[f"{tf}_DD_Duration_days", f"{tf}_Worst_Rolling_%"])
    dfp["Pain_%"] = -dfp[f"{tf}_Worst_Rolling_%"]
    fig = px.scatter(
        dfp, x=f"{tf}_DD_Duration_days", y=f"{tf}_Worst_Rolling_%", size=dfp[f"{tf}_MaxDD_%"].abs(),
        color=f"{tf}_Best_Rolling_%", text="fund", title="Mental Pain Map"
    )
    fig.update_layout(font=dict(size=10))
    st.plotly_chart(fig, use_container_width=True, height=400)

# ================= FOOTER =================
st.caption("""
CAGR = โตจริง  
Volatility = ผันผวน  
Sharpe = คุ้มเสี่ยง  
MaxDD = เจ็บสุด  
Worst Rolling = ช่วงนรก  
Best Rolling = ช่วงฟิน  
DD Duration = ทรมานกี่วัน
""")
