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
    df = nav_df[nav_df["fund"].isin(funds)] \
            .pivot(index="date", columns="fund", values="nav") \
            .sort_index()

    start_date = df.dropna().index.min()
    df = df[df.index >= start_date]
    df = df.ffill()

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
        vol = row[f"{tf}_Volatility_%"]
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
    "Return_%",
    "CAGR_%",
    "Volatility_%",
    "Sharpe",
    "MaxDD_%",
    "Worst_Rolling_%",
    "Best_Rolling_%",
    "DD_Duration_days"
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
st.set_page_config(page_title="Fund Dashboard", layout="wide")
st.title("📊 Fund Performance Dashboard (Investor View)")

with st.sidebar:
    tf = st.radio(
        "📅 Timeframe",
        ["MTD","YTD","1M","3M","6M","1Y","3Y","5Y","MAX"],
        index=5
    )

    # ===== แสดงช่วงวันที่ =====
    nav_tmp = nav_df.set_index("date")["nav"]
    nav_tmp = filter_by_timeframe(nav_tmp, tf)

    if len(nav_tmp) > 0:
        start_date = nav_tmp.index.min().date()
        end_date = nav_tmp.index.max().date()

        st.caption(f"📆 {start_date} → {end_date}")
    else:
        st.caption("ไม่มีข้อมูลในช่วงนี้")
    funds = st.multiselect(
        "เลือกกองทุน",
        df["fund"].unique(),
        default=list(df["fund"].unique())
    )

dff = df[df["fund"].isin(funds)]

tab_overview, tab_pain, tab_port, tab_diver = st.tabs(
    ["📊 Overview", "😈 Mental Pain", "🧾 Portfolio", "🔗 Diversification"]
)

# ================= OVERVIEW =================
with tab_overview:
    st.subheader(f"Overview ({tf})")

    ycol = f"{tf}_Return_%" if tf in ["MTD","YTD"] else f"{tf}_CAGR_%"
    dfp = dff.dropna(subset=[ycol]).copy()

    # ================= DECISION ENGINE (คำนวณครั้งเดียว) =================
    df_engine = dfp.copy()
    df_engine["3M"] = df_engine.apply(lambda r: recommend(r, "3M"), axis=1)
    df_engine["6M"] = df_engine.apply(lambda r: recommend(r, "6M"), axis=1)
    df_engine["1Y"] = df_engine.apply(lambda r: recommend(r, "1Y"), axis=1)
    df_engine["3Y"] = df_engine.apply(lambda r: recommend(r, "3Y"), axis=1)
    df_engine["Final Action"] = df_engine.apply(multi_vote, axis=1)

    # ================= DECISION TABLE (ขึ้นบนสุด) =================
    st.subheader("🧭 Decision (Multi-Timeframe Voting)")
    decision_cols = ["fund","3M","6M","1Y","3Y","Final Action"]
    st.dataframe(df_engine[decision_cols], use_container_width=True)

    # ================= RISK vs RETURN =================
    fig = px.scatter(
        dfp,
        x=f"{tf}_Volatility_%",
        y=ycol,
        size=dfp[f"{tf}_MaxDD_%"].abs(),
        text="fund",
        title="Risk vs Return"
    )

    xm = dfp[f"{tf}_Volatility_%"].mean()
    ym = dfp[ycol].mean()
    fig.add_vline(x=xm, line_dash="dash")
    fig.add_hline(y=ym, line_dash="dash")

    xmin = dfp[f"{tf}_Volatility_%"].min()
    xmax = dfp[f"{tf}_Volatility_%"].max()
    ymin = dfp[ycol].min()
    ymax = dfp[ycol].max()

    fig.add_annotation(x=(xmin+xm)/2, y=(ym+ymax)/2,
        text="💎 ของดีหายาก<br>กำไรดี เสี่ยงต่ำ", showarrow=False)
    fig.add_annotation(x=(xm+xmax)/2, y=(ym+ymax)/2,
        text="🏆 ตัวแรง<br>โตไว ใจต้องนิ่ง", showarrow=False)
    fig.add_annotation(x=(xmin+xm)/2, y=(ymin+ym)/2,
        text="🧘 ปลอดภัย<br>ไม่รวยแต่ไม่เจ็บ", showarrow=False)
    fig.add_annotation(x=(xm+xmax)/2, y=(ymin+ym)/2,
        text="😵 เสี่ยงฟรี<br>ควรหลีกเลี่ยง", showarrow=False)

    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    # ================= METRICS =================
    if tf in ["MTD","YTD"]:
        metric_cols = ["fund", f"{tf}_Return_%", f"{tf}_Sharpe",
                   f"{tf}_Volatility_%", f"{tf}_MaxDD_%"]
    else:
        metric_cols = ["fund", f"{tf}_CAGR_%", f"{tf}_Sharpe",
                    f"{tf}_Volatility_%", f"{tf}_MaxDD_%"]

    st.subheader(f"📊 Metrics ({tf})")
    st.dataframe(df_engine[metric_cols].round(2), use_container_width=True)

    st.subheader("📈 Fund NAV Curve")
    df_plot = nav_df[nav_df["fund"].isin(dff)]
    fig = px.line(
        df_plot,
        x="date",
        y="nav",
        color="fund",
        title="Fund NAV Curve"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📉 Drawdown Curve")
    dd_all = []
    for f in dff:
        fdf = nav_df[nav_df["fund"] == f].copy()
        fdf["cummax"] = fdf["nav"].cummax()
        fdf["drawdown"] = (fdf["nav"] / fdf["cummax"] - 1) * 100
        dd_all.append(fdf)

    dd_df = pd.concat(dd_all)
    fig_dd = px.line(
        dd_df,
        x="date",
        y="drawdown",
        color="fund",
        title="Drawdown (%)"
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    st.subheader("🔥 Buy / Overheat Zone")
    win = 60
    z_all = []
    for f in dff:
        fdf = nav_df[nav_df["fund"] == f].copy()
        fdf["ma"] = fdf["nav"].rolling(win).mean()
        fdf["std"] = fdf["nav"].rolling(win).std()
        fdf["z"] = (fdf["nav"] - fdf["ma"]) / fdf["std"]
        z_all.append(fdf)
    z_df = pd.concat(z_all)
    fig_z = px.line(
        z_df,
        x="date",
        y="z",
        color="fund",
        title="Z-Score (Buy / Overheat)"
    )
    fig_z.add_hline(y=2, line_dash="dash")
    fig_z.add_hline(y=-2, line_dash="dash")
    st.plotly_chart(fig_z, use_container_width=True)

# ================= MENTAL PAIN =================
with tab_pain:
    st.subheader(f"Mental Pain ({tf})")

    dfp = dff.dropna(subset=[
        f"{tf}_DD_Duration_days",
        f"{tf}_Worst_Rolling_%"
    ])
    dfp["Pain_%"] = -dfp[f"{tf}_Worst_Rolling_%"]
    fig = px.scatter(
        dfp,
        x=f"{tf}_DD_Duration_days",
        y=f"{tf}_Worst_Rolling_%",
        size=dfp[f"{tf}_MaxDD_%"].abs(),
        color=f"{tf}_Best_Rolling_%",
        text="fund",
        title="Mental Pain Map"
    )
    
    xm = dfp[f"{tf}_DD_Duration_days"].mean()
    ym = dfp[f"{tf}_Worst_Rolling_%"].mean()

    fig.add_vline(x=xm, line_dash="dash")
    fig.add_hline(y=ym, line_dash="dash")

    # fig.add_annotation(
    #     x=xm, y=ym,
    #     text="ขึ้น = เจ็บแรง\nขวา = เจ็บนาน",
    #     showarrow=False,
    #     font=dict(size=14)
    # )

    fig.add_annotation(x=xm*0.6, y=ym*0.6, text="🧘 Zen\nถือแล้วสบายใจ", showarrow=False)
    fig.add_annotation(x=xm*0.6, y=ym*1.4, text="💥 Shock\nตกแรงแต่หายไว", showarrow=False)
    fig.add_annotation(x=xm*1.4, y=ym*0.6, text="🐢 Slow Burn\nทรมานยาว", showarrow=False)
    fig.add_annotation(x=xm*1.4, y=ym*1.4, text="🔥 Hell Mode\nใจพัง", showarrow=False)

    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    loss_rows = []
    for fund in funds:
        g = nav_df[nav_df["fund"] == fund].sort_values("date")
        nav = filter_by_timeframe(g.set_index("date")["nav"], tf)
        ret = nav.pct_change().dropna()
        roll = (1+ret).rolling(252).apply(np.prod, raw=True) - 1
        loss_rows.append({
            "fund": fund,
            "Loss_Prob_%": (roll < 0).mean()*100
        })

    loss_df = pd.DataFrame(loss_rows)
    st.dataframe(loss_df.round(2))


# ================= PORTFOLIO =================
with tab_port:

    # ---------- Load transactions ----------
    if not os.path.exists("transactions.csv"):
        pd.DataFrame(columns=["date","fund","amount","price"]).to_csv("transactions.csv", index=False)

    nav_cut = nav_df.set_index("date")
    nav_cut = nav_df.groupby("fund").apply(
        lambda x: filter_by_timeframe(
            x.set_index("date")["nav"], tf
        )
    ).reset_index(name="nav")
    
    tx_df = pd.read_csv("transactions.csv")
    tx_df["date"] = pd.to_datetime(tx_df["date"], errors="coerce")

    if len(tx_df) > 0:
        tx_df["units"] = tx_df["amount"] / tx_df["price"]
        port = tx_df.groupby("fund").agg({
            "amount":"sum",
            "units":"sum"
        }).reset_index()

        # 🔹 FILTER ตาม sidebar (หัวใจของคำตอบ)
        port = port[port["fund"].isin(funds)]

        latest_nav = nav_cut.sort_values("date").groupby("fund").tail(1)[["fund","nav"]]
        port = port.merge(latest_nav, on="fund", how="left")
        port["current_value"] = port["units"] * port["nav"]
        port["profit"] = port["current_value"] - port["amount"]
        port["profit_%"] = port["profit"] / port["amount"] * 100

        # ---------- Merge Volatility ----------
        risk_col = f"{tf}_Volatility_%"
        vol_cols = df[["fund", risk_col]]
        port = port.merge(vol_cols, on="fund", how="left")

        # ---------- Risk Weight ----------
        port["risk_weight"] = port["current_value"] * port[risk_col]

        total_amount = port["amount"].sum()
        total_value = port["current_value"].sum()
        total_profit = total_value - total_amount
        total_profit_pct = total_profit / total_amount * 100

        summary_row = pd.DataFrame([{
            "fund": "TOTAL",
            "amount": total_amount,
            "units": np.nan,
            "nav": np.nan,
            "current_value": total_value,
            "profit": total_profit,
            "profit_%": total_profit_pct,
            risk_col: np.nan,
            "risk_weight": port["risk_weight"].sum()
        }])

        port_show = pd.concat([port, summary_row], ignore_index=True)

        # ================= Money & Risk Pie (2 Columns) =================
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("🥧 Money Allocation (เงินอยู่ตรงไหน)")
            fig1 = px.pie(
                port,
                values="current_value",
                names="fund",
                title="สัดส่วนเงินลงทุนปัจจุบัน"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            st.subheader("⚠️ Risk Exposure (ความเสี่ยงมาจากไหน)")
            fig2 = px.pie(
                port,
                values="risk_weight",
                names="fund",
                title=f"สัดส่วนความเสี่ยง (Money × {tf} Volatility)"
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

    # ---------- Add Transaction ----------
    st.subheader("➕ บันทึกการซื้อกองทุน")
    with st.form("add_tx"):
        c1, c2 = st.columns(2)
        with c1:
            tx_date = st.date_input("วันที่")
            tx_fund = st.selectbox("กองทุน", df["fund"].unique())
        with c2:
            tx_amount = st.number_input("เงินลงทุน", min_value=0.0)
            tx_price = st.number_input("ราคาต่อหน่วย", min_value=0.0)

        if st.form_submit_button("➕ เพิ่ม"):
            new = pd.DataFrame([{
                "date": pd.to_datetime(tx_date),
                "fund": tx_fund,
                "amount": tx_amount,
                "price": tx_price
            }])
            tx_df = pd.concat([tx_df, new])
            tx_df.to_csv("transactions.csv", index=False)
            st.success("บันทึกเรียบร้อย")

    st.divider()

    st.subheader("📜 ประวัติ")
    st.dataframe(tx_df.sort_values("date", ascending=False))

    if len(tx_df) > 0:
        tx_df["units"] = tx_df["amount"] / tx_df["price"]
        st.divider()
        st.subheader("📊 สรุปพอร์ต")

        show_cols = [
            "fund",
            "amount",
            "current_value",
            "profit",
            "profit_%",
            risk_col
        ]

        st.dataframe(port_show[show_cols].round(2))

    st.subheader("📈 Portfolio NAV Curve (Equal Weight)")

    df_norm, port_nav = build_equal_weight_nav(nav_cut, funds)

    fig = px.line(title="Portfolio vs Each Fund (Normalized = 100)")
    for f in funds:
        fig.add_scatter(
            x=df_norm.index,
            y=df_norm[f],
            name=f,
            opacity=0.4
        )

    fig.add_scatter(
        x=port_nav.index,
        y=port_nav.values,
        name="PORTFOLIO",
        line=dict(width=4)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
# ================= DIVERSIFICATION =================
with tab_diver:
    st.subheader("Correlation Heatmap")

    # ----- เตรียม NAV ตาม timeframe แบบ robust -----
    nav_cut = nav_df.groupby("fund").apply(
        lambda x: filter_by_timeframe(
            x.set_index("date")["nav"], tf
        )
    ).reset_index(name="nav")

    # ----- คำนวณ return และ align -----
    df_ret = nav_cut[nav_cut["fund"].isin(funds)] \
        .pivot(index="date", columns="fund", values="nav") \
        .ffill() \
        .pct_change() \
        .dropna()

    # ----- Guardrail: sample size -----
    if len(df_ret) < 60:
        st.warning("ข้อมูลน้อยเกินไป Correlation อาจไม่น่าเชื่อถือ (ควร ≥ 60 วัน)")

    # ----- Correlation matrix -----
    corr = df_ret.corr()

    # ----- Heatmap -----
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap (Return)"
    )

    fig.add_annotation(
        x=0.5, y=1.08,
        xref="paper", yref="paper",
        text="ใกล้ +1 = ไปทิศเดียวกัน | ใกล้ 0 = ไม่เกี่ยวกัน | ใกล้ -1 = สวนทางกัน",
        showarrow=False,
        font=dict(size=14)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ===== วิเคราะห์เป็นคู่ =====
    def interpret_corr(val):
        if val > 0.8:
            return "ซ้ำกันมาก"
        elif val > 0.5:
            return "สัมพันธ์สูง"
        elif val > 0.2:
            return "สัมพันธ์ต่ำ"
        elif val > -0.2:
            return "แทบไม่เกี่ยว"
        else:
            return "สวนทาง"

    if len(funds) >= 2:
        pairs = list(itertools.combinations(funds, 2))
        results = []

        for f1, f2 in pairs:
            v = corr.loc[f1, f2]
            results.append({
                "คู่กอง": f"{f1} vs {f2}",
                "Correlation": round(v, 2),
                "ความหมาย": interpret_corr(v)
            })

        result_df = pd.DataFrame(results)

        st.subheader("📊 วิเคราะห์ความสัมพันธ์ของกองที่เลือก")
        st.dataframe(result_df, use_container_width=True)

        # ----- Insight -----
        avg_corr = result_df["Correlation"].mean()
        max_row = result_df.loc[result_df["Correlation"].idxmax()]
        min_row = result_df.loc[result_df["Correlation"].idxmin()]

        st.markdown("### 🧠 Insight รวม")
        st.write(f"• ค่าเฉลี่ย Correlation: **{avg_corr:.2f}**")
        st.write(f"• คู่ที่ซ้ำสุด: **{max_row['คู่กอง']} ({max_row['Correlation']})**")
        st.write(f"• คู่ที่กระจายสุด: **{min_row['คู่กอง']} ({min_row['Correlation']})**")

        if avg_corr > 0.7:
            st.error("พอร์ตนี้ซ้ำสูงมาก → เสี่ยงพร้อมกัน")
        elif avg_corr > 0.4:
            st.warning("พอร์ตนี้ยังซ้ำพอสมควร → ควรเพิ่มกองที่อิสระกว่านี้")
        else:
            st.success("พอร์ตนี้กระจายตัวดี → ความเสี่ยงถ่วงกันได้")
    else:
        st.info("กรุณาเลือกอย่างน้อย 2 กอง")

    st.subheader("⚠️ Correlation-adjusted Portfolio Risk")

    # ----- weights จากเงินจริงในพอร์ต -----
    latest_nav = nav_df.sort_values("date").groupby("fund").tail(1)
    latest_nav = latest_nav[latest_nav["fund"].isin(funds)]
    latest_nav = latest_nav.set_index("fund")

    weights = latest_nav["nav"] / latest_nav["nav"].sum()

    # ----- align returns -----
    ret_use = df_ret[weights.index].dropna()

    # ----- covariance -----
    cov = ret_use.cov()

    # ----- portfolio variance -----
    w = weights.values
    port_var = np.dot(w.T, np.dot(cov, w))

    # ----- annualized volatility -----
    port_vol = np.sqrt(port_var * 252)

    st.metric("Portfolio Volatility (Corr-adjusted)", f"{port_vol*100:.2f}%")
    
    # ----- Diversification Ratio -----
    indiv_vol = ret_use.std() * np.sqrt(252)
    weighted_avg = np.sum(w * indiv_vol)

    div_ratio = weighted_avg / port_vol

    st.metric("Diversification Ratio", f"{div_ratio:.2f}")
    
    st.markdown("""
    ### 🧭 วิธีอ่านค่า

    **Portfolio Volatility (Corr-adjusted)**  
    ความผันผวนของพอร์ตจริงทั้งระบบต่อปี  
    คำนวณจาก *น้ำหนักเงิน + ความผันผวน + ความสัมพันธ์ของกอง*

    > 10% = ปีปกติขึ้นลงราว ±10%  
    > 15% = ปีปกติขึ้นลงราว ±15%  
    > 20%+ = พอร์ตเหวี่ยงสูง ต้องรับความเสี่ยงได้

    ---

    **Diversification Ratio**  
    ระดับการกระจายความเสี่ยงจริงของพอร์ต

    > 1.0 = แทบไม่กระจาย (ซ้ำกัน)  
    > 1.2 = กระจายพอใช้  
    > 1.4 = กระจายดี  
    > 1.6+ = กระจายระดับกองทุน

    ---

    **สรุปสั้น ๆ**

    Portfolio Volatility = *พอร์ตเสี่ยงแค่ไหน*  
    Diversification Ratio = *ความเสี่ยงนั้นจำเป็นจริงหรือซ้ำกันเอง*
    """)
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



