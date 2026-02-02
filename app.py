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
    if tf == "MTD": start = end.replace(day=1)
    elif tf == "YTD": start = end.replace(month=1, day=1)
    elif tf == "1M": start = end - pd.DateOffset(months=1)
    elif tf == "3M": start = end - pd.DateOffset(months=3)
    elif tf == "6M": start = end - pd.DateOffset(months=6)
    elif tf == "1Y": start = end - pd.DateOffset(years=1)
    elif tf == "3Y": start = end - pd.DateOffset(years=3)
    elif tf == "5Y": start = end - pd.DateOffset(years=5)
    elif tf == "MAX": start = nav_series.index.min()
    else: return nav_series
    return nav_series[nav_series.index >= start]

def recommend(row, tf):
    try:
        cagr = row[f"{tf}_CAGR_%"]
        sharpe = row[f"{tf}_Sharpe"]
        dd = abs(row[f"{tf}_MaxDD_%"])
    except: return "ข้อมูลไม่พอ"
    if pd.isna(cagr) or pd.isna(sharpe): return "ข้อมูลไม่พอ"
    if cagr > 8 and sharpe > 1 and dd < 25: return "🟢 ควรเพิ่ม"
    elif cagr > 5 and sharpe > 0.5: return "🟡 ถือรอ"
    elif sharpe < 0.3 or dd > 40: return "🟠 ควรลด"
    else: return "🔴 ควรขาย/สับ"

def multi_vote(row):
    frames = ["3M","6M","1Y","3Y"]
    votes = []
    for tf in frames:
        try:
            v = recommend(row, tf)
            if v != "ข้อมูลไม่พอ": votes.append(v)
        except: pass
    if len(votes) == 0: return "ข้อมูลไม่พอ"
    return max(set(votes), key=votes.count)

# ================= PRE-CALC =================
timeframes = ["MTD","YTD","1M","3M","6M","1Y","3Y","5Y","MAX"]
METRICS = ["Return_%","CAGR_%","Volatility_%","Sharpe","MaxDD_%","Worst_Rolling_%","Best_Rolling_%","DD_Duration_days"]
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
                if tf=="YTD" and k in ["CAGR_%","Sharpe"]: data[f"{tf}_{k}"]=np.nan
                else: data[f"{tf}_{k}"]=m[k]
        else:
            for k in METRICS: data[f"{tf}_{k}"]=np.nan
    rows.append(data)
df = pd.DataFrame(rows)

# ================= UI =================
st.set_page_config(page_title="Fund Dashboard", layout="centered")
st.title("📊 Fund Performance Dashboard (Mobile)")

# ================= EXPANDER: FILTER =================
with st.expander("🔧 ตัวเลือกกองทุน / Timeframe", expanded=True):
    tf = st.radio("📅 Timeframe", timeframes, index=5, horizontal=True)
    funds = st.multiselect("เลือกกองทุน", df["fund"].unique(), default=list(df["fund"].unique()))
dff = df[df["fund"].isin(funds)]

# ================= TABS =================
tab_overview, tab_pain, tab_port, tab_diver = st.tabs(
    ["📊 Overview", "😈 Mental Pain", "🧾 Portfolio", "🔗 Diversification"]
)

# ================= OVERVIEW =================
with tab_overview:
    st.subheader(f"Overview ({tf})")

    # ------------------- NAV / Drawdown / Z-Score -------------------
    st.subheader("📈 Fund NAV Curve + 📉 Drawdown + 🔥 Buy/Overheat Zone")

    df_plot = nav_df[nav_df["fund"].isin(dff["fund"])].copy()

    # ---------- NAV Curve ----------
    fig_nav = px.line(
        df_plot,
        x="date",
        y="nav",
        color="fund",
        title="📈 Fund NAV Curve",
        labels={"nav":"NAV (หน่วย)", "date":"วันที่"}
    )
    fig_nav.update_layout(legend_title="Fund")
    fig_nav.add_annotation(
        x=df_plot['date'].min(),
        y=df_plot['nav'].max(),
        text="📌 NAV = มูลค่าหน่วยลงทุน",
        showarrow=False,
        font=dict(size=12, color="blue")
    )
    st.plotly_chart(fig_nav, use_container_width=True, height=300)

    # ---------- Drawdown ----------
    dd_all=[]
    for f in dff["fund"]:
        fdf = df_plot[df_plot["fund"]==f].copy()
        fdf["cummax"]=fdf["nav"].cummax()
        fdf["drawdown"]=(fdf["nav"]/fdf["cummax"]-1)*100
        dd_all.append(fdf)
    dd_df=pd.concat(dd_all)

    fig_dd=px.line(
        dd_df,
        x="date",
        y="drawdown",
        color="fund",
        title="📉 Drawdown Curve (%)",
        labels={"drawdown":"Drawdown (%)", "date":"วันที่"}
    )
    fig_dd.update_traces(line=dict(width=2))
    fig_dd.add_hline(y=0,line_dash="dash",line_color="black")
    fig_dd.add_annotation(
        x=dd_df['date'].min(),
        y=dd_df["drawdown"].min(),
        text="💥 Drawdown = % การลดลงจากจุดสูงสุด",
        showarrow=False,
        font=dict(size=12, color="red")
    )
    st.plotly_chart(fig_dd, use_container_width=True, height=300)

    # ---------- Z-Score ----------
    win = 60
    z_all=[]
    for f in dff["fund"]:
        fdf = df_plot[df_plot["fund"]==f].copy()
        fdf["ma"] = fdf["nav"].rolling(win).mean()
        fdf["std"] = fdf["nav"].rolling(win).std()
        fdf["z"] = (fdf["nav"]-fdf["ma"])/fdf["std"]
        z_all.append(fdf)
    z_df=pd.concat(z_all)

    fig_z = px.line(
        z_df,
        x="date",
        y="z",
        color="fund",
        title="🔥 Z-Score (Buy / Overheat Zone)",
        labels={"z":"Z-Score", "date":"วันที่"}
    )
    fig_z.update_traces(line=dict(width=2))
    # Buy/Overheat zones
    fig_z.add_hline(y=2,line_dash="dash",line_color="red", annotation_text="Overheat", annotation_position="top left")
    fig_z.add_hline(y=-2,line_dash="dash",line_color="green", annotation_text="Buy Zone", annotation_position="bottom left")
    fig_z.add_annotation(
        x=z_df['date'].min(),
        y=z_df['z'].max(),
        text="📌 Z-Score = (NAV - MA60)/STD60\nสูง → overheat / ต่ำ → ซื้อ",
        showarrow=False,
        font=dict(size=12, color="purple")
    )
    st.plotly_chart(fig_z, use_container_width=True, height=300)

    st.divider()

    # ------------------- Decision Engine & Risk vs Return -------------------
    ycol = f"{tf}_Return_%" if tf in ["MTD","YTD"] else f"{tf}_CAGR_%"
    dfp = dff.dropna(subset=[ycol]).copy()

    if dfp.empty:
        st.info("ไม่มีข้อมูลเพียงพอสำหรับ Overview Risk vs Return")
    else:
        # ================= DECISION ENGINE =================
        df_engine = dfp.copy()
        df_engine["3M"] = df_engine.apply(lambda r: recommend(r, "3M"), axis=1)
        df_engine["6M"] = df_engine.apply(lambda r: recommend(r, "6M"), axis=1)
        df_engine["1Y"] = df_engine.apply(lambda r: recommend(r, "1Y"), axis=1)
        df_engine["3Y"] = df_engine.apply(lambda r: recommend(r, "3Y"), axis=1)
        df_engine["Final Action"] = df_engine.apply(multi_vote, axis=1)

        # ================= DECISION TABLE =================
        st.subheader("🧭 Decision (Multi-Timeframe Voting)")
        decision_cols = ["fund","3M","6M","1Y","3Y","Final Action"]
        st.dataframe(df_engine[decision_cols], use_container_width=True)

        # ================= RISK vs RETURN =================
        fig = px.scatter(
            dfp,
            x=f"{tf}_Volatility_%",
            y=ycol,
            size=dfp[f"{tf}_MaxDD_%"].abs(),
            color=ycol,
            text="fund",
            title="Risk vs Return",
            hover_data={
                "fund": True,
                f"{tf}_Volatility_%": True,
                ycol: True,
                f"{tf}_MaxDD_%": True,
                f"{tf}_Sharpe": True
            },
            color_continuous_scale="Viridis",
        )

        # Mean lines
        xm = dfp[f"{tf}_Volatility_%"].mean()
        ym = dfp[ycol].mean()
        fig.add_vline(x=xm, line_dash="dash", line_color="gray", annotation_text="Avg Volatility", annotation_position="top left")
        fig.add_hline(y=ym, line_dash="dash", line_color="gray", annotation_text="Avg Return", annotation_position="top right")

        # Quadrant annotations
        xmin = dfp[f"{tf}_Volatility_%"].min()
        xmax = dfp[f"{tf}_Volatility_%"].max()
        ymin = dfp[ycol].min()
        ymax = dfp[ycol].max()

        fig.add_annotation(x=(xmin+xm)/2, y=(ym+ymax)/2,
            text="💎 ของดีหายาก\nกำไรดี เสี่ยงต่ำ", showarrow=False)
        fig.add_annotation(x=(xm+xmax)/2, y=(ym+ymax)/2,
            text="🏆 ตัวแรง\nโตไว ใจต้องนิ่ง", showarrow=False)
        fig.add_annotation(x=(xmin+xm)/2, y=(ymin+ym)/2,
            text="🧘 ปลอดภัย\nไม่รวยแต่ไม่เจ็บ", showarrow=False)
        fig.add_annotation(x=(xm+xmax)/2, y=(ymin+ym)/2,
            text="😵 เสี่ยงฟรี\nควรหลีกเลี่ยง", showarrow=False)

        fig.update_traces(textposition="top center")
        fig.update_layout(
            xaxis_title="Volatility (%)",
            yaxis_title="Return (%)",
            legend_title=ycol
        )

        st.plotly_chart(fig, use_container_width=True)


# ================= MENTAL PAIN TAB =================
with tab_pain:
    st.subheader(f"Mental Pain Map ({tf})")

    # Filter funds with enough data
    dfp = dff.dropna(subset=[
        f"{tf}_DD_Duration_days",
        f"{tf}_Worst_Rolling_%",
        f"{tf}_MaxDD_%",
        f"{tf}_Best_Rolling_%"
    ]).copy()

    if dfp.empty:
        st.info("ไม่มีข้อมูลเพียงพอสำหรับแสดง Mental Pain Map")
    else:
        # Pain calculation (negative of Worst Rolling %)
        dfp["Pain_%"] = -dfp[f"{tf}_Worst_Rolling_%"]

        # Scatter plot
        fig = px.scatter(
            dfp,
            x=f"{tf}_DD_Duration_days",
            y=f"{tf}_Worst_Rolling_%",
            size=dfp[f"{tf}_MaxDD_%"].abs(),
            color=f"{tf}_Best_Rolling_%",
            text="fund",
            title="Mental Pain Map",
            hover_data={
                f"{tf}_DD_Duration_days": True,
                f"{tf}_Worst_Rolling_%": True,
                f"{tf}_MaxDD_%": True,
                f"{tf}_Best_Rolling_%": True,
            }
        )

        # Mean lines
        xm = dfp[f"{tf}_DD_Duration_days"].mean()
        ym = dfp[f"{tf}_Worst_Rolling_%"].mean()
        fig.add_vline(x=xm, line_dash="dash", line_color="gray")
        fig.add_hline(y=ym, line_dash="dash", line_color="gray")

        # Annotations (เหมือนเดิม)
        fig.add_annotation(x=xm*0.6, y=ym*0.6, text="🧘 Zen\nถือแล้วสบายใจ", showarrow=False)
        fig.add_annotation(x=xm*0.6, y=ym*1.4, text="💥 Shock\nตกแรงแต่หายไว", showarrow=False)
        fig.add_annotation(x=xm*1.4, y=ym*0.6, text="🐢 Slow Burn\nทรมานยาว", showarrow=False)
        fig.add_annotation(x=xm*1.4, y=ym*1.4, text="🔥 Hell Mode\nใจพัง", showarrow=False)

        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ---------- Loss probability table ----------
        loss_rows = []
        for fund in funds:
            g = nav_df[nav_df["fund"] == fund].sort_values("date")
            nav_series = filter_by_timeframe(g.set_index("date")["nav"], tf)
            if len(nav_series) >= 20:
                ret = nav_series.pct_change().dropna()
                roll = (1 + ret).rolling(252).apply(np.prod, raw=True) - 1
                loss_rows.append({
                    "fund": fund,
                    "Loss_Prob_%": (roll < 0).mean() * 100
                })

        loss_df = pd.DataFrame(loss_rows)
        if not loss_df.empty:
            st.subheader("📉 Loss Probability (ความน่าจะเป็นขาดทุนช่วง Rolling 252 วัน)")
            st.dataframe(loss_df.round(2), use_container_width=True)

# ================= PORTFOLIO TAB =================
with tab_port:
    st.subheader(f"Portfolio Overview ({tf})")

    # ---------- Load transactions ----------
    if not os.path.exists("transactions.csv"):
        pd.DataFrame(columns=["date","fund","amount","price"]).to_csv("transactions.csv", index=False)

    tx_df = pd.read_csv("transactions.csv")
    tx_df["date"] = pd.to_datetime(tx_df["date"], errors="coerce")

    # ---------- Filter NAV by selected funds & timeframe ----------
    nav_cut = nav_df[nav_df["fund"].isin(funds)].copy()
    nav_cut = nav_cut.groupby("fund").apply(
        lambda x: filter_by_timeframe(x.set_index("date")["nav"], tf)
    ).reset_index(name="nav")

    # ---------- Compute portfolio ----------
    if len(tx_df) > 0 and len(nav_cut) > 0:
        tx_df["units"] = tx_df["amount"] / tx_df["price"]
        port = tx_df.groupby("fund").agg({
            "amount": "sum",
            "units": "sum"
        }).reset_index()

        # Filter by sidebar selection
        port = port[port["fund"].isin(funds)]

        # Merge latest NAV
        latest_nav = nav_cut.sort_values("date").groupby("fund").tail(1)[["fund","nav"]]
        port = port.merge(latest_nav, on="fund", how="left")
        port["current_value"] = port["units"] * port["nav"]
        port["profit"] = port["current_value"] - port["amount"]
        port["profit_%"] = port["profit"] / port["amount"] * 100

        # Merge Volatility for risk-weight
        risk_col = f"{tf}_Volatility_%"
        vol_cols = df[["fund", risk_col]]
        port = port.merge(vol_cols, on="fund", how="left")
        port["risk_weight"] = port["current_value"] * port[risk_col]

        # Summary row
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

        # ================= Money & Risk Pie =================
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🥧 Money Allocation")
            fig1 = px.pie(
                port,
                values="current_value",
                names="fund",
                title="สัดส่วนเงินลงทุนปัจจุบัน"
            )
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            st.subheader("⚠️ Risk Exposure")
            fig2 = px.pie(
                port,
                values="risk_weight",
                names="fund",
                title=f"สัดส่วนความเสี่ยง ({tf} Volatility)"
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # ================= Add Transaction Form =================
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

        # ================= Transaction History =================
        st.subheader("📜 ประวัติ")
        st.dataframe(tx_df.sort_values("date", ascending=False))

        # ================= Portfolio Summary =================
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

        # ================= Portfolio NAV Curve (Equal Weight) =================
        st.subheader("📈 Portfolio NAV Curve (Equal Weight)")
        df_norm, port_nav = build_equal_weight_nav(nav_cut, funds)
        fig = px.line(title="Portfolio vs Each Fund (Normalized = 100)")
        for f in funds:
            if f in df_norm.columns:
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

        # ================= Volatility / Diversification =================
        st.subheader("🔗 Portfolio Risk & Diversification")

        # Align NAV for return calculation
        df_pivot = nav_cut.pivot(index="date", columns="fund", values="nav")
        df_pivot = df_pivot.reindex(columns=port["fund"]).ffill().pct_change().dropna()

        w = port.set_index("fund")["current_value"] / port["current_value"].sum()
        w = w.reindex(df_pivot.columns).fillna(0).values

        cov = df_pivot.cov()
        port_var = np.dot(w.T, np.dot(cov, w))
        port_vol = np.sqrt(port_var * 252)
        st.metric("Portfolio Volatility (Corr-adjusted)", f"{port_vol*100:.2f}%")

        indiv_vol = df_pivot.std() * np.sqrt(252)
        weighted_avg = np.sum(w * indiv_vol)
        div_ratio = weighted_avg / port_vol if port_vol > 0 else np.nan
        st.metric("Diversification Ratio", f"{div_ratio:.2f}")

        st.markdown("""
        • Portfolio Volatility = ความผันผวนของพอร์ตต่อปี  
        • Diversification Ratio = การกระจายความเสี่ยง (สูง = กระจายดี)
        """)
    else:
        st.info("ไม่มีข้อมูล NAV หรือ Transaction ให้แสดง")

# ================= DIVERSIFICATION =================
with tab_diver:
    st.subheader("Correlation Heatmap")
    nav_cut=nav_df.groupby("fund").apply(lambda x: filter_by_timeframe(x.set_index("date")["nav"], tf)).reset_index(name="nav")
    df_ret=nav_cut[nav_cut["fund"].isin(funds)].pivot(index="date", columns="fund", values="nav").ffill().pct_change().dropna()
    corr=df_ret.corr()
    fig=px.imshow(corr,text_auto=".2f",color_continuous_scale="RdBu",zmin=-1,zmax=1)
    st.plotly_chart(fig,use_container_width=True,height=400)





