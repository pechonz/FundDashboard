import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import itertools
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

SHEET_URL = "https://docs.google.com/spreadsheets/d/1rPF7yvVHpZtqewgdnXPE9u5slxlYgnV-epEla3svG0s"

# ================= LOAD NAV =================
url = "https://raw.githubusercontent.com/pechonz/FundDashboard/main/fund_nav_5y.csv"
nav_df = pd.read_csv(url)
nav_df["date"] = pd.to_datetime(nav_df["date"], errors="coerce")
nav_df = nav_df.sort_values(["fund","date"])

# ================= FUNCTIONS =================
@st.cache_data(ttl=30)
def load_data():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    gc = gspread.authorize(creds)
    sh = gc.open("transactions")
    ws = sh.sheet1
    data = ws.get_all_records()
    return pd.DataFrame(data)


def save_data(df):
    import gspread
    from google.oauth2.service_account import Credentials

    creds = Credentials.from_service_account_info(
        st.secrets["gcp"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )

    gc = gspread.authorize(creds)
    sh = gc.open("transactions")
    ws = sh.sheet1

    # 🔴 FIX สำคัญ: แปลง datetime → string ก่อนส่ง
    df2 = df.copy()
    for c in ["trade_date","settle_from","settle_to"]:
        if c in df2.columns:
            df2[c] = pd.to_datetime(df2[c], errors="coerce").dt.strftime("%Y-%m-%d")

    ws.clear()
    ws.update([df2.columns.tolist()] + df2.fillna("").values.tolist())

def filter_by_tf(df, tf):
    end = df["date"].max()
    
    if tf == "MTD":
        start = end.replace(day=1)
    elif tf == "YTD":
        start = end.replace(month=1, day=1)
    if tf == "3M":
        start = end - pd.DateOffset(months=3)
    elif tf == "6M":
        start = end - pd.DateOffset(months=6)
    elif tf == "1Y":
        start = end - pd.DateOffset(years=1)
    elif tf == "3Y":
        start = end - pd.DateOffset(years=3)
    elif tf == "5Y":
        start = end - pd.DateOffset(years=5)
    else:  # ALL
        start = df["date"].min()
    return df[df["date"] >= start]
    
# ================= NAV FUNCTION =================
def get_nav_price(fund, trade_date, nav_df):
    if pd.isna(fund) or pd.isna(trade_date):
        return None

    df = nav_df[
        (nav_df["fund"] == fund) &
        (nav_df["date"] <= trade_date)
    ].sort_values("date")

    if len(df) == 0:
        return None

    return df.iloc[-1]["nav"]
# ================= EXPLODE ENGINE =================
def explode_transactions(tx_df):
    rows = []

    for _, r in tx_df.iterrows():

        amount = pd.to_numeric(r["amount"], errors="coerce")
        p_from = pd.to_numeric(r["price_from"], errors="coerce")
        p_to   = pd.to_numeric(r["price_to"], errors="coerce")

        # BUY
        if r["action"] == "BUY" and pd.notna(p_to) and pd.notna(amount):
            units = amount / p_to
            rows.append({
                "date": r["settle_to"],
                "fund": r["fund_to"],
                "units": units
            })

        # SELL
        elif r["action"] == "SELL" and pd.notna(p_from) and pd.notna(amount):
            units = - amount / p_from
            rows.append({
                "date": r["settle_from"],
                "fund": r["fund_from"],
                "units": units
            })

        # SWITCH / SWAP
        elif r["action"] in ["SWITCH","SWAP"]:
            if pd.notna(p_from) and pd.notna(amount):
                units_out = - amount / p_from
                rows.append({
                    "date": r["settle_from"],
                    "fund": r["fund_from"],
                    "units": units_out
                })
            if pd.notna(p_to) and pd.notna(amount):
                units_in = amount / p_to
                rows.append({
                    "date": r["settle_to"],
                    "fund": r["fund_to"],
                    "units": units_in
                })

    return pd.DataFrame(rows)
    
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
st.title("📊 FPDP")
# ===== SHOW LAST UPDATE TIME =====
file_path = "fund_nav_5y.csv"
if os.path.exists(file_path):
    ts = os.path.getmtime(file_path)
    last_update = datetime.fromtimestamp(ts)
    st.caption(f"🕒 Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
else:
    st.caption("⚠️ ไม่พบไฟล์ fund_nav_5y.csv")
    
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
    # st.caption(f"ช่วงข้อมูล: {df_plot['date'].min().date()} → {df_plot['date'].max().date()}")
    df_plot = nav_df[nav_df["fund"].isin(dff["fund"])].copy()
    df_plot = filter_by_tf(df_plot, tf)

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
    fig_nav.update_layout(
        legend=dict(orientation="h", y=-0.25, x=0, xanchor="left"),
        margin=dict(t=50, b=80)
    )
    fig.update_xaxes(range=[start, end])
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
    fig_dd.update_layout(
        legend=dict(orientation="h", y=-0.25, x=0, xanchor="left"),
        margin=dict(t=50, b=80)
    )
    fig.update_xaxes(range=[start, end])
    st.plotly_chart(fig_dd, use_container_width=True, height=300)

    # ---------- Z-Score ----------
    win = min(60, len(fdf)//2)
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
    fig_z.update_layout(
        legend=dict(orientation="h", y=-0.25, x=0, xanchor="left"),
        margin=dict(t=50, b=80)
    )
    fig.update_xaxes(range=[start, end])
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
with tab_port:
    st.subheader(f"Portfolio Overview ({tf})")

    COLS = [
        "trade_date","action",
        "fund_from","fund_to",
        "settle_from","settle_to",
        "amount","price_from","price_to"
    ]

    # ================= LOAD FROM GOOGLE SHEET =================
    tx_df = load_data()
    
    if tx_df.empty:
        tx_df = pd.DataFrame(columns=COLS)

    if tx_df.empty:
        tx_df = pd.DataFrame(columns=COLS)

    tx_df["action"] = tx_df["action"].astype(str).str.strip().str.upper()
    tx_df = tx_df.dropna(subset=["action"])

    for c in ["trade_date","settle_from","settle_to"]:
        tx_df[c] = pd.to_datetime(tx_df[c], errors="coerce")

    # ================= COMBINE =================
    buy_df    = tx_df[tx_df["action"]=="BUY"].copy()
    sell_df   = tx_df[tx_df["action"]=="SELL"].copy()
    switch_df = tx_df[tx_df["action"].isin(["SWITCH","SWAP"])].copy()

    edited_df = pd.concat([buy_df, sell_df, switch_df], ignore_index=True)
    edited_df = edited_df[COLS]

    # ================= AUTO PRICE =================
    for i, row in edited_df.iterrows():
        d = row["trade_date"]
        if row["action"] == "BUY":
            edited_df.at[i, "price_to"] = get_nav_price(row["fund_to"], d, nav_df)
        elif row["action"] == "SELL":
            edited_df.at[i, "price_from"] = get_nav_price(row["fund_from"], d, nav_df)
        elif row["action"] == "SWITCH":
            edited_df.at[i, "price_from"] = get_nav_price(row["fund_from"], d, nav_df)
            edited_df.at[i, "price_to"]   = get_nav_price(row["fund_to"], d, nav_df)

    # ================= Portfolio Engine =================
    if len(edited_df) > 0:
        for c in ["amount","price_from","price_to"]:
            edited_df[c] = pd.to_numeric(edited_df[c], errors="coerce")

        pos_df = explode_transactions(edited_df)

        if len(pos_df) > 0:
            # -------- SUMMARY --------
            port = pos_df.groupby("fund")["units"].sum().reset_index()
            port = port[port["fund"].isin(funds)]

            latest_nav = nav_df.sort_values("date").groupby("fund").tail(1)[["fund","nav"]]
            port = port.merge(latest_nav, on="fund", how="left")

            port["current_value"] = port["units"] * port["nav"]

            cost = []
            for f in port["fund"]:
                buys = edited_df[edited_df["fund_to"] == f]
                sells = edited_df[edited_df["fund_from"] == f]
                cost.append(buys["amount"].sum() - sells["amount"].sum())

            port["amount"] = cost
            port["profit"] = port["current_value"] - port["amount"]
            port["profit_%"] = port["profit"] / port["amount"] * 100

            # -------- TOTAL --------
            total_row = pd.DataFrame([{
                "fund": "TOTAL",
                "units": port["units"].sum(),
                "nav": None,
                "current_value": port["current_value"].sum(),
                "amount": port["amount"].sum(),
                "profit": port["profit"].sum(),
                "profit_%": (
                    port["profit"].sum() / port["amount"].sum() * 100
                    if port["amount"].sum() != 0 else 0
                )
            }])
            port = pd.concat([port, total_row], ignore_index=True)

            # ================= SHOW TOP =================
            st.subheader("📊 Portfolio Summary")
            st.dataframe(port.round(2), use_container_width=True)

            # -------- VOL & RISK --------
            nav_df_sorted = nav_df.sort_values(["fund","date"])
            nav_df_sorted["ret"] = nav_df_sorted.groupby("fund")["nav"].pct_change()

            vol_df = nav_df_sorted.groupby("fund")["ret"].std().reset_index()
            vol_df.columns = ["fund","vol"]

            port = port.merge(vol_df, on="fund", how="left")
            port["risk_weight"] = port["current_value"] * port["vol"]

            port_no_total = port[port["fund"] != "TOTAL"]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🥧 Money Allocation")
                fig1 = px.pie(
                    port_no_total,
                    values="current_value",
                    names="fund"
                )
                fig1.update_traces(textinfo="percent+label")
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                st.subheader("⚠️ Risk Exposure")
                fig2 = px.pie(
                    port_no_total,
                    values="risk_weight",
                    names="fund"
                )
                fig2.update_traces(textinfo="percent+label")
                st.plotly_chart(fig2, use_container_width=True)

            st.divider()
    
    st.subheader("✏️ Transaction Manager")

    # ---------------- BUY ----------------
    st.markdown("### 🟢 BUY")
    if buy_df.empty:
        buy_df = pd.DataFrame(columns=["trade_date","fund_to","settle_to","amount"])

    buy_edit = st.data_editor(
        buy_df[["trade_date","fund_to","settle_to","amount"]],
        num_rows="dynamic",
        key="buy_editor",
        use_container_width=True,
        column_config={
            "trade_date": st.column_config.DateColumn("Trade Date"),
            "settle_to":  st.column_config.DateColumn("Settle Date"),
            "fund_to": st.column_config.SelectboxColumn("Fund", options=funds)
        }
    )
    buy_edit["action"] = "BUY"
    buy_edit["fund_from"] = None
    buy_edit["settle_from"] = None
    buy_edit["price_from"] = None
    buy_edit["price_to"] = None

    # ---------------- SELL ----------------
    st.markdown("### 🔴 SELL")
    if sell_df.empty:
        sell_df = pd.DataFrame(columns=["trade_date","fund_from","settle_from","amount"])

    sell_edit = st.data_editor(
        sell_df[["trade_date","fund_from","settle_from","amount"]],
        num_rows="dynamic",
        key="sell_editor",
        use_container_width=True,
        column_config={
            "trade_date":  st.column_config.DateColumn("Trade Date"),
            "settle_from": st.column_config.DateColumn("Settle Date"),
            "fund_from": st.column_config.SelectboxColumn("Fund", options=funds)
        }
    )
    sell_edit["action"] = "SELL"
    sell_edit["fund_to"] = None
    sell_edit["settle_to"] = None
    sell_edit["price_to"] = None
    sell_edit["price_from"] = None

    # ---------------- SWITCH ----------------
    st.markdown("### 🔄 SWITCH / SWAP")
    if switch_df.empty:
        switch_df = pd.DataFrame(columns=[
            "trade_date","fund_from","fund_to",
            "settle_from","settle_to","amount"
        ])

    switch_edit = st.data_editor(
        switch_df[["trade_date","fund_from","fund_to","settle_from","settle_to","amount"]],
        num_rows="dynamic",
        key="switch_editor",
        use_container_width=True,
        column_config={
            "trade_date":  st.column_config.DateColumn("Trade Date"),
            "settle_from": st.column_config.DateColumn("Settle From"),
            "settle_to":   st.column_config.DateColumn("Settle To"),
            "fund_from": st.column_config.SelectboxColumn("From", options=funds),
            "fund_to":   st.column_config.SelectboxColumn("To", options=funds)
        }
    )
    switch_edit["action"] = "SWITCH"
    switch_edit["price_from"] = None
    switch_edit["price_to"] = None

    # ================= COMBINE =================
    edited_df = pd.concat([buy_edit, sell_edit, switch_edit], ignore_index=True)
    edited_df = edited_df[COLS]

    for c in ["trade_date","settle_from","settle_to"]:
        edited_df[c] = pd.to_datetime(edited_df[c], errors="coerce")

    # ================= AUTO PRICE FROM NAV =================
    for i, row in edited_df.iterrows():
        d = row["trade_date"]

        if row["action"] == "BUY":
            edited_df.at[i, "price_to"] = get_nav_price(
                row["fund_to"], d, nav_df
            )

        elif row["action"] == "SELL":
            edited_df.at[i, "price_from"] = get_nav_price(
                row["fund_from"], d, nav_df
            )

        elif row["action"] == "SWITCH":
            edited_df.at[i, "price_from"] = get_nav_price(
                row["fund_from"], d, nav_df
            )
            edited_df.at[i, "price_to"] = get_nav_price(
                row["fund_to"], d, nav_df
            )

    # auto memory
    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Save"):
            save_data(edited_df)
            st.success("บันทึกลง Google Sheet แล้ว")
            st.rerun()   # ดึงข้อมูลใหม่จาก sheet ทันที
    
    with col2:
        if st.button("🗑️ Reset All"):
            empty = pd.DataFrame(columns=COLS)
            save_data(empty)
            st.warning("ล้างข้อมูลทั้งหมดแล้ว")
            st.rerun()

    st.divider()
# ================= DIVERSIFICATION =================
with tab_diver:
    st.subheader(f"🔗 Diversification Analysis ({tf})")

    # ----- เตรียม NAV ตาม timeframe แบบ robust -----
    nav_cut = nav_df.groupby("fund").apply(
        lambda x: filter_by_timeframe(
            x.set_index("date")["nav"], tf
        )
    ).reset_index(name="nav")

    nav_cut = nav_cut[nav_cut["fund"].isin(funds)]

    if nav_cut.empty or len(funds) < 2:
        st.info("กรุณาเลือกอย่างน้อย 2 กองทุน และต้องมีข้อมูล NAV เพียงพอ")
    else:
        # ----- Pivot เพื่อคำนวณ return -----
        df_ret = nav_cut.pivot(index="date", columns="fund", values="nav").ffill().pct_change().dropna()

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
            zmin=-1, zmax=1,
            title="Correlation Heatmap (Return)"
        )
        fig.update_layout(
            legend=dict(orientation="h", y=-0.25, x=0, xanchor="left"),
            margin=dict(t=50, b=80)
        )
        fig.add_annotation(
            x=0.5, y=1.08,
            xref="paper", yref="paper",
            text="ใกล้ +1 = ไปทิศเดียวกัน | ใกล้ 0 = ไม่เกี่ยวกัน | ใกล้ -1 = สวนทางกัน",
            showarrow=False,
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True, height=350)

        # ===== วิเคราะห์เป็นคู่ =====
        def interpret_corr(val):
            if val > 0.8: return "ซ้ำกันมาก"
            elif val > 0.5: return "สัมพันธ์สูง"
            elif val > 0.2: return "สัมพันธ์ต่ำ"
            elif val > -0.2: return "แทบไม่เกี่ยว"
            else: return "สวนทาง"

        pairs = list(itertools.combinations(funds, 2))
        results = []
        for f1, f2 in pairs:
            v = corr.loc[f1, f2]
            results.append({
                "คู่กอง": f"{f1} vs {f2}",
                "Correlation": round(v,2),
                "ความหมาย": interpret_corr(v)
            })
        result_df = pd.DataFrame(results)

        st.subheader("📊 วิเคราะห์ความสัมพันธ์ของกองที่เลือก")
        st.dataframe(result_df, use_container_width=True)

        # ----- Portfolio Volatility (Correlation-adjusted) -----
        latest_nav = nav_df.sort_values("date").groupby("fund").tail(1)
        latest_nav = latest_nav[latest_nav["fund"].isin(funds)].set_index("fund")
        weights = latest_nav["nav"] / latest_nav["nav"].sum()

        # Align returns
        ret_use = df_ret[weights.index].dropna()
        cov = ret_use.cov()
        w = weights.values
        port_var = np.dot(w.T, np.dot(cov, w))
        port_vol = np.sqrt(port_var * 252)

        # Diversification Ratio
        indiv_vol = ret_use.std() * np.sqrt(252)
        weighted_avg = np.sum(w * indiv_vol)
        div_ratio = weighted_avg / port_vol

        # ----- Metrics display -----
        st.metric("Portfolio Volatility (Corr-adjusted)", f"{port_vol*100:.2f}%")
        st.metric("Diversification Ratio", f"{div_ratio:.2f}")

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

        st.markdown("""
        ### 📌 วิธีอ่านค่า
        **Portfolio Volatility (Corr-adjusted)**  
        ความผันผวนของพอร์ตจริงทั้งระบบต่อปี  
        > 10% = ปีปกติขึ้นลงราว ±10%  
        > 15% = ปีปกติขึ้นลงราว ±15%  
        > 20%+ = พอร์ตเหวี่ยงสูง ต้องรับความเสี่ยงได้

        **Diversification Ratio**  
        ระดับการกระจายความเสี่ยงจริงของพอร์ต  
        > 1.0 = แทบไม่กระจาย (ซ้ำกัน)  
        > 1.2 = กระจายพอใช้  
        > 1.4 = กระจายดี  
        > 1.6+ = กระจายระดับกองทุน
        """)











