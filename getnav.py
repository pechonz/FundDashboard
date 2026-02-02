import requests
import pandas as pd
import time
from datetime import datetime, timedelta

# ================= CONFIG =================
target = ['K-US500X','K-GTECH','K-USXNDQ','K-WPULTIMATE','K-INDIA','K-GINCOME','K-SF'] 

headers_fundlist = {
    "Ocp-Apim-Subscription-Key": "a1abb45800d747668d53b3ea50a06734"
}

headers_nav = {
    "Ocp-Apim-Subscription-Key": "3901836c6fcf45e680ef29584f8cf4bd"
}

urlFundList = "https://api.sec.or.th/FundFactsheet/fund/amc/C0000000021"
urlTemplate = "https://api.sec.or.th/FundDailyInfo/{proj_id}/dailynav/{date}"

# ================= SAFE GET =================
def safe_get_json(url, headers):
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200 or r.text.strip() == "":
            return None
        return r.json()
    except:
        return None

# ================= LOAD EXISTING NAV =================
try:
    old_df = pd.read_csv("fund_nav_5y.csv")
    old_df["date"] = pd.to_datetime(old_df["date"])
except:
    old_df = pd.DataFrame(columns=["date","nav","fund","proj_id"])

# ================= GET FUND LIST =================
fundslist = safe_get_json(urlFundList, headers_fundlist)
target_funds = [f for f in fundslist if f["proj_abbr_name"] in target]

# ================= GET LATEST NAV =================
def get_latest_nav(proj_id, max_back=15):
    today = datetime.today()
    for i in range(max_back):
        d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        url = urlTemplate.format(proj_id=proj_id, date=d)
        data = safe_get_json(url, headers_nav)
        if data:
            row = data[0]
            return {
                "date": pd.to_datetime(row["nav_date"]),
                "nav": float(row["last_val"])
            }
    return None

latest_rows = []
for f in target_funds:
    res = get_latest_nav(f["proj_id"])
    if res:
        latest_rows.append({
            "fund": f["proj_abbr_name"],
            "proj_id": f["proj_id"],
            "date": res["date"],
            "nav": res["nav"]
        })
        print("LATEST OK:", f["proj_abbr_name"], res["date"], res["nav"])
    time.sleep(0.3)

df = pd.DataFrame(latest_rows)

# ================= LAST DATE =================
def get_last_date(old_df, fund):
    sub = old_df[old_df["fund"] == fund]
    if len(sub) == 0:
        return None
    return sub["date"].max()

# ================= INCREMENTAL NAV =================
def get_nav_series_incremental(old_df, proj_id, fund, end_date):
    last_date = get_last_date(old_df, fund)

    if last_date is None:
        start_date = end_date - timedelta(days=365*10)
    else:
        start_date = last_date + timedelta(days=1)

    rows = []
    d = start_date
    while d <= end_date:
        url = urlTemplate.format(proj_id=proj_id, date=d.strftime("%Y-%m-%d"))
        data = safe_get_json(url, headers_nav)
        if data:
            row = data[0]
            rows.append({
                "date": pd.to_datetime(row["nav_date"]),
                "nav": float(row["last_val"]),
                "fund": fund,
                "proj_id": proj_id
            })
        d += timedelta(days=1)
        time.sleep(0.05)

    return pd.DataFrame(rows)

# ================= UPDATE NAV FILE =================
all_new = []
for _, row in df.iterrows():
    ts = get_nav_series_incremental(old_df, row["proj_id"], row["fund"], row["date"])
    if len(ts) > 0:
        all_new.append(ts)

if all_new:
    new_df = pd.concat(all_new)
    final_nav = pd.concat([old_df, new_df]).drop_duplicates(
        subset=["date","fund"], keep="last"
    )
    final_nav = final_nav.sort_values(["fund","date"])
    final_nav.to_csv("fund_nav_5y.csv", index=False)
    print("NAV UPDATED:", len(new_df), "rows")
else:
    final_nav = old_df.copy()
    print("NO NEW NAV")

# ================= CAGR =================
def calc_cagr(nav_start, nav_end, years):
    return (nav_end / nav_start) ** (1/years) - 1

def get_nav_near_date(proj_id, target_date, max_back=30):
    for i in range(max_back):
        d = (target_date - timedelta(days=i)).strftime("%Y-%m-%d")
        url = urlTemplate.format(proj_id=proj_id, date=d)
        data = safe_get_json(url, headers_nav)
        if data:
            row = data[0]
            return {
                "date": pd.to_datetime(row["nav_date"]),
                "nav": float(row["last_val"])
            }
    return None

# ================= MAX DRAWDOWN =================
def calc_max_drawdown_with_dates(df_ts):
    df_ts = df_ts.sort_values("date").reset_index(drop=True)
    nav = df_ts["nav"]
    dates = df_ts["date"]

    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax

    idx_trough = drawdown.idxmin()
    idx_peak = nav[:idx_trough+1].idxmax()

    return {
        "mdd_pct": round(drawdown.min() * 100, 2),
        "peak_date": dates[idx_peak].strftime("%Y-%m-%d"),
        "trough_date": dates[idx_trough].strftime("%Y-%m-%d")
    }

def get_nav_series_local(final_nav, fund, start_date, end_date):
    sub = final_nav[final_nav["fund"] == fund]
    return sub[(sub["date"] >= start_date) & (sub["date"] <= end_date)]

def get_max_drawdown_by_year(final_nav, fund, end_date, years):
    start_date = end_date - timedelta(days=365*years)
    ts = get_nav_series_local(final_nav, fund, start_date, end_date)
    if len(ts) < 30:
        return None
    return calc_max_drawdown_with_dates(ts)

# ================= FINAL METRICS =================
periods = [1, 3, 5]
results = []

for _, row in df.iterrows():
    fund = row["fund"]
    proj_id = row["proj_id"]
    nav_end = row["nav"]
    date_end = row["date"]

    out = {
        "fund": fund,
        "nav_latest": nav_end,
        "date_latest": date_end.strftime("%Y-%m-%d")
    }

    for y in periods:
        target_date = date_end - timedelta(days=365*y)
        start_data = get_nav_near_date(proj_id, target_date)

        if start_data:
            nav_start = start_data["nav"]
            real_years = (date_end - start_data["date"]).days / 365
            cagr = calc_cagr(nav_start, nav_end, real_years)
            out[f"{y}Y_CAGR_%"] = round(cagr * 100, 2)
        else:
            out[f"{y}Y_CAGR_%"] = None

        time.sleep(0.2)

    for y in periods:
        res = get_max_drawdown_by_year(final_nav, fund, date_end, y)
        if res:
            out[f"MaxDD_{y}Y_%"] = res["mdd_pct"]
            out[f"MaxDD_{y}Y_Peak"] = res["peak_date"]
            out[f"MaxDD_{y}Y_Trough"] = res["trough_date"]
        else:
            out[f"MaxDD_{y}Y_%"] = None

    results.append(out)

df_final = pd.DataFrame(results)
print("\n==== FINAL RESULT ====")
print(df_final)
