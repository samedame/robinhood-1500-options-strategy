"""Fetch SPY option_chain rows from DoltHub (post-no-preference/options) one trading day at a time.
Narrow (date, symbol) queries are fast; aggregate queries time out. Output: data/dolthub/SPY_<date>.csv
"""
import os, sys, time, json, urllib.parse, requests
import pandas as pd

OUT = "/home/claude/research/data/dolthub"
os.makedirs(OUT, exist_ok=True)
BASE = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"

def q(sql, tries=4):
    for i in range(tries):
        try:
            r = requests.get(BASE, params={"q": sql}, timeout=90)
            j = r.json()
            if j.get("query_execution_status") == "Success":
                return j["rows"]
            time.sleep(2 + 3 * i)
        except Exception:
            time.sleep(2 + 3 * i)
    return None

def fetch_day(sym, d):
    fn = f"{OUT}/{sym}_{d}.csv"
    if os.path.exists(fn):
        return
    rows, off = [], 0
    while True:
        got = q(f"SELECT * FROM option_chain WHERE date='{d}' AND act_symbol='{sym}' ORDER BY expiration, strike, call_put LIMIT 1000 OFFSET {off}")
        if got is None:
            print("FAIL", sym, d, flush=True); return
        rows += got
        if len(got) < 1000:
            break
        off += 1000
    pd.DataFrame(rows).to_csv(fn, index=False)
    print(sym, d, len(rows), flush=True)

if __name__ == "__main__":
    sym, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
    spy = pd.read_csv("/home/claude/data/mkt/yf_SPY.csv", index_col=0, parse_dates=True)
    days = [d.strftime("%Y-%m-%d") for d in spy.index if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    for d in days:
        fetch_day(sym, d)
