"""
Build a unified, compact SPY/QQQ option-chain store from the raw vendor files.

Sources (all public, anonymous downloads from Kaggle, provenance logged in the book):
  - optionsdx SPY EOD 2010-2023 (dudesurfin/spy-options-eod-volatility-surface-2010-2023, parquet)
  - optionsdx QQQ EOD 2020-2022 (kylegraupe/qqq-daily-option-chains-q1-2020-to-q4-2022, csv)
  - Alpha Vantage-format SPY EOD 2014-2025 (shankerabhigyan/s-and-p500-options-spy-implied-volatility-2019-24, json)

Output: one parquet per (vendor, symbol, year) with long-format rows:
  date, expiry, dte (calendar days), strike, cp ('C'/'P'), bid, ask, iv, delta, volume, oi, und (underlying close)
Filters (to keep the store small; nothing a strategy in this study needs is dropped):
  dte <= 75 calendar days, 0.75*S <= strike <= 1.25*S, bid/ask not both missing.
"""
import os, sys, io, re, zipfile, json
import numpy as np, pandas as pd
import pyarrow.parquet as pq

RAW = "/home/claude/data/kaggle"
OUT = "/home/claude/research/data/chains"
os.makedirs(OUT, exist_ok=True)

def _finish(df, tag):
    df = df[(df.dte >= 0) & (df.dte <= 75)]
    df = df[(df.strike >= 0.75 * df.und) & (df.strike <= 1.25 * df.und)]
    df = df[~(df.bid.isna() & df.ask.isna())]
    df = df.sort_values(["date", "expiry", "cp", "strike"]).reset_index(drop=True)
    for c in ["strike", "bid", "ask", "iv", "delta", "und"]:
        df[c] = df[c].astype("float32")
    df["dte"] = df["dte"].astype("int16")
    df["volume"] = df["volume"].astype("float32")
    df["oi"] = df["oi"].astype("float32")
    df.to_parquet(f"{OUT}/{tag}.parquet", index=False)
    print(tag, len(df), df.date.min(), df.date.max(), flush=True)

def optionsdx_frame(df):
    df.columns = [c.strip().strip("[]").strip() for c in df.columns]
    base = pd.DataFrame({
        "date": pd.to_datetime(df["QUOTE_DATE"].astype(str).str.strip()),
        "expiry": pd.to_datetime(df["EXPIRE_DATE"].astype(str).str.strip()),
        "strike": pd.to_numeric(df["STRIKE"], errors="coerce"),
        "und": pd.to_numeric(df["UNDERLYING_LAST"], errors="coerce"),
    })
    base["dte"] = (base.expiry - base.date).dt.days
    out = []
    for cp, p in [("C", "C_"), ("P", "P_")]:
        f = base.copy()
        f["cp"] = cp
        for src, dst in [("BID", "bid"), ("ASK", "ask"), ("IV", "iv"), ("DELTA", "delta"), ("VOLUME", "volume")]:
            f[dst] = pd.to_numeric(df[p + src], errors="coerce")
        f["oi"] = np.nan
        out.append(f)
    return pd.concat(out, ignore_index=True)

def build_spy_optionsdx():
    for y in range(2010, 2024):
        fn = f"{RAW}/spy_odx/spy_eod_{y}.parquet"
        cols = ["[QUOTE_DATE]", "[EXPIRE_DATE]", "[STRIKE]", "[UNDERLYING_LAST]", "[C_BID]", "[C_ASK]", "[C_IV]", "[C_DELTA]",
                "[C_VOLUME]", "[P_BID]", "[P_ASK]", "[P_IV]", "[P_DELTA]", "[P_VOLUME]"]
        t = pq.read_table(fn, columns=cols).to_pandas()
        _finish(optionsdx_frame(t), f"odx_SPY_{y}")

def build_qqq_optionsdx():
    with zipfile.ZipFile(f"{RAW}/qqq_optionsdx_2020_2022.zip") as z:
        with z.open("qqq_2020_2022.csv") as fh:
            df = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8-sig"), skipinitialspace=True, low_memory=False)
    f = optionsdx_frame(df)
    for y, g in f.groupby(f.date.dt.year):
        _finish(g.copy(), f"odx_QQQ_{y}")

def build_spy_av(und_close):
    obj_re = re.compile(rb"\{[^{}]*\}")
    keep = ["expiration", "strike", "type", "bid", "ask", "volume", "open_interest", "date", "implied_volatility", "delta"]
    with zipfile.ZipFile(f"{RAW}/spy_av_2014_2025.zip") as z:
        for name in sorted(z.namelist()):
            yy = int(re.findall(r"_(\d\d)\.json", name)[0]) + 2000
            rows = {k: [] for k in keep}
            buf = b""
            with z.open(name) as fh:
                while True:
                    chunk = fh.read(64 << 20)
                    if not chunk:
                        break
                    buf += chunk
                    last = 0
                    for m in obj_re.finditer(buf):
                        d = json.loads(m.group(0))
                        for k in keep:
                            rows[k].append(d.get(k))
                        last = m.end()
                    buf = buf[last:]
            df = pd.DataFrame(rows)
            out = pd.DataFrame({
                "date": pd.to_datetime(df["date"]),
                "expiry": pd.to_datetime(df["expiration"]),
                "strike": pd.to_numeric(df["strike"], errors="coerce"),
                "cp": np.where(df["type"].str.lower().str.startswith("c"), "C", "P"),
                "bid": pd.to_numeric(df["bid"], errors="coerce"),
                "ask": pd.to_numeric(df["ask"], errors="coerce"),
                "iv": pd.to_numeric(df["implied_volatility"], errors="coerce"),
                "delta": pd.to_numeric(df["delta"], errors="coerce"),
                "volume": pd.to_numeric(df["volume"], errors="coerce"),
                "oi": pd.to_numeric(df["open_interest"], errors="coerce"),
            })
            out["dte"] = (out.expiry - out.date).dt.days
            out["und"] = out["date"].map(und_close)
            out = out[out.und.notna()]
            _finish(out, f"av_SPY_{yy}")
            del df, out, rows

if __name__ == "__main__":
    what = sys.argv[1]
    if what == "spy_odx":
        build_spy_optionsdx()
    elif what == "qqq_odx":
        build_qqq_optionsdx()
    elif what == "spy_av":
        spy = pd.read_csv("/home/claude/data/mkt/yf_SPY.csv", index_col=0, parse_dates=True)
        build_spy_av(spy["Close"].to_dict())
