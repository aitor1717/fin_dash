# Portfolio Dashboard

**Live demo: [aitor1717.github.io/fin_dash](https://aitor1717.github.io/fin_dash/)**

![Dashboard preview](preview.png)

A simple but powerful finance dashboard for a trade-log CSV feed: historic and current-day
performance vs. benchmarks, with standard portfolio/trade metrics plus an
evaluation-style compliance panel (profit target, max loss, position
concentration, price/volume floors).

See [`pipeline/generate_sample_feed.py`](pipeline/generate_sample_feed.py)
for how the sample data is built. Real, liquid tickers priced at their
actual historical closes. The trading activity itself (timing, side, win
rate) is generated, not derived from any real account. Real trade history
has no place in this repo and is gitignored; see "Running your own book"
below.

The data shown is a synthetic sample made for this demo. The page is static; the data
is only as fresh as the last local run that was committed.

## How it works

```
feed/*.csv  --(pipeline/build.py)-->  docs/data*/dashboard.json  --(docs/index.html)-->  browser
```

The feed has no position-size or account-size field by design, so `build.py`
applies a **buying-power-capped allocator**: each trade risks a configurable
% of a configurable account size, and a new trade is skipped if it would push
total open exposure over 100% of the account. Both are set in the config
YAML.

Benchmark history and per-ticker live prices / average volume come from
`yfinance` (no API key required, no guaranteed real-time accuracy).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas yfinance pyyaml numpy
```

## Run

```bash
python pipeline/build.py --config config.sample.yaml
```

This writes `docs/data-sample/dashboard.json` (the demo dataset `docs/app.js`
loads by default). Open `docs/index.html` directly in a browser, or serve
the `docs/` folder locally:

```bash
python -m http.server --directory docs 8000
```

To regenerate the synthetic feed itself (new seed, different trade count,
etc.):

```bash
python pipeline/generate_sample_feed.py --out-feed feed/sample_feed.csv --seed 7
python pipeline/build.py --config config.sample.yaml
```

## Feed schema

`open_datetime,close_datetime,ticker,open_bid,open_ask,close_bid,close_ask,change,position,status,notes`

- Entry cost is `open_ask`, exit proceeds are `close_bid` (long-only
  reconciliation, regardless of the `position` tag).
- `status` is the authoritative open/closed signal: exactly `open` or
  `closed` (case-insensitive). `close_bid`/`close_ask` show `pending` on
  an open row, for readability only. A real broker-exported feed needs
  its own `status` column cleaned up to these two values before parsing
  it here.
