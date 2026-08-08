# Portfolio Dashboard

A self-refreshing dashboard for a trade-log CSV feed: historic and current-day
performance vs. benchmarks, with standard portfolio/trade metrics plus an
evaluation-style compliance panel (profit target, max loss, position
concentration, price/volume floors).

The demo data is **fully synthetic** (see
[`pipeline/generate_sample_feed.py`](pipeline/generate_sample_feed.py)) —
real, liquid tickers priced at their actual historical closes, but the
trading activity itself (timing, side, win rate) is generated, not derived
from any real account. Real trade history has no place in this repo and is
gitignored; see "Running your own book" below.

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

## Running your own book

Copy `config.sample.yaml` (e.g. to `config.personal.yaml`), point `feed_path`
at your own real CSV, and adjust `account_size` / `position_weight_pct`. Give
it its own `output_dir` (e.g. `docs/data-personal`) so it doesn't clobber the
demo data, then point `DATA_PATH` in `docs/app.js` at that same path.

If your book is actively trading (new trades logged over time), leave
`freeze_asof` unset — the dashboard will show real current benchmark data
and live mark-to-market. If it's a frozen/static snapshot instead, set
`freeze_asof: true` so the "today" view stays pinned to the feed's own last
activity rather than drifting with real time (see the config schema note in
`CLAUDE.md`, kept locally and not published).

`.gitignore` already excludes common real-data filenames
(`feed/example_feed*.csv`, `docs/data/`, etc.) — if you name your own feed
or output directory differently, add it there too before committing.

## Feed schema

`open_datetime,close_datetime,ticker,open_bid,open_ask,close_bid,close_ask,change,position,status,notes`

- Entry cost is `open_ask`, exit proceeds are `close_bid` (long-only
  reconciliation, regardless of the `position` tag).
- `close_bid == "pending"` means the trade is still open — this is what the
  pipeline uses to separate open from closed, not the `status` column. In
  the synthetic sample feed, `status` is just `open`/`closed` derived from
  that same signal; a real broker-exported feed's status values may not map
  as cleanly and are only ever carried through as a display tag.

## Deploying to GitHub Pages

Commit `docs/` (including the generated `docs/data-sample/dashboard.json`),
then in the repo's Settings → Pages, set the source branch with `/docs` as
the folder. The page is static — its data is only as fresh as the last local
`build.py` run that was committed. Do not commit real trade data or its
generated output; keep those local and gitignored.
