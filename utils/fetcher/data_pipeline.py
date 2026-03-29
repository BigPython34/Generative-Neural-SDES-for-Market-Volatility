"""
Unified data regeneration pipeline.

Goals:
- Rebuild datasets under data/ from Yahoo + CBOE
- Keep canonical organized files under data/market and data/rates
"""

from __future__ import annotations

import io
import json
import requests
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from quant.loader.timeseries_merge import add_datetime_column, coerce_time_column, merge_timeseries_frames

warnings.filterwarnings("ignore", category=FutureWarning)

today=datetime.now().strftime("%Y-%m-%d")
NYFED_SOFR_URL = (
    "https://markets.newyorkfed.org/read?startDt=2016-03-01&endDt=" + today +
    "&eventCodes=520&productCode=50&sort=postDt:-1,eventCode:1&format=xlsx"
)


YAHOO_REFRESH_CATALOG = {
    "equity_indices": {
        "spx": {"ticker": "^GSPC", "intervals": {"1m", "5m", "15m", "30m", "1h", "1d"}},
        "cac40": {"ticker": "^FCHI", "intervals": {"1m", "5m", "15m", "30m", "1h", "1d"}},
    },
    "equity_etfs": {
        "spy": {"ticker": "SPY", "intervals": {"1m", "5m", "15m", "30m", "1h", "1d"}},
    },
    "volatility": {
        "vix": {"ticker": "^VIX", "intervals": {"1m", "5m", "15m", "30m", "1h", "1d"}},
        "vvix": {"ticker": "^VVIX", "intervals": {"1m", "5m", "15m", "30m", "1h", "1d"}},
        "vix1d": {"fixed_reason": "No Yahoo Finance equivalent for CBOE fixed-tenor VIX1D"},
        "vix9d": {"fixed_reason": "No Yahoo Finance equivalent for CBOE fixed-tenor VIX9D"},
        "vix3m": {"fixed_reason": "No Yahoo Finance equivalent for CBOE fixed-tenor VIX3M"},
        "vix6m": {"fixed_reason": "No Yahoo Finance equivalent for CBOE fixed-tenor VIX6M"},
        "vix1y": {"fixed_reason": "No Yahoo Finance equivalent for CBOE fixed-tenor VIX1Y"},
    },
    "rates": {
        "us02y": {"fixed_reason": "No reliable Yahoo Finance series found for the 2Y Treasury yield in this environment"},
        "us05y": {"ticker": "^FVX", "intervals": {"1m", "5m", "15m", "30m", "1h", "1d"}},
        "us10y": {"ticker": "^TNX", "intervals": {"1m", "5m", "15m", "30m", "1h", "1d"}},
        "us30y": {"ticker": "^TYX", "intervals": {"1m", "5m", "15m", "30m", "1h", "1d"}},
    },
    "sp_futures": {
        "es1": {"ticker": "ES=F", "intervals": {"1m", "5m", "15m", "30m", "1h", "1d"}},
        "es2": {"fixed_reason": "Yahoo Finance exposes only the continuous front-month ES futures"},
    },
    "vix_futures": {
        "vx1": {"fixed_reason": "Yahoo Finance does not expose a reliable continuous VX front-month history here"},
        "vx2": {"fixed_reason": "Yahoo Finance exposes only the continuous front-month VX futures"},
    },
}

INTERVAL_SECONDS = {
    "5m": 5 * 60,
    "10m": 10 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 60 * 60,
    "1d": 24 * 60 * 60,
}

INTERVAL_PERIODS = {
    "1m": "7d",
    "2m": "60d",
    "5m": "60d",
    "10m": "60d",
    "15m": "60d",
    "30m": "60d",
    "45m": "60d",
    "60m": "730d",
    "90m": "60d",
    "1h": "730d",
    "1d": "max",
}


def _infer_interval_from_filename(name: str) -> Optional[str]:
    stem = Path(name).stem.lower()
    if stem.endswith("_daily"):
        return "1d"
    for tag in ["1m", "2m", "5m", "10m", "15m", "30m", "45m", "60m", "90m", "1h"]:
        if stem.endswith(f"_{tag}"):
            return tag
    return None


@dataclass
class PipelineResult:
    generated: Dict[str, str]
    skipped: Dict[str, str]
    metadata: Dict[str, str]


class DataRegenerator:
    """Regenerate market datasets into canonical data/ locations."""

    def __init__(self, root: str | Path = ".", force_download: bool = False):
        self.root = Path(root).resolve()
        self.data_dir = self.root / "data"
        self.force_download = bool(force_download)

        self.market_dir = self.data_dir / "market"
        # Market must mirror trading_view categories
        self.vix_dir = self.market_dir / "volatility"
        self.vvix_dir = self.market_dir / "volatility"
        self.spx_dir = self.market_dir / "equity_indices"
        self.rates_dir = self.market_dir / "rates"
        self.vix_fut_dir = self.market_dir / "cboe_vix_futures_full"

        for d in [
            self.data_dir,
            self.market_dir,
            self.vix_dir,
            self.vvix_dir,
            self.spx_dir,
            self.rates_dir,
            self.vix_fut_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # TradingView organized directory (by category)
        self.tv_dir = self.data_dir / "trading_view"
        self.tv_input_dir = self.data_dir / "tradingview"
        self.tv_input_dir_alt = self.tv_dir

        # Category subdirectories within trading_view/
        self.tv_volatility = self.tv_dir / "volatility"
        self.tv_equity_indices = self.tv_dir / "equity_indices"
        self.tv_equity_etfs = self.tv_dir / "equity_etfs"
        self.tv_vol_etfs = self.tv_dir / "vol_etfs"
        self.tv_vix_futures = self.tv_dir / "vix_futures"
        self.tv_sp_futures = self.tv_dir / "sp_futures"
        self.tv_rates = self.tv_dir / "rates"
        self.tv_fx = self.tv_dir / "fx"
        self.tv_sentiment = self.tv_dir / "sentiment"

        self.generated: Dict[str, str] = {}
        self.skipped: Dict[str, str] = {}
        self.metadata: Dict[str, str] = {}

        self.cleanup_redundant = True

        self.market_targets = {
            "vix_5m": self.vix_dir / "vix_5m.csv",
            "vix_10m": self.vix_dir / "vix_10m.csv",
            "vix_15m": self.vix_dir / "vix_15m.csv",
            "vix_30m": self.vix_dir / "vix_30m.csv",
            "spx_5m": self.spx_dir / "spx_5m.csv",
            "spx_30m": self.spx_dir / "spx_30m.csv",
            "spx_daily": self.spx_dir / "spx_daily.csv",
            "vvix_daily": self.vvix_dir / "vvix_daily.csv",
        }

        # TradingView source paths (organized structure)
        self.tv_sources = {
            "vix_5m": self.tv_volatility / "vix_5m.csv",
            "vix_15m": self.tv_volatility / "vix_15m.csv",
            "vix_30m": self.tv_volatility / "vix_30m.csv",
            "vix_1h": self.tv_volatility / "vix_1h.csv",
            "vix_daily": self.tv_volatility / "vix_daily.csv",
            "vvix_5m": self.tv_volatility / "vvix_5m.csv",
            "vvix_15m": self.tv_volatility / "vvix_15m.csv",
            "vvix_30m": self.tv_volatility / "vvix_30m.csv",
            "vvix_1h": self.tv_volatility / "vvix_1h.csv",
            "vvix_daily": self.tv_volatility / "vvix_daily.csv",
            "spx_5m": self.tv_equity_indices / "spx_5m.csv",
            "spx_15m": self.tv_equity_indices / "spx_15m.csv",
            "spx_30m": self.tv_equity_indices / "spx_30m.csv",
            "spx_1h": self.tv_equity_indices / "spx_1h.csv",
            "spx_daily": self.tv_equity_indices / "spx_daily.csv",
            "spx_5s": self.tv_equity_indices / "spx_5s.csv",
            "spy_5m": self.tv_equity_etfs / "spy_5m.csv",
            "spy_15m": self.tv_equity_etfs / "spy_15m.csv",
            "spy_daily": self.tv_equity_etfs / "spy_daily.csv",
        }

    # ------------------------------------------------------------------
    # Generic I/O
    # ------------------------------------------------------------------
    def _register(self, label: str, path: Path):
        self.generated[label] = str(path.relative_to(self.root))

    def _register_skip(self, label: str, reason: str):
        self.skipped[label] = reason

    def _to_canonical_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Already in canonical format — pass through
        needed = {"time", "open", "high", "low", "close"}
        if needed.issubset(out.columns):
            out = out[["time", "open", "high", "low", "close"]].copy()
            out["time"] = pd.to_numeric(out["time"], errors="coerce")
            for c in ["open", "high", "low", "close"]:
                out[c] = pd.to_numeric(out[c], errors="coerce")
            out = out.dropna()
            return out.sort_values("time").drop_duplicates("time")

        # Normalize date column
        out = out.rename(columns={"Date": "date", "Datetime": "date", "DATE": "date"})

        if "date" not in out.columns:
            idx_name = out.index.name
            out = out.reset_index()
            if idx_name and idx_name in out.columns:
                out = out.rename(columns={idx_name: "date"})
            elif "index" in out.columns:
                out = out.rename(columns={"index": "date"})
            elif "Datetime" in out.columns:
                out = out.rename(columns={"Datetime": "date"})
            elif "Date" in out.columns:
                out = out.rename(columns={"Date": "date"})

        if "date" not in out.columns:
            out = out.rename(columns={out.columns[0]: "date"})

        dt = pd.to_datetime(out["date"], errors="coerce", utc=True)
        out["time"] = (dt.view("int64") // 10**9).astype("int64")

        # Normalize OHLC: handle both capitalized and lowercase
        col_map = {}
        for canonical, alts in [("open", ["Open", "OPEN"]),
                                ("high", ["High", "HIGH"]),
                                ("low", ["Low", "LOW"]),
                                ("close", ["Close", "CLOSE"])]:
            if canonical in out.columns:
                continue
            for alt in alts:
                if alt in out.columns:
                    col_map[alt] = canonical
                    break
            else:
                out[canonical] = np.nan
        if col_map:
            out = out.rename(columns=col_map)

        out = out[["time", "open", "high", "low", "close"]].dropna()
        return out.sort_values("time").drop_duplicates("time")

    def _save_csv(self, df: pd.DataFrame, path: Path, label: str, force_overwrite: bool = False):
        if path.exists() and not self.force_download and not force_overwrite:
            self._register_skip(label, f"exists: {path.relative_to(self.root)}")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        self._register(label, path)

    def _read_csv_safe(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    def _latest_timestamp(self, path: Path):
        df = self._read_csv_safe(path)
        if len(df) == 0:
            return None
        try:
            df = coerce_time_column(df)
            if len(df) == 0:
                return None
            return pd.to_datetime(int(df["time"].max()), unit="s", utc=True)
        except Exception:
            return None

    def _interval_seconds(self, interval: Optional[str]) -> Optional[int]:
        if interval is None:
            return None
        return INTERVAL_SECONDS.get(interval)

    def _latest_is_stale(self, latest_ts, interval: Optional[str]) -> bool:
        if latest_ts is None:
            return True
        now = pd.Timestamp.now(tz="UTC")
        if interval == "1d":
            return latest_ts.normalize() < now.normalize()
        sec = self._interval_seconds(interval)
        if sec is None:
            return latest_ts.normalize() < now.normalize()
        return latest_ts < (now - pd.Timedelta(seconds=sec * 2))

    def _catalog_for(self, category: str, prefix: str) -> dict:
        return YAHOO_REFRESH_CATALOG.get(category, {}).get(prefix, {})

    def _yahoo_refreshable(self, category: str, prefix: str, interval: Optional[str]) -> tuple[bool, str | None, str | None]:
        spec = self._catalog_for(category, prefix)
        if not spec:
            return False, None, f"No Yahoo Finance equivalent for {category}/{prefix}"
        if "fixed_reason" in spec:
            return False, None, spec["fixed_reason"]
        if interval is None:
            return False, None, f"Unsupported filename interval for {category}/{prefix}"
        if interval not in spec["intervals"]:
            return False, spec.get("ticker"), f"Yahoo ticker exists but interval {interval} is not supported for {category}/{prefix}"
        return True, spec.get("ticker"), None

    def _refresh_yahoo_series(self, path: Path, ticker: str, interval: str, label: str):
        """Refresh a single CSV by fetching only data newer than the on-disk tail."""
        existing = self._read_csv_safe(path)
        latest_ts = self._latest_timestamp(path)
        if latest_ts is not None:
            start = (latest_ts + pd.Timedelta(seconds=1)).to_pydatetime()
            df_new = yf.Ticker(ticker).history(
                start=start,
                end=datetime.now(),
                interval=interval,
                auto_adjust=False,
            )
        else:
            period = INTERVAL_PERIODS.get(interval, "max")
            df_new = yf.Ticker(ticker).history(
                period=period,
                interval=interval,
                auto_adjust=False,
            )

        if df_new is None or len(df_new) == 0:
            return {
                "label": label,
                "path": str(path.relative_to(self.root)).replace("\\", "/"),
                "ticker": ticker,
                "interval": interval,
                "status": "empty",
                "updated": False,
                "rows_existing": len(existing),
                "rows_new": 0,
                "rows_final": len(existing),
                "latest": str(latest_ts) if latest_ts is not None else None,
            }

        df_new = self._to_canonical_ohlc(df_new)
        if len(df_new) == 0:
            return {
                "label": label,
                "path": str(path.relative_to(self.root)).replace("\\", "/"),
                "ticker": ticker,
                "interval": interval,
                "status": "empty-after-normalize",
                "updated": False,
                "rows_existing": len(existing),
                "rows_new": 0,
                "rows_final": len(existing),
                "latest": str(latest_ts) if latest_ts is not None else None,
            }

        merged = merge_timeseries_frames(existing, df_new, prefer="incoming")
        merged = add_datetime_column(merged)
        merged = merged.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)

        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(path, index=False)

        new_latest = self._latest_timestamp(path)
        return {
            "label": label,
            "path": str(path.relative_to(self.root)).replace("\\", "/"),
            "ticker": ticker,
            "interval": interval,
            "status": "updated",
            "updated": True,
            "rows_existing": len(existing),
            "rows_new": len(df_new),
            "rows_final": len(merged),
            "latest_before": str(latest_ts) if latest_ts is not None else None,
            "latest_after": str(new_latest) if new_latest is not None else None,
        }

    def build_refresh_inventory(self) -> dict:
        """Inspect every CSV in data/market and classify it for refresh vs fixed."""
        inventory = []
        for path in sorted(self.market_dir.rglob("*.csv")):
            if path.name == "regeneration_report.json":
                continue
            rel = path.relative_to(self.market_dir)
            category = rel.parts[0] if len(rel.parts) > 1 else rel.parts[0]
            prefix = path.stem.lower().split("_", 1)[0]
            interval = _infer_interval_from_filename(path.name)
            refreshable, ticker, fixed_reason = self._yahoo_refreshable(category, prefix, interval)
            latest_ts = self._latest_timestamp(path)
            inventory.append({
                "path": str(rel).replace("\\", "/"),
                "category": category,
                "prefix": prefix,
                "interval": interval,
                "latest": str(latest_ts) if latest_ts is not None else None,
                "stale": self._latest_is_stale(latest_ts, interval) if refreshable else None,
                "refreshable": refreshable,
                "ticker": ticker,
                "reason": fixed_reason,
            })
        return {
            "refreshable": [x for x in inventory if x["refreshable"]],
            "fixed": [x for x in inventory if not x["refreshable"]],
        }

    def print_refresh_inventory(self):
        plan = self.build_refresh_inventory()
        print("\n[DATA PIPELINE] REFRESH PLAN")
        print("  Refreshable (Yahoo Finance):")
        for x in plan["refreshable"]:
            stale_flag = "STALE" if x["stale"] else "fresh"
            print(f"    - {x['path']}  [{x['ticker']} / {x['interval']}]  latest={x['latest']}  {stale_flag}")
        print("  Fixed (no Yahoo Finance equivalent):")
        for x in plan["fixed"]:
            reason = x["reason"]
            if x["category"] == "rates" and x["prefix"] == "sofr":
                reason = "Managed by NY Fed SOFR fetcher (not Yahoo)"
            print(f"    - {x['path']}  ({reason})")

    def refresh_market_smart(self, *, verbose: bool = True) -> dict:
        """Refresh only refreshable market files, fetching only the missing tail."""
        plan = self.build_refresh_inventory()
        results = []
        for x in plan["refreshable"]:
            if not x["stale"] and not self.force_download:
                results.append({**x, "status": "up-to-date", "updated": False})
                continue
            path = self.market_dir / x["path"]
            res = self._refresh_yahoo_series(path, x["ticker"], x["interval"], x["path"])
            res.update({"refreshable": True, "stale": x["stale"]})
            results.append(res)
            if verbose:
                if res["updated"]:
                    print(f"  OK  {x['path']}  rows={res['rows_final']}  (added {res['rows_new']})")
                else:
                    print(f"  SKIP {x['path']}  {res['status']}")

        return {
            "refreshable": plan["refreshable"],
            "fixed": plan["fixed"],
            "results": results,
        }

    # ------------------------------------------------------------------
    # Yahoo download helpers
    # ------------------------------------------------------------------
    def _download_yahoo(self,
                        ticker: str,
                        interval: str,
                        period: Optional[str] = None,
                        start: Optional[str] = None,
                        end: Optional[str] = None) -> pd.DataFrame:
        kwargs = {
            "tickers": ticker,
            "interval": interval,
            "auto_adjust": False,
            "progress": False,
        }
        if period:
            kwargs["period"] = period
        else:
            kwargs["start"] = start
            kwargs["end"] = end

        df = yf.download(**kwargs)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df.dropna(how="all")

    def _save_market_series(self,
                            df: pd.DataFrame,
                            organized_path: Path,
                            label_prefix: str):
        if organized_path.exists() and not self.force_download:
            self._register_skip(label_prefix, f"exists: {organized_path.relative_to(self.root)}")
            return

        if df is None or len(df) == 0:
            self._register_skip(label_prefix, "empty dataframe")
            return

        canonical = self._to_canonical_ohlc(df)
        if len(canonical) == 0:
            self._register_skip(label_prefix, "no valid OHLC rows")
            return

        self._save_csv(canonical, organized_path, f"{label_prefix}:canonical")

    def _has_market_series(self, organized_path: Path) -> bool:
        return organized_path.exists() and (not self.force_download)

    def _series_time_info(self, path: Path, mode: str = "epoch") -> Optional[dict]:
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
            if len(df) == 0:
                return None

            if mode == "epoch":
                if "time" not in df.columns:
                    return None
                dt = pd.to_datetime(df["time"], unit="s", errors="coerce", utc=True).dropna().sort_values()
            else:
                cands = ["Date", "Datetime", "date", "datetime", "Trade Date"]
                c = next((x for x in cands if x in df.columns), None)
                if c is None:
                    c = df.columns[0]
                dt = pd.to_datetime(df[c], errors="coerce", utc=True).dropna().sort_values()

            if len(dt) == 0:
                return None

            med_min = None
            if len(dt) > 1:
                med_min = float(dt.diff().dropna().dt.total_seconds().median() / 60.0)

            return {
                "path": str(path.relative_to(self.root)),
                "rows": int(len(df)),
                "start": str(dt.min()),
                "end": str(dt.max()),
                "median_dt_min": med_min,
            }
        except Exception:
            return None

    def _normalize_intraday_file(self, path: Path, label: str):
        """Ensure canonical intraday files use [time, open, high, low, close] schema."""
        if not path.exists():
            return
        try:
            df = pd.read_csv(path)
            needed = {"time", "open", "high", "low", "close"}
            if needed.issubset(df.columns):
                return
            normalized = self._to_canonical_ohlc(df)
            if len(normalized) == 0:
                self._register_skip(f"normalize:{label}", "failed: no valid rows")
                return
            normalized.to_csv(path, index=False)
            self._register(f"normalize:{label}", path)
        except Exception as e:
            self._register_skip(f"normalize:{label}", str(e))

    def normalize_existing_canonical_files(self):
        intraday_targets = [
            ("vix_5m", self.market_targets["vix_5m"]),
            ("vix_10m", self.market_targets["vix_10m"]),
            ("vix_15m", self.market_targets["vix_15m"]),
            ("vix_30m", self.market_targets["vix_30m"]),
            ("spx_5m", self.market_targets["spx_5m"]),
            ("spx_30m", self.market_targets["spx_30m"]),
        ]
        for label, p in intraday_targets:
            self._normalize_intraday_file(p, label)

    def _build_temporal_coherence_report(self) -> dict:
        report = {
            "intraday": {},
            "daily": {},
            "overlap": {},
        }

        intraday_map = {
            "vix_5m": self.market_targets["vix_5m"],
            "vix_10m": self.market_targets["vix_10m"],
            "vix_15m": self.market_targets["vix_15m"],
            "vix_30m": self.market_targets["vix_30m"],
            "spx_5m": self.market_targets["spx_5m"],
            "spx_30m": self.market_targets["spx_30m"],
        }
        for k, p in intraday_map.items():
            info = self._series_time_info(p, mode="epoch")
            if info:
                report["intraday"][k] = info

        daily_map = {
            "spx_daily": self.market_targets["spx_daily"],
            "vvix_daily": self.market_targets["vvix_daily"],
            "sofr_nyfed": self.rates_dir / "sofr_daily_nyfed.csv",
            "vix_futures": self.vix_fut_dir / "vix_futures_all.csv",
        }
        for k, p in daily_map.items():
            info = self._series_time_info(p, mode="date")
            if info:
                report["daily"][k] = info

        # Intraday overlap window between SPX and VIX frequencies
        for f in [5, 15, 30]:
            vix_k = f"vix_{f}m"
            spx_k = f"spx_{f}m"
            if vix_k in report["intraday"] and spx_k in report["intraday"]:
                s1 = pd.to_datetime(report["intraday"][vix_k]["start"], utc=True)
                e1 = pd.to_datetime(report["intraday"][vix_k]["end"], utc=True)
                s2 = pd.to_datetime(report["intraday"][spx_k]["start"], utc=True)
                e2 = pd.to_datetime(report["intraday"][spx_k]["end"], utc=True)
                s = max(s1, s2)
                e = min(e1, e2)
                report["overlap"][f"vix_vs_spx_{f}m"] = {
                    "start": str(s),
                    "end": str(e),
                    "days": int((e - s).days) if e >= s else -1,
                }

        return report

    def _write_data_ranges_manifest(self):
        rows = []

        paths = {
            "vix_5m": self.market_targets["vix_5m"],
            "vix_10m": self.market_targets["vix_10m"],
            "vix_15m": self.market_targets["vix_15m"],
            "vix_30m": self.market_targets["vix_30m"],
            "spx_5m": self.market_targets["spx_5m"],
            "spx_30m": self.market_targets["spx_30m"],
            "spx_daily": self.market_targets["spx_daily"],
            "vvix_daily": self.market_targets["vvix_daily"],
            "sofr_nyfed": self.rates_dir / "sofr_daily_nyfed.csv",
            "vix_futures_all": self.vix_fut_dir / "vix_futures_all.csv",
        }

        for name, p in paths.items():
            mode = "epoch" if "_5m" in name or "_10m" in name or "_15m" in name or "_30m" in name else "date"
            info = self._series_time_info(p, mode=mode)
            if not info:
                continue
            rows.append({
                "dataset": name,
                "path": info["path"],
                "rows": info["rows"],
                "start": info["start"],
                "end": info["end"],
                "median_dt_min": info["median_dt_min"],
            })

        if not rows:
            return

        out = self.data_dir / "data_ranges.csv"
        pd.DataFrame(rows).sort_values("dataset").to_csv(out, index=False)
        self._register("data_ranges", out)

    def _find_tv_file(self, base_dir: Path, *candidates: str) -> Optional[Path]:
        """Find the first existing TradingView file among naming variants.
        
        Searches in:
        1. The given base_dir directly
        2. All subdirectories of trading_view/ (organized by category)
        """
        # Direct match in base_dir
        for name in candidates:
            p = base_dir / name
            if p.exists():
                return p
        # Search in organized subdirectories
        if self.tv_dir.exists():
            for subdir in self.tv_dir.iterdir():
                if subdir.is_dir():
                    for name in candidates:
                        p = subdir / name
                        if p.exists():
                            return p
        return None

    def _ingest_tv_file(self, input_name: str, output_path: Path, label: str,
                        alt_names: Optional[List[str]] = None):
        base_dir = self.tv_input_dir if self.tv_input_dir.exists() else self.tv_input_dir_alt
        candidates = [input_name] + (alt_names or [])
        in_path = self._find_tv_file(base_dir, *candidates)
        if in_path is None:
            self._register_skip(label, f"missing tradingview input (tried: {', '.join(candidates)})")
            return
        df = pd.read_csv(in_path)
        self._save_market_series(df, output_path, label)

    def ingest_tradingview_exports(self):
        """
        Build canonical files from user-provided TradingView exports.
        
        Supports organized structure (primary):
          data/trading_view/volatility/vix_5m.csv
          data/trading_view/equity_indices/spx_5m.csv
                Also accepts historical flat naming:
          "TVC_VIX, 5.csv", "TVC_VIX,5min.csv"
        
        Searches in data/trading_view/<category>/ first, then flat.
        """
        # VIX intraday
        self._ingest_tv_file(
            "vix_5m.csv", self.market_targets["vix_5m"], "vix_5m_tv",
            alt_names=["TVC_VIX, 5.csv", "TVC_VIX,5min.csv", "CBOE_DLY_VIX, 5.csv"],
        )
        self._ingest_tv_file(
            "vix_10m.csv", self.market_targets["vix_10m"], "vix_10m_tv_direct",
            alt_names=["TVC_VIX, 10.csv", "TVC_VIX,10min.csv", "CBOE_DLY_VIX, 10.csv"],
        )
        self._ingest_tv_file(
            "vix_15m.csv", self.market_targets["vix_15m"], "vix_15m_tv",
            alt_names=["TVC_VIX, 15.csv", "TVC_VIX,15min.csv", "CBOE_DLY_VIX, 15.csv"],
        )
        self._ingest_tv_file(
            "vix_30m.csv", self.market_targets["vix_30m"], "vix_30m_tv",
            alt_names=["TVC_VIX, 30.csv", "TVC_VIX,30min.csv", "CBOE_DLY_VIX, 30.csv"],
        )
        # SPX intraday
        self._ingest_tv_file(
            "spx_5m.csv", self.market_targets["spx_5m"], "spx_5m_tv",
            alt_names=["SP_SPX, 5.csv", "SP_SPX,5min.csv", "TVC_SPX, 5.csv"],
        )
        self._ingest_tv_file(
            "spx_30m.csv", self.market_targets["spx_30m"], "spx_30m_tv",
            alt_names=["SP_SPX, 30.csv", "SP_SPX,30min.csv", "TVC_SPX, 30.csv"],
        )

        # VVIX (daily or intraday exports accepted)
        vvix_path = self._find_tv_file(
            self.tv_volatility if self.tv_volatility.exists() else self.tv_dir,
            "vvix_daily.csv", "vvix_5m.csv",
            "CBOE_DLY_VVIX, 5.csv", "VVIX_daily.csv",
        )
        if vvix_path is not None:
            df = pd.read_csv(vvix_path)
            out = self.market_targets["vvix_daily"]
            self._save_csv(df, out, "vvix_daily_tv")
        else:
            self._register_skip("vvix_daily_tv", "no VVIX file found in tradingview folder")

        # Build VIX 10m from VIX 5m if direct 10m not ingested
        p5 = self.market_targets["vix_5m"]
        p10 = self.market_targets["vix_10m"]
        if not p10.exists() and p5.exists():
            l5 = pd.read_csv(p5)
            dt = pd.to_datetime(l5["time"], unit="s", utc=True)
            base = pd.DataFrame({
                "Open": l5["open"].values,
                "High": l5["high"].values,
                "Low": l5["low"].values,
                "Close": l5["close"].values,
            }, index=dt)
            vix10 = base.resample("10min").agg(
                {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
            ).dropna()
            self._save_market_series(
                vix10, p10, "vix_10m_tv_resampled",
            )
        elif not p5.exists() and not p10.exists():
            self._register_skip("vix_10m_tv", "missing VIX 5m source for resampling")

    # ------------------------------------------------------------------
    # VIX / VVIX / SPX
    # ------------------------------------------------------------------
    def fetch_vix_and_spx(self):
        """
        Regenerate VIX/SPX/VVIX datasets used by loaders.

        Notes:
        - Yahoo intraday depth is limited by interval.
        """
        # VIX intraday (5m/15m/30m). 10m is resampled from 5m.
        vix_5 = None
        if self._has_market_series(self.market_targets["vix_5m"]):
            self._register_skip("vix_5m", "exists")
        else:
            vix_5 = self._download_yahoo("^VIX", interval="5m", period="60d")
            self._save_market_series(
                vix_5,
                self.market_targets["vix_5m"],
                "vix_5m",
            )

        if self._has_market_series(self.market_targets["vix_15m"]):
            self._register_skip("vix_15m", "exists")
        else:
            vix_15 = self._download_yahoo("^VIX", interval="15m", period="60d")
            self._save_market_series(
                vix_15,
                self.market_targets["vix_15m"],
                "vix_15m",
            )

        if self._has_market_series(self.market_targets["vix_30m"]):
            self._register_skip("vix_30m", "exists")
        else:
            vix_30 = self._download_yahoo("^VIX", interval="30m", period="60d")
            self._save_market_series(
                vix_30,
                self.market_targets["vix_30m"],
                "vix_30m",
            )

        if self._has_market_series(self.market_targets["vix_10m"]):
            self._register_skip("vix_10m", "exists")
        else:
            if vix_5 is None:
                # Build from already-downloaded canonical 5m file to avoid network call.
                p5 = self.market_targets["vix_5m"]
                if p5.exists():
                    l5 = pd.read_csv(p5)
                    dt = pd.to_datetime(l5["time"], unit="s", utc=True)
                    base = pd.DataFrame({
                        "Open": l5["open"].values,
                        "High": l5["high"].values,
                        "Low": l5["low"].values,
                        "Close": l5["close"].values,
                    }, index=dt)
                    vix10 = base.resample("10min").agg(
                        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
                    ).dropna()
                    self._save_market_series(
                        vix10,
                        self.market_targets["vix_10m"],
                        "vix_10m",
                    )
                else:
                    self._register_skip("vix_10m", "missing source 5m data")
            elif len(vix_5) > 0:
                vix10 = vix_5[["Open", "High", "Low", "Close"]].resample("10min").agg(
                    {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
                ).dropna()
                self._save_market_series(
                    vix10,
                    self.market_targets["vix_10m"],
                    "vix_10m",
                )
            else:
                self._register_skip("vix_10m", "5m VIX not available for resampling")

        # SPX intraday + daily long history (extend beyond 2023)
        if self._has_market_series(self.market_targets["spx_5m"]):
            self._register_skip("spx_5m", "exists")
        else:
            spx_5 = self._download_yahoo("^GSPC", interval="5m", period="60d")
            self._save_market_series(
                spx_5,
                self.market_targets["spx_5m"],
                "spx_5m",
            )

        if self._has_market_series(self.market_targets["spx_30m"]):
            self._register_skip("spx_30m", "exists")
        else:
            spx_30 = self._download_yahoo("^GSPC", interval="30m", period="60d")
            self._save_market_series(
                spx_30,
                self.market_targets["spx_30m"],
                "spx_30m",
            )

        fresh_daily = self.market_targets["spx_daily"]
        if fresh_daily.exists() and not self.force_download:
            self._register_skip("spx_daily", "exists")
        else:
            spx_daily = self._download_yahoo("^GSPC", interval="1d", start="2010-01-01", end=datetime.now().strftime("%Y-%m-%d"))
            if spx_daily is not None and len(spx_daily) > 0:
                out_daily = spx_daily.reset_index()
                self._save_csv(out_daily, fresh_daily, "spx_daily:canonical")
            else:
                self._register_skip("spx_daily", "Yahoo download returned empty")

        # VVIX daily from Yahoo
        pod = self.market_targets["vvix_daily"]
        if pod.exists() and not self.force_download:
            self._register_skip("vvix", "exists")
        else:
            vvix_d = self._download_yahoo("^VVIX", interval="1d", start="2013-01-01", end=datetime.now().strftime("%Y-%m-%d"))
            if vvix_d is not None and len(vvix_d) > 0:
                self._save_csv(vvix_d.reset_index(), pod, "vvix_daily")
            else:
                self._register_skip("vvix", "Yahoo ^VVIX unavailable")

    def build_combined_panels(self):
        """Create joined market panels from canonical files for easier downstream use."""
        panel_specs = [
            ("vix", [5, 10, 15, 30], self.vix_dir / "vix_combined.csv"),
            ("spx", [5, 30], self.spx_dir / "spx_combined.csv"),
        ]

        for symbol, freqs, out_path in panel_specs:
            rows = []
            for f in freqs:
                p = (self.vix_dir / f"vix_{f}m.csv") if symbol == "vix" else (self.spx_dir / f"spx_{f}m.csv")
                if not p.exists():
                    continue
                try:
                    df = pd.read_csv(p)
                    needed = {"time", "open", "high", "low", "close"}
                    if not needed.issubset(df.columns):
                        continue
                    sub = df[["time", "open", "high", "low", "close"]].copy()
                    sub["frequency_min"] = int(f)
                    sub["symbol"] = symbol.upper()
                    rows.append(sub)
                except Exception:
                    continue

            if not rows:
                self._register_skip(f"{symbol}_combined", "no source files")
                continue

            combo = pd.concat(rows, ignore_index=True)
            combo = combo.sort_values(["frequency_min", "time"]).drop_duplicates(["frequency_min", "time"])
            self._save_csv(combo, out_path, f"{symbol}_combined")

    # ------------------------------------------------------------------
    # SOFR from NY Fed
    # ------------------------------------------------------------------
    def fetch_sofr_nyfed(self,
                         nyfed_url: Optional[str] = None,
                         start: str = "2016-03-01",
                         end: Optional[str] = None):
        """
        Download SOFR from NY Fed link.

        Works with the provided xlsx URL and falls back to csv format automatically
        when Excel dependencies are unavailable.
        """
        out_path = self.rates_dir / "sofr_daily_nyfed.csv"

        existing = self._read_csv_safe(out_path)
        latest_existing = None
        if len(existing) > 0:
            date_cols = [c for c in existing.columns if c.lower() in {"date", "datetime"}]
            if date_cols:
                try:
                    latest_existing = pd.to_datetime(existing[date_cols[0]], errors="coerce").max()
                except Exception:
                    latest_existing = None

        if latest_existing is not None and not self.force_download:
            start = (pd.to_datetime(latest_existing).normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        import re

        base_url = nyfed_url or NYFED_SOFR_URL
        base_url = re.sub(r"startDt=[0-9-]+", f"startDt={start}", base_url)
        base_url = re.sub(r"endDt=[0-9-]+", f"endDt={end}", base_url)

        df = None
        used_url = None

        # 1) Try direct URL (xlsx typically)
        try:
            df = pd.read_excel(base_url)
            used_url = base_url
        except Exception:
            df = None

        # 2) Try csv form (no openpyxl needed)
        if df is None:
            csv_url = base_url.replace("format=xlsx", "format=csv")
            try:
                df = pd.read_csv(csv_url)
                used_url = csv_url
            except Exception:
                df = None

        if df is None or len(df) == 0:
            self._register_skip("sofr_nyfed", "Failed to download/parse NY Fed SOFR file")
            return

        # Normalize likely column names
        cols_lower = {c.lower(): c for c in df.columns}
        date_col = cols_lower.get("effective date") or cols_lower.get("effective_date") or cols_lower.get("date")
        rate_col = cols_lower.get("rate (%)") or cols_lower.get("rate") or cols_lower.get("percent rate")

        if date_col is None or rate_col is None:
            # fallback: try first two columns
            if df.shape[1] >= 2:
                date_col = df.columns[0]
                rate_col = df.columns[1]
            else:
                self._register_skip("sofr_nyfed", "SOFR columns not found in NY Fed file")
                return

        out = pd.DataFrame({
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "rate": pd.to_numeric(df[rate_col], errors="coerce"),
        }).dropna()

        if len(out) == 0:
            self._register_skip("sofr_nyfed", "Parsed NY Fed SOFR data is empty")
            return

        if len(existing) > 0:
            existing = existing.copy()
            if "date" not in existing.columns and "Date" in existing.columns:
                existing = existing.rename(columns={"Date": "date"})
            if "rate" not in existing.columns and "Close" in existing.columns:
                existing = existing.rename(columns={"Close": "rate"})
            out = pd.concat([existing, out], ignore_index=True, sort=False)

        out = out.sort_values("date").drop_duplicates("date", keep="last")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        self._register("sofr_nyfed", out_path)

        self.metadata["sofr_source"] = "nyfed"
        self.metadata["sofr_url"] = used_url

    def fetch_sofr_yahoo(self):
        """
        Download SOFR-like series from Yahoo first.

        Tries multiple symbols and picks the first with usable data.
        """
        out_path = self.rates_dir / "sofr_daily_yahoo.csv"
        if out_path.exists() and not self.force_download:
            self._register_skip("sofr", "exists")
            return

        candidates = ["^SOFR", "SOFR", "SR3=F"]
        used = None
        df_used = None

        for t in candidates:
            try:
                df = self._download_yahoo(t, interval="1d", start="2016-03-01", end=datetime.now().strftime("%Y-%m-%d"))
                if df is not None and len(df) > 30 and "Close" in df.columns:
                    used = t
                    df_used = df
                    break
            except Exception:
                continue

        if df_used is None:
            self._register_skip("sofr_yahoo", f"No Yahoo symbol found among {candidates}")
            # Fallback to NY Fed source
            self.fetch_sofr_nyfed()
            return

        out = df_used.reset_index()
        self._save_csv(out, out_path, "sofr_yahoo")

        self.metadata["sofr_symbol"] = used
        self.metadata["sofr_source"] = "yahoo"

    # ------------------------------------------------------------------
    # CBOE VIX futures history
    # ------------------------------------------------------------------
    def _month_expiries(self, start: str = "2013-01-01", months_forward: int = 12) -> List[pd.Timestamp]:
        s = pd.Timestamp(start)
        e = pd.Timestamp.today().normalize() + pd.DateOffset(months=months_forward)
        months = pd.period_range(s, e, freq="M")
        expiries = []
        for m in months:
            # Third Wednesday approximation
            first = pd.Timestamp(m.start_time.date())
            wednesdays = pd.date_range(first, first + pd.offsets.MonthEnd(0), freq="W-WED")
            if len(wednesdays) >= 3:
                expiries.append(wednesdays[2])
        return expiries

    def fetch_vix_futures_cboe(self):
        """Rebuild vix_futures_all.csv and front/2M/3M files from CBOE endpoint.
        
        Only downloads missing trade dates (optimized for incremental updates).
        """
        full = self.vix_fut_dir / "vix_futures_all.csv"
        p_front = self.vix_fut_dir / "vix_futures_front_month.csv"
        p_2m = self.vix_fut_dir / "vix_futures_2M.csv"
        p_3m = self.vix_fut_dir / "vix_futures_3M.csv"
        
        # Get existing trade dates to avoid re-downloading
        existing_dates = set()
        df_existing = None
        if full.exists() and not self.force_download:
            try:
                df_existing = pd.read_csv(full)
                if "Trade Date" in df_existing.columns:
                    existing_dates = set(pd.to_datetime(df_existing["Trade Date"]).dt.strftime("%Y-%m-%d"))
            except Exception:
                pass

        expiries = self._month_expiries("2013-01-01", months_forward=12)
        all_rows = []
        downloaded_count = 0

        for exp in expiries:
            ds = exp.strftime("%Y-%m-%d")
            
            # Skip if already have this date
            if ds in existing_dates and not self.force_download:
                continue
            
            url = f"https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_{ds}.csv"
            try:
                r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 200 and r.text.strip():
                    df = pd.read_csv(io.StringIO(r.text))
                    if "Futures" in df.columns and "expiration_date" not in df.columns:
                        df["expiration_date"] = pd.to_datetime(
                            df["Futures"].astype(str).str.extract(r"\((.*?)\)")[0],
                            format="%b %Y",
                            errors="coerce",
                        )
                    all_rows.append(df)
                    downloaded_count += 1
            except Exception:
                continue

        if not all_rows and not df_existing:
            # No new data and no existing file
            self._register_skip("vix_futures_cboe", "No CSV downloaded from CBOE")
            return
        
        if not all_rows and df_existing is not None:
            # No new data, but have existing file
            self._register_skip("vix_futures_cboe", f"No new dates (have {len(existing_dates)} existing)")
            return

        # Merge with existing
        if df_existing is not None and not self.force_download:
            all_rows.insert(0, df_existing)
        
        df_all = pd.concat(all_rows, ignore_index=True).drop_duplicates()
        
        # Sort by Trade Date to maintain chronological order
        df_all["Trade Date"] = pd.to_datetime(df_all["Trade Date"])
        df_all = df_all.sort_values("Trade Date").reset_index(drop=True)
        df_all["Trade Date"] = df_all["Trade Date"].dt.strftime("%Y-%m-%d")
        
        # For incremental updates, force overwrite the vix_futures_all.csv
        self._save_csv(df_all, full, "vix_futures_all", force_overwrite=True)
        
        if downloaded_count > 0:
            print(f"  → Downloaded {downloaded_count} new trade dates from CBOE")

        # Build front/2M/3M
        if "Trade Date" not in df_all.columns:
            self._register_skip("vix_futures_front_curves", "Trade Date column missing")
            return

        # Ensure expiration_date is consistently datetime
        if "expiration_date" not in df_all.columns and "Futures" in df_all.columns:
            df_all["expiration_date"] = pd.to_datetime(
                df_all["Futures"].astype(str).str.extract(r"\((.*?)\)")[0], format="%b %Y", errors="coerce"
            )
        else:
            # Convert any existing expiration_date to datetime (might be mixed types)
            df_all["expiration_date"] = pd.to_datetime(df_all["expiration_date"], errors="coerce")

        df_sorted = df_all.dropna(subset=["expiration_date"]).copy()
        df_sorted = df_sorted.groupby("Trade Date", group_keys=False).apply(lambda g: g.sort_values("expiration_date"))

        front = df_sorted.groupby("Trade Date", group_keys=False).nth(0).reset_index()
        sec = df_sorted.groupby("Trade Date", group_keys=False).nth(1).reset_index()
        third = df_sorted.groupby("Trade Date", group_keys=False).nth(2).reset_index()

        # For incremental updates, force overwrite the curve files
        self._save_csv(front, p_front, "vix_futures_front_month", force_overwrite=True)
        self._save_csv(sec, p_2m, "vix_futures_2m", force_overwrite=True)
        self._save_csv(third, p_3m, "vix_futures_3m", force_overwrite=True)

    # ------------------------------------------------------------------
    # Daily supplements (for tradingview mode)
    # ------------------------------------------------------------------
    def _fetch_daily_supplements(self):
        """Fetch daily SPX, VVIX, and SOFR that TradingView exports don't include."""
        # SPX daily (2010-present) for P-measure RV extension
        fresh_daily = self.market_targets["spx_daily"]
        if not fresh_daily.exists() or self.force_download:
            try:
                spx_daily = self._download_yahoo(
                    "^GSPC", interval="1d",
                    start="2010-01-01",
                    end=datetime.now().strftime("%Y-%m-%d"),
                )
                if spx_daily is not None and len(spx_daily) > 0:
                    self._save_csv(spx_daily.reset_index(), fresh_daily, "spx_daily:supplement")
            except Exception as e:
                self._register_skip("spx_daily:supplement", str(e))

        # VVIX daily for eta calibration
        pod = self.market_targets["vvix_daily"]
        if not pod.exists() or self.force_download:
            try:
                vvix_d = self._download_yahoo(
                    "^VVIX", interval="1d",
                    start="2013-01-01",
                    end=datetime.now().strftime("%Y-%m-%d"),
                )
                if vvix_d is not None and len(vvix_d) > 0:
                    self._save_csv(vvix_d.reset_index(), pod, "vvix_daily:supplement")
            except Exception as e:
                self._register_skip("vvix_daily:supplement", str(e))

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run(self,
            sofr_url: Optional[str] = None,
            mode: str = "yahoo",
            cleanup_redundant: bool = True) -> PipelineResult:
        self.cleanup_redundant = bool(cleanup_redundant)

        # Migrate previous file layout to canonical intraday schema when needed
        self.normalize_existing_canonical_files()

        # Refresh from selected market source
        if mode == "tradingview":
            self.ingest_tradingview_exports()
            # Also fetch daily data + VVIX that TradingView doesn't provide
            self._fetch_daily_supplements()
        elif mode == "merge":
            # TradingView first (richer), then Yahoo to fill gaps
            self.ingest_tradingview_exports()
            self._fetch_daily_supplements()
            self.fetch_vix_and_spx()  # Yahoo fills any missing intraday
        else:
            self.fetch_vix_and_spx()

        self.fetch_sofr_nyfed(nyfed_url=sofr_url)
        self.build_combined_panels()

        # Rebuild CBOE futures every run (deterministic self-regeneration)
        self.fetch_vix_futures_cboe()
        self._write_data_ranges_manifest()

        report = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "generated": self.generated,
            "skipped": self.skipped,
            "metadata": self.metadata,
            "temporal_coherence": self._build_temporal_coherence_report(),
        }
        out = self.data_dir / "regeneration_report.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        self._register("regeneration_report", out)

        return PipelineResult(generated=self.generated, skipped=self.skipped, metadata=self.metadata)
