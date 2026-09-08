#!/usr/bin/env python3
"""Download Oslo stock history and compute pairwise correlations."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

OBX_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/OBX_Index"
FALLBACK_TICKERS = [
    "DNB.OL",
    "EQNR.OL",
    "TEL.OL",
    "YAR.OL",
    "MOWI.OL",
    "NHY.OL",
    "SALM.OL",
    "ORK.OL",
    "AKRBP.OL",
    "STB.OL",
]


@dataclass
class Config:
    period: str
    interval: str
    min_history_points: int
    min_history_coverage: float
    min_pair_observations: int
    rolling_window: int
    stable_corr_threshold: float
    stable_std_threshold: float
    output_dir: Path
    max_pairs_csv_rows: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Hent historiske priser for Oslo-aksjer, beregn korrelasjon mellom aksje-par "
            "og lagre resultater i CSV."
        )
    )
    parser.add_argument("--period", default="5y", help="Historikk-periode for Yahoo Finance.")
    parser.add_argument("--interval", default="1d", help="Intervall for historikk (f.eks. 1d).")
    parser.add_argument("--min-history-points", type=int, default=252)
    parser.add_argument("--min-history-coverage", type=float, default=0.7)
    parser.add_argument("--min-pair-observations", type=int, default=200)
    parser.add_argument("--rolling-window", type=int, default=60)
    parser.add_argument("--stable-corr-threshold", type=float, default=0.7)
    parser.add_argument("--stable-std-threshold", type=float, default=0.15)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-pairs-csv-rows", type=int, default=50)
    args = parser.parse_args()

    if not (0 < args.min_history_coverage <= 1):
        raise ValueError("--min-history-coverage må være mellom 0 og 1.")

    return Config(
        period=args.period,
        interval=args.interval,
        min_history_points=args.min_history_points,
        min_history_coverage=args.min_history_coverage,
        min_pair_observations=args.min_pair_observations,
        rolling_window=args.rolling_window,
        stable_corr_threshold=args.stable_corr_threshold,
        stable_std_threshold=args.stable_std_threshold,
        output_dir=Path(args.output_dir),
        max_pairs_csv_rows=args.max_pairs_csv_rows,
    )


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _normalize_oslo_ticker(raw: str) -> str:
    ticker = raw.strip().upper()
    if not ticker:
        return ""
    if ticker.endswith(".OL"):
        return ticker
    return f"{ticker}.OL"


def _extract_tickers_from_table(table: pd.DataFrame) -> list[str]:
    candidate_columns = [
        col
        for col in table.columns
        if any(key in str(col).lower() for key in ["ticker", "symbol", "ric"])
    ]
    for col in candidate_columns:
        series = table[col].astype(str).str.split().str[0]
        normalized = [_normalize_oslo_ticker(x) for x in series]
        normalized = [x for x in normalized if x and x.replace(".OL", "").isalnum()]
        if len(normalized) >= 5:
            return sorted(set(normalized))
    return []


def get_oslo_universe() -> list[str]:
    logging.info("Henter aksjeunivers fra %s", OBX_WIKIPEDIA_URL)
    try:
        tables = pd.read_html(OBX_WIKIPEDIA_URL)
    except Exception as exc:
        logging.warning("Klarte ikke hente OBX-tabell (%s). Bruker fallback-univers.", exc)
        return FALLBACK_TICKERS

    for table in tables:
        tickers = _extract_tickers_from_table(table)
        if tickers:
            logging.info("Fant %d tickere fra OBX-univers.", len(tickers))
            return tickers

    logging.warning("Ingen gyldige tickere funnet i OBX-tabeller. Bruker fallback-univers.")
    return FALLBACK_TICKERS


def download_adjusted_close(
    tickers: Iterable[str], period: str, interval: str
) -> tuple[pd.DataFrame, dict[str, str]]:
    series_by_ticker: dict[str, pd.Series] = {}
    failure_reasons: dict[str, str] = {}

    for ticker in tickers:
        logging.info("Laster ned historikk for %s", ticker)
        try:
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            failure_reasons[ticker] = f"Nedlasting feilet: {exc}"
            logging.warning("%s ekskludert: %s", ticker, failure_reasons[ticker])
            continue

        if data.empty:
            failure_reasons[ticker] = "Ingen data returnert"
            logging.warning("%s ekskludert: %s", ticker, failure_reasons[ticker])
            continue

        price_column = "Adj Close" if "Adj Close" in data.columns else "Close"
        if price_column not in data.columns:
            failure_reasons[ticker] = "Mangler Close/Adj Close"
            logging.warning("%s ekskludert: %s", ticker, failure_reasons[ticker])
            continue

        series = data[price_column].copy()
        series.index = pd.to_datetime(series.index).tz_localize(None)
        series.name = ticker
        series_by_ticker[ticker] = series

    prices = pd.concat(series_by_ticker.values(), axis=1) if series_by_ticker else pd.DataFrame()
    return prices.sort_index(), failure_reasons


def filter_valid_stocks(
    prices: pd.DataFrame,
    min_history_points: int,
    min_history_coverage: float,
) -> tuple[pd.DataFrame, list[str]]:
    if prices.empty:
        return prices, []

    max_points = prices.notna().sum().max()
    min_points_by_coverage = int(max_points * min_history_coverage)
    min_required = max(min_history_points, min_points_by_coverage)

    valid_tickers = [
        ticker for ticker in prices.columns if prices[ticker].notna().sum() >= min_required
    ]

    excluded = [ticker for ticker in prices.columns if ticker not in valid_tickers]
    for ticker in excluded:
        logging.info(
            "%s ekskludert pga for kort historikk (%d datapunkter)",
            ticker,
            prices[ticker].notna().sum(),
        )

    filtered = prices[valid_tickers].copy()
    return filtered, excluded


def compute_pairwise_correlations(
    returns: pd.DataFrame,
    min_pair_observations: int,
    rolling_window: int,
    stable_corr_threshold: float,
    stable_std_threshold: float,
) -> pd.DataFrame:
    records: list[dict[str, float | int | str | bool]] = []
    columns = list(returns.columns)

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            ticker_a = columns[i]
            ticker_b = columns[j]
            pair_data = returns[[ticker_a, ticker_b]].dropna()
            observations = len(pair_data)
            if observations < min_pair_observations:
                continue

            corr = pair_data[ticker_a].corr(pair_data[ticker_b])

            rolling_corr = (
                pair_data[ticker_a]
                .rolling(rolling_window)
                .corr(pair_data[ticker_b])
                .dropna()
            )
            rolling_mean = float(rolling_corr.mean()) if not rolling_corr.empty else float("nan")
            rolling_std = float(rolling_corr.std()) if not rolling_corr.empty else float("nan")

            stable_candidate = (
                pd.notna(rolling_mean)
                and pd.notna(rolling_std)
                and abs(rolling_mean) >= stable_corr_threshold
                and rolling_std <= stable_std_threshold
            )

            records.append(
                {
                    "stock_a": ticker_a,
                    "stock_b": ticker_b,
                    "correlation": float(corr),
                    "observations": observations,
                    "rolling_corr_mean": rolling_mean,
                    "rolling_corr_std": rolling_std,
                    "stable_candidate": bool(stable_candidate),
                }
            )

    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result

    return result.sort_values("correlation", ascending=False).reset_index(drop=True)


def save_outputs(pair_df: pd.DataFrame, output_dir: Path, max_rows: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    full_path = output_dir / "pairwise_correlations.csv"
    pair_df.to_csv(full_path, index=False)

    strongest_path = output_dir / "top_strongest_pairs.csv"
    weakest_path = output_dir / "top_weakest_pairs.csv"
    candidates_path = output_dir / "stable_candidate_pairs.csv"

    pair_df.sort_values("correlation", ascending=False).head(max_rows).to_csv(strongest_path, index=False)
    pair_df.sort_values("correlation", ascending=True).head(max_rows).to_csv(weakest_path, index=False)
    pair_df[pair_df["stable_candidate"]].sort_values("correlation", ascending=False).to_csv(
        candidates_path, index=False
    )

    logging.info("Lagret: %s", full_path)
    logging.info("Lagret: %s", strongest_path)
    logging.info("Lagret: %s", weakest_path)
    logging.info("Lagret: %s", candidates_path)


def validate_output(pair_df: pd.DataFrame, valid_ticker_count: int) -> None:
    if pair_df.empty:
        raise ValueError("Ingen aksjepar med nok observasjoner for korrelasjonsanalyse.")

    expected_pairs = valid_ticker_count * (valid_ticker_count - 1) // 2
    analysed_pairs = len(pair_df)

    if analysed_pairs == 0:
        raise ValueError("Output er tomt. Sjekk datakvalitet og terskler.")

    if analysed_pairs < expected_pairs:
        logging.warning(
            "Analyserte %d av %d mulige par pga minstekrav til observasjoner.",
            analysed_pairs,
            expected_pairs,
        )
    else:
        logging.info("Analyserte alle %d mulige aksjepar.", analysed_pairs)


def main() -> None:
    setup_logging()
    cfg = parse_args()

    tickers = get_oslo_universe()
    logging.info("Starter med %d tickere i universet.", len(tickers))

    prices, failures = download_adjusted_close(tickers, cfg.period, cfg.interval)
    if failures:
        logging.info("Ekskludert %d tickere under nedlasting.", len(failures))

    valid_prices, excluded_short_history = filter_valid_stocks(
        prices,
        min_history_points=cfg.min_history_points,
        min_history_coverage=cfg.min_history_coverage,
    )
    if excluded_short_history:
        logging.info("Ekskludert %d tickere med utilstrekkelig historikk.", len(excluded_short_history))

    if valid_prices.shape[1] < 2:
        raise ValueError("For få gyldige aksjer til å lage korrelasjonspar.")

    returns = valid_prices.pct_change(fill_method=None).dropna(how="all")
    pair_df = compute_pairwise_correlations(
        returns=returns,
        min_pair_observations=cfg.min_pair_observations,
        rolling_window=cfg.rolling_window,
        stable_corr_threshold=cfg.stable_corr_threshold,
        stable_std_threshold=cfg.stable_std_threshold,
    )

    validate_output(pair_df, valid_ticker_count=valid_prices.shape[1])
    save_outputs(pair_df, cfg.output_dir, cfg.max_pairs_csv_rows)


if __name__ == "__main__":
    main()
