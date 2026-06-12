"""Deterministic synthetic stock universe used by demos and tests."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


INDUSTRIES = ["Semiconductors", "Banks", "Healthcare", "Clean Energy", "Consumer", "Software", "Industrials", "Chemicals"]


def generate_mock_data(n_stocks: int = 100, seed: int = 42) -> pd.DataFrame:
    """Return a reproducible 19-factor universe labelled as synthetic data."""
    if n_stocks < 10:
        raise ValueError("n_stocks must be at least 10")

    rng = np.random.default_rng(seed)
    n_sh = n_stocks // 3
    n_sz = n_stocks // 3
    codes = [f"{600000 + i:06d}.SH" for i in range(n_sh)]
    codes += [f"{1 + i:06d}.SZ" for i in range(n_sz)]
    codes += [f"{300001 + i:06d}.SZ" for i in range(n_stocks - n_sh - n_sz)]

    industry = rng.choice(INDUSTRIES, size=n_stocks)
    value = rng.normal(size=n_stocks)
    quality = rng.normal(size=n_stocks)
    growth = rng.normal(size=n_stocks)
    momentum = rng.normal(size=n_stocks)
    expectation = rng.normal(size=n_stocks)

    frame = pd.DataFrame(
        {
            "code": codes,
            "name": [f"Demo Stock {i + 1:03d}" for i in range(n_stocks)],
            "sw_industry_name": industry,
            "market": "DEMO",
            "currency": "N/A",
            "data_source": "synthetic_demo",
            "is_mock": True,
            "as_of_date": date.today().isoformat(),
            "pe_ttm": np.clip(24 - 5 * value + rng.normal(0, 3, n_stocks), 2, None),
            "pb_lf": np.clip(3.2 - 0.7 * value + rng.normal(0, 0.5, n_stocks), 0.2, None),
            "ps_ttm": np.clip(4.0 - 0.8 * value + rng.normal(0, 0.7, n_stocks), 0.2, None),
            "ev_ebitda": np.clip(15 - 3 * value + rng.normal(0, 2, n_stocks), 1, None),
            "dividend_yield": np.clip(2 + 0.8 * value + rng.normal(0, 0.6, n_stocks), 0, None),
            "roe_deducted": 11 + 6 * quality + rng.normal(0, 2, n_stocks),
            "roa": 5 + 3 * quality + rng.normal(0, 1.5, n_stocks),
            "gross_margin": 32 + 8 * quality + rng.normal(0, 4, n_stocks),
            "net_margin": 12 + 5 * quality + rng.normal(0, 3, n_stocks),
            "debt_to_equity": np.clip(70 - 15 * quality + rng.normal(0, 12, n_stocks), 1, None),
            "revenue_yoy": 12 + 12 * growth + rng.normal(0, 6, n_stocks),
            "profit_yoy": 10 + 16 * growth + rng.normal(0, 8, n_stocks),
            "fcf_yield": 3 + 1.5 * growth + rng.normal(0, 1, n_stocks),
            "return_1m": 4 * momentum + rng.normal(0, 5, n_stocks),
            "return_3m": 8 * momentum + rng.normal(0, 8, n_stocks),
            "return_12m": 12 + 15 * momentum + rng.normal(0, 12, n_stocks),
            "sue": expectation + rng.normal(0, 0.35, n_stocks),
            "eps_revision": 0.15 * expectation + rng.normal(0, 0.08, n_stocks),
            "rating_revision": 0.25 * expectation + rng.normal(0, 0.12, n_stocks),
        }
    )
    return frame
