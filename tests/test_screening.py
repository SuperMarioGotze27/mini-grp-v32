import numpy as np

from core.factor_engine import calculate_factors
from core.main import score_universe
from utils.mock import generate_mock_data


def test_mock_data_is_reproducible_and_labelled():
    first = generate_mock_data(60, seed=7)
    second = generate_mock_data(60, seed=7)
    assert first.equals(second)
    assert first["is_mock"].all()
    assert set(first["data_source"]) == {"synthetic_demo"}


def test_screening_returns_unique_dimension_columns():
    scored, top = score_universe(generate_mock_data(80, seed=11), top_n=15)
    assert len(top) == 15
    assert top["composite_score"].is_monotonic_decreasing
    assert not top.columns.duplicated().any()
    assert scored["composite_score"].between(0, 100).all()


def test_unavailable_expectation_factors_are_excluded():
    data = generate_mock_data(60)
    data[["sue", "eps_revision", "rating_revision"]] = np.nan
    factors = calculate_factors(data)
    assert "sue_z" not in factors
    assert factors["expectation_score"].isna().all()


def test_cross_market_standardization_is_market_local():
    cn = generate_mock_data(50, seed=1).assign(market="CN")
    us = generate_mock_data(50, seed=2).assign(market="US")
    us["pe_ttm"] = us["pe_ttm"] * 20
    factors = calculate_factors(__import__("pandas").concat([cn, us], ignore_index=True))
    means = factors.groupby("market")["pe_ttm_z"].mean().abs()
    assert (means < 1e-10).all()
