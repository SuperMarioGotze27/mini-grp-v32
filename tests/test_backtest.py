from backtest.engine import BacktestConfig, analyze_backtest_results, run_backtest


def test_monthly_backtest_produces_trades_and_finite_nav():
    config = BacktestConfig(
        start_date="2022-01-01",
        end_date="2024-12-31",
        top_n=10,
        transaction_cost=0.001,
    )
    results = run_backtest(config, demo_seed=3, demo_n_stocks=60)
    metrics = analyze_backtest_results(results)
    assert len(results) > 20
    assert metrics["trade_count"] > 0
    assert metrics["data_mode"] == "synthetic_demo"
    assert results[["portfolio_nav", "benchmark_nav"]].notna().all().all()


def test_quarterly_backtest_uses_quarterly_annualization():
    config = BacktestConfig(
        start_date="2021-01-01",
        end_date="2024-12-31",
        rebalance_freq="quarterly",
        top_n=10,
    )
    results = run_backtest(config, demo_seed=5, demo_n_stocks=60)
    metrics = analyze_backtest_results(results)
    assert metrics["periods_per_year"] == 4
    assert metrics["num_periods"] == len(results)
