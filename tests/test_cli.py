from pathlib import Path

from core.main import main


def test_demo_screening_cli_writes_artifacts(tmp_path: Path):
    exit_code = main(
        [
            "--mode", "screen",
            "--data-mode", "demo",
            "--max-stocks", "50",
            "--top-n", "10",
            "--output-dir", str(tmp_path),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "screening" / "top_picks.csv").exists()
    assert (tmp_path / "screening" / "screening_manifest.json").exists()
