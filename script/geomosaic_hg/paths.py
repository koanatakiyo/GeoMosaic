"""Project-local path helpers.

No default path in this package points outside the GeoMosaic repository.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "script"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "0_raw"
SCORE_DIR = DATA_DIR / "3_direct_scores"
BENCH_DIR = DATA_DIR / "geomosaic_bench"
REPORT_DIR = DATA_DIR / "reports"
INDEX_DIR = DATA_DIR / "indexes"


def project_path(*parts: str | Path) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def data_path(*parts: str | Path) -> Path:
    return DATA_DIR.joinpath(*parts)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def relative_to_project(path: str | Path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()
