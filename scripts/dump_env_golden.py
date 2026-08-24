"""Dump frozen env golden parquet/csv for Rust parity tests."""
from __future__ import annotations

from pathlib import Path

from betatrend.config import ROOT
from betatrend.nn.env import dump_golden


def main() -> None:
    out = ROOT / "crates" / "betatrend-env" / "golden"
    path = dump_golden(out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
