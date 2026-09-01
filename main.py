"""Compatibility wrapper for running the HINN simulation from the repository root."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from hinn.simulation import main


if __name__ == "__main__":
    main()
