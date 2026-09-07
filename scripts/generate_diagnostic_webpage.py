#!/usr/bin/env python3
"""CLI script to generate/regenerate the diagnostics viewer webpage for a figures directory."""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import esp_lab
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from esp_lab.diagnostics.web import generate_diagnostics_webpage


def main():
    parser = argparse.ArgumentParser(
        description="Generate an interactive index.html diagnostics viewer page in the target directory."
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to the directory containing the saved diagnostic figures.",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Rebuild figures.json from actual fig_* image files before generating HTML.",
    )
    args = parser.parse_args()

    diag_dir = Path(args.dir)
    print(f"Generating diagnostic webpage in: {diag_dir}")
    try:
        output_path = generate_diagnostics_webpage(
            diag_dir, discover_figures=args.discover
        )
        print(f"Success! Webpage created at: {output_path}")
    except Exception as e:
        print(f"Error generating webpage: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
