#!/usr/bin/env python3
"""Reproduce all analysis tables and figures for the groundwater-network study.

Required input
--------------
1. The relation-matrix workbook (default: ``All global networks.xlsx``).
2. The catchment groundwater-contribution GeoPackage used for Fig. 1c
   (default: ``Rgw0_data.gpkg``). This file is not needed with ``--skip-fig1``.

Examples
--------
Run the full workflow::

    python reproduce_all_outputs.py \
        --input "All global networks.xlsx" \
        --rgw0 "Rgw0_data.gpkg" \
        --output reproduced_outputs

Run only analysis and supplementary outputs::

    python reproduce_all_outputs.py \
        --input "All global networks.xlsx" \
        --output reproduced_outputs \
        --steps network,matrix,centrality,sensitivity

Python requirement: >= 3.10.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from water_cycle_workflows import (
    create_centrality_workflow,
    create_fraction_matrix_workflow,
    create_main_figures_workflow,
    create_network_graph_workflow,
    create_sensitivity_workflow,
)

ALL_STEPS = ("network", "matrix", "centrality", "sensitivity", "main")
PACKAGE_NAMES = (
    "numpy",
    "pandas",
    "matplotlib",
    "networkx",
    "openpyxl",
    "geopandas",
    "shapely",
    "pyproj",
    "basemap",
)


def parse_steps(text: str) -> list[str]:
    requested = [item.strip().lower() for item in text.split(",") if item.strip()]
    if not requested or requested == ["all"]:
        return list(ALL_STEPS)

    unknown = sorted(set(requested) - set(ALL_STEPS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown step(s): {', '.join(unknown)}. "
            f"Choose from: {', '.join(ALL_STEPS)}, or 'all'."
        )

    # Preserve canonical execution order and remove duplicates.
    return [step for step in ALL_STEPS if step in requested]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_manifest_path(path: Path) -> str:
    """Return a portable path without exposing host-specific parent directories."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def package_versions(names: Iterable[str] = PACKAGE_NAMES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def write_manifest(output_root: Path, input_file: Path, rgw0_file: Path | None, args) -> None:
    records = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name in {
            "reproduction_manifest.json",
            "reproduction_manifest.tsv",
        }:
            continue
        records.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    tsv_path = output_root / "reproduction_manifest.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("path\tsize_bytes\tsha256\n")
        for item in records:
            handle.write(f"{item['path']}\t{item['size_bytes']}\t{item['sha256']}\n")

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "steps": args.steps,
        "skip_fig1": args.skip_fig1,
        "input_workbook": {
            "path": portable_manifest_path(input_file),
            "size_bytes": input_file.stat().st_size,
            "sha256": sha256_file(input_file),
        },
        "rgw0_file": None
        if rgw0_file is None
        else {
            "path": portable_manifest_path(rgw0_file),
            "size_bytes": rgw0_file.stat().st_size,
            "sha256": sha256_file(rgw0_file),
        },
        "package_versions": package_versions(),
        "outputs": records,
    }
    (output_root / "reproduction_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def validate_inputs(input_file: Path, rgw0_file: Path | None, steps: list[str], skip_fig1: bool) -> None:
    if not input_file.is_file():
        raise FileNotFoundError(f"Input workbook not found: {input_file}")
    if input_file.suffix.lower() not in {".xlsx", ".xlsm"}:
        raise ValueError(f"Input workbook must be .xlsx or .xlsm: {input_file}")

    if "main" in steps and not skip_fig1:
        if rgw0_file is None:
            raise ValueError("--rgw0 is required for Fig. 1. Use --skip-fig1 to omit Fig. 1.")
        if not rgw0_file.is_file():
            raise FileNotFoundError(f"Rgw0 GeoPackage not found: {rgw0_file}")


def run_pipeline(
    input_file: Path,
    output_root: Path,
    rgw0_file: Path | None,
    steps: list[str],
    skip_fig1: bool = False,
) -> None:
    """Create the workflow namespaces and execute the selected stages."""
    output_root.mkdir(parents=True, exist_ok=True)

    # Main Fig. 2 and Fig. 3 require the network, centrality and matrix products.
    expanded_steps = set(steps)
    if "main" in expanded_steps:
        expanded_steps.update({"network", "matrix", "centrality"})

    print("Input workbook:", input_file.resolve())
    if rgw0_file is not None:
        print("Rgw0 data:", rgw0_file.resolve())
    print("Output root:", output_root.resolve())
    print("Workflow steps:", ", ".join(step for step in ALL_STEPS if step in expanded_steps))

    network = create_network_graph_workflow(input_file, output_root)
    fraction = create_fraction_matrix_workflow(input_file, output_root)
    centrality = create_centrality_workflow(input_file, output_root)
    sensitivity = create_sensitivity_workflow(input_file, output_root)

    if "network" in expanded_steps:
        print("\n=== Network diagrams ===")
        network.main()

    if "matrix" in expanded_steps:
        print("\n=== Contribution, delivery and delta matrices ===")
        fraction.compute_all_cases_from_excel()

    if "centrality" in expanded_steps:
        print("\n=== Node-level centrality metrics ===")
        centrality.main()

    if "sensitivity" in expanded_steps:
        print("\n=== Groundwater-routing sensitivity outputs ===")
        sensitivity.main()

    if "main" in expanded_steps:
        print("\n=== Main manuscript figures ===")
        mainfig = create_main_figures_workflow(
            input_file=input_file,
            output_root=output_root,
            rgw0_file=rgw0_file,
            sensitivity_workflow=sensitivity,
            network_graph_workflow=network,
        )

        mainfig.run_FIG2()
        mainfig.run_FIG3()

        # Fig. 1 and Fig. 4 share the sensitivity records, so calculate once.
        long_df, _ = mainfig.build_sensitivity_long_dataframe(sensitivity, workbook=input_file)
        if not skip_fig1:
            mainfig.run_FIG1(long_df=long_df.copy(), workflow=sensitivity)
        else:
            print("Fig. 1 skipped by request.")
        mainfig.run_fig4(long_df=long_df.copy(), workflow=sensitivity)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce all tables and figures for the terrestrial water-cycle network study."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("All global networks.xlsx"),
        help="Path to the relation-matrix workbook.",
    )
    parser.add_argument(
        "--rgw0",
        type=Path,
        default=None,
        help="Path to Rgw0_data.gpkg, required for main Fig. 1 unless --skip-fig1 is used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reproduced_outputs"),
        help="Root directory for all generated files.",
    )
    parser.add_argument(
        "--steps",
        type=parse_steps,
        default=list(ALL_STEPS),
        help="Comma-separated stages: network,matrix,centrality,sensitivity,main; default: all.",
    )
    parser.add_argument(
        "--skip-fig1",
        action="store_true",
        help="Skip the map-based Fig. 1, allowing execution without the Rgw0 GeoPackage.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_file = args.input.expanduser().resolve()
    rgw0_file = args.rgw0.expanduser().resolve() if args.rgw0 is not None else None
    output_root = args.output.expanduser().resolve()

    try:
        validate_inputs(input_file, rgw0_file, args.steps, args.skip_fig1)
        started = time.perf_counter()
        run_pipeline(
            input_file=input_file,
            output_root=output_root,
            rgw0_file=rgw0_file,
            steps=args.steps,
            skip_fig1=args.skip_fig1,
        )
        write_manifest(output_root, input_file, rgw0_file, args)
        elapsed = time.perf_counter() - started
        print(f"\nReproduction completed in {elapsed / 60:.1f} minutes.")
        print("Manifest:", output_root / "reproduction_manifest.json")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
