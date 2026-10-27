#!/usr/bin/env python3
"""Reproduce all analysis tables and figures for the groundwater-network study.

Required input
--------------
1. The relation-matrix workbook (default: ``All global networks.xlsx``).
2. The catchment groundwater-contribution GeoPackage used for Fig. 4
   (default: ``Rgw0_data.gpkg``). This file is not needed with ``--skip-fig4``.

Examples
--------
Run the full workflow::

    python reproduce_all_outputs.py \
        --input "All global networks.xlsx" \
        --rgw0 "Rgw0_data.gpkg" \
        --output reproduced_outputs

Run the complete manuscript figure package::

    python reproduce_all_outputs.py \
        --input "All global networks.xlsx" \
        --rgw0 "Rgw0_data.gpkg" \
        --output reproduced_outputs_fresh \
        --steps main,supplementary \
        --clean-output

Python requirement: >= 3.10.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
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

ALL_STEPS = ("network", "matrix", "centrality", "sensitivity", "main", "supplementary")
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

SUPPLEMENTARY_FIGURES = (
    ("FigS1", None, "Relation matrices for six global water-cycle assessments"),
    ("FigS2", "centrality_output/radar_bar_overall.png", "Complete node-level centrality profiles across six global water-cycle assessments"),
    ("FigS3", "matrix_output/first6_original_contribution_heatmaps.png", "Contribution matrices across six global water-cycle assessments"),
    ("FigS4", "matrix_output/first6_relavg_contribution_heatmaps.png", "Relative contribution matrices across six global water-cycle assessments"),
    ("FigS5", "matrix_output/first6_original_delivery_heatmaps.png", "Delivery matrices across six global water-cycle assessments"),
    ("FigS6", "matrix_output/first6_relavg_delivery_heatmaps.png", "Relative delivery matrices across six global water-cycle assessments"),
    ("FigS7", "centrality_output/FigS_delta_std_ranking_excluding_SS85.png", "Sensitivity of cross-network pathway variability to exclusion of SS-85"),
    ("FigS8", "network_plots/Various_datasets.png", "Dataset-based network structures used for alphaGW sensitivity experiments"),
    ("FigS9", "sensitivity_outputs/sensitivity_centrality_metrics.png", "Full dataset-based sensitivity of node-level network metrics to alphaGW"),
    ("FigS10", "sensitivity_outputs/sensitivity_contribution_matrix.png", "Dataset-based sensitivity of relative contribution matrices to alphaGW"),
    ("FigS11", "sensitivity_outputs/sensitivity_delivery_matrix.png", "Dataset-based sensitivity of relative delivery matrices to alphaGW"),
    ("FigS12", "sensitivity_outputs/sensitivity_delta_matrix.png", "Dataset-based sensitivity of the Delta matrix to alphaGW"),
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


def clean_output_directory(output_root: Path, input_file: Path) -> None:
    """Remove one explicitly requested generated-output tree after safety checks."""
    resolved = output_root.resolve()
    protected = {
        Path(resolved.anchor).resolve(),
        Path.cwd().resolve(),
        input_file.resolve().parent,
    }
    if resolved in protected:
        raise ValueError(f"Refusing to clean protected path: {resolved}")
    if resolved.exists():
        # Network shares can report directories as non-empty for a short time
        # after their children have been removed. Retry a bottom-up cleanup
        # instead of leaving a partially deleted output tree.
        for attempt in range(6):
            try:
                shutil.rmtree(resolved)
            except OSError:
                if resolved.exists():
                    paths = sorted(
                        resolved.rglob("*"),
                        key=lambda path: len(path.parts),
                        reverse=True,
                    )
                    for path in paths:
                        try:
                            if path.is_file() or path.is_symlink():
                                path.unlink(missing_ok=True)
                            elif path.is_dir():
                                path.rmdir()
                        except OSError:
                            pass
                    try:
                        resolved.rmdir()
                    except OSError:
                        pass
            if not resolved.exists():
                break
            time.sleep(0.25 * (attempt + 1))
        if resolved.exists():
            raise OSError(f"Could not fully clean generated-output tree: {resolved}")
        print("Removed previous generated-output tree:", resolved)


def remove_large_figure_exports(output_root: Path) -> int:
    """Remove reproducible TIFF exports before writing a compact repository manifest."""
    removed = 0
    for pattern in ("*.tif", "*.tiff"):
        for path in output_root.rglob(pattern):
            if path.is_file():
                path.unlink()
                removed += 1
    print(f"Removed {removed} reproducible TIFF export(s) from the compact snapshot.")
    return removed


def organize_supplementary_figures(
    output_root: Path,
    figs1_source: Path,
) -> Path:
    """Collect manuscript-ordered supplementary PNGs and write their provenance."""
    destination_dir = output_root / "supplementary_figures"
    destination_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for figure, relative_source, content in SUPPLEMENTARY_FIGURES:
        source = figs1_source if relative_source is None else output_root / relative_source
        if not source.is_file():
            raise FileNotFoundError(f"Required source for {figure} not found: {source}")

        destination = destination_dir / f"{figure}.png"
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        try:
            source_label = source.resolve().relative_to(output_root.resolve()).as_posix()
        except ValueError:
            source_label = portable_manifest_path(source)
        rows.append(
            {
                "figure": figure,
                "file": destination.name,
                "source": source_label,
                "content": content,
            }
        )

    order_path = destination_dir / "figure_order.tsv"
    with order_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("figure", "file", "source", "content"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Supplementary figures collected in manuscript order:", destination_dir)
    print("Supplementary-figure order manifest:", order_path)
    return order_path


def write_figure_inventory(output_root: Path, include_supplementary: bool) -> Path:
    """Write one compact index covering all manuscript figure deliverables."""
    rows: list[dict[str, str]] = []
    for number in range(1, 5):
        path = output_root / "main_figures" / f"Fig{number}.png"
        if path.is_file():
            rows.append(
                {
                    "section": "main",
                    "figure": f"Fig{number}",
                    "file": path.relative_to(output_root).as_posix(),
                }
            )

    if include_supplementary:
        for figure, _relative_source, _content in SUPPLEMENTARY_FIGURES:
            path = output_root / "supplementary_figures" / f"{figure}.png"
            if path.is_file():
                rows.append(
                    {
                        "section": "supplementary",
                        "figure": figure,
                        "file": path.relative_to(output_root).as_posix(),
                    }
                )

    inventory_path = output_root / "figure_inventory.tsv"
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("section", "figure", "file"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    print("Figure inventory:", inventory_path)
    return inventory_path


def write_manifest(
    output_root: Path,
    input_file: Path,
    rgw0_file: Path | None,
    figs1_source: Path | None,
    args,
) -> None:
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
        "skip_fig4": args.skip_fig4,
        "clean_output": args.clean_output,
        "compact_output": args.compact_output,
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
        "supplementary_fig1_source": None
        if figs1_source is None
        else {
            "path": portable_manifest_path(figs1_source),
            "size_bytes": figs1_source.stat().st_size,
            "sha256": sha256_file(figs1_source),
        },
        "package_versions": package_versions(),
        "outputs": records,
    }
    (output_root / "reproduction_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def validate_inputs(
    input_file: Path,
    rgw0_file: Path | None,
    figs1_source: Path | None,
    steps: list[str],
    skip_fig4: bool,
) -> None:
    if not input_file.is_file():
        raise FileNotFoundError(f"Input workbook not found: {input_file}")
    if input_file.suffix.lower() not in {".xlsx", ".xlsm"}:
        raise ValueError(f"Input workbook must be .xlsx or .xlsm: {input_file}")

    if "main" in steps and not skip_fig4:
        if rgw0_file is None:
            raise ValueError("--rgw0 is required for Fig. 4. Use --skip-fig4 to omit Fig. 4.")
        if not rgw0_file.is_file():
            raise FileNotFoundError(f"Rgw0 GeoPackage not found: {rgw0_file}")

    if "supplementary" in steps:
        if figs1_source is None or not figs1_source.is_file():
            raise FileNotFoundError(
                "Supplementary Fig. S1 source not found. "
                "Provide --figs1-source or restore source_figures/FigS1_relation_matrices.png."
            )


def run_pipeline(
    input_file: Path,
    output_root: Path,
    rgw0_file: Path | None,
    figs1_source: Path | None,
    steps: list[str],
    skip_fig4: bool = False,
) -> None:
    """Create the workflow namespaces and execute the selected stages."""
    output_root.mkdir(parents=True, exist_ok=True)

    # Main Figs. 1 and 2 require the matrix, network and centrality products.
    expanded_steps = set(steps)
    if "main" in expanded_steps:
        expanded_steps.update({"network", "matrix", "centrality"})
    if "supplementary" in expanded_steps:
        expanded_steps.update({"network", "matrix", "centrality", "sensitivity"})

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

    mainfig = None
    if "main" in expanded_steps or "supplementary" in expanded_steps:
        mainfig = create_main_figures_workflow(
            input_file=input_file,
            output_root=output_root,
            rgw0_file=rgw0_file,
            sensitivity_workflow=sensitivity,
            network_graph_workflow=network,
        )

    if "main" in expanded_steps:
        print("\n=== Main manuscript figures ===")
        mainfig.run_revised_FIG1()
        mainfig.run_revised_FIG2()

        # Fig. 3 combines routing-sensitivity and variance-attribution panels.
        long_df, _ = mainfig.build_sensitivity_long_dataframe(sensitivity, workbook=input_file)
        mainfig.run_revised_FIG3(long_df=long_df.copy(), workflow=sensitivity)

        if not skip_fig4:
            mainfig.run_revised_FIG4()
        else:
            print("Fig. 4 skipped by request.")

        order_path = mainfig.write_main_figure_order_manifest(include_fig4=not skip_fig4)
        print("Main-figure order manifest:", order_path)

    if "supplementary" in expanded_steps:
        print("\n=== Supplementary manuscript figures ===")
        if "main" not in expanded_steps:
            mainfig.run_supplementary_delta_ranking()
        organize_supplementary_figures(output_root, figs1_source)

    if "main" in expanded_steps or "supplementary" in expanded_steps:
        write_figure_inventory(
            output_root,
            include_supplementary="supplementary" in expanded_steps,
        )


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
        help="Path to Rgw0_data.gpkg, required for main Fig. 4 unless --skip-fig4 is used.",
    )
    parser.add_argument(
        "--figs1-source",
        type=Path,
        default=Path("source_figures/FigS1_relation_matrices.png"),
        help="Static source image for Supplementary Fig. S1.",
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
        help="Comma-separated stages: network,matrix,centrality,sensitivity,main,supplementary; default: all.",
    )
    parser.add_argument(
        "--skip-fig4",
        "--skip-fig1",
        dest="skip_fig4",
        action="store_true",
        help="Skip map/distribution Fig. 4, allowing execution without the Rgw0 GeoPackage. "
        "The deprecated --skip-fig1 spelling remains accepted for compatibility.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete the selected generated-output directory before running. "
        "Safety checks prevent cleaning the repository root or input directory.",
    )
    parser.add_argument(
        "--compact-output",
        action="store_true",
        help="Remove reproducible TIFF exports after figure generation and before manifesting. "
        "PNG figures and all analysis tables are retained.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_file = args.input.expanduser().resolve()
    rgw0_file = args.rgw0.expanduser().resolve() if args.rgw0 is not None else None
    figs1_source = args.figs1_source.expanduser().resolve() if args.figs1_source is not None else None
    output_root = args.output.expanduser().resolve()

    try:
        validate_inputs(input_file, rgw0_file, figs1_source, args.steps, args.skip_fig4)
        if args.clean_output:
            clean_output_directory(output_root, input_file)
        started = time.perf_counter()
        run_pipeline(
            input_file=input_file,
            output_root=output_root,
            rgw0_file=rgw0_file,
            figs1_source=figs1_source,
            steps=args.steps,
            skip_fig4=args.skip_fig4,
        )
        if args.compact_output:
            remove_large_figure_exports(output_root)
        write_manifest(output_root, input_file, rgw0_file, figs1_source, args)
        elapsed = time.perf_counter() - started
        print(f"\nReproduction completed in {elapsed / 60:.1f} minutes.")
        print("Manifest:", output_root / "reproduction_manifest.json")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
