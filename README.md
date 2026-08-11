# Global terrestrial water-cycle network analysis 

Reproducibility resources for a network-based analysis of the global terrestrial water cycle, prepared to accompany a manuscript submitted to *Nature Water*.

This repository contains the source data, Python workflows, and compact reference outputs needed to reproduce the network diagrams, contribution/delivery/delta matrices, centrality analyses, groundwater-routing sensitivity analyses.

## Repository contents

| Path | Description |
| --- | --- |
| `reproduce_all_outputs.py` | Command-line entry point for the complete or stage-specific workflow. |
| `water_cycle_workflows.py` | Integrated analysis and plotting implementations. |
| `All global networks.xlsx` | Relation matrices in the `Network matrices` worksheet, with supporting sheets. |
| `Rgw0_data.gpkg` | Catchment groundwater-contribution data used to map main Fig. 1. |
| `Rgw0_data.txt` | Plain-text groundwater-contribution table supplied for inspection and reuse. |
| `requirements.txt` | Tested Python package versions. |
| `reproduced_outputs/` | Compact reference workbooks, tables, manifests, and PNG figures. |


## Quick start

Python 3.10 or later is required.

```bash
git clone https://github.com/RanMo-SU/groundwater-network-analysis.git
cd groundwater-network-analysis
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the complete workflow from the repository root:

```bash
python reproduce_all_outputs.py \
  --input "All global networks.xlsx" \
  --rgw0 "Rgw0_data.gpkg" \
  --output reproduced_outputs_fresh
```

Using a new or empty output directory is recommended. The generated manifest inventories all files present under the selected output directory.

## Run selected stages

Available stages are `network`, `matrix`, `centrality`, `sensitivity`, and `main`.

```bash
python reproduce_all_outputs.py \
  --input "All global networks.xlsx" \
  --output reproduced_outputs_fresh \
  --steps network,matrix,centrality,sensitivity
```

To generate main Figs. 2-4 without the GeoPackage-dependent catchment map in Fig. 1:

```bash
python reproduce_all_outputs.py \
  --input "All global networks.xlsx" \
  --output reproduced_outputs_fresh \
  --steps main \
  --skip-fig1
```

The `main` stage automatically runs the network, matrix, and centrality prerequisites used by Figs. 2 and 3.

## Output structure

```text
reproduced_outputs_fresh/
|-- All global networks-results-1.xlsx
|-- All global networks-results-2.xlsx
|-- network_plots/
|-- matrix_output/
|-- centrality_output/
|-- sensitivity_outputs/
|-- reproduction_manifest.json
`-- reproduction_manifest.tsv
```

The output directories contain analysis tables plus PNG and TIFF figure exports. The JSON manifest records the command, execution environment, input SHA-256 checksums, package versions, and output checksums; the TSV manifest provides a compact file inventory.

## Reproducibility notes

- The scripts do not modify the source workbook or GeoPackage.
- Workbook layout, worksheet names, and node labels must remain consistent with the supplied publication data.
- Matrix calculations are deterministic; no random seed is required.
- Figures request Arial. If Arial is unavailable, the scripts use compatible sans-serif fallbacks, so small typographic differences may occur.
- Runtime depends on the system and can be several minutes because the supplementary sensitivity figures contain many panels.
- The committed `reproduced_outputs/` directory is a compact reference snapshot. Run the workflow in a fresh directory to create a complete local set including high-resolution TIFF files and a new provenance manifest.

## Verification performed

The complete workflow was executed successfully with the supplied workbook and GeoPackage on 11 August 2026. All analysis stages, compact tables, high-resolution exports, the provenance manifest, and main Figs. 1-4 were regenerated in approximately 2.5 minutes on the test system.

## Citation

When using these resources, please cite the associated manuscript and the archived release of this repository. The formal article citation and repository DOI can be added here when they become available.
