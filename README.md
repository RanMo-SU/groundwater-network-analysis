# Global terrestrial water-cycle network analysis

Reproducibility resources for a network-based analysis of the global terrestrial water cycle, prepared to accompany a manuscript submitted to *Nature Water*.

This repository contains the source data, Python workflows, and compact reference outputs needed to reproduce the network diagrams, contribution/delivery/delta matrices, centrality analyses, groundwater-routing sensitivity analyses, four revised main figures, and twelve supplementary figures in manuscript order.

## Repository contents

| Path | Description |
| --- | --- |
| `reproduce_all_outputs.py` | Command-line entry point for the complete or stage-specific workflow. |
| `water_cycle_workflows.py` | Integrated analysis and plotting implementations. |
| `All global networks.xlsx` | Relation matrices in the `Network matrices` worksheet, with supporting sheets. |
| `Rgw0_data.gpkg` | Catchment groundwater-contribution data used for main Fig. 4. |
| `Rgw0_data.txt` | Plain-text groundwater-contribution table supplied for inspection and reuse. |
| `source_figures/FigS1_relation_matrices.png` | Static source image for Supplementary Fig. S1; its numerical matrices are also contained in the workbook. |
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
  --output reproduced_outputs_fresh \
  --clean-output
```

The default run executes all analysis stages, generates main Figs. 1-4 and Supplementary Figs. S1-S12, collects the supplementary PNGs in manuscript order, and writes checksummed manifests. `--clean-output` applies only to the selected generated-output directory; safety checks prevent cleaning the repository root or input directory. Add `--compact-output` to remove reproducible TIFF exports before writing the manifest while retaining PNG figures and analysis tables.

## Run selected stages

Available stages are `network`, `matrix`, `centrality`, `sensitivity`, `main`, and `supplementary`.

```bash
python reproduce_all_outputs.py \
  --input "All global networks.xlsx" \
  --output reproduced_outputs_fresh \
  --steps network,matrix,centrality,sensitivity
```

To generate main Figs. 1-3 without the GeoPackage-dependent map and distribution in Fig. 4:

```bash
python reproduce_all_outputs.py \
  --input "All global networks.xlsx" \
  --output reproduced_outputs_fresh \
  --steps main \
  --skip-fig4
```

The `main` stage automatically runs the network, matrix, and centrality prerequisites used by Figs. 1 and 2.
The `supplementary` stage automatically runs the network, matrix, centrality, and sensitivity prerequisites.

## Manuscript figure order

The `main` stage writes the revised figures in manuscript order to `main_figures/`:

| Figure | Panels | Content |
| --- | --- | --- |
| Fig. 1 | a-c | Alternative global networks, centrality variability, and metric profiles. |
| Fig. 2 | a-g | Delta matrices for six groundwater-contribution assessments and ranked pathway uncertainty. |
| Fig. 3 | a-d | Groundwater-routing structure, node sensitivity, and node/pathway variance attribution. |
| Fig. 4 | a-b | Catchment-scale groundwater contribution map and cross-scale observational constraint. Fig. 4b identifies AD-23 as the area-weighted aggregation of the catchment distribution and keeps the independent estimate `0.59 +/- 0.07` visually separate. |

The workflow collects the final supplementary PNGs in `supplementary_figures/` and writes `supplementary_figures/figure_order.tsv` with source provenance.

| Supplementary figure | Generated from | Content |
| --- | --- | --- |
| Fig. S1 | `source_figures/FigS1_relation_matrices.png` | Relation matrices for six global water-cycle assessments. |
| Fig. S2 | `centrality_output/radar_bar_overall.png` | Complete node-level centrality profiles. |
| Fig. S3 | `matrix_output/first6_original_contribution_heatmaps.png` | Contribution matrices. |
| Fig. S4 | `matrix_output/first6_relavg_contribution_heatmaps.png` | Relative contribution matrices. |
| Fig. S5 | `matrix_output/first6_original_delivery_heatmaps.png` | Delivery matrices. |
| Fig. S6 | `matrix_output/first6_relavg_delivery_heatmaps.png` | Relative delivery matrices. |
| Fig. S7 | `centrality_output/FigS_delta_std_ranking_excluding_SS85.png` | Pathway variability after excluding SS-85. |
| Fig. S8 | `network_plots/Various_datasets.png` | Dataset-based network structures used for groundwater-routing sensitivity experiments. |
| Fig. S9 | `sensitivity_outputs/sensitivity_centrality_metrics.png` | Node-level metric sensitivity to groundwater routing. |
| Fig. S10 | `sensitivity_outputs/sensitivity_contribution_matrix.png` | Relative-contribution sensitivity. |
| Fig. S11 | `sensitivity_outputs/sensitivity_delivery_matrix.png` | Relative-delivery sensitivity. |
| Fig. S12 | `sensitivity_outputs/sensitivity_delta_matrix.png` | Delta-matrix sensitivity. |

## Output structure

```text
reproduced_outputs_fresh/
|-- All global networks-results-1.xlsx
|-- All global networks-results-2.xlsx
|-- network_plots/
|-- matrix_output/
|-- centrality_output/
|-- sensitivity_outputs/
|-- main_figures/
|   |-- Fig1.png
|   |-- Fig2.png
|   |-- Fig3.png
|   |-- Fig4.png
|   `-- figure_order.tsv
|-- supplementary_figures/
|   |-- FigS1.png
|   |-- ...
|   |-- FigS12.png
|   `-- figure_order.tsv
|-- figure_inventory.tsv
|-- reproduction_manifest.json
`-- reproduction_manifest.tsv
```

The stage-specific directories retain the analysis tables and source plots used to assemble the manuscript figures. The JSON manifest records the command, execution environment, input SHA-256 checksums, package versions, and output checksums; the TSV manifest provides a compact file inventory.

## Reproducibility notes

- The scripts do not modify the source workbook or GeoPackage.
- The static Fig. S1 source is copied without modification; its numerical matrices remain available in the workbook.
- Workbook layout, worksheet names, and node labels must remain consistent with the supplied publication data.
- The six assessments use the uniform order `SS-85, Ab-19, OK-06, Wr-13, Tr-14, AD-23`.
- Matrix calculations are deterministic; no random seed is required.
- Figures request Arial. If Arial is unavailable, the scripts use compatible sans-serif fallbacks, so small typographic differences may occur.
- Runtime depends on the system and can be several minutes because the supplementary sensitivity figures contain many panels.
- The committed `reproduced_outputs/` directory is a compact reference snapshot. Run the workflow in a fresh directory to create a complete local set including high-resolution TIFF files and a new provenance manifest.

## Verification performed

The complete reorganized workflow was executed with the supplied workbook, GeoPackage, and Fig. S1 source on 27 October 2026. Main Figs. 1-4, Supplementary Figs. S1-S12, compact data tables, and provenance manifests were regenerated and inspected.

## Citation

When using these resources, please cite the associated manuscript and the archived release of this repository. The formal article citation and repository DOI can be added here when they become available.
