"""Integrated computational workflows for the global terrestrial water-cycle network study.

This module consolidates the original network-graph, fraction-matrix, centrality,
sensitivity-analysis and main-figure scripts into reproducible workflow factories.
It contains no input data. Use ``reproduce_all_outputs.py`` as the command-line entry point.

Python requirement: >= 3.10.
"""

from __future__ import annotations

import types
from pathlib import Path

__all__ = [
    "create_network_graph_workflow",
    "create_fraction_matrix_workflow",
    "create_centrality_workflow",
    "create_sensitivity_workflow",
    "create_main_figures_workflow",
]


def create_network_graph_workflow(input_file: str | Path, output_root: str | Path):
    """Create the integrated network-graph workflow namespace."""
    import re
    from pathlib import Path
    from typing import Dict, Iterable, List, Optional, Tuple
    from collections import defaultdict

    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyArrowPatch, Ellipse
    from matplotlib.text import Text
    import numpy as np
    import pandas as pd


    # =========================================================
    # GLOBAL FIGURE FONT CONFIGURATION
    # =========================================================
    # Apply Arial to ordinary text and to Matplotlib math-text notation.
    FIGURE_FONT_FAMILY = "Arial"

    plt.rcParams.update({
        "font.family": FIGURE_FONT_FAMILY,
        "font.sans-serif": [FIGURE_FONT_FAMILY, "Arimo", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": FIGURE_FONT_FAMILY,
        "mathtext.it": f"{FIGURE_FONT_FAMILY}:italic",
        "mathtext.bf": f"{FIGURE_FONT_FAMILY}:bold",
        "mathtext.sf": FIGURE_FONT_FAMILY,
        "mathtext.tt": FIGURE_FONT_FAMILY,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


    def _apply_arial_font(fig) -> None:
        """Force Arial on every Matplotlib Text artist before export."""
        for text_artist in fig.findobj(match=Text):
            text_artist.set_fontfamily(FIGURE_FONT_FAMILY)


    # ==================================================
    # CONFIGURATION
    # ==================================================
    INPUT_FILE = Path(input_file)
    INPUT_SHEET = "Network matrices"
    OUTPUT_DIR = Path(output_root) / "network_plots"
    OUTPUT_DIR.mkdir(exist_ok=True)

    TITLE_KEYWORD = "RELATION MATRIX"
    N_NODES = 8
    NODE_ROW_OFFSET = 3
    DATA_ROW_OFFSET = 4
    KEEP_FIRST_N_NETWORKS = 6
    REFERENCE_ASSESSMENT_ORDER = [
        "SS-85", "Ab-19", "OK-06", "Wr-13", "Tr-14", "AD-23"
    ]

    CLEAR_OLD_NETWORK_PLOTS = True
    LABEL_EDGES = True
    EDGE_VALUE_DECIMALS = 1

    # Export controls
    CREATE_INDIVIDUAL_FIGURES = False
    CREATE_REFERENCE_PANEL = True
    CREATE_DATASET_PANEL = True

    SHOW_TITLE = True
    SHOW_SUMMARY = False

    # White-background publication style
    BACKGROUND = "#FFFFFF"
    TITLE_TEXT = "#111111"
    SUBTLE_TEXT = "#333333"
    NODE_TEXT = "#111111"

    # Same naming convention as the centrality script.
    # The workbook order is assumed to be: _0.5, _0.0, _1.0 for each dataset family.
    NETWORK_SUFFIX_BY_INDEX = {
        1: "0.0",
        0: "0.5",
        2: "1.0",
    }
    DATASET_FAMILIES = ["Obs", "Mixed-GLEAM", "GLDAS", "ERA5"]

    # ==================================================
    # Node and edge layout
    # ==================================================
    # Coordinates use a 0-1 plotting coordinate system.
    NODE_POS = {
        "SM":     (0.15, 0.65),
        "VEG":    (0.05, 0.28),
        "GW":     (0.34, 0.09),
        "SP/ENG": (0.45, 0.84),
        "SW":     (0.61, 0.31),
        "SEA":    (0.84, 0.09),
        "GL":     (0.93, 0.43),
        "Aw":     (0.74, 0.69),
    }

    PREFERRED_NODE_ORDER = ["Aw", "SM", "VEG", "GW", "SW", "GL", "SP/ENG", "SEA"]

    # Node colors. Link colors are taken from the source/input node edge color.
    NODE_STYLE = {
        "Aw":     dict(edge="#1D9BF0", fill="#E0F2FE"),
        "SM":     dict(edge="#B45309", fill="#FFF7ED"),
        "VEG":    dict(edge="#4C9A2A", fill="#EAF7E3"),
        "GW":     dict(edge="#D62728", fill="#FEE2E2"),
        "SW":     dict(edge="#2563EB", fill="#DBEAFE"),
        "GL":     dict(edge="#6B7280", fill="#F3F4F6"),
        "SP/ENG": dict(edge="#D946EF", fill="#FAE8FF"),
        "SEA":    dict(edge="#551DD8", fill="#EDE9FE"),
    }

    # Curvature of each pathway. Keep this block independent from label positions.
    EDGE_RAD = {
        ("Aw", "SM"): 0.10,
        ("VEG", "Aw"): -0.28,
        ("SEA", "Aw"): -0.10,

        ("SM", "GW"): -0.10,
        ("SM", "SW"): -0.28,

        ("GW", "SW"): 0.26,

        ("SW", "Aw"): 0.20,
        ("SW", "SP/ENG"): 0.30,

        ("SP/ENG", "SW"): 0.22,
        ("SP/ENG", "SEA"): -0.28,
    }

    # --------------------------------------------------
    # Manual solid/dashed styling for the six reference networks.
    # These entries reproduce the Mixed-GLEAM solid/dashed convention in the
    # original hand-drawn/reference diagrams while retaining the cleaner
    # publication layout used by this script.
    #
    # Edit this dictionary if a specific reference pathway should be changed.
    # Keys must match the cleaned sheet names generated by clean_title().
    # --------------------------------------------------
    DASHED_EDGES_BY_NETWORK = {
        "SS-85": {
            ("SP/ENG", "SM"),
            ("SP/ENG", "SW"),
            ("Aw", "SP/ENG"),
            ("SEA", "Aw"),
        },
        "Ab-19": {
            ("SP/ENG", "SM"),
            ("Aw", "SP/ENG"),
            ("SEA", "SP/ENG"),
            ("GW", "SP/ENG"),
            ("SM", "VEG"),
            ("SM", "SW"),
            ("Aw", "GL"),
        },
        "OK-06": {
            ("SM", "VEG"),
            ("SM", "GW"),
        },
        "Tr-14":{
            ("SM", "VEG"),
        },
        "Wr-13": {
            ("SM", "VEG"),
            ("SM", "GW"),
        },
        "AD-23": {
            ("SM", "GW"),
            ("SM", "SW"),
            ("SM", "VEG"),
            ("SEA", "Aw"),
        },
    }

    # --------------------------------------------------
    # Manual solid/dashed styling for the dataset-comparison panel.
    # Keys must match the display names used in build_dataset_panel_items().
    # These initial settings follow the old dataset-comparison figure as closely
    # as possible while using the same cleaner layout as the new script.
    # --------------------------------------------------
    DASHED_EDGES_BY_DATASET = {
        "Obs": {
            ("SP/ENG", "SM"),
            ("SM", "SP/ENG"),
            ("SP/ENG", "Aw"),
            ("Aw", "SP/ENG"),
            ("SEA", "Aw"),
            ("GW", "SW"),
            ("SM", "GW"),
            ("SM", "SW"),
            ("SM", "VEG"),
        },
        "Mixed-GLEAM": {
            ("SP/ENG", "SM"),
            ("SM", "SP/ENG"),
            ("SP/ENG", "Aw"),
            ("Aw", "SP/ENG"),
            ("SEA", "Aw"),
            ("GW", "SW"),
            ("SM", "GW"),
            ("SM", "SW"),
            ("SM", "VEG"),
        },
        "GLDAS": {
            ("SP/ENG", "SM"),
            ("SM", "SP/ENG"),
            ("SP/ENG", "Aw"),
            ("Aw", "SP/ENG"),
            ("SEA", "Aw"),
            ("GW", "SW"),
            ("SM", "GW"),
            ("SM", "SW"),
            ("SM", "VEG"),
        },
        "ERA5": {
            ("SP/ENG", "SM"),
            ("SM", "SP/ENG"),
            ("SP/ENG", "Aw"),
            ("Aw", "SP/ENG"),
            ("SEA", "Aw"),
            ("GW", "SW"),
            ("SM", "GW"),
            ("SM", "SW"),
            ("SM", "VEG"),
        },
    }

    # Matplotlib dash pattern: on/off lengths in points.
    DASHED_LINESTYLE = (0, (4.0, 2.4))

    # ==================================================
    # Manual edge-label placement
    # ==================================================
    # No automatic collision detection or Bezier-path label search is used.
    # Every edge label is placed according to this dictionary.
    # Coordinates use the same 0-1 plotting coordinate system as NODE_POS.
    # Edit these numbers only when you want to move labels manually.
    REQUIRE_MANUAL_LABEL_POSITIONS = False

    MANUAL_LABEL_POS = {
        # Atmosphere / land-surface links
        ("Aw", "SM"):         (0.43, 0.71),
        ("VEG", "Aw"):        (0.32, 0.45),
        ("SEA", "Aw"):        (0.77, 0.40),
        ("Aw", "GL"):         (0.83, 0.55),
        ("GL", "SEA"):        (0.90, 0.25),

        # SM-related links
        ("SM", "VEG"):        (0.10, 0.46),
        ("SM", "GW"):         (0.27, 0.36),
        ("SM", "SW"):         (0.50, 0.53),
        ("SM", "SP/ENG"):     (0.61, 0.75),

        # Groundwater / surface-water / sea links
        ("GW", "SW"):         (0.47, 0.22),
        ("GW", "SEA"):        (0.59, 0.10),
        ("SW", "Aw"):         (0.66, 0.49),
        ("SW", "SEA"):        (0.72, 0.20),
        ("SW", "SP/ENG"):     (0.58, 0.58),

        # SP/ENG-related links
        ("SP/ENG", "SM"):     (0.30, 0.75),
        ("SP/ENG", "SW"):     (0.52, 0.61),
        ("SP/ENG", "VEG"):    (0.30, 0.61),
        ("GW", "SP/ENG"):     (0.43, 0.43),
        ("SEA", "SP/ENG"):    (0.67, 0.45),
        ("SP/ENG", "Aw"):     (0.31, 0.79),
        ("Aw", "SP/ENG"):     (0.61, 0.75),
    }

    # Optional manual label rotation. Keep 0 for horizontal labels.
    # Positive values rotate counter-clockwise. Values are degrees.
    MANUAL_LABEL_ROTATION = {
        ("Aw", "SM"):        0,
        ("VEG", "Aw"):       0,
        ("SEA", "Aw"):       0,
        ("Aw", "GL"):        0,
        ("GL", "SEA"):       0,
        ("Aw", "SP/ENG"):    0,

        ("SM", "VEG"):       0,
        ("SM", "GW"):        0,
        ("SM", "SW"):        0,
        ("SM", "SP/ENG"):    0,

        ("GW", "SW"):        0,
        ("GW", "SEA"):       0,
        ("SW", "Aw"):        0,
        ("SW", "SEA"):       0,
        ("SW", "SP/ENG"):    0,

        ("SP/ENG", "SM"):    0,
        ("SP/ENG", "SW"):    0,
        ("SP/ENG", "SEA"):   0,
    }

    LABEL_BOX_ALPHA = 0.6


    # ==================================================
    # Basic utilities
    # ==================================================
    def save_figure_both(fig, output_path, dpi=300, **savefig_kwargs):
        """
        Save the same figure as both TIFF and PNG with the same dpi.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure object.
        output_path : str or Path
            Usually a .tiff path. The PNG path will use the same stem.
        dpi : int
            Export dpi for both formats.
        **savefig_kwargs
            Extra keyword arguments passed to fig.savefig().
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = output_path.suffix.lower()
        if suffix not in [".tif", ".tiff", ".png"]:
            tiff_path = output_path.with_suffix(".tiff")
        elif suffix == ".png":
            tiff_path = output_path.with_suffix(".tiff")
        else:
            tiff_path = output_path

        png_path = tiff_path.with_suffix(".png")

        _apply_arial_font(fig)

        fig.savefig(tiff_path, dpi=dpi, **savefig_kwargs)
        fig.savefig(png_path, dpi=dpi, **savefig_kwargs)

        return tiff_path, png_path


    def ordered_nodes(nodes: Iterable[str]) -> List[str]:
        nodes = [
            str(n).strip()
            for n in nodes
            if str(n).strip() and str(n).strip().lower() != "nan"
        ]
        known = [n for n in PREFERRED_NODE_ORDER if n in nodes]
        rest = [n for n in nodes if n not in known]
        return known + rest


    def clean_title(title: str) -> str:
        """
        Use a compact network-name convention.

        Examples
        --------
        RELATION MATRIX – SS 85                 -> SS-85
        RELATION MATRIX – Abbot 19              -> Ab-19
        RELATION MATRIX – Althoff&Destouni 23   -> AD-23
        """
        title = str(title).strip()
        title = re.sub(r"RELATION\s+MATRIX\s*[–—-]?\s*", "", title, flags=re.I)
        title = re.sub(r"matrix", "", title, flags=re.I)
        title = re.sub(r"[^A-Za-z0-9 _-]+", " ", title)
        title = re.sub(r"\s+", "_", title).strip("_")
        return title or "Network"


    def safe_filename(text: str) -> str:
        text = str(text).strip().replace("&", "_and_")
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text


    def extract_matrix_family(title_text: str) -> str:
        """
        Detect the dataset/family name from the workbook title.
        """
        t = str(title_text).strip()

        for family in DATASET_FAMILIES:
            if re.search(rf"\b{re.escape(family)}\b", t, flags=re.IGNORECASE):
                return family

        cleaned = re.sub(r"RELATION\s+MATRIX", "", t, flags=re.IGNORECASE)
        cleaned = re.sub(r"matrix", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[^A-Za-z0-9 _-]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned if cleaned else "Network"


    def assign_case_names_from_blocks(blocks: List[dict]) -> List[dict]:
        """
        Assign consistent case names such as:
            Obs_0.5, Obs_0.0, Obs_1.0,
            Mixed-GLEAM_0.5, ...
        while preserving non-dataset names for the reference networks.
        """
        counters = defaultdict(int)

        for block in blocks:
            family = extract_matrix_family(block["title"])
            idx = counters[family]
            counters[family] += 1

            if family in DATASET_FAMILIES:
                suffix = NETWORK_SUFFIX_BY_INDEX.get(idx, str(idx))
                block["case_name"] = f"{family}_{suffix}"
            else:
                block["case_name"] = clean_title(block["title"])

        return blocks


    def format_edge_value(value: float) -> str:
        if abs(value) >= 100:
            return f"{value:.0f}"
        if abs(value) >= 10:
            return f"{value:.1f}".rstrip("0").rstrip(".")
        return f"{value:.{EDGE_VALUE_DECIMALS}f}".rstrip("0").rstrip(".")


    # ==================================================
    # Workbook reading
    # ==================================================
    def find_relation_matrix_blocks(raw: pd.DataFrame) -> List[dict]:
        blocks = []
        seen = set()

        for r in range(raw.shape[0]):
            for c in range(raw.shape[1]):
                val = str(raw.iat[r, c]).strip()
                if TITLE_KEYWORD.upper() in val.upper():
                    start_row = r
                    start_col = max(c - 1, 0)
                    key = (start_row, start_col)
                    if key in seen:
                        continue
                    seen.add(key)
                    blocks.append({"start_row": start_row, "start_col": start_col, "title": val})

        return sorted(blocks, key=lambda x: (x["start_row"], x["start_col"]))


    def read_relation_matrix(raw: pd.DataFrame, block: dict) -> pd.DataFrame:
        start_row = block["start_row"]
        start_col = block["start_col"]

        node_row = start_row + NODE_ROW_OFFSET
        data_row = start_row + DATA_ROW_OFFSET
        col0 = start_col + 2
        col1 = col0 + N_NODES

        nodes = [str(x).strip() for x in raw.iloc[node_row, col0:col1].tolist()]
        matrix = raw.iloc[data_row:data_row + N_NODES, col0:col1].copy()
        matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        matrix.index = nodes
        matrix.columns = nodes

        nodes = ordered_nodes(nodes)
        return matrix.loc[nodes, nodes].astype(float)


    # ==================================================
    # Network construction
    # ==================================================
    def matrix_to_edges(matrix: pd.DataFrame) -> List[Tuple[str, str, float]]:
        edges = []

        for source in matrix.index:
            for target in matrix.columns:
                value = float(matrix.loc[source, target])
                if not np.isfinite(value) or value <= 0:
                    continue
                if str(source) == str(target):
                    continue
                if source not in NODE_POS or target not in NODE_POS:
                    continue
                edges.append((str(source), str(target), value))

        return edges


    def active_nodes_from_edges(edges: List[Tuple[str, str, float]]) -> List[str]:
        active = set()
        for source, target, _ in edges:
            active.add(source)
            active.add(target)
        return ordered_nodes(active)


    def edge_color(source: str, target: str) -> str:
        """The link color follows the input/source node color."""
        return NODE_STYLE.get(source, {"edge": "#333333"})["edge"]


    def edge_linestyle(case_name: str, source: str, target: str):
        """Return solid or dashed style for a pathway in a named network or dataset panel."""
        name = str(case_name)
        dashed_edges = DASHED_EDGES_BY_NETWORK.get(name, set())
        if not dashed_edges:
            dashed_edges = DASHED_EDGES_BY_DATASET.get(name, set())
        if (source, target) in dashed_edges:
            return DASHED_LINESTYLE
        return "-"


    # ==================================================
    # Dataset-panel label overrides
    # ==================================================
    def build_alpha_label_overrides_from_R(
        matrix: pd.DataFrame,
        source_node: str = "SM",
        sw_node: str = "SW",
        gw_node: str = "GW",
    ) -> Dict[Tuple[str, str], str]:
        """
        Build labels using the actual R value from SM -> SW.

        Labels:
            SM -> GW : alpha_GW * R_value
            GW   -> SW : alpha_GW * R_value
            SM -> SW : (1 - alpha_GW) * R_value
        """
        if source_node not in matrix.index or sw_node not in matrix.columns:
            raise ValueError(f"Cannot find R = {source_node} -> {sw_node} in the matrix.")

        R = 2 * float(matrix.loc[source_node, sw_node])

        if not np.isfinite(R):
            raise ValueError(f"R value is not finite: {source_node} -> {sw_node} = {R}")

        R_label = format_edge_value(R)

        return {
            (source_node, gw_node): rf"$\alpha_{{GW}}\times {R_label}$",
            (gw_node, sw_node): rf"$\alpha_{{GW}}\times {R_label}$",
            (source_node, sw_node): rf"$(1-\alpha_{{GW}})\times {R_label}$",
        }


    def build_dataset_panel_items(raw: pd.DataFrame, blocks: List[dict]) -> List[tuple]:
        """
        Build a 2x2 panel for:
            Obs_0.5, Mixed-GLEAM_0.5, GLDAS_0.5, ERA5_0.5

        Each panel uses the original _0.5 network structure. Only the labels of
        three pathways are replaced:
            SM -> GW : alpha_GW * R_value
            GW   -> SW : alpha_GW * R_value
            SM -> SW : (1 - alpha_GW) * R_value
        """
        wanted = [f"{family}_0.5" for family in DATASET_FAMILIES]
        items = []

        for case_name in wanted:
            hit = next((b for b in blocks if b.get("case_name") == case_name), None)
            if hit is None:
                raise ValueError(f"Cannot find the required matrix block for: {case_name}")

            matrix = read_relation_matrix(raw, hit)
            family = case_name.split("_")[0]
            label_overrides = build_alpha_label_overrides_from_R(matrix)

            R = 2 * float(matrix.loc["SM", "SW"])
            print(f"{case_name}: R = {R:.4f}")

            items.append((family, matrix, label_overrides))

        return items


    # ==================================================
    # Drawing functions
    # ==================================================
    def draw_arrow_only(
        ax,
        source: str,
        target: str,
        value: float,
        min_weight: float,
        max_weight: float,
        scale: float = 1.0,
        label_text: Optional[str] = None,
        linestyle="-",
    ):
        x1, y1 = NODE_POS[source]
        x2, y2 = NODE_POS[target]

        if max_weight > min_weight:
            norm = (value - min_weight) / (max_weight - min_weight)
        else:
            norm = 0.5

        lw = scale * (1.10 + 3.80 * np.sqrt(max(norm, 0.0)))
        mutation = scale * (12.0 + 6.0 * np.sqrt(max(norm, 0.0)))
        rad = EDGE_RAD.get((source, target), 0.08)
        color = edge_color(source, target)

        # The white underlay improves readability on white background.
        underlay = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=mutation * 1.02,
            linewidth=lw + 2.4 * scale,
            linestyle=linestyle,
            color="#FFFFFF",
            alpha=0.95,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=50 * scale,
            shrinkB=50 * scale,
            zorder=1,
        )
        ax.add_patch(underlay)

        patch = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=mutation,
            linewidth=lw,
            linestyle=linestyle,
            color=color,
            alpha=0.90,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=50 * scale,
            shrinkB=50 * scale,
            zorder=2,
        )
        ax.add_patch(patch)

        if label_text is None:
            label_text = format_edge_value(value)

        label_key = (source, target)

        if label_key not in MANUAL_LABEL_POS:
            msg = f"No manual label position defined for edge: {source} -> {target}"
            if REQUIRE_MANUAL_LABEL_POSITIONS:
                raise ValueError(msg)
            print("Warning:", msg)
            return None

        lx, ly = MANUAL_LABEL_POS[label_key]
        rotation = MANUAL_LABEL_ROTATION.get(label_key, 0)

        return {
            "x": lx,
            "y": ly,
            "text": label_text,
            "color": color,
            "rotation": rotation,
        }


    def draw_edge_labels(ax, label_items: List[dict], scale: float = 1.0):
        """
        Draw edge labels at manually specified positions.
        No automatic collision detection or path search is used.
        """
        if not LABEL_EDGES:
            return

        for item in label_items:
            if item is None:
                continue

            ax.text(
                item["x"],
                item["y"],
                item["text"],
                color=item["color"],
                fontsize=22 * scale,
                fontweight="bold",
                ha="center",
                va="center",
                rotation=item.get("rotation", 0),
                rotation_mode="anchor",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=LABEL_BOX_ALPHA,
                    boxstyle="round,pad=0.10",
                ),
                path_effects=[
                    pe.withStroke(
                        linewidth=2.2 * scale,
                        foreground="white",
                    )
                ],
                zorder=8,
                clip_on=False,
            )


    def draw_nodes(ax, nodes: List[str], scale: float = 1.0):
        for node in nodes:
            if node not in NODE_POS:
                continue

            x, y = NODE_POS[node]
            style = NODE_STYLE.get(node, dict(edge="#333333", fill="#F3F4F6"))

            width = 0.118 * 1.5
            height = 0.080 * 1.5

            if node == "SP/ENG":
                width = 0.165 * 1.5
            elif node == "VEG":
                width = 0.150 * 1.5
            elif node == "SM":
                width = 0.140 * 1.5
            if node in {"Aw", "SEA", "SW", "GW", "GL"}:
                width = 0.112 * 1.5

            node_patch = Ellipse(
                (x, y),
                width,
                height,
                facecolor=style["fill"],
                edgecolor=style["edge"],
                linewidth=2.3 * scale,
                zorder=5,
            )
            node_patch.set_path_effects([
                pe.SimplePatchShadow(offset=(1.6 * scale, -1.6 * scale), alpha=0.16),
                pe.Normal(),
            ])
            ax.add_patch(node_patch)

            ax.text(
                x,
                y,
                node,
                ha="center",
                va="center",
                fontsize=18 * scale,
                fontweight="bold",
                color=NODE_TEXT,
                zorder=7,
            )


    def draw_network_on_ax(
        ax,
        case_name: str,
        matrix: pd.DataFrame,
        panel_mode: bool = False,
        label_overrides: Optional[Dict[Tuple[str, str], str]] = None,
    ):
        edges = matrix_to_edges(matrix)
        nodes = active_nodes_from_edges(edges)
        weights = [value for _, _, value in edges]
        min_weight = min(weights) if weights else 0.0
        max_weight = max(weights) if weights else 1.0
        scale = 0.7 if panel_mode else 1.0

        ax.set_facecolor(BACKGROUND)
        ax.set_xlim(-0.075, 1.020)
        ax.set_ylim(0.020, 0.965)
        ax.set_aspect("equal")
        ax.axis("off")

        title_size = 15.0 if panel_mode else 20.0
        subtitle_size = 15.0 if panel_mode else 20.0

        if SHOW_TITLE:
            ax.text(
                0.10,
                0.80,
                case_name,
                fontsize=title_size,
                fontweight="bold",
                color=TITLE_TEXT,
                ha="center",
                va="top",
                transform=ax.transAxes,
            )

        if SHOW_SUMMARY:
            ax.text(
                0.50,
                0.945,
                f"{len(edges)} active links | total = {matrix.values.sum():.1f}",
                fontsize=subtitle_size,
                color=SUBTLE_TEXT,
                ha="center",
                va="top",
                transform=ax.transAxes,
            )

        label_items = []
        edges_sorted = sorted(edges, key=lambda e: abs(EDGE_RAD.get((e[0], e[1]), 0.0)))

        for source, target, value in edges_sorted:
            custom_label = None
            if label_overrides is not None:
                custom_label = label_overrides.get((source, target), None)

            link_style = edge_linestyle(case_name, source, target)

            label_item = draw_arrow_only(
                ax,
                source,
                target,
                value,
                min_weight,
                max_weight,
                scale=scale,
                label_text=custom_label,
                linestyle=link_style,
            )

            if label_item is not None:
                label_items.append(label_item)

        draw_edge_labels(ax, label_items, scale=scale)
        draw_nodes(ax, nodes, scale=scale)


    def draw_network(case_name: str, matrix: pd.DataFrame, output_stem: str) -> List[Path]:
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor(BACKGROUND)
        draw_network_on_ax(ax, case_name, matrix, panel_mode=False)

        saved_paths = []
        path = OUTPUT_DIR / f"{output_stem}.tiff"
        save_figure_both(fig, path, dpi=300, bbox_inches="tight", facecolor=BACKGROUND)
        saved_paths.append(path)
        plt.close(fig)

        return saved_paths


    def draw_panel(
        network_items: List[tuple],
        filename: str,
        nrows: int,
        ncols: int,
        figsize: tuple,
        panel_title: Optional[str] = None,
        alias_filenames: Optional[List[str]] = None,
    ) -> List[Path]:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        fig.patch.set_facecolor(BACKGROUND)

        axes = np.array(axes).reshape(-1)

        for ax, item in zip(axes, network_items):
            if len(item) == 2:
                case_name, matrix = item
                label_overrides = None
            else:
                case_name, matrix, label_overrides = item

            draw_network_on_ax(
                ax,
                case_name,
                matrix,
                panel_mode=True,
                label_overrides=label_overrides,
            )

        for ax in axes[len(network_items):]:
            ax.axis("off")
            ax.set_facecolor(BACKGROUND)

        if panel_title:
            fig.suptitle(panel_title, fontsize=24, fontweight="bold", x=0.1, y=0.9)

        fig.subplots_adjust(
            left=0.005,
            right=0.995,
            top=0.97,
            bottom=0.01,
            wspace=0.02,
            hspace=0.02,
        )

        saved_paths = []
        main_path = OUTPUT_DIR / filename
        save_figure_both(fig, main_path, dpi=300, bbox_inches="tight", facecolor=BACKGROUND)
        saved_paths.append(main_path)

        plt.close(fig)
        return saved_paths


    # ==================================================
    # Output management and workflow
    # ==================================================
    def clean_output_folder():
        if not CLEAR_OLD_NETWORK_PLOTS:
            return

        patterns = [
            "*_network.png",
            "*_network.tiff",
            "overall.png",
            "overall.tiff",
            "Fig 1.tiff",
            "Reference_networks.tiff",
            "Various_datasets.tiff",
        ]

        for pattern in patterns:
            for p in OUTPUT_DIR.glob(pattern):
                p.unlink(missing_ok=True)


    def main():
        if not INPUT_FILE.exists():
            raise FileNotFoundError(f"Cannot find input workbook: {INPUT_FILE.resolve()}")

        clean_output_folder()

        raw = pd.read_excel(INPUT_FILE, sheet_name=INPUT_SHEET, header=None, engine="openpyxl")
        blocks = find_relation_matrix_blocks(raw)

        if not blocks:
            raise ValueError(f"No relation matrix blocks found in {INPUT_FILE.name}")

        blocks = assign_case_names_from_blocks(blocks)

        saved_all = []

        # --------------------------------------------------
        # 1) Reference networks panel
        # --------------------------------------------------
        reference_blocks = blocks[:KEEP_FIRST_N_NETWORKS]

        if len(reference_blocks) < KEEP_FIRST_N_NETWORKS:
            raise ValueError(
                f"Expected at least {KEEP_FIRST_N_NETWORKS} reference networks, "
                f"found {len(reference_blocks)}."
            )

        reference_items = []
        for block in reference_blocks:
            case_name = clean_title(block["title"])
            matrix = read_relation_matrix(raw, block)
            reference_items.append((case_name, matrix))

            if CREATE_INDIVIDUAL_FIGURES:
                output_stem = f"{safe_filename(case_name)}_network"
                saved_all.extend(draw_network(case_name, matrix, output_stem))

        reference_by_name = dict(reference_items)
        missing_reference = [
            name for name in REFERENCE_ASSESSMENT_ORDER if name not in reference_by_name
        ]
        if missing_reference:
            raise ValueError(
                "Reference networks missing from the canonical assessment order: "
                f"{missing_reference}"
            )
        reference_items = [
            (name, reference_by_name[name]) for name in REFERENCE_ASSESSMENT_ORDER
        ]

        if CREATE_REFERENCE_PANEL:
            saved_all.extend(
                draw_panel(
                    reference_items,
                    filename="Reference_networks.tiff",
                    nrows=2,
                    ncols=3,
                    figsize=(16.5, 10),
                    panel_title="",
                )
            )

        # --------------------------------------------------
        # 2) Various datasets panel
        # --------------------------------------------------
        if CREATE_DATASET_PANEL:
            dataset_items = build_dataset_panel_items(raw, blocks)
            saved_all.extend(
                draw_panel(
                    dataset_items,
                    filename="Various_datasets.tiff",
                    nrows=2,
                    ncols=2,
                    figsize=(13.5, 10),
                    panel_title="",
                )
            )

        print("\nCreated files:")
        for p in saved_all:
            print("  ", p.resolve())

        print(f"\nFinished. Outputs are in: {OUTPUT_DIR.resolve()}")

    _public = {k: v for k, v in locals().items() if not k.startswith("__")}
    return types.SimpleNamespace(**_public)


def create_fraction_matrix_workflow(input_file: str | Path, output_root: str | Path):
    """Create the integrated contribution/delivery matrix workflow namespace."""
    """
    Integrated workflow for hydrological network matrix results and figures.

    This script combines the fraction-matrix workflow and the matrix-comparison/
    figure workflow into one file.

    Main change requested:
        Cp_percent and Cpp_percent are NOT row/column share matrices anymore.
        They are relative-average matrices:

            Cp_percent[i, j]  = Cf[i, j]  / mean_i_nonzero(Cf[:, j])
            Cpp_percent[i, j] = Df[i, j] / mean_j_nonzero(Df[i, :])

    Interpretation:
        100%  = average effective non-zero cumulative effect
        >100% = stronger-than-average effective cumulative effect
        <100% = weaker-than-average effective cumulative effect

    Notes:
        - The variable names Cp_percent and Cpp_percent are kept for compatibility
          with the previous workflow and downstream plotting code.
        - Delta is calculated from these relative-average values:
              Delta = Cp_percent - Cpp_percent
          Therefore Delta is now a relative-average contrast, not a percentage-share
          contrast.
    """

    import re
    from pathlib import Path
    from collections import defaultdict

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    from numpy.linalg import inv, pinv, svd
    from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
    from openpyxl.utils import get_column_letter



    # =========================================================
    # GLOBAL FIGURE FONT CONFIGURATION
    # =========================================================
    # Apply Arial to ordinary text and to Matplotlib math-text notation.
    FIGURE_FONT_FAMILY = "Arial"

    plt.rcParams.update({
        "font.family": FIGURE_FONT_FAMILY,
        "font.sans-serif": [FIGURE_FONT_FAMILY, "Arimo", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": FIGURE_FONT_FAMILY,
        "mathtext.it": f"{FIGURE_FONT_FAMILY}:italic",
        "mathtext.bf": f"{FIGURE_FONT_FAMILY}:bold",
        "mathtext.sf": FIGURE_FONT_FAMILY,
        "mathtext.tt": FIGURE_FONT_FAMILY,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


    def _apply_arial_font(fig) -> None:
        """Force Arial on every Matplotlib Text artist before export."""
        for text_artist in fig.findobj(match=Text):
            text_artist.set_fontfamily(FIGURE_FONT_FAMILY)

    # =========================================================
    # 0. CONFIGURATION
    # =========================================================
    # Input workbook containing the original relation matrices.
    INPUT_EXCEL = Path(input_file)
    INPUT_SHEET = "Network matrices"

    # All tables and plots will be written to this folder.
    OUTPUT_DIR = Path(output_root) / "matrix_output"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Main output workbook.
    OUTPUT_WORKBOOK = Path(output_root) / "All global networks-results-1.xlsx"

    TITLE_KEYWORD = "RELATION MATRIX"
    NODE_ROW_OFFSET = 3
    DATA_ROW_OFFSET = 4
    N_NODES = 8

    KEEP_NODE_ALIASES = [
        ["Aw"],
        ["VEG"],
        ["SM"],
        ["SW"],
        ["GW"],
        ["SP/ENG"],
        ["GL"],
    ]

    ADD_TOTALS_TO_A_B_CP_CPP = True

    # Relative-average calculation settings.
    # Cp_percent is relative to the effective non-zero column mean of Cf.
    # Cpp_percent is relative to the effective non-zero row mean of Df.
    # Entries with absolute values <= RELATIVE_EPS are treated as structurally absent
    # when calculating the relative-average denominator.
    RELATIVE_EPS = 1e-6

    # Delta heatmap options.
    GENERATE_DELTA_HEATMAPS = True
    MASK_DELTA_DIAGONAL = False  # False means diagonal/self-effects are included.
    DELTA_CMAP = "RdBu_r"
    CONTRIBUTION_CMAP = "YlOrRd"
    DELIVERY_CMAP = "YlGnBu"
    DELTA_NCOLS = 3
    DELTA_FIG_DPI = 300
    FIRST_N_NETWORKS_FOR_DELTA_FIG = 6
    DATASET_GROUP_ORDER = ["Obs", "Mixed-GLEAM", "GLDAS", "ERA5"]

    # If True, the script exports heatmaps for contribution, delivery, and Delta together.
    # This replaces the earlier Delta-only plotting behavior.
    GENERATE_ALL_MATRIX_HEATMAPS = True

    # Additional 2 x 2 dataset comparison figure using one selected network scenario.
    # With the current naming convention this selects Obs_0.5, Mixed-GLEAM_0.5, GLDAS_0.5, ERA5_0.5.
    GENERATE_DATASET_FOCUS_2X2_HEATMAPS = True
    DATASET_FOCUS_SUFFIX = "0.5"

    MATRIX_PLOT_SPECS = {
        "contribution": {
            "result_key": "Cp_percent",
            "title": "Contribution",
            "cbar_label": r"$Cf_{rel.avg}$",
            "cmap": CONTRIBUTION_CMAP,
            "symmetric": False,
        },
        "delivery": {
            "result_key": "Cpp_percent",
            "title": "Delivery",
            "cbar_label": r"$Df_{rel.avg}$",
            "cmap": DELIVERY_CMAP,
            "symmetric": False,
        },
        "delta": {
            "result_key": "Delta_Cp_minus_Cpp_percent",
            "title": "Delta",
            "cbar_label": r"$\Delta$ = $Cf_{rel.avg} - Df_{rel.avg}$",
            "cmap": DELTA_CMAP,
            "symmetric": True,
        },
    }

    # Output table names.
    DELTA_LONG_CSV = OUTPUT_DIR / "long_results.csv"
    DELTA_TOP_CSV = OUTPUT_DIR / "top_pathways.csv"


    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.titlesize": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


    # =========================================================
    # 1. BASIC MATRIX FUNCTIONS
    # =========================================================
    def save_figure_both(fig, output_path, dpi=300, **savefig_kwargs):
        """
        Save the same figure as both TIFF and PNG with the same dpi.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure object.
        output_path : str or Path
            Usually a .tiff path. The PNG path will use the same stem.
        dpi : int
            Export dpi for both formats.
        **savefig_kwargs
            Extra keyword arguments passed to fig.savefig().
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = output_path.suffix.lower()
        if suffix not in [".tif", ".tiff", ".png"]:
            tiff_path = output_path.with_suffix(".tiff")
        elif suffix == ".png":
            tiff_path = output_path.with_suffix(".tiff")
        else:
            tiff_path = output_path

        png_path = tiff_path.with_suffix(".png")

        _apply_arial_font(fig)

        fig.savefig(tiff_path, dpi=dpi, **savefig_kwargs)
        fig.savefig(png_path, dpi=dpi, **savefig_kwargs)

        return tiff_path, png_path


    def build_flux_matrix(node_count, flux_list):
        """Build original weighted relation matrix A from a 1-based flux list."""
        A = np.zeros((node_count, node_count), dtype=float)
        for i, j, flow in flux_list:
            A[i - 1, j - 1] += flow
        return A


    def build_link_matrix(node_count, flux_list):
        """Build binary link matrix B from a 1-based flux list."""
        B = np.zeros((node_count, node_count), dtype=int)
        for i, j, flow in flux_list:
            if flow > 0:
                B[i - 1, j - 1] = 1
        return B


    def normalize_rows(A):
        """
        Row-normalize a matrix.

        Used only for the original relation matrix A to derive Bpp:
            Bpp[i, j] = A[i, j] / sum_j A[i, j]
        """
        A = np.asarray(A, dtype=float)
        row_sum = A.sum(axis=1)
        out = np.zeros_like(A, dtype=float)
        valid = np.abs(row_sum) > RELATIVE_EPS
        out[valid, :] = A[valid, :] / row_sum[valid, None]
        return out, row_sum


    def normalize_cols(A):
        """
        Column-normalize a matrix.

        Used only for the original relation matrix A to derive Bp:
            Bp[i, j] = A[i, j] / sum_i A[i, j]
        """
        A = np.asarray(A, dtype=float)
        col_sum = A.sum(axis=0)
        out = np.zeros_like(A, dtype=float)
        valid = np.abs(col_sum) > RELATIVE_EPS
        out[:, valid] = A[:, valid] / col_sum[None, valid]
        return out, col_sum


    def relative_to_axis_mean(A, axis, eps=RELATIVE_EPS):
        """
        Convert raw cumulative-effect values to relative-average values.

        Important modification:
            The denominator is calculated from effective non-zero entries only.
            Values with abs(value) <= eps are ignored when calculating the mean.

        This avoids artificial inflation caused by structurally absent pathways.
        Example: if a 7-node column has only one effective non-zero value, that
        value is divided by itself and becomes 100%, not 700%.

        Parameters
        ----------
        A : array-like
            Raw matrix, usually Cf or Df.
        axis : {0, 1}
            axis=0: compare each cell with the effective non-zero mean of its column.
                    Used for Cp_percent: Cf[i, j] / mean_i_nonzero(Cf[:, j]).
            axis=1: compare each cell with the effective non-zero mean of its row.
                    Used for Cpp_percent: Df[i, j] / mean_j_nonzero(Df[i, :]).
        eps : float
            Effective-zero tolerance. Entries with abs(value) <= eps are ignored
            in the denominator and are reported as 0 in the relative matrix when
            the corresponding denominator is valid.

        Returns
        -------
        out : np.ndarray
            Relative-average matrix in percent units. Among effective non-zero
            entries, the mean along the selected axis is 100 wherever the
            denominator is valid. Effective zero entries are kept as 0.
        means : np.ndarray
            Effective non-zero column or row means used as denominators.
        counts : np.ndarray
            Number of effective non-zero values used in each column or row mean.
        """
        A = np.asarray(A, dtype=float)
        out = np.full_like(A, np.nan, dtype=float)

        finite = np.isfinite(A)
        effective = finite & (np.abs(A) > eps)

        if axis == 0:
            counts = effective.sum(axis=0)
            sums = np.where(effective, A, 0.0).sum(axis=0)
            means = np.full(A.shape[1], np.nan, dtype=float)
            valid = counts > 0
            means[valid] = sums[valid] / counts[valid]

            # Preserve structural zero cells as 0, while scaling only effective values.
            out[:, valid] = 0.0
            valid_effective = effective[:, valid]
            out[:, valid] = np.where(
                valid_effective,
                A[:, valid] / means[None, valid] * 100.0,
                0.0,
            )

        elif axis == 1:
            counts = effective.sum(axis=1)
            sums = np.where(effective, A, 0.0).sum(axis=1)
            means = np.full(A.shape[0], np.nan, dtype=float)
            valid = counts > 0
            means[valid] = sums[valid] / counts[valid]

            # Preserve structural zero cells as 0, while scaling only effective values.
            out[valid, :] = 0.0
            valid_effective = effective[valid, :]
            out[valid, :] = np.where(
                valid_effective,
                A[valid, :] / means[valid, None] * 100.0,
                0.0,
            )

        else:
            raise ValueError("axis must be 0 for column mean or 1 for row mean.")

        return out, means, counts

    def safe_inverse(M, tol=1e-9):
        """Return inv(M), or pinv(M) if M is near singular."""
        M = np.asarray(M, dtype=float)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"safe_inverse expects a square matrix, got shape {M.shape}")
        if M.shape[0] == 0:
            raise ValueError("safe_inverse received an empty matrix.")

        s = svd(M, compute_uv=False)
        if s.min() < tol:
            return pinv(M), True
        return inv(M), False


    def add_totals(obj):
        """Add row and column totals to a DataFrame or Series."""
        if isinstance(obj, pd.DataFrame):
            out = obj.copy()
            out["total"] = out.sum(axis=1, numeric_only=True)
            total_row = out.sum(axis=0, numeric_only=True)
            total_row.name = "total"
            out = pd.concat([out, total_row.to_frame().T])
            return out

        if isinstance(obj, pd.Series):
            out = obj.copy()
            out.loc["total"] = out.sum()
            return out

        return obj


    def add_sum_column_only(df, col_name="total_column"):
        """Add row sums as a final column."""
        out = df.copy()
        out[col_name] = out.sum(axis=1, numeric_only=True)
        return out


    def add_sum_row_only(df, row_name="total_row"):
        """Add column sums as a final row."""
        out = df.copy()
        sum_row = out.sum(axis=0, numeric_only=True)
        sum_row.name = row_name
        out = pd.concat([out, sum_row.to_frame().T])
        return out


    def matrix_to_flux_list(matrix_df):
        """Convert a matrix DataFrame to the 1-based flux-list format."""
        node_names = list(matrix_df.index)
        flux_list = []
        n = len(node_names)

        for i in range(n):
            for j in range(n):
                val = matrix_df.iat[i, j]
                if pd.notna(val) and float(val) != 0:
                    flux_list.append((i + 1, j + 1, float(val)))

        return flux_list, node_names


    # =========================================================
    # 2. MAIN MATRIX CALCULATION
    # =========================================================
    def compute_all(node_count, flux_list, keep_nodes, node_names=None, add_total=False):
        """
        Calculate A, B, Cf, Cp_percent, Df, Cpp_percent, and Delta.

        Cf and Df:
            Cf  = inv(I - Bp)
            Df = inv(I - Bpp)

        Relative-average matrices:
            Cp_percent[i, j]  = Cf[i, j] / mean_i_nonzero(Cf[:, j])
            Cpp_percent[i, j] = Df[i, j] / mean_j_nonzero(Df[i, :])

        Delta:
            Delta_Cp_minus_Cpp_percent = Cp_percent - Cpp_percent

        Interpretation of Cp_percent / Cpp_percent:
            100%  = average effective non-zero cumulative effect among retained nodes
            >100% = above-average effective cumulative effect
            <100% = below-average effective cumulative effect
        """
        if node_names is None:
            node_names = [f"Node{i}" for i in range(1, node_count + 1)]

        missing_nodes = [node for node in keep_nodes if node not in node_names]
        if missing_nodes:
            raise ValueError(f"These keep_nodes are not in node_names: {missing_nodes}")

        A = build_flux_matrix(node_count, flux_list)
        B = build_link_matrix(node_count, flux_list)

        # Direct-flow normalization is still needed to obtain Bp and Bpp.
        Bpp_full, _ = normalize_rows(A)
        Bp_full, _ = normalize_cols(A)

        def df(X):
            return pd.DataFrame(X, index=node_names, columns=node_names)

        A_df = df(A)
        B_df = df(B)
        Bp_full_df = df(Bp_full)
        Bpp_full_df = df(Bpp_full)

        # Use only selected terrestrial nodes for Cf/Df calculation.
        Bp_sub = Bp_full_df.loc[keep_nodes, keep_nodes].copy()
        Bpp_sub = Bpp_full_df.loc[keep_nodes, keep_nodes].copy()

        I_sub = np.eye(len(keep_nodes))
        Cp_sub_arr, used_pinv_contribution = safe_inverse(I_sub - Bp_sub.values)
        Cpp_sub_arr, used_pinv_delivery = safe_inverse(I_sub - Bpp_sub.values)

        Cp_sub = pd.DataFrame(Cp_sub_arr, index=keep_nodes, columns=keep_nodes)
        Cpp_sub = pd.DataFrame(Cpp_sub_arr, index=keep_nodes, columns=keep_nodes)

        # -----------------------------------------------------
        # MAIN CHANGE: relative-average values, not share values
        # -----------------------------------------------------
        Cp_percent_arr, Cp_col_mean, Cp_col_nonzero_count = relative_to_axis_mean(Cp_sub.values, axis=0)
        Cpp_percent_arr, Cpp_row_mean, Cpp_row_nonzero_count = relative_to_axis_mean(Cpp_sub.values, axis=1)

        Cp_percent = pd.DataFrame(Cp_percent_arr, index=keep_nodes, columns=keep_nodes)
        Cpp_percent = pd.DataFrame(Cpp_percent_arr, index=keep_nodes, columns=keep_nodes)

        Delta_Cp_minus_Cpp_percent = Cp_percent - Cpp_percent

        Cp_col_mean_series = pd.Series(Cp_col_mean, index=keep_nodes, name="Cp_effective_nonzero_column_mean")
        Cpp_row_mean_series = pd.Series(Cpp_row_mean, index=keep_nodes, name="Cpp_effective_nonzero_row_mean")
        Cp_col_count_series = pd.Series(Cp_col_nonzero_count, index=keep_nodes, name="Cp_effective_nonzero_column_count")
        Cpp_row_count_series = pd.Series(Cpp_row_nonzero_count, index=keep_nodes, name="Cpp_effective_nonzero_row_count")

        results = {
            "A": A_df,
            "B": B_df,
            "Cf": Cp_sub,
            "Cp_percent": Cp_percent,
            "Df": Cpp_sub,
            "Cpp_percent": Cpp_percent,
            "Delta_Cp_minus_Cpp_percent": Delta_Cp_minus_Cpp_percent,
            "Cp_column_mean_used_for_relative_average": Cp_col_mean_series,
            "Cpp_row_mean_used_for_relative_average": Cpp_row_mean_series,
            "Cp_column_nonzero_count_used_for_relative_average": Cp_col_count_series,
            "Cpp_row_nonzero_count_used_for_relative_average": Cpp_row_count_series,
            "used_pinv_contribution": used_pinv_contribution,
            "used_pinv_delivery": used_pinv_delivery,
        }

        if add_total:
            # Add totals only to A and B for workbook reporting.
            # Keep Cf and Df as pure node x node matrices so exported original
            # contribution and delivery heatmaps do not include total_row or
            # total_column. Relative-average and Delta matrices also remain pure
            # node x node matrices.
            results["A"] = add_totals(results["A"])
            results["B"] = add_totals(results["B"])

        return results


    # =========================================================
    # 3. NODE ALIAS MATCHING
    # =========================================================
    def resolve_keep_nodes(node_names):
        """Resolve preferred hydrological nodes that are actually present in a matrix."""
        keep_nodes = []
        for aliases in KEEP_NODE_ALIASES:
            found = None
            for alias in aliases:
                if alias in node_names:
                    found = alias
                    break
            if found is not None:
                keep_nodes.append(found)
        return keep_nodes


    # =========================================================
    # 4. EXCEL BLOCK READING
    # =========================================================
    def read_relation_matrix_from_excel(
        excel_path,
        sheet_name,
        start_row,
        start_col,
        node_row_offset=NODE_ROW_OFFSET,
        data_row_offset=DATA_ROW_OFFSET,
        n_nodes=N_NODES,
    ):
        """Read one relation matrix block from the source workbook."""
        raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

        node_row = start_row + node_row_offset
        data_row = start_row + data_row_offset
        col0 = start_col + 2
        col1 = col0 + n_nodes

        node_names = raw.iloc[node_row, col0:col1].tolist()
        node_names = [str(x).strip() for x in node_names]

        matrix = raw.iloc[data_row:data_row + n_nodes, col0:col1].copy()
        matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        matrix.index = node_names
        matrix.columns = node_names

        if len(set(node_names)) != len(node_names):
            raise ValueError(f"Duplicated node names detected: {node_names}")
        if any(x == "" or str(x).lower() == "nan" for x in node_names):
            raise ValueError(f"Invalid node names detected: {node_names}")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Relation matrix must be square, got {matrix.shape}")

        return matrix.astype(float)


    def find_all_relation_matrix_blocks(excel_path, sheet_name, title_keyword=TITLE_KEYWORD):
        """Find all relation-matrix blocks in the source workbook."""
        raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
        blocks = []
        seen = set()
        keyword = str(title_keyword).upper()

        for r in range(raw.shape[0]):
            for c in range(raw.shape[1]):
                val = str(raw.iat[r, c]).strip()
                if keyword in val.upper():
                    start_row = r
                    start_col = max(c - 1, 0)
                    key = (start_row, start_col)
                    if key not in seen:
                        seen.add(key)
                        blocks.append({
                            "start_row": start_row,
                            "start_col": start_col,
                            "title_cell_text": val,
                        })

        return sorted(blocks, key=lambda x: (x["start_row"], x["start_col"]))


    # =========================================================
    # 5. MATRIX NAME PARSING
    # =========================================================
    def clean_name_for_sheet(name):
        name = str(name).strip()
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"[\\/*?:\[\]]", "-", name)
        return name


    def safe_sheet_name(name):
        invalid = ["\\", "/", "*", "?", ":", "[", "]"]
        name = str(name)
        for ch in invalid:
            name = name.replace(ch, "-")
        return name[:31]


    def extract_matrix_family(title_text):
        t = str(title_text).strip()

        for family in ["Obs", "Mixed-GLEAM", "GLDAS", "ERA5"]:
            if re.search(rf"\b{re.escape(family)}\b", t, flags=re.IGNORECASE):
                return family

        cleaned = re.sub(r"RELATION\s+MATRIX", "", t, flags=re.IGNORECASE)
        cleaned = re.sub(r"matrix", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[^A-Za-z0-9 _-]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if not cleaned:
            return "Matrix"

        return clean_name_for_sheet(cleaned)


    def assign_case_names_from_blocks(blocks):
        """Assign custom suffix names such as Obs_0.5, Mixed-GLEAM_0.0, GLDAS_1.0."""
        counters = defaultdict(int)

        # Map the count index to your required specific suffix strings
        suffix_mapping = {0: "0.5", 1: "0.0", 2: "1.0"}

        for block in blocks:
            family = extract_matrix_family(block["title_cell_text"])
            idx = counters[family]
            counters[family] += 1

            # Retrieve the mapped suffix.
            # Fallback to the standard integer string if a family appears more than 3 times.
            suffix = suffix_mapping.get(idx, str(idx))

            if family in {"Obs", "Mixed-GLEAM", "GLDAS", "ERA5"}:
                case_name = f"{family}_{suffix}"
            else:
                # If it's a non-standard family, you can decide whether the first item
                # should still just be the family name, or take the _0.5 suffix.
                # Here, it is preserved to remain blank on the first instance.
                case_name = family if idx == 0 else f"{family}_{suffix}"

            block["case_name"] = case_name

        return blocks


    # =========================================================
    # 6. EXCEL STYLING HELPERS
    # =========================================================
    TITLE_FILL = PatternFill("solid", fgColor="D9EAF7")
    SECTION_FILL = PatternFill("solid", fgColor="EAF2F8")
    HEADER_FILL = PatternFill("solid", fgColor="F4F6F6")

    THIN_BORDER = Border(
        left=Side(style="thin", color="BFBFBF"),
        right=Side(style="thin", color="BFBFBF"),
        top=Side(style="thin", color="BFBFBF"),
        bottom=Side(style="thin", color="BFBFBF"),
    )

    CENTER = Alignment(horizontal="center", vertical="center")
    LEFT = Alignment(horizontal="left", vertical="center")


    def style_cell(cell, *, bold=False, fill=None, border=None, align=None, number_format=None, font_size=None):
        cell.font = Font(bold=bold, size=font_size if font_size else 11)
        if fill is not None:
            cell.fill = fill
        if border is not None:
            cell.border = border
        if align is not None:
            cell.alignment = align
        if number_format is not None:
            cell.number_format = number_format


    def write_block_to_sheet(ws, title, obj, start_row, start_col=1, float_format="0.000"):
        """Write a matrix or Series block to an Excel worksheet."""
        ws.cell(row=start_row, column=start_col, value=title)
        style_cell(
            ws.cell(row=start_row, column=start_col),
            bold=True,
            fill=SECTION_FILL,
            border=THIN_BORDER,
            align=LEFT,
            font_size=12,
        )

        row = start_row + 1

        if isinstance(obj, pd.Series):
            df = obj.to_frame()
        else:
            df = obj.copy()

        headers = ["index"] + list(df.columns)

        for j, h in enumerate(headers, start=start_col):
            cell = ws.cell(row=row, column=j, value=str(h))
            style_cell(cell, bold=True, fill=HEADER_FILL, border=THIN_BORDER, align=CENTER)

        for i, idx in enumerate(df.index, start=row + 1):
            idx_cell = ws.cell(row=i, column=start_col, value=str(idx))
            style_cell(idx_cell, border=THIN_BORDER, align=LEFT)

            for j, col in enumerate(df.columns, start=start_col + 1):
                val = df.loc[idx, col]
                cell = ws.cell(row=i, column=j)

                if pd.isna(val):
                    cell.value = None
                    style_cell(cell, border=THIN_BORDER, align=CENTER)
                else:
                    if isinstance(val, (int, np.integer)):
                        cell.value = int(val)
                        style_cell(cell, border=THIN_BORDER, align=CENTER, number_format="0")
                    else:
                        try:
                            cell.value = float(val)
                            style_cell(cell, border=THIN_BORDER, align=CENTER, number_format=float_format)
                        except Exception:
                            cell.value = str(val)
                            style_cell(cell, border=THIN_BORDER, align=LEFT)

        next_row = row + len(df.index) + 3
        return next_row


    def autofit_columns(ws, min_width=10, max_width=28):
        widths = {}
        for row in ws.iter_rows():
            for cell in row:
                if cell.value is None:
                    continue
                widths[cell.column] = max(widths.get(cell.column, 0), len(str(cell.value)) + 2)

        for col_idx, width in widths.items():
            letter = get_column_letter(col_idx)
            ws.column_dimensions[letter].width = max(min_width, min(width, max_width))


    def add_case_header(ws, case_name):
        ws["A1"] = f"{case_name} results"
        style_cell(
            ws["A1"],
            bold=True,
            fill=TITLE_FILL,
            border=THIN_BORDER,
            align=LEFT,
            font_size=14,
        )
        ws.row_dimensions[1].height = 22


    def add_method_sheet(writer):
        """Add a compact method sheet explaining the relative-average matrices."""
        method_rows = [
            ["Item", "Definition / interpretation"],
            ["Cf", "Contribution cumulative-effect matrix: inv(I - Bp)."],
            ["Df", "Delivery cumulative-effect matrix: inv(I - Bpp)."],
            ["Cp_percent", "Relative-average contribution value: 100 * Cf[i,j] / mean_i_nonzero(Cf[:,j]). Effective zero or near-zero values are excluded from the denominator."],
            ["Cpp_percent", "Relative-average delivery value: 100 * Df[i,j] / mean_j_nonzero(Df[i,:]). Effective zero or near-zero values are excluded from the denominator."],
            ["Delta", "Cp_percent - Cpp_percent. Positive values indicate relatively stronger contribution effect; negative values indicate relatively stronger delivery effect."],
            ["Interpretation", "100 = average among effective non-zero cumulative effects; >100 = above average; <100 = below average. Structural zero cells remain 0."],
        ]
        df = pd.DataFrame(method_rows[1:], columns=method_rows[0])
        df.to_excel(writer, sheet_name="method_note", index=False)

        ws = writer.book["method_note"]
        for cell in ws[1]:
            style_cell(cell, bold=True, fill=HEADER_FILL, border=THIN_BORDER, align=CENTER)
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                style_cell(cell, border=THIN_BORDER, align=LEFT)
        autofit_columns(ws, min_width=14, max_width=95)


    # =========================================================
    # 7. LONG TABLES AND HEATMAP PLOTS
    # =========================================================
    def matrix_values_to_long(case_name, results, mask_diagonal=MASK_DELTA_DIAGONAL):
        """
        Export contribution, delivery, and Delta values in one long table.

        Cp_relative_average and Cpp_relative_average are the relative-average
        matrices in percent units, i.e. 100 means the average effective non-zero cumulative effect.
        """
        cp_percent_df = results["Cp_percent"]
        cpp_percent_df = results["Cpp_percent"]
        delta_df = results["Delta_Cp_minus_Cpp_percent"]

        rows = []
        for i in delta_df.index:
            for j in delta_df.columns:
                if mask_diagonal and i == j:
                    continue

                cp_val = float(cp_percent_df.loc[i, j])
                cpp_val = float(cpp_percent_df.loc[i, j])
                delta_val = float(delta_df.loc[i, j])

                if not np.isfinite(delta_val):
                    dominance = "Undefined"
                    abs_delta = np.nan
                else:
                    abs_delta = abs(delta_val)
                    dominance = (
                        "Contribution-dominant" if delta_val > 0 else
                        "Delivery-dominant" if delta_val < 0 else
                        "Balanced"
                    )

                rows.append({
                    "case_name": case_name,
                    "From": i,
                    "To": j,
                    "Cp_relative_average": cp_val,
                    "Cpp_relative_average": cpp_val,
                    "Delta_Cp_relative_minus_Cpp_relative": delta_val,
                    "Abs_Delta": abs_delta,
                    "Dominance": dominance,
                })

        return pd.DataFrame(rows)


    # Backward-compatible alias for older calls.
    def delta_matrix_to_long(case_name, delta_df, cp_percent_df, cpp_percent_df, mask_diagonal=MASK_DELTA_DIAGONAL):
        results = {
            "Cp_percent": cp_percent_df,
            "Cpp_percent": cpp_percent_df,
            "Delta_Cp_minus_Cpp_percent": delta_df,
        }
        return matrix_values_to_long(case_name, results, mask_diagonal=mask_diagonal)


    def export_delta_tables(all_results):
        """
        Export long and top-difference tables.

        The long table contains contribution, delivery, and Delta values together;
        the top table is still ranked by absolute Delta.
        """
        parts = []
        for case_name, results in all_results.items():
            parts.append(matrix_values_to_long(
                case_name=case_name,
                results=results,
                mask_diagonal=MASK_DELTA_DIAGONAL,
            ))

        if not parts:
            return None, None

        delta_long = pd.concat(parts, ignore_index=True)
        delta_long = delta_long.sort_values(["case_name", "Abs_Delta"], ascending=[True, False])
        delta_long.to_csv(DELTA_LONG_CSV, index=False, encoding="utf-8-sig")

        top_parts = []
        for _, g in delta_long.groupby("case_name"):
            top_parts.append(g.sort_values("Abs_Delta", ascending=False).head(12))
        delta_top = pd.concat(top_parts, ignore_index=True) if top_parts else pd.DataFrame()
        delta_top.to_csv(DELTA_TOP_CSV, index=False, encoding="utf-8-sig")

        return delta_long, delta_top


    def _mask_diagonal_for_plot(matrix_df):
        out = matrix_df.copy()
        if MASK_DELTA_DIAGONAL:
            common = [node for node in out.index if node in out.columns]
            for node in common:
                out.loc[node, node] = np.nan
        return out


    def annotate_heatmap(ax, data, scale_max, symmetric=False, fmt=".0f"):
        arr = np.asarray(data.values, dtype=float)
        threshold = scale_max * 0.55 if scale_max > 0 else 0.0

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                if not np.isfinite(val):
                    continue

                if symmetric:
                    strong_value = abs(val) >= threshold
                else:
                    strong_value = val >= threshold
                color = "white" if strong_value else "black"

                ax.text(
                    j,
                    i,
                    format(val, fmt),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=color,
                )


    # Backward-compatible alias.
    def annotate_delta_heatmap(ax, data, vmax, fmt=".2f"):
        annotate_heatmap(ax, data, scale_max=vmax, symmetric=True, fmt=fmt)


    def infer_dataset_group(case_name):
        """Infer dataset group from case name, e.g. Obs_0.5 -> Obs."""
        case_name = str(case_name)
        for group in DATASET_GROUP_ORDER:
            if case_name.lower().startswith(group.lower()):
                return group
        if "_" in case_name:
            return case_name.split("_")[0]
        if "-" in case_name:
            return case_name.split("-")[0]
        return case_name


    def get_global_vmax(plot_mats, symmetric=False):
        """Use one common color scale for all panels in one figure."""
        vmax = 0.0
        for mat in plot_mats.values():
            values = mat.values.astype(float)
            if not np.isfinite(values).any():
                continue
            if symmetric:
                vmax = max(vmax, float(np.nanmax(np.abs(values))))
            else:
                vmax = max(vmax, float(np.nanmax(values)))

        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        return vmax


    # Backward-compatible alias.
    def get_global_delta_vmax(plot_mats):
        return get_global_vmax(plot_mats, symmetric=True)



    def add_subplot_letter(ax, letter, x=-0.14, y=1.06, fontsize=16):
        """
        Add bold subplot notation such as a, b, c, ... at the upper-left.
        """
        ax.text(
            x,
            y,
            letter,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight="bold",
            ha="left",
            va="top",
            clip_on=False,
        )


    def _drop_total_labels(mat: pd.DataFrame) -> pd.DataFrame:
        """Remove workbook-added total rows/columns if present."""
        out = mat.copy()
        drop_rows = [idx for idx in out.index if str(idx).strip().lower() in {"total", "total_row"}]
        drop_cols = [col for col in out.columns if str(col).strip().lower() in {"total", "total_column"}]
        if drop_rows:
            out = out.drop(index=drop_rows, errors="ignore")
        if drop_cols:
            out = out.drop(columns=drop_cols, errors="ignore")
        return out


    def _apply_outer_axis_labels(ax, idx: int):
        """Show only the requested outer axis labels for the 2 x 3 layout."""
        # Bold 'From' only for panels a and d (idx 0 and 3).
        if idx in (0, 3):
            ax.set_ylabel("From", fontsize=15, fontweight="bold")
        else:
            ax.set_ylabel("")

        # Bold 'To' only for panels d, e, f (idx 3, 4, 5).
        if idx in (3, 4, 5):
            ax.set_xlabel("To", fontsize=15, fontweight="bold")
        else:
            ax.set_xlabel("")


    def plot_first6_matrix_heatmap_grid_decimal(
        all_results,
        result_key,
        basename,
        cbar_label,
        cmap,
        n_first=FIRST_N_NETWORKS_FOR_DELTA_FIG,
        value_scale=1.0,
    ):
        """
        Export one 2 x 3 heatmap figure for the first six reference networks.

        This is used for four additional decimal-value figures:
            1. original contribution matrix Cf
            2. relative-average contribution matrix Cp_percent / 100
            3. original delivery matrix Df
            4. relative-average delivery matrix Cpp_percent / 100

        Notes
        -----
        The internal Cp_percent and Cpp_percent matrices are stored as 100-based
        relative-average values for workbook compatibility. For the decimal
        relative-average figures, pass value_scale=0.01.
        """
        if not all_results:
            return None

        selected_cases = list(all_results.keys())[:n_first]
        if not selected_cases:
            return None

        plot_mats = {}
        for case in selected_cases:
            mat = _drop_total_labels(_mask_diagonal_for_plot(all_results[case][result_key]).copy())
            mat = mat.astype(float) * float(value_scale)
            plot_mats[case] = mat

        vmax = get_global_vmax(plot_mats, symmetric=False)

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(16.5, 9.8),
            dpi=DELTA_FIG_DPI,
            squeeze=False,
        )

        fig.subplots_adjust(
            left=0.07,
            right=0.90,
            top=0.93,
            bottom=0.08,
            wspace=0.28,
            hspace=0.34,
        )

        axes_flat = axes.ravel()
        letters = list("abcdef")
        im = None

        for idx, case_name in enumerate(selected_cases):
            ax = axes_flat[idx]
            mat = plot_mats[case_name]
            nodes = list(mat.index)

            im = ax.imshow(
                mat.values,
                cmap=cmap,
                vmin=0.0,
                vmax=vmax,
                aspect="auto",
            )

            ax.set_title(case_name, fontsize=16, fontweight="normal", pad=10)
            ax.set_xticks(range(len(nodes)))
            ax.set_yticks(range(len(nodes)))
            ax.set_xticklabels(nodes, fontsize=14, rotation=45, ha="right")
            ax.set_yticklabels(nodes, fontsize=14)
            _apply_outer_axis_labels(ax, idx)

            annotate_heatmap(ax, mat, scale_max=vmax, symmetric=False, fmt=".2f")
            add_subplot_letter(ax, letters[idx])

        for idx in range(len(selected_cases), 6):
            axes_flat[idx].axis("off")

        if im is not None:
            cbar = fig.colorbar(
                im,
                ax=axes_flat[:len(selected_cases)],
                shrink=0.82,
                aspect=28,
                pad=0.02,
            )
            cbar.set_label(cbar_label)

        tiff_path = OUTPUT_DIR / f"{basename}.tiff"
        save_figure_both(fig, tiff_path, dpi=DELTA_FIG_DPI, bbox_inches="tight")
        plt.close(fig)

        return tiff_path

    def plot_matrix_heatmap_grid(
        plot_mats,
        basename,
        matrix_title,
        cbar_label,
        cmap,
        symmetric=False,
        ncols=DELTA_NCOLS,
    ):
        """
        Generic multi-panel matrix heatmap figure.

        This is used for contribution, delivery, and Delta so that the 6-network
        comparison are exported consistently.
        """
        if not plot_mats:
            return None

        case_names = list(plot_mats.keys())
        vmax = get_global_vmax(plot_mats, symmetric=symmetric)
        vmin = -vmax if symmetric else 0.0

        n = len(case_names)
        ncols = min(max(1, ncols), n)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.5 * ncols, 5.0 * nrows),
            dpi=DELTA_FIG_DPI,
            squeeze=False,
            constrained_layout=True,
        )
        axes_flat = axes.ravel()

        im = None
        for idx, case_name in enumerate(case_names):
            ax = axes_flat[idx]
            mat = plot_mats[case_name]
            nodes = list(mat.index)

            im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(f"{case_name}", fontweight="bold", pad=12)
            ax.set_xticks(range(len(nodes)))
            ax.set_yticks(range(len(nodes)))
            ax.set_xticklabels(nodes, rotation=45, ha="right")
            ax.set_yticklabels(nodes)
            annotate_heatmap(ax, mat, scale_max=vmax, symmetric=symmetric, fmt=".0f")
            ax.set_xlabel("To")
            ax.set_ylabel("From")

        for ax in axes_flat[n:]:
            ax.axis("off")

        if im is not None:
            cbar = fig.colorbar(im, ax=axes_flat[:n], shrink=0.75, aspect=25, pad=0.03)
            cbar.set_label(cbar_label)

        tiff_path = OUTPUT_DIR / f"{basename}.tiff"
        save_figure_both(fig, tiff_path, dpi=DELTA_FIG_DPI, bbox_inches="tight")
        plt.close(fig)

        return tiff_path


    # Backward-compatible Delta-only wrapper.
    def plot_delta_heatmap_grid(plot_mats, basename, ncols=DELTA_NCOLS):
        return plot_matrix_heatmap_grid(
            plot_mats=plot_mats,
            basename=basename,
            matrix_title="Delta",
            cbar_label=r"$\Delta$ = $Cf_{rel} - Df_{rel}$",
            cmap=DELTA_CMAP,
            symmetric=True,
            ncols=ncols,
        )


    def build_first_n_matrices(all_results, result_key, n_first=FIRST_N_NETWORKS_FOR_DELTA_FIG):
        """Return masked matrices for the first N networks."""
        selected = list(all_results.keys())[:n_first]
        return {
            case: _mask_diagonal_for_plot(all_results[case][result_key])
            for case in selected
        }


    # Backward-compatible alias.
    def build_first_n_delta_matrices(all_results, n_first=FIRST_N_NETWORKS_FOR_DELTA_FIG):
        return build_first_n_matrices(
            all_results,
            result_key="Delta_Cp_minus_Cpp_percent",
            n_first=n_first,
        )



    def build_dataset_focus_matrices(all_results, result_key, suffix=DATASET_FOCUS_SUFFIX):
        """
        Select one network scenario for the four datasets and order it as:
            Obs, Mixed-GLEAM, GLDAS, ERA5

        With suffix="0.5", this returns Obs_0.5, Mixed-GLEAM_0.5, GLDAS_0.5,
        and ERA5_0.5 when those cases are present.

        This function is retained for the optional dataset_0p5_2x2_* figures.
        All dataset_assessment_* code has been removed.
        """
        suffix = str(suffix).lstrip("_")
        selected_mats = {}

        for group in DATASET_GROUP_ORDER:
            target_name = f"{group}_{suffix}"
            case_name = None

            if target_name in all_results:
                case_name = target_name
            else:
                candidates = [
                    name for name in all_results.keys()
                    if infer_dataset_group(name) == group and str(name).endswith(f"_{suffix}")
                ]
                if candidates:
                    case_name = sorted(candidates)[0]

            if case_name is not None:
                selected_mats[case_name] = _mask_diagonal_for_plot(all_results[case_name][result_key])

        return selected_mats


    def plot_all_matrix_heatmap_subplots(all_results):
        """
        Generate only the four requested first-six 2 x 3 heatmap figures:

            first6_original_contribution_heatmaps.tiff/png
            first6_relavg_contribution_heatmaps.tiff/png
            first6_original_delivery_heatmaps.tiff/png
            first6_relavg_delivery_heatmaps.tiff/png

        Removed:
            - original first6_contribution/delivery/delta heatmaps
            - dataset_0p5_2x2_* heatmaps
            - dataset_assessment_* heatmaps
        """
        if not all_results:
            return []

        output_paths = []

        fig_path = plot_first6_matrix_heatmap_grid_decimal(
            all_results=all_results,
            result_key="Cf",
            basename="first6_original_contribution_heatmaps",
            cbar_label=r"$Cf$",
            cmap=CONTRIBUTION_CMAP,
            n_first=FIRST_N_NETWORKS_FOR_DELTA_FIG,
            value_scale=1.0,
        )
        if fig_path is not None:
            output_paths.append(fig_path)

        fig_path = plot_first6_matrix_heatmap_grid_decimal(
            all_results=all_results,
            result_key="Cp_percent",
            basename="first6_relavg_contribution_heatmaps",
            cbar_label=r"$Cf_{rel}$",
            cmap=CONTRIBUTION_CMAP,
            n_first=FIRST_N_NETWORKS_FOR_DELTA_FIG,
            value_scale=0.01,
        )
        if fig_path is not None:
            output_paths.append(fig_path)

        fig_path = plot_first6_matrix_heatmap_grid_decimal(
            all_results=all_results,
            result_key="Df",
            basename="first6_original_delivery_heatmaps",
            cbar_label=r"$Df$",
            cmap=DELIVERY_CMAP,
            n_first=FIRST_N_NETWORKS_FOR_DELTA_FIG,
            value_scale=1.0,
        )
        if fig_path is not None:
            output_paths.append(fig_path)

        fig_path = plot_first6_matrix_heatmap_grid_decimal(
            all_results=all_results,
            result_key="Cpp_percent",
            basename="first6_relavg_delivery_heatmaps",
            cbar_label=r"$Df_{rel}$",
            cmap=DELIVERY_CMAP,
            n_first=FIRST_N_NETWORKS_FOR_DELTA_FIG,
            value_scale=0.01,
        )
        if fig_path is not None:
            output_paths.append(fig_path)

        return output_paths


    # Backward-compatible name used by the previous main workflow.
    def plot_delta_heatmap_subplots(all_results):
        """
        Backward-compatible plotting entry point.

        When GENERATE_ALL_MATRIX_HEATMAPS is True, export all configured matrix
        figures. When it is False, export only the first-six Delta figure.
        Only the four first6_original/relavg heatmap exports are preserved.
        """
        if GENERATE_ALL_MATRIX_HEATMAPS:
            return plot_all_matrix_heatmap_subplots(all_results)

        output_paths = []

        first6_mats = build_first_n_delta_matrices(
            all_results,
            n_first=FIRST_N_NETWORKS_FOR_DELTA_FIG,
        )
        fig1_path = plot_delta_heatmap_grid(
            plot_mats=first6_mats,
            basename="Fig 2",
            ncols=3,
        )
        if fig1_path is not None:
            output_paths.append(fig1_path)

        return output_paths


    # =========================================================
    # 8. EXCEL EXPORT
    # =========================================================
    def export_all_cases_one_sheet_per_case(output_path, all_results):
        """Export all matrix tables to one formatted workbook in OUTPUT_DIR."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            summary_rows = []

            for case_name, results in all_results.items():
                sheet_name = safe_sheet_name(case_name)

                pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
                ws = writer.book[sheet_name]

                add_case_header(ws, case_name)

                row = 3
                row = write_block_to_sheet(ws, "A", results["A"], row, float_format="0.0")
                row = write_block_to_sheet(ws, "B", results["B"], row, float_format="0")

                row = write_block_to_sheet(ws, "Cf", results["Cf"], row, float_format="0.000")
                row = write_block_to_sheet(
                    ws,
                    "Cp_percent_relative_average",
                    results["Cp_percent"],
                    row,
                    float_format="0.0",
                )
                row = write_block_to_sheet(
                    ws,
                    "Cp_effective_nonzero_column_mean_used_for_relative_average",
                    results["Cp_column_mean_used_for_relative_average"],
                    row,
                    float_format="0.000",
                )
                row = write_block_to_sheet(
                    ws,
                    "Cp_effective_nonzero_column_count_used_for_relative_average",
                    results["Cp_column_nonzero_count_used_for_relative_average"],
                    row,
                    float_format="0",
                )

                row = write_block_to_sheet(ws, "Df", results["Df"], row, float_format="0.000")
                row = write_block_to_sheet(
                    ws,
                    "Cpp_percent_relative_average",
                    results["Cpp_percent"],
                    row,
                    float_format="0.0",
                )
                row = write_block_to_sheet(
                    ws,
                    "Cpp_effective_nonzero_row_mean_used_for_relative_average",
                    results["Cpp_row_mean_used_for_relative_average"],
                    row,
                    float_format="0.000",
                )
                row = write_block_to_sheet(
                    ws,
                    "Cpp_effective_nonzero_row_count_used_for_relative_average",
                    results["Cpp_row_nonzero_count_used_for_relative_average"],
                    row,
                    float_format="0",
                )

                row = write_block_to_sheet(
                    ws,
                    "Delta_Cp_relative_average_minus_Cpp_relative_average",
                    results["Delta_Cp_minus_Cpp_percent"],
                    row,
                    float_format="0.0",
                )

                ws.freeze_panes = "A3"
                autofit_columns(ws)

                summary_rows.append({
                    "case_name": case_name,
                    "used_pinv_contribution": results["used_pinv_contribution"],
                    "used_pinv_delivery": results["used_pinv_delivery"],
                    "Cp_percent_definition": "100 * Cf[i,j] / mean_i(Cf[:,j])",
                    "Cpp_percent_definition": "100 * Df[i,j] / mean_j(Df[i,:])",
                })

            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_excel(writer, sheet_name="summary", index=False)

            ws = writer.book["summary"]
            for cell in ws[1]:
                style_cell(cell, bold=True, fill=HEADER_FILL, border=THIN_BORDER, align=CENTER)

            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    style_cell(cell, border=THIN_BORDER, align=CENTER)

            ws.freeze_panes = "A2"
            autofit_columns(ws, min_width=12, max_width=42)

            add_method_sheet(writer)


    # =========================================================
    # 9. BATCH COMPUTATION
    # =========================================================
    def compute_all_cases_from_excel(
        excel_path=INPUT_EXCEL,
        sheet_name=INPUT_SHEET,
        output_path=OUTPUT_WORKBOOK,
        title_keyword=TITLE_KEYWORD,
        node_row_offset=NODE_ROW_OFFSET,
        data_row_offset=DATA_ROW_OFFSET,
        n_nodes=N_NODES,
        add_total=ADD_TOTALS_TO_A_B_CP_CPP,
    ):
        """
        Full workflow:
        1. Read all original relation matrices.
        2. Calculate A, B, Cf, relative-average Cp_percent,
           Df, relative-average Cpp_percent, and Delta.
        3. Export formatted Excel workbook.
        4. Export Delta long/top CSV tables.
        5. Export contribution, delivery, and Delta heatmap figures, including the first-six decimal original/relative-average figures.
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        excel_path = Path(excel_path)
        output_path = Path(output_path)

        if not excel_path.exists():
            raise FileNotFoundError(
                f"Input workbook not found: {excel_path.resolve()}\n"
                "Please put 'All global networks.xlsx' in the same folder as this script, "
                "or edit INPUT_EXCEL at the top of the script."
            )

        blocks = find_all_relation_matrix_blocks(excel_path, sheet_name, title_keyword)
        if not blocks:
            raise ValueError(f"No blocks found with title keyword: {title_keyword}")

        blocks = assign_case_names_from_blocks(blocks)

        print(f"Detected {len(blocks)} matrix blocks.")
        print("Assigned names:")
        for block in blocks:
            print(f"  {block['case_name']}  <-  {block['title_cell_text']}")

        all_results = {}

        for block in blocks:
            start_row = block["start_row"]
            start_col = block["start_col"]
            case_name = block["case_name"]

            matrix_df = read_relation_matrix_from_excel(
                excel_path=excel_path,
                sheet_name=sheet_name,
                start_row=start_row,
                start_col=start_col,
                node_row_offset=node_row_offset,
                data_row_offset=data_row_offset,
                n_nodes=n_nodes,
            )

            flux_list, node_names = matrix_to_flux_list(matrix_df)
            keep_nodes = resolve_keep_nodes(node_names)

            if not keep_nodes:
                print(f"Skip {case_name}: no compatible keep_nodes. node_names={node_names}")
                continue

            results = compute_all(
                node_count=len(node_names),
                flux_list=flux_list,
                keep_nodes=keep_nodes,
                node_names=node_names,
                add_total=add_total,
            )

            all_results[case_name] = results
            print(f"{case_name} finished | keep_nodes={keep_nodes}")

        if not all_results:
            raise ValueError("No valid cases were computed.")

        export_all_cases_one_sheet_per_case(output_path, all_results)
        print(f"\nTables exported to:\n  {output_path}")

        delta_long, delta_top = export_delta_tables(all_results)
        if delta_long is not None:
            print(f"Delta long table exported to:\n  {DELTA_LONG_CSV}")
            print(f"Delta top-pathway table exported to:\n  {DELTA_TOP_CSV}")

        if GENERATE_DELTA_HEATMAPS:
            fig_paths = plot_delta_heatmap_subplots(all_results)
            if fig_paths is not None:
                print("Contribution, delivery, and Delta heatmap figures exported to:")
                for p in fig_paths:
                    print(f"  {p}")

        print(f"\nFinished. All tables and plots are in:\n  {OUTPUT_DIR.resolve()}")
        return all_results


    # =========================================================
    # 10. MAIN
    # =========================================================

    _public = {k: v for k, v in locals().items() if not k.startswith("__")}
    return types.SimpleNamespace(**_public)


def create_centrality_workflow(input_file: str | Path, output_root: str | Path):
    """Create the integrated node-centrality workflow namespace."""
    """
    Combined centrality + poster plotting workflow.

    This script combines two tasks in one workflow:

    1. Calculate node-level metrics from each original relationship matrix:
       - Strength
       - Closeness
       - Clustering
       - Eigenvector

    2. Generate poster-ready outputs:
       - Excel workbook with long-format metrics, Summary_all table, mean metrics, and matrices
       - Radar plots for the five centrality metrics
       - Mean barplots across all nodes for each network/model
       - Mean barplots colored consistently with radar plots


    Edit only the CONFIGURATION section unless you need to change the method.
    """

    import re
    import sys
    import warnings
    from collections import defaultdict
    from pathlib import Path
    from typing import Dict, Iterable, List, Optional, Tuple

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.text import Text
    import networkx as nx
    import numpy as np
    import pandas as pd
    from numpy.linalg import inv, pinv, svd
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    warnings.filterwarnings("ignore")

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass



    # =========================================================
    # GLOBAL FIGURE FONT CONFIGURATION
    # =========================================================
    # Apply Arial to ordinary text and to Matplotlib math-text notation.
    FIGURE_FONT_FAMILY = "Arial"

    plt.rcParams.update({
        "font.family": FIGURE_FONT_FAMILY,
        "font.sans-serif": [FIGURE_FONT_FAMILY, "Arimo", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": FIGURE_FONT_FAMILY,
        "mathtext.it": f"{FIGURE_FONT_FAMILY}:italic",
        "mathtext.bf": f"{FIGURE_FONT_FAMILY}:bold",
        "mathtext.sf": FIGURE_FONT_FAMILY,
        "mathtext.tt": FIGURE_FONT_FAMILY,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


    def _apply_arial_font(fig) -> None:
        """Force Arial on every Matplotlib Text artist before export."""
        for text_artist in fig.findobj(match=Text):
            text_artist.set_fontfamily(FIGURE_FONT_FAMILY)

    # ==================================================
    # 1. CONFIGURATION
    # ==================================================
    INPUT_FILE = Path(input_file)
    INPUT_SHEET = "Network matrices"
    OUTPUT_DIR = Path(output_root) / "centrality_output"
    OUTPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_EXCEL = Path(output_root) / "All global networks-results-2.xlsx"

    TITLE_KEYWORD = "RELATION MATRIX"
    NODE_ROW_OFFSET = 3
    DATA_ROW_OFFSET = 4
    N_NODES = 8

    # Preferred plotting/order list. The code preserves whatever nodes are actually
    # present in each matrix, but orders known nodes according to this list.
    PREFERRED_NODE_ORDER = ["Aw", "VEG", "SM", "SW", "GW", "SP/ENG", "SEA", "GL"]

    # Calculate Contribution and Delivery from derived Cp and Cpp matrices.
    # These are optional poster summary metrics, not graph centrality metrics.
    INCLUDE_CONTRIBUTION_DELIVERY = True

    # If True, Cp_percent and Cpp_percent are exported as 0-100 values.
    # If False, they are exported as 0-1 fractions.
    PERCENT_AS_100 = False

    # Original matrix direction.
    # True means matrix row node -> column node.
    ROW_TO_COLUMN = True

    # Self-loops are usually excluded for graph centrality to prevent diagonal
    # entries from dominating metrics.
    INCLUDE_SELF_LOOPS = False

    # Strength uses the Opsahl-style tuning:
    #     strength = degree^alpha * total_weight^(1-alpha)
    STRENGTH_ALPHA = 0.5

    # Closeness direction.
    # "in" reproduces NetworkX directed closeness on G.
    # "out" uses G.reverse() and is often better for propagation/reachability.
    # "undirected" ignores edge direction.
    CLOSENESS_MODE = "in"

    # Clustering mode.
    # "undirected" is usually more stable for poster-level interpretation.
    CLUSTERING_MODE = "undirected"

    # Eigenvector centrality mode.
    # "undirected" is recommended for poster-level hydrological interpretation.
    # "in" emphasizes nodes receiving influence from important upstream nodes.
    # "out" emphasizes nodes pointing toward important downstream nodes.
    EIGENVECTOR_MODE = "undirected"

    PAGERANK_DAMPING = 0.85
    EPS_DISTANCE = 1e-4

    CENTRALITY_METRICS = [
        "Strength",
        "Closeness",
        "Clustering",
        "Eigenvector",
    ]

    # Only the first N models are used in the main radar_metric_axis.tiff,
    # the first-six mean barplot, and the first-six 5x2 radar-bar figure.
    # Grouped plots are exported separately.
    RADAR_MODEL_LIMIT = 6


    # Letter annotations on mean barplots.
    # Set to False to remove a/b/c/d-style notation.
    ADD_SAME_PERFORMANCE_LETTERS = False
    SAME_PERFORMANCE_TOLERANCE_FRACTION = 0.05

    EXPORT_MATRIX_LONG = True
    EXPORT_FIGURES = True
    EXPORT_GROUPED_RADAR_PLOTS = False

    # --------------------------------------------------
    # Figure styling
    # --------------------------------------------------
    FONT_SIZE = 22
    LABEL_SIZE = 24
    TICK_SIZE = 20
    LEGEND_SIZE = 24
    SMALL_TICK_SIZE = 20
    SUPTITLE_SIZE = 24

    # Mapping used when renaming automatically detected model/network suffixes.
    # Example: Obs_0 -> Obs_0.5, Obs_1 -> Obs_0.0, Obs_2 -> Obs_1.0.
    NETWORK_SUFFIX_BY_INDEX = {
        1: "0.0",
        0: "0.5",
        2: "1.0",
    }

    NETWORK_CODE_LETTERS = list("abcdefghijklmnopqrstuvwxyz")

    # Ten-color palette reserved for figures:
    # the first 6 colors are used for the six network structures, and the last
    # 4 colors are reserved for the four datasets.
    COLOR_PALETTE = [
        "#4C78A8",  # network a - steel blue
        "#F58518",  # network b - orange
        "#54A24B",  # network c - muted green
        "#E45756",  # network d - muted red
        "#72B7B2",  # network e - teal
        "#B279A2",  # network f - purple

        "#1f77b4",  # dataset 1
        "#2ca02c",  # dataset 2
        "#666000",  # dataset 3
        "#ff66b2",  # dataset 4
    ]

    NETWORK_COLORS = COLOR_PALETTE[:6]
    DATASET_COLORS = COLOR_PALETTE[6:]

    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "font.family": "Arial",
        "axes.titlesize": SUPTITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    TITLE_FILL = PatternFill("solid", fgColor="D9EAF7")
    HEADER_FILL = PatternFill("solid", fgColor="F4F6F6")
    THIN_BORDER = Border(
        left=Side(style="thin", color="BFBFBF"),
        right=Side(style="thin", color="BFBFBF"),
        top=Side(style="thin", color="BFBFBF"),
        bottom=Side(style="thin", color="BFBFBF"),
    )
    CENTER = Alignment(horizontal="center", vertical="center")


    # ==================================================
    # 2. BASIC UTILITIES
    # ==================================================
    def save_figure_both(fig, output_path, dpi=300, **savefig_kwargs):
        """
        Save the same figure as both TIFF and PNG with the same dpi.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure object.
        output_path : str or Path
            Usually a .tiff path. The PNG path will use the same stem.
        dpi : int
            Export dpi for both formats.
        **savefig_kwargs
            Extra keyword arguments passed to fig.savefig().
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = output_path.suffix.lower()
        if suffix not in [".tif", ".tiff", ".png"]:
            tiff_path = output_path.with_suffix(".tiff")
        elif suffix == ".png":
            tiff_path = output_path.with_suffix(".tiff")
        else:
            tiff_path = output_path

        png_path = tiff_path.with_suffix(".png")

        _apply_arial_font(fig)

        fig.savefig(tiff_path, dpi=dpi, **savefig_kwargs)
        fig.savefig(png_path, dpi=dpi, **savefig_kwargs)

        return tiff_path, png_path


    def clean_name(x) -> str:
        if pd.isna(x):
            return ""
        return str(x).strip()


    def clean_name_for_sheet(name: str) -> str:
        name = str(name).strip()
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"[\\/*?:\[\]]", "-", name)
        return name or "Matrix"


    def safe_sheet_name(name: str) -> str:
        return clean_name_for_sheet(name)[:31]


    def safe_float(x) -> float:
        try:
            if pd.isna(x):
                return np.nan
            return float(x)
        except Exception:
            return np.nan


    def ordered_nodes(nodes: Iterable[str], preferred: List[str] = PREFERRED_NODE_ORDER) -> List[str]:
        """Return nodes ordered by preferred hydrological order, then any remaining nodes."""
        nodes = [str(n).strip() for n in nodes if str(n).strip()]
        preferred_present = [n for n in preferred if n in nodes]
        remaining = [n for n in nodes if n not in preferred_present]
        return preferred_present + remaining


    def style_cell(cell, *, bold=False, fill=None, border=None, align=None, number_format=None, font_size=None):
        cell.font = Font(bold=bold, size=font_size if font_size else 12)
        if fill is not None:
            cell.fill = fill
        if border is not None:
            cell.border = border
        if align is not None:
            cell.alignment = align
        if number_format is not None:
            cell.number_format = number_format


    def autofit_columns(ws, min_width=10, max_width=32):
        widths = {}
        for row in ws.iter_rows():
            for cell in row:
                if cell.value is not None:
                    widths[cell.column] = max(widths.get(cell.column, 0), len(str(cell.value)) + 2)
        for col_idx, width in widths.items():
            ws.column_dimensions[get_column_letter(col_idx)].width = max(min_width, min(width, max_width))


    def format_dataframe_sheet(ws, freeze: str = "A2", float_format: str = "0.000"):
        for cell in ws[1]:
            style_cell(cell, bold=True, fill=HEADER_FILL, border=THIN_BORDER, align=CENTER)
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, float):
                    style_cell(cell, border=THIN_BORDER, align=CENTER, number_format=float_format)
                else:
                    style_cell(cell, border=THIN_BORDER, align=CENTER)
        ws.freeze_panes = freeze
        autofit_columns(ws)


    # ==================================================
    # 3. MATRIX BLOCK READING FROM ORIGINAL WORKBOOK
    # ==================================================
    def find_all_relation_matrix_blocks(
        excel_path: str | Path,
        sheet_name: str,
        title_keyword: str = TITLE_KEYWORD,
    ) -> List[dict]:
        raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None, engine="openpyxl")
        blocks = []
        seen = set()
        keyword = title_keyword.upper()

        for r in range(raw.shape[0]):
            for c in range(raw.shape[1]):
                val = str(raw.iat[r, c]).strip()
                if keyword in val.upper():
                    # Workbook layout: title cell is one column right of block origin.
                    start_row = r
                    start_col = max(c - 1, 0)
                    key = (start_row, start_col)
                    if key not in seen:
                        seen.add(key)
                        blocks.append({
                            "start_row": start_row,
                            "start_col": start_col,
                            "title_cell_text": val,
                        })

        return sorted(blocks, key=lambda x: (x["start_row"], x["start_col"]))


    def extract_matrix_family(title_text: str) -> str:
        t = str(title_text).strip()
        for family in ["Obs", "Mixed-GLEAM", "GLDAS", "ERA5"]:
            if re.search(rf"\b{re.escape(family)}\b", t, flags=re.IGNORECASE):
                return family

        cleaned = re.sub(r"RELATION\s+MATRIX", "", t, flags=re.IGNORECASE)
        cleaned = re.sub(r"matrix", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[^A-Za-z0-9 _-]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return clean_name_for_sheet(cleaned) if cleaned else "Matrix"


    def assign_case_names_from_blocks(blocks: List[dict]) -> List[dict]:
        counters = defaultdict(int)
        for block in blocks:
            family = extract_matrix_family(block["title_cell_text"])
            idx = counters[family]
            counters[family] += 1
            suffix = NETWORK_SUFFIX_BY_INDEX.get(idx, str(idx))
            if family in {"Obs", "Mixed-GLEAM", "GLDAS", "ERA5"}:
                block["case_name"] = f"{family}_{suffix}"
            else:
                block["case_name"] = family if idx == 0 else f"{family}_{suffix}"
        return blocks


    def read_relation_matrix_from_excel(
        excel_path: str | Path,
        sheet_name: str,
        start_row: int,
        start_col: int,
        node_row_offset: int = NODE_ROW_OFFSET,
        data_row_offset: int = DATA_ROW_OFFSET,
        n_nodes: int = N_NODES,
    ) -> pd.DataFrame:
        raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None, engine="openpyxl")

        node_row = start_row + node_row_offset
        data_row = start_row + data_row_offset
        col0 = start_col + 2
        col1 = col0 + n_nodes

        node_names = raw.iloc[node_row, col0:col1].tolist()
        node_names = [str(x).strip() for x in node_names]

        matrix = raw.iloc[data_row:data_row + n_nodes, col0:col1].copy()
        matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        matrix.index = node_names
        matrix.columns = node_names

        if len(set(node_names)) != len(node_names):
            raise ValueError(f"Duplicated node names detected: {node_names}")
        if any(x.lower() == "nan" or x == "" for x in node_names):
            raise ValueError(f"Invalid node names detected: {node_names}")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Matrix must be square, got {matrix.shape}")

        nodes = ordered_nodes(node_names)
        matrix = matrix.loc[nodes, nodes]

        return matrix.astype(float)


    # ==================================================
    # 4. CONTRIBUTION / DELIVERY MATRICES
    # ==================================================
    def normalize_rows(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        A = np.asarray(A, dtype=float)
        row_sum = A.sum(axis=1)
        out = np.zeros_like(A, dtype=float)
        valid = row_sum > 0
        out[valid, :] = A[valid, :] / row_sum[valid, None]
        return out, row_sum


    def normalize_cols(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        A = np.asarray(A, dtype=float)
        col_sum = A.sum(axis=0)
        out = np.zeros_like(A, dtype=float)
        valid = col_sum > 0
        out[:, valid] = A[:, valid] / col_sum[None, valid]
        return out, col_sum


    def safe_inverse(M: np.ndarray, tol: float = 1e-9) -> Tuple[np.ndarray, bool]:
        M = np.asarray(M, dtype=float)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"safe_inverse expects a square matrix, got shape {M.shape}")
        if M.shape[0] == 0:
            raise ValueError("safe_inverse received an empty matrix.")

        singular_values = svd(M, compute_uv=False)
        if singular_values.min() < tol:
            return pinv(M), True
        return inv(M), False


    def dataframe_from_array(A: np.ndarray, nodes: List[str]) -> pd.DataFrame:
        return pd.DataFrame(A, index=nodes, columns=nodes)


    def derive_contribution_delivery_matrices(
        original_matrix: pd.DataFrame,
        percent_as_100: bool = PERCENT_AS_100,
    ) -> Tuple[Dict[str, pd.DataFrame], dict]:
        """
        Derive Bp/Bpp, Cp/Cpp, and Cp_percent/Cpp_percent from the original matrix.
        Centrality is still calculated only on Original_A.
        """
        nodes = list(original_matrix.index)
        A = original_matrix.astype(float).values
        n = A.shape[0]

        Bp, Tin = normalize_cols(A)
        Bpp, Tout = normalize_rows(A)

        Cp_arr, used_pinv_contribution = safe_inverse(np.eye(n) - Bp)
        Cpp_arr, used_pinv_delivery = safe_inverse(np.eye(n) - Bpp)

        Cp = dataframe_from_array(Cp_arr, nodes)
        Cpp = dataframe_from_array(Cpp_arr, nodes)

        Cp_percent_arr, _ = normalize_cols(Cp_arr)
        Cpp_percent_arr, _ = normalize_rows(Cpp_arr)

        scale = 100.0 if percent_as_100 else 1.0

        matrices = {
            "Original_A": original_matrix.copy(),
            "Bp_direct_contribution": dataframe_from_array(Bp, nodes),
            "Bpp_direct_delivery": dataframe_from_array(Bpp, nodes),
            "Cp_total_contribution": Cp,
            "Cpp_total_delivery": Cpp,
            "Cp_percent_column_normalized": dataframe_from_array(Cp_percent_arr * scale, nodes),
            "Cpp_percent_row_normalized": dataframe_from_array(Cpp_percent_arr * scale, nodes),
        }

        diagnostics = {
            "used_pinv_contribution": used_pinv_contribution,
            "used_pinv_delivery": used_pinv_delivery,
            "Tin_min": float(np.min(Tin)) if len(Tin) else np.nan,
            "Tin_max": float(np.max(Tin)) if len(Tin) else np.nan,
            "Tout_min": float(np.min(Tout)) if len(Tout) else np.nan,
            "Tout_max": float(np.max(Tout)) if len(Tout) else np.nan,
            "Cp_percent_col_sum_min": float(matrices["Cp_percent_column_normalized"].sum(axis=0).min()),
            "Cp_percent_col_sum_max": float(matrices["Cp_percent_column_normalized"].sum(axis=0).max()),
            "Cpp_percent_row_sum_min": float(matrices["Cpp_percent_row_normalized"].sum(axis=1).min()),
            "Cpp_percent_row_sum_max": float(matrices["Cpp_percent_row_normalized"].sum(axis=1).max()),
        }
        return matrices, diagnostics


    def calculate_contribution_delivery(Cp: pd.DataFrame, Cpp: pd.DataFrame) -> pd.DataFrame:
        """
        Poster summary metrics:
            Contribution = column sum of Cp
            Delivery     = row sum of Cpp
        """
        nodes = ordered_nodes([n for n in Cp.index if n in Cp.columns])
        Cp = Cp.reindex(index=nodes, columns=nodes).fillna(0.0)
        Cpp = Cpp.reindex(index=nodes, columns=nodes).fillna(0.0)

        return pd.DataFrame({
            "Contribution": Cp.sum(axis=0),
            "Delivery": Cpp.sum(axis=1),
        })


    # ==================================================
    # 5. GRAPH AND CENTRALITY METRICS
    # ==================================================
    def build_graph_from_matrix(matrix_df: pd.DataFrame, include_self_loops: bool = INCLUDE_SELF_LOOPS) -> nx.DiGraph:
        G = nx.DiGraph()
        nodes = list(matrix_df.index)
        G.add_nodes_from([str(n) for n in nodes])

        for source in matrix_df.index:
            for target in matrix_df.columns:
                if not include_self_loops and str(source) == str(target):
                    continue
                value = safe_float(matrix_df.loc[source, target])
                if pd.isna(value) or value <= 0:
                    continue
                if ROW_TO_COLUMN:
                    u, v = str(source), str(target)
                else:
                    u, v = str(target), str(source)
                G.add_edge(u, v, weight=float(value), distance=1.0 / (float(value) + EPS_DISTANCE))
        return G


    def pagerank_ranking_vector(G: nx.DiGraph, node_order: List[str], damping: float = PAGERANK_DAMPING) -> Dict[str, float]:
        if G.number_of_nodes() == 0:
            return {}

        if G.number_of_edges() == 0:
            uniform = 1.0 / max(len(node_order), 1)
            return {node: uniform for node in node_order}

        try:
            ranking = nx.pagerank(G, alpha=damping, weight="weight", max_iter=1000, tol=1e-12)
        except Exception:
            total_weight = {}
            for node in G.nodes():
                in_w = sum(G[p][node]["weight"] for p in G.predecessors(node))
                out_w = sum(G[node][s]["weight"] for s in G.successors(node))
                total_weight[node] = in_w + out_w
            denom = sum(total_weight.values())
            if denom <= 0:
                uniform = 1.0 / max(len(node_order), 1)
                ranking = {node: uniform for node in node_order}
            else:
                ranking = {node: total_weight.get(node, 0.0) / denom for node in node_order}

        ranking = {node: max(float(ranking.get(node, 0.0)), 0.0) for node in node_order}
        denom = sum(ranking.values())
        if denom <= 0:
            uniform = 1.0 / max(len(node_order), 1)
            return {node: uniform for node in node_order}
        return {node: val / denom for node, val in ranking.items()}


    def eigenvector_centrality_weighted(
        G: nx.DiGraph,
        node_order: List[str],
        mode: str = EIGENVECTOR_MODE,
        weight: str = "weight",
        max_iter: int = 1000,
        tol: float = 1e-12,
    ) -> Dict[str, float]:
        """
        Robust weighted eigenvector centrality.

        Values are normalized by their maximum value within each network so the most
        eigenvector-central node equals 1.0. If the eigenvector algorithm fails, the
        function falls back to normalized weighted degree.
        """
        if not node_order:
            return {}

        if G.number_of_edges() == 0:
            return {node: 0.0 for node in node_order}

        if mode == "undirected":
            H = G.to_undirected()
        elif mode == "in":
            H = G.copy()
        elif mode == "out":
            H = G.reverse(copy=True)
        else:
            raise ValueError("EIGENVECTOR_MODE must be 'undirected', 'in', or 'out'.")

        try:
            ev = nx.eigenvector_centrality(H, max_iter=max_iter, tol=tol, weight=weight)
        except Exception:
            try:
                ev = nx.eigenvector_centrality_numpy(H, weight=weight)
            except Exception:
                ev = {}
                for node in node_order:
                    if node in G:
                        in_w = sum(G[p][node].get(weight, 1.0) for p in G.predecessors(node))
                        out_w = sum(G[node][s].get(weight, 1.0) for s in G.successors(node))
                        ev[node] = float(in_w + out_w)
                    else:
                        ev[node] = 0.0

        ev = {node: max(float(ev.get(node, 0.0)), 0.0) for node in node_order}
        max_val = max(ev.values()) if ev else 0.0
        if max_val > 0:
            ev = {node: val / max_val for node, val in ev.items()}
        return ev


    def calculate_original_matrix_metrics(
        A: pd.DataFrame,
        case_name: str,
        alpha: float = STRENGTH_ALPHA,
        include_self_loops: bool = INCLUDE_SELF_LOOPS,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Calculate five metrics from the original relationship matrix only:
            Strength, Closeness, Clustering, Eigenvector.
        """
        nodes = ordered_nodes(A.index)
        A = A.reindex(index=nodes, columns=nodes).fillna(0.0)
        nodes = [str(n) for n in A.index]

        G = build_graph_from_matrix(A, include_self_loops=include_self_loops)

        # ---------- Strength ----------
        strength = {}
        for node in nodes:
            in_weight = sum(G[p][node]["weight"] for p in G.predecessors(node)) if node in G else 0.0
            out_weight = sum(G[node][s]["weight"] for s in G.successors(node)) if node in G else 0.0
            total_weight = in_weight + out_weight
            total_degree = G.in_degree(node) + G.out_degree(node) if node in G else 0
            strength[node] = (
                (total_degree ** alpha) * (total_weight ** (1 - alpha))
                if total_degree > 0 and total_weight > 0
                else 0.0
            )

        # ---------- Closeness ----------
        if G.number_of_edges() > 0:
            if CLOSENESS_MODE == "in":
                closeness = nx.closeness_centrality(G, distance="distance")
            elif CLOSENESS_MODE == "out":
                closeness = nx.closeness_centrality(G.reverse(copy=True), distance="distance")
            elif CLOSENESS_MODE == "undirected":
                closeness = nx.closeness_centrality(G.to_undirected(), distance="distance")
            else:
                raise ValueError("CLOSENESS_MODE must be 'in', 'out', or 'undirected'.")
        else:
            closeness = {node: 0.0 for node in nodes}


        # ---------- Clustering ----------
        if G.number_of_edges() > 0:
            if CLUSTERING_MODE == "undirected":
                Gu = G.to_undirected()
                for u, v in Gu.edges():
                    wuv = G[u][v]["weight"] if G.has_edge(u, v) else 0.0
                    wvu = G[v][u]["weight"] if G.has_edge(v, u) else 0.0
                    Gu[u][v]["weight"] = (wuv + wvu) / 2.0 if wuv > 0 and wvu > 0 else max(wuv, wvu)
                clustering = nx.clustering(Gu, weight="weight")
            elif CLUSTERING_MODE == "directed":
                clustering = nx.clustering(G, weight="weight")
            else:
                raise ValueError("CLUSTERING_MODE must be 'undirected' or 'directed'.")
        else:
            clustering = {node: 0.0 for node in nodes}

        # ---------- Eigenvector ----------
        eigenvector = eigenvector_centrality_weighted(
            G,
            node_order=nodes,
            mode=EIGENVECTOR_MODE,
            weight="weight",
        )

        result = pd.DataFrame({
            "Strength": pd.Series(strength),
            "Closeness": pd.Series(closeness),
            "Clustering": pd.Series(clustering),
            "Eigenvector": pd.Series(eigenvector),
        }).reindex(nodes)

        edge_weights = [data["weight"] for _, _, data in G.edges(data=True)]
        summary = {
            "Model": case_name,
            "N_nodes": G.number_of_nodes(),
            "N_edges_positive": G.number_of_edges(),
            "Self_Loops_Included": include_self_loops,
            "Density": nx.density(G) if G.number_of_nodes() > 1 else 0.0,
            "Total_Weight": float(sum(edge_weights)),
            "Mean_Weight": float(np.mean(edge_weights)) if edge_weights else 0.0,
            "Max_Weight": float(np.max(edge_weights)) if edge_weights else 0.0,
            "Eigenvector_Mode": EIGENVECTOR_MODE,
            "Mean_Strength": float(result["Strength"].mean()),
            "Mean_Closeness": float(result["Closeness"].mean()),
            "Mean_Clustering": float(result["Clustering"].mean()),
            "Mean_Eigenvector": float(result["Eigenvector"].mean()),
        }
        return result, summary


    # ==================================================
    # 6. LONG FORMAT, SUMMARY_ALL, AND NORMALIZATION
    # ==================================================
    def metrics_to_long(model_name: str, metrics_df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for node in metrics_df.index:
            for metric in metrics_df.columns:
                value = metrics_df.loc[node, metric]
                if pd.isna(value):
                    continue
                records.append({
                    "Model": model_name,
                    "Metric": metric,
                    "Node": node,
                    "Value": float(value),
                })
        return pd.DataFrame(records)


    def matrix_to_long(model_name: str, matrix_name: str, matrix_df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for i in matrix_df.index:
            for j in matrix_df.columns:
                records.append({
                    "Model": model_name,
                    "Matrix": matrix_name,
                    "From": i,
                    "To": j,
                    "Value": float(matrix_df.loc[i, j]),
                })
        return pd.DataFrame(records)


    def add_normalized_values(long_df: pd.DataFrame, radar_metrics: Iterable[str] = CENTRALITY_METRICS) -> pd.DataFrame:
        long_df = long_df.copy()
        long_df["Normalized"] = np.nan

        max_by_metric = (
            long_df[long_df["Metric"].isin(radar_metrics)]
            .groupby("Metric")["Value"]
            .max()
        )

        for metric in radar_metrics:
            max_value = max_by_metric.get(metric, np.nan)
            if pd.isna(max_value) or max_value == 0:
                continue
            mask = long_df["Metric"] == metric
            long_df.loc[mask, "Normalized"] = long_df.loc[mask, "Value"] / max_value

        return long_df


    def build_summary_all(long_df: pd.DataFrame, metric_order: Optional[List[str]] = None) -> pd.DataFrame:
        if metric_order is None:
            metric_order = CENTRALITY_METRICS

        rows = []
        model_order = long_df["Model"].drop_duplicates().tolist()

        for model in model_order:
            available_nodes = ordered_nodes(long_df.loc[long_df["Model"] == model, "Node"].unique())
            rows.append([model] + available_nodes)

            for metric in metric_order:
                d = long_df[(long_df["Model"] == model) & (long_df["Metric"] == metric)]
                if d.empty:
                    continue
                value_map = dict(zip(d["Node"], d["Value"]))
                rows.append([metric] + [value_map.get(node, np.nan) for node in available_nodes])

            rows.append([""])

        max_len = max(len(r) for r in rows)
        rows = [r + [""] * (max_len - len(r)) for r in rows]
        return pd.DataFrame(rows)

    def get_radar_axis_limits(values, n_ticks=2):
        """
        Create readable radial-axis limits and ticks for actual-value radar plots.
        Each metric gets its own radial scale.
        """
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return 1.0, [0.50, 1.00]

        vmax = float(np.nanmax(values))

        if vmax <= 0:
            return 1.0, [0.50, 1.00]

        # Add headroom
        ymax = vmax * 1.15

        # Choose a readable rounding step
        if ymax <= 0.1:
            step = 0.02
        elif ymax <= 0.5:
            step = 0.1
        elif ymax <= 1:
            step = 0.2
        elif ymax <= 5:
            step = 1
        elif ymax <= 20:
            step = 5
        elif ymax <= 50:
            step = 10
        else:
            step = 20

        ymax = np.ceil(ymax / step) * step
        ticks = np.linspace(ymax / n_ticks, ymax, n_ticks)

        return ymax, ticks


    # ==================================================
    # 7. RADAR PLOTS
    # ==================================================
    def make_overall_metric_axis_radar_plot(
        long_df: pd.DataFrame,
        model_order: List[str],
        available_nodes: List[str],
        save_path: str | Path,
    ):
        metrics = [metric for metric in CENTRALITY_METRICS if metric in long_df["Metric"].unique()]
        models = [model for model in model_order if model in long_df["Model"].unique()]

        if not metrics:
            raise ValueError("No radar metrics were found.")
        if not models:
            raise ValueError("No models were found for radar plotting.")

        labels = available_nodes
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        fig, axes = plt.subplots(
            1,
            len(metrics),
            figsize=(4.0 * len(metrics), 4.7),
            dpi=300,
            subplot_kw={"polar": True},
        )
        if len(metrics) == 1:
            axes = [axes]

        for idx, (ax, metric) in enumerate(zip(axes, metrics)):
            d = long_df[(long_df["Metric"] == metric) & (long_df["Model"].isin(models))].copy()
            # Use actual metric values, not normalized values
            value_col = "Value"

            pivot = (
                d.pivot_table(index="Model", columns="Node", values=value_col, aggfunc="mean")
                .reindex(models)
                .reindex(columns=available_nodes)
                .fillna(0)
            )

            radar_ymax, radar_yticks = get_radar_axis_limits(pivot.values.flatten())

            for model in models:
                values = pivot.loc[model, available_nodes].tolist()
                values += values[:1]
                ax.plot(angles, values, linewidth=2.0, marker="o", markersize=3.0, label=model)
                ax.fill(angles, values, alpha=0.015)

            ax.set_title(metric, pad=14)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=TICK_SIZE)
            ax.set_ylim(0, radar_ymax)
            ax.set_yticks(radar_yticks)

            if idx == 0:
                ax.set_yticklabels([f"{v:g}" for v in radar_yticks], fontsize=SMALL_TICK_SIZE)
            else:
                ax.set_yticklabels([])
            ax.tick_params(pad=2)
            ax.grid(alpha=0.30)

        handles, labels_legend = axes[0].get_legend_handles_labels()
        leg = fig.legend(
            handles,
            labels_legend,
            loc="lower center",
            ncol=len(models),
            frameon=False,
            bbox_to_anchor=(0.5, -0.15),
            handlelength=1.8,
        )

        for line in leg.get_lines():
            line.set_linewidth(5.0)

        save_figure_both(fig, save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


    def infer_group_name(model_name: str) -> str:
        if "_" in model_name:
            return model_name.split("_")[0]
        if "-" in model_name:
            return model_name.split("-")[0]
        return "All"


    def make_grouped_radar_plots(long_df: pd.DataFrame, available_nodes: List[str], output_dir: str | Path):
        groups: Dict[str, List[str]] = {}
        for model in long_df["Model"].drop_duplicates():
            group = infer_group_name(model)
            groups.setdefault(group, []).append(model)

        for group, models in groups.items():
            if len(models) < 2:
                continue
            save_path = Path(output_dir) / f"radar_metric_axis_{clean_name_for_sheet(group)}.tiff"
            make_overall_metric_axis_radar_plot(
                long_df=long_df,
                model_order=models,
                available_nodes=available_nodes,
                save_path=save_path,
            )


    # ==================================================
    # 8. MEAN BARPLOTS ACROSS ALL NODES
    # ==================================================
    def build_mean_metric_table(
        long_df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Calculate the mean value of each metric across all available nodes for each model/network.
        """
        if metrics is None:
            metrics = CENTRALITY_METRICS

        d = long_df[long_df["Metric"].isin(metrics)].copy()
        if d.empty:
            return pd.DataFrame(columns=["Model", "Metric", "Mean_Value", "N_Nodes"])

        mean_df = (
            d.groupby(["Model", "Metric"], as_index=False)
            .agg(
                Mean_Value=("Value", "mean"),
                N_Nodes=("Node", "nunique"),
            )
        )

        model_order = long_df["Model"].drop_duplicates().tolist()
        metric_order = [m for m in metrics if m in mean_df["Metric"].unique()]

        mean_df["Model"] = pd.Categorical(mean_df["Model"], categories=model_order, ordered=True)
        mean_df["Metric"] = pd.Categorical(mean_df["Metric"], categories=metric_order, ordered=True)
        mean_df = mean_df.sort_values(["Metric", "Model"]).reset_index(drop=True)

        mean_df["Model"] = mean_df["Model"].astype(str)
        mean_df["Metric"] = mean_df["Metric"].astype(str)
        return mean_df


    def assign_same_performance_letters(values: List[float], tolerance_fraction: float = SAME_PERFORMANCE_TOLERANCE_FRACTION) -> List[str]:
        """
        Assign letters to bars with similar values.

        This is a descriptive grouping, not a statistical test. Values within
        tolerance_fraction * range(values) are assigned the same letter.
        """
        arr = np.array(values, dtype=float)
        letters = ["" for _ in values]
        finite_idx = np.where(np.isfinite(arr))[0]
        if len(finite_idx) == 0:
            return letters

        finite_vals = arr[finite_idx]
        value_range = float(np.nanmax(finite_vals) - np.nanmin(finite_vals))
        tol = tolerance_fraction * value_range if value_range > 0 else 0.0

        sorted_idx = sorted(finite_idx, key=lambda i: arr[i], reverse=True)
        groups: List[List[int]] = []
        group_refs: List[float] = []

        for idx in sorted_idx:
            placed = False
            for g_idx, ref_val in enumerate(group_refs):
                if abs(arr[idx] - ref_val) <= tol:
                    groups[g_idx].append(idx)
                    placed = True
                    break
            if not placed:
                groups.append([idx])
                group_refs.append(float(arr[idx]))

        alphabet = "abcdefghijklmnopqrstuvwxyz"
        for g_idx, group in enumerate(groups):
            label = alphabet[g_idx] if g_idx < len(alphabet) else f"g{g_idx+1}"
            for idx in group:
                letters[idx] = label
        return letters


    def split_model_name(model: str) -> Tuple[str, str]:
        """Split a model name such as ERA5_0.5 into dataset and network suffix."""
        model = str(model)
        if "_" in model:
            dataset, suffix = model.split("_", 1)
            return dataset, suffix
        return model, model


    def sort_models_by_suffix(models: List[str]) -> List[str]:
        """
        Sort models within each dataset group as:
        *_0.0, *_0.5, *_1.0
        """
        suffix_order = {
            "0.0": 0,
            "0.5": 1,
            "1.0": 2,
        }

        def sort_key(model: str):
            dataset, suffix = split_model_name(model)
            return suffix_order.get(suffix, 999), suffix

        return sorted(models, key=sort_key)


    def network_code_label(index: int) -> str:
        if index < len(NETWORK_CODE_LETTERS):
            return NETWORK_CODE_LETTERS[index]
        return f"n{index + 1}"


    def build_network_code_maps(models: List[str], include_model_name: bool = True) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Return compact x-tick labels and expanded legend labels for model/network names."""
        tick_label_map = {}
        legend_label_map = {}
        for i, model in enumerate(models):
            code = network_code_label(i)
            tick_label_map[model] = code
            if include_model_name:
                legend_label_map[model] = f"{code}: {model}"
            else:
                legend_label_map[model] = f"{code}"
        return tick_label_map, legend_label_map


    def build_model_color_map(models: List[str]) -> Dict[str, tuple]:
        """Create a consistent model-to-color mapping shared by radar and bar plots."""
        models = list(models)
        cmap_name = "tab10" if len(models) <= 10 else "tab20"
        cmap = plt.get_cmap(cmap_name)
        return {model: cmap(i % cmap.N) for i, model in enumerate(models)}


    def build_fig1_color_map(models: List[str]) -> Dict[str, str]:
        """Use the first six colors of the ten-color Figure-1 palette for networks."""
        return {model: NETWORK_COLORS[i % len(NETWORK_COLORS)] for i, model in enumerate(models)}


    def build_grouped_model_color_map(models, group=None) -> Dict[str, Tuple]:
        models = list(models)
        if not models:
            return {}

        if group is None:
            group = split_model_name(models[0])[0]

        # Map your dataset prefixes to the specific hex colors from DATASET_COLORS
        # Update the dictionary keys to match your actual dataset names
        hex_map = {
            "Obs": DATASET_COLORS[0],
            "Mixed-GLEAM": DATASET_COLORS[1],
            "ERA5": DATASET_COLORS[2],
            "GLDAS": DATASET_COLORS[3],
        }

        # Fetch the base color, falling back to a neutral dark grey if the dataset is unknown
        base_color = hex_map.get(group, "#333333")

        # Generate a custom sequential colormap from white, through the base color, to black
        custom_cmap = LinearSegmentedColormap.from_list("custom",["#F7F7F7", base_color])
        # Avoid very pale and very dark extremes so lines, markers, and bars remain visible
        sample_positions = np.linspace(0.3, 0.95, len(models))
        return {model: custom_cmap(pos) for model, pos in zip(models, sample_positions)}


    def make_mean_metric_barplots(
        mean_df: pd.DataFrame,
        model_order: List[str],
        save_path: str | Path,
        metrics: Optional[List[str]] = None,
        ncols: int = 2,
        add_letters: bool = ADD_SAME_PERFORMANCE_LETTERS,
    ):
        """
        Draw barplots of model/network mean values for each metric.

        Each subplot represents one metric. Each bar represents one model/network.
        Bar height is the average across all available nodes for that metric.
        If add_letters=True, similar values receive the same letter above the bar.
        """
        if metrics is None:
            metrics = CENTRALITY_METRICS

        metrics = [m for m in metrics if m in mean_df["Metric"].unique()]
        models = [m for m in model_order if m in mean_df["Model"].unique()]

        if not metrics:
            raise ValueError("No mean-barplot metrics were found.")
        if not models:
            raise ValueError("No models were found for mean-barplot plotting.")

        ncols = max(1, min(ncols, len(metrics)))
        nrows = int(np.ceil(len(metrics) / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(max(7.5, 0.48 * len(models) * ncols), 3.9 * nrows),
            dpi=300,
            squeeze=False,
            constrained_layout=True,
        )

        axes_flat = axes.ravel()
        x = np.arange(len(models))
        color_map = build_model_color_map(models)

        for ax, metric in zip(axes_flat, metrics):
            d = mean_df[mean_df["Metric"] == metric].copy()
            value_map = dict(zip(d["Model"], d["Mean_Value"]))
            values = [value_map.get(model, np.nan) for model in models]

            bar_colors = [color_map.get(model) for model in models]
            bars = ax.bar(x, values, color=bar_colors, edgecolor="black", linewidth=0.6)
            ax.set_title(metric)
            ax.set_ylabel("Mean value across nodes")
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=55, ha="right")
            ax.grid(True, axis="y", alpha=0.25, linestyle="--")

            finite_values = [v for v in values if pd.notna(v)]
            if finite_values:
                ymin = min(finite_values)
                ymax = max(finite_values)
                yrange = ymax - ymin
                pad = 0.12 * yrange if yrange > 0 else (0.12 * ymax if ymax > 0 else 0.1)
                if ymin >= 0:
                    ax.set_ylim(0, ymax + pad * 2.3)
                else:
                    ax.set_ylim(ymin - pad, ymax + pad * 2.3)

                if add_letters:
                    letters = assign_same_performance_letters(values)
                    text_offset = pad if pad > 0 else 0.05
                    for bar, val, letter in zip(bars, values, letters):
                        if not pd.notna(val) or not letter:
                            continue
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            val + text_offset,
                            letter,
                            ha="center",
                            va="bottom",
                            fontsize=TICK_SIZE,
                            fontweight="bold",
                        )

        for ax in axes_flat[len(metrics):]:
            ax.axis("off")

        save_figure_both(fig, save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


    def make_grouped_mean_metric_barplots(
        mean_df: pd.DataFrame,
        output_dir: str | Path,
        metrics: Optional[List[str]] = None,
    ):
        """Generate separate mean-metric barplots by model prefix."""
        groups: Dict[str, List[str]] = {}
        for model in mean_df["Model"].drop_duplicates():
            group = infer_group_name(model)
            groups.setdefault(group, []).append(model)

        for group, models in groups.items():
            if len(models) < 2:
                continue
            save_path = Path(output_dir) / f"mean_metric_barplots_{clean_name_for_sheet(group)}.tiff"
            make_mean_metric_barplots(
                mean_df=mean_df,
                model_order=models,
                save_path=save_path,
                metrics=metrics,
                ncols=2,
                add_letters=ADD_SAME_PERFORMANCE_LETTERS,
            )


    # ==================================================
    # 9. COMBINED 5 x 2 RADAR + BAR FIGURES
    # ==================================================
    def add_subplot_notation(
        ax,
        label: str,
        x: float = -0.12,
        y: float = 1.08,
        fontsize: float = 24,
    ):
        """
        Add subplot notation such as a, b, c, ... at the upper-left of an axis.
        Works for both normal axes and polar/radar axes.
        """
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight="bold",
            ha="left",
            va="top",
            clip_on=False,
        )


    def make_radar_bar_5x2_plot(
        long_df: pd.DataFrame,
        mean_df: pd.DataFrame,
        model_order: List[str],
        available_nodes: List[str],
        save_path: str | Path,
        title: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        add_letters: bool = ADD_SAME_PERFORMANCE_LETTERS,
        color_map: Optional[Dict[str, tuple]] = None,
        tick_label_map: Optional[Dict[str, str]] = None,
        legend_label_map: Optional[Dict[str, str]] = None,
        legend_title: Optional[str] = None,
        show_legend: bool = True,
        show_bar_metric_titles: bool = False,
    ):
        """
        Create a 5 x 2 poster-style figure.

        Layout
        ------
        - Columns: metrics, normally the five centrality metrics.
        - First row: radar plots, one radar per metric, with nodes as axes and
          networks/models as lines.
        - Second row: mean barplots, one barplot per metric, where each bar is the
          mean metric value across all available nodes for one network/model.

        This figure is designed for directly comparing the same set of networks
        using both node-level distribution patterns (radar) and network-level mean
        metric performance (barplot).
        """
        if metrics is None:
            metrics = CENTRALITY_METRICS

        metrics = [m for m in metrics if m in long_df["Metric"].unique()]
        models = [m for m in model_order if m in long_df["Model"].unique()]

        if not metrics:
            raise ValueError("No metrics were found for the 5x2 radar-bar plot.")
        if not models:
            raise ValueError("No models were found for the 5x2 radar-bar plot.")

        # Keep the intended 5-column structure when possible. If fewer metrics are
        # available, the function still works with the available number of columns.
        n_metrics = len(metrics)
        labels = available_nodes
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        fig = plt.figure(
            figsize=(4 * n_metrics, 9),
            dpi=300,
            constrained_layout=True,
        )
        gs = fig.add_gridspec(
            2,
            n_metrics,
            height_ratios=[0.95, 1.05],
            hspace=0.1,
            wspace=0.1,
        )

        legend_handles = None
        legend_labels = None
        if color_map is None:
            color_map = build_model_color_map(models)
        if tick_label_map is None:
            tick_label_map = {model: model for model in models}
        if legend_label_map is None:
            legend_label_map = {model: model for model in models}

        subplot_labels = list("abcdefghij")
        subplot_id = 0

        for col, metric in enumerate(metrics):
            # --------------------------------------------------
            # Row 1: radar plot
            # --------------------------------------------------
            ax_radar = fig.add_subplot(gs[0, col], projection="polar")

            add_subplot_notation(
                ax_radar,
                subplot_labels[subplot_id],
                x=-0.12,
                y=1.12,
                fontsize=24,
            )
            subplot_id += 1

            d = long_df[
                (long_df["Metric"] == metric) &
                (long_df["Model"].isin(models))
            ].copy()

            # Use actual metric values, not normalized values
            value_col = "Value"

            pivot = (
                d.pivot_table(
                    index="Model",
                    columns="Node",
                    values=value_col,
                    aggfunc="mean",
                )
                .reindex(models)
                .reindex(columns=labels)
                .fillna(0.0)
            )

            radar_ymax, radar_yticks = get_radar_axis_limits(pivot.values.flatten())

            for model in models:
                values = pivot.loc[model, labels].tolist()
                values += values[:1]
                color = color_map.get(model)
                ax_radar.plot(
                    angles,
                    values,
                    linewidth=2,
                    marker="o",
                    markersize=2.8,
                    label=legend_label_map.get(model, model),
                    color=color,
                )
                ax_radar.fill(angles, values, alpha=0.012, color=color)

            ax_radar.set_title(metric, pad=13, fontweight="bold")
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(labels, fontsize=TICK_SIZE)
            ax_radar.set_ylim(0, radar_ymax)
            ax_radar.set_yticks(radar_yticks)

            ax_radar.set_yticklabels([f"{v:g}" for v in radar_yticks], fontsize=TICK_SIZE)

            ax_radar.grid(alpha=0.30)

            if col == 0:
                legend_handles, legend_labels = ax_radar.get_legend_handles_labels()

            # --------------------------------------------------
            # Row 2: mean barplot across nodes
            # --------------------------------------------------
            ax_bar = fig.add_subplot(gs[1, col])

            add_subplot_notation(
                ax_bar,
                subplot_labels[subplot_id],
                x=-0.12,
                y=1.12,
                fontsize=24,
            )
            subplot_id += 1

            d_mean = mean_df[mean_df["Metric"] == metric].copy()
            value_map = dict(zip(d_mean["Model"], d_mean["Mean_Value"]))
            values = [value_map.get(model, np.nan) for model in models]

            x = np.arange(len(models))
            bar_colors = [color_map.get(model) for model in models]
            bars = ax_bar.bar(x, values, color=bar_colors, edgecolor="black", linewidth=0.6)
            if show_bar_metric_titles:
                ax_bar.set_title(metric, fontweight="bold")
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(
                [tick_label_map.get(model, model) for model in models],
                rotation=0,
                ha="center",
                fontsize=TICK_SIZE,
            )
            ax_bar.grid(True, axis="y", alpha=0.25, linestyle="--")

            if col == 0:
                ax_bar.set_ylabel("Mean value across nodes", fontsize=LABEL_SIZE, labelpad=8)
            else:
                ax_bar.set_ylabel("")

            finite_values = [v for v in values if pd.notna(v)]
            if finite_values:
                ymin = min(finite_values)
                ymax = max(finite_values)
                yrange = ymax - ymin
                pad = 0.12 * yrange if yrange > 0 else (0.12 * ymax if ymax > 0 else 0.1)
                if ymin >= 0:
                    ax_bar.set_ylim(0, ymax + pad * 2.5)
                else:
                    ax_bar.set_ylim(ymin - pad, ymax + pad * 2.5)

                if add_letters:
                    letters = assign_same_performance_letters(values)
                    text_offset = pad if pad > 0 else 0.05
                    for bar, val, letter in zip(bars, values, letters):
                        if not pd.notna(val) or not letter:
                            continue
                        ax_bar.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            val + text_offset,
                            letter,
                            ha="center",
                            va="bottom",
                            fontsize=TICK_SIZE,
                            fontweight="bold",
                        )

        if title:
            fig.suptitle(title, fontsize=SUPTITLE_SIZE, fontweight="bold")

        if show_legend and legend_handles and legend_labels:
            leg = fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                ncol=len(models),
                frameon=False,
                bbox_to_anchor=(0.5, -0.15),
                handlelength=1.7,
                title=legend_title,
                title_fontsize=LEGEND_SIZE,
                fontsize=LEGEND_SIZE,
            )

            for line in leg.get_lines():
                line.set_linewidth(5.0)

        save_figure_both(fig, save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


    def make_grouped_radar_bar_5x2_plots(
        long_df: pd.DataFrame,
        mean_df: pd.DataFrame,
        available_nodes: List[str],
        output_dir: str | Path,
        metrics: Optional[List[str]] = None,
    ):
        """Generate separate 5 x 2 radar-bar figures by dataset/model prefix."""
        groups: Dict[str, List[str]] = {}
        for model in long_df["Model"].drop_duplicates():
            group = infer_group_name(model)
            groups.setdefault(group, []).append(model)

        for group, models in groups.items():
            models = sort_models_by_suffix(models)

            if len(models) < 2:
                continue
            save_path = Path(output_dir) / f"radar_bar_{clean_name_for_sheet(group)}.tiff"
            tick_label_map, legend_label_map = build_network_code_maps(models, include_model_name=True)
            make_radar_bar_5x2_plot(
                long_df=long_df,
                mean_df=mean_df,
                model_order=models,
                available_nodes=available_nodes,
                save_path=save_path,
                title=None,
                metrics=metrics,
                add_letters=ADD_SAME_PERFORMANCE_LETTERS,
                color_map=build_grouped_model_color_map(models, group=group),
                tick_label_map=tick_label_map,
                legend_label_map=legend_label_map,
                legend_title="Global assessment networks",
                show_bar_metric_titles=False,
            )


    # ==================================================
    # 10. EXPORT
    # ==================================================
    def export_excel(
        long_df: pd.DataFrame,
        summary_all: pd.DataFrame,
        case_summary_df: pd.DataFrame,
        diagnostics_df: pd.DataFrame,
        matrix_long_df: Optional[pd.DataFrame],
        mean_metric_df: Optional[pd.DataFrame],
        output_excel: str | Path = OUTPUT_EXCEL,
    ):
        with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
            long_df.to_excel(writer, sheet_name="Metrics_long", index=False)
            summary_all.to_excel(writer, sheet_name="Summary_all", index=False, header=False)
            case_summary_df.to_excel(writer, sheet_name="Case_summary", index=False)
            diagnostics_df.to_excel(writer, sheet_name="Matrix_diagnostics", index=False)

            if matrix_long_df is not None and not matrix_long_df.empty:
                matrix_long_df.to_excel(writer, sheet_name="Matrices_long", index=False)

            if mean_metric_df is not None and not mean_metric_df.empty:
                mean_metric_df.to_excel(writer, sheet_name="Mean_by_model_metric", index=False)
                mean_wide = (
                    mean_metric_df.pivot_table(
                        index="Model",
                        columns="Metric",
                        values="Mean_Value",
                        aggfunc="mean",
                    )
                )
                mean_wide.to_excel(writer, sheet_name="Mean_metrics_wide")

            available_nodes = ordered_nodes(long_df["Node"].unique())
            for metric in CENTRALITY_METRICS:
                d = long_df[long_df["Metric"] == metric]
                if d.empty:
                    continue
                wide = (
                    d.pivot_table(index="Model", columns="Node", values="Value", aggfunc="mean")
                    .reindex(columns=available_nodes)
                )
                sheet_name = metric.replace("-", "_")[:31]
                wide.to_excel(writer, sheet_name=sheet_name)

            for sheet_name in writer.book.sheetnames:
                ws = writer.book[sheet_name]
                if sheet_name != "Summary_all":
                    format_dataframe_sheet(ws, freeze="A2", float_format="0.000")
                else:
                    autofit_columns(ws)


    # ==================================================
    # 11. MAIN WORKFLOW
    # ==================================================
    def main():
        blocks = find_all_relation_matrix_blocks(INPUT_FILE, INPUT_SHEET, TITLE_KEYWORD)
        if not blocks:
            raise ValueError(f"No relation matrix blocks found using keyword: {TITLE_KEYWORD}")

        blocks = assign_case_names_from_blocks(blocks)

        print(f"Detected {len(blocks)} relation matrix blocks:")
        for block in blocks:
            print(f"  - {block['case_name']} <- {block['title_cell_text']}")

        all_long = []
        case_summaries = []
        diagnostics = []
        matrix_long_parts = []

        for block in blocks:
            case_name = block["case_name"]
            print(f"\nProcessing: {case_name}")

            try:
                A = read_relation_matrix_from_excel(
                    excel_path=INPUT_FILE,
                    sheet_name=INPUT_SHEET,
                    start_row=block["start_row"],
                    start_col=block["start_col"],
                    node_row_offset=NODE_ROW_OFFSET,
                    data_row_offset=DATA_ROW_OFFSET,
                    n_nodes=N_NODES,
                )

                matrices, diag = derive_contribution_delivery_matrices(A, percent_as_100=PERCENT_AS_100)
                diag.update({
                    "Model": case_name,
                    "Title": block["title_cell_text"],
                    "N_nodes": int(A.shape[0]),
                    "N_edges_positive_raw_including_diagonal": int((A.values > 0).sum()),
                    "Total_Original_Weight": float(A.values.sum()),
                })
                diagnostics.append(diag)

                centrality_df, summary = calculate_original_matrix_metrics(
                    A,
                    case_name=case_name,
                    alpha=STRENGTH_ALPHA,
                    include_self_loops=INCLUDE_SELF_LOOPS,
                )
                summary.update({"Title": block["title_cell_text"]})
                case_summaries.append(summary)

                combined_df = centrality_df.copy()
                if INCLUDE_CONTRIBUTION_DELIVERY:
                    cd_df = calculate_contribution_delivery(
                        matrices["Cp_total_contribution"],
                        matrices["Cpp_total_delivery"],
                    )
                    combined_df = pd.concat([cd_df, combined_df], axis=1)

                all_long.append(metrics_to_long(case_name, combined_df))

                if EXPORT_MATRIX_LONG:
                    for matrix_name, matrix_df in matrices.items():
                        matrix_long_parts.append(matrix_to_long(case_name, matrix_name, matrix_df))

                print("  Done.")

            except Exception as exc:
                print(f"  Skipped {case_name}: {exc}")

        if not all_long:
            raise ValueError("No model case was successfully processed.")

        long_df = pd.concat(all_long, ignore_index=True)
        long_df = add_normalized_values(long_df, radar_metrics=CENTRALITY_METRICS)

        available_nodes = ordered_nodes(long_df["Node"].unique())
        summary_all = build_summary_all(long_df, metric_order=CENTRALITY_METRICS)
        case_summary_df = pd.DataFrame(case_summaries)
        diagnostics_df = pd.DataFrame(diagnostics)
        matrix_long_df = pd.concat(matrix_long_parts, ignore_index=True) if matrix_long_parts else None
        mean_metric_df = build_mean_metric_table(long_df, metrics=CENTRALITY_METRICS)

        export_excel(
            long_df=long_df,
            summary_all=summary_all,
            case_summary_df=case_summary_df,
            diagnostics_df=diagnostics_df,
            matrix_long_df=matrix_long_df,
            mean_metric_df=mean_metric_df,
            output_excel=OUTPUT_EXCEL,
        )
        print(f"\nExported combined workbook to: {OUTPUT_EXCEL}")

        if EXPORT_FIGURES:
            model_order_all = long_df["Model"].drop_duplicates().tolist()
            model_order_limited = model_order_all[:RADAR_MODEL_LIMIT]

            print(f"\nMain radar plot includes the first {len(model_order_limited)} models:")
            for model in model_order_limited:
                print("  -", model)

            fig1_tick_label_map, fig1_legend_label_map = build_network_code_maps(
                model_order_limited,
                include_model_name=True,
            )
            make_radar_bar_5x2_plot(
                long_df=long_df,
                mean_df=mean_metric_df,
                model_order=model_order_limited,
                available_nodes=available_nodes,
                save_path=OUTPUT_DIR / "radar_bar_overall.tiff",
                metrics=CENTRALITY_METRICS,
                add_letters=ADD_SAME_PERFORMANCE_LETTERS,
                color_map=build_fig1_color_map(model_order_limited),
                tick_label_map=fig1_tick_label_map,
                legend_label_map=fig1_legend_label_map,
                legend_title="Global assessment networks",
                show_bar_metric_titles=False,
            )
            print("Saved:", OUTPUT_DIR / "radar_bar_overall.tiff")

            if EXPORT_GROUPED_RADAR_PLOTS:
                make_grouped_radar_bar_5x2_plots(
                    long_df=long_df,
                    mean_df=mean_metric_df,
                    available_nodes=available_nodes,
                    output_dir=OUTPUT_DIR,
                    metrics=CENTRALITY_METRICS,
                )
                print("Saved grouped radar-bar plots in:", OUTPUT_DIR)

        print("\nFinished. All outputs are in:", OUTPUT_DIR)

    _public = {k: v for k, v in locals().items() if not k.startswith("__")}
    return types.SimpleNamespace(**_public)


def create_sensitivity_workflow(input_file: str | Path, output_root: str | Path):
    """Create the integrated groundwater-routing sensitivity workflow namespace."""
    import re
    import warnings
    from pathlib import Path
    from typing import Dict, List, Optional, Tuple

    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    import networkx as nx
    import numpy as np
    import pandas as pd
    from numpy.linalg import inv, pinv, svd

    warnings.filterwarnings("ignore")


    # =========================================================
    # GLOBAL FIGURE FONT CONFIGURATION
    # =========================================================
    # Apply Arial to ordinary text and to Matplotlib math-text notation.
    FIGURE_FONT_FAMILY = "Arial"

    plt.rcParams.update({
        "font.family": FIGURE_FONT_FAMILY,
        "font.sans-serif": [FIGURE_FONT_FAMILY, "Arimo", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": FIGURE_FONT_FAMILY,
        "mathtext.it": f"{FIGURE_FONT_FAMILY}:italic",
        "mathtext.bf": f"{FIGURE_FONT_FAMILY}:bold",
        "mathtext.sf": FIGURE_FONT_FAMILY,
        "mathtext.tt": FIGURE_FONT_FAMILY,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


    def _apply_arial_font(fig) -> None:
        """Force Arial on every Matplotlib Text artist before export."""
        for text_artist in fig.findobj(match=Text):
            text_artist.set_fontfamily(FIGURE_FONT_FAMILY)

    # =========================================================
    # 1. DATA AND ANALYSIS CONFIGURATION
    # =========================================================
    INPUT = Path(input_file)
    SHEET = "Network matrices"
    OUTDIR = Path(output_root) / "sensitivity_outputs"
    OUTDIR.mkdir(exist_ok=True)

    # The source workbook contains 8-node matrices, including SEA.
    # For Cp/Cpp/Cdelta we keep the same no-SEA retained-node system as the 1-Fraction workflow.
    # Important: SEA is not deleted before normalization; Bp/Bpp are first computed from the full
    # original matrix, then the no-SEA terrestrial nodes are extracted for the inverse calculation.
    N_NODES = 8
    KEEP_NODES = ["Aw", "VEG", "SM", "SW", "GW", "SP/ENG", "GL"]
    KEEP_METRIC_NODES = ["Aw", "VEG", "SM", "SW", "GW", "SP/ENG", "SEA", "GL"]

    # Start rows of the alpha=0.5/reference matrices in the workbook.
    DATASETS = {
        "Obs": 136,
        "Mixed-GLEAM": 158,
        "GLDAS": 180,
        "ERA5": 202,
    }

    # Existing workbook layout:
    # start_col 15 => alpha=0, start_col 0 => alpha=0.5, start_col 30 => alpha=1.
    MID_START_COL = 0

    # Sensitivity experiment: redistribute SM -> GW and SM -> SW.
    ALPHAS = np.round(np.linspace(0, 1, 21), 2)
    PLOT_X_MODE = "GW_OVER_SW_PLUS_GW"
    PLOT_X_COLUMN = "Recharge_Rate_GW_fraction"
    PLOT_X_LABEL = r"$\alpha_{GW}$"

    # Fixed/reference-network overlay. These are the six non-dataset networks
    # detected from the workbook and shown as star points on the same x-axis.
    INCLUDE_REFERENCE_NETWORKS = True
    REFERENCE_NETWORK_LIMIT = 6
    REFERENCE_NETWORK_NAME_FILTERS: List[str] = []  # optional exact control, e.g. ["Oki", "Trenberth"]
    REFERENCE_TITLE_KEYWORD = "RELATION MATRIX"

    # Relative-average settings copied from the validated 1-Fraction workflow.
    # Near-zero cumulative-effect values are excluded from the denominator.
    # With the no-SEA 7-node system, one effective non-zero value in a row/column becomes 100%, not 700%.
    RELATIVE_EPS = 1e-6
    PERCENT_AS_100 = False  # False: average = 1.0; True: average = 100.

    # Centrality settings.
    INCLUDE_SELF_LOOPS = False
    PAGERANK_DAMPING = 0.85
    STRENGTH_ALPHA = 0.5
    DISTANCE_EPS = 1e-4
    EIGENVECTOR_MODE = "undirected"  # "undirected", "in", or "out"

    CENTRALITY_METRICS = [
        "Strength",
        "Closeness",
        "Clustering",
        "Eigenvector",
    ]

    # =========================================================
    # 2. SEPARATE FIGURE CONFIGURATIONS
    # =========================================================
    COMMON_STYLE = {
        "font_family": "Arial",
        "dataset_linewidth": 2,
        "dataset_markersize": 5,
        "grid_alpha": 0.18,
        "grid_linewidth": 0.45,
        "legend_ncol": 5,
        "legend_markerscale": 2.0,
        "reference_marker": "*",
        "reference_size": 150,
        "reference_linewidth": 1,
        "reference_alpha": 0.88,
        "colors": {"Obs": "#1f77b4", "Mixed-GLEAM": "#2ca02c", "GLDAS": "#666000", "ERA5": "#ff66b2"},
        "markers": {"Obs": "o", "Mixed-GLEAM": "D", "GLDAS": "s", "ERA5": "^"},
    }

    CENTRALITY_FIG = {
        "basename": "sensitivity_centrality_metrics",
        "dpi": 300,
        "figsize_per_col": 2.5,
        "figsize_per_row": 2.5,
        "figsize_extra_y": 0.85,
        "font_title": 25,
        "font_tick": 25,
        "font_axis_label": 25,
        "font_legend": 25,
        "layout": {
            "left": 0.045,
            "right": 0.99,
            "top": 0.925,
            "bottom": 0.125,
            "hspace": 0.50,
            "wspace": 0.30,
        },
        "legend_bbox": (0.5, -0.06),
        "xticks": [0, 0.50, 1.0],
        "share_row_ylim": True,
        "show_y_ticks_first_col_only": True,
    }

    MATRIX_FIG = {
        "dpi": 300,
        "figsize_per_col": 2.8,
        "figsize_per_row": 2.4,
        "figsize_extra_y": 1.15,
        "font_title": 25,
        "font_tick": 25,
        "font_axis_label": 25,
        "font_legend": 25,
        "layout": {
            "left": 0.060,
            "right": 0.992,
            "top": 0.905,
            "bottom": 0.140,
            "hspace": 0.50,
            "wspace": 0.30,
        },
        "legend_bbox": (0.5, 0.02),
        "xticks": [0, 0.50, 1.0],
        "share_row_ylim": True,
        "show_y_ticks_first_col_only": True,
    }

    MATRIX_PLOTS = [
        {
            "basename": "sensitivity_contribution_matrix",
            "metric_group": "Contribution_Matrix",
            "metric_prefix": "CpPercent",
        },
        {
            "basename": "sensitivity_delivery_matrix",
            "metric_group": "Delivery_Matrix",
            "metric_prefix": "CppPercent",
        },
        {
            "basename": "sensitivity_delta_matrix",
            "metric_group": "Delta_Matrix",
            "metric_prefix": "DeltaPercent",
        },
    ]

    plt.rcParams.update(
        {
            "font.family": COMMON_STYLE["font_family"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": CENTRALITY_FIG["dpi"],
        }
    )


    # =========================================================
    # 3. MATRIX READING AND TRANSFORMATION
    # =========================================================
    def save_figure_both(fig, output_path, dpi=300, **savefig_kwargs):
        """
        Save the same figure as both TIFF and PNG with the same dpi.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure object.
        output_path : str or Path
            Usually a .tiff path. The PNG path will use the same stem.
        dpi : int
            Export dpi for both formats.
        **savefig_kwargs
            Extra keyword arguments passed to fig.savefig().
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = output_path.suffix.lower()
        if suffix not in [".tif", ".tiff", ".png"]:
            tiff_path = output_path.with_suffix(".tiff")
        elif suffix == ".png":
            tiff_path = output_path.with_suffix(".tiff")
        else:
            tiff_path = output_path

        png_path = tiff_path.with_suffix(".png")

        _apply_arial_font(fig)

        fig.savefig(tiff_path, dpi=dpi, **savefig_kwargs)
        fig.savefig(png_path, dpi=dpi, **savefig_kwargs)

        return tiff_path, png_path


    def read_matrix(raw: pd.DataFrame, start_row: int, start_col: int, n_nodes: int = N_NODES) -> pd.DataFrame:
        """Read one relation-matrix block from the workbook."""
        node_row = start_row + 3
        data_row = start_row + 4
        col0 = start_col + 2
        col1 = col0 + n_nodes

        nodes = [str(x).strip() for x in raw.iloc[node_row, col0:col1].tolist()]
        matrix = raw.iloc[data_row:data_row + n_nodes, col0:col1].copy()
        matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        matrix.index = nodes
        matrix.columns = nodes

        if len(set(nodes)) != len(nodes):
            raise ValueError(f"Duplicated node names detected: {nodes}")
        if any(node == "" or node.lower() == "nan" for node in nodes):
            raise ValueError(f"Invalid node names detected: {nodes}")

        return matrix.astype(float)



    def clean_reference_name(text: str) -> str:
        """Convert a relation-matrix title into a clean reference-network name."""
        name = str(text).strip()

        # Remove the matrix-title wording.
        name = re.sub(r"RELATION\s+MATRIX", "", name, flags=re.IGNORECASE)
        name = re.sub(r"\bmatrix\b", "", name, flags=re.IGNORECASE)

        # Remove all leading punctuation, dashes, underscores and spaces.
        name = re.sub(r"^[^A-Za-z0-9]+", "", name)

        # Convert remaining separators to a single underscore.
        name = re.sub(r"[^A-Za-z0-9]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")

        return name if name else "Reference"


    def infer_dataset_group(name: str) -> str:
        """Infer whether a workbook block belongs to one of the four dataset groups."""
        text = str(name).strip()
        for group in DATASETS:
            if re.search(rf"\b{re.escape(group)}\b", text, flags=re.IGNORECASE):
                return group
        if "_" in text:
            return text.split("_")[0]
        if "-" in text:
            return text.split("-")[0]
        return "Reference"


    def find_relation_matrix_blocks(raw: pd.DataFrame, title_keyword: str = REFERENCE_TITLE_KEYWORD) -> List[dict]:
        """Find all relation-matrix blocks in the workbook."""
        blocks: List[dict] = []
        seen = set()
        keyword = title_keyword.upper()

        for r in range(raw.shape[0]):
            for c in range(raw.shape[1]):
                value = str(raw.iat[r, c]).strip()
                if keyword in value.upper():
                    start_row = r
                    start_col = max(c - 1, 0)
                    key = (start_row, start_col)
                    if key in seen:
                        continue
                    seen.add(key)
                    name = clean_reference_name(value)
                    blocks.append({
                        "start_row": start_row,
                        "start_col": start_col,
                        "title_cell_text": value,
                        "reference_name": name,
                        "reference_group": infer_dataset_group(name),
                    })

        return sorted(blocks, key=lambda item: (item["start_row"], item["start_col"]))


    def make_unique_reference_names(blocks: List[dict]) -> List[dict]:
        """Ensure reference names are unique while preserving workbook order."""
        counters: Dict[str, int] = {}
        output: List[dict] = []

        for block in blocks:
            base = block.get("reference_name", "Reference") or "Reference"
            count = counters.get(base, 0)
            counters[base] = count + 1
            block = dict(block)
            block["reference_name"] = base if count == 0 else f"{base}_{count}"
            block["reference_group"] = infer_dataset_group(block["reference_name"])
            output.append(block)

        return output


    def select_reference_blocks(blocks: List[dict]) -> List[dict]:
        """Select up to six fixed/reference networks for star overlays."""
        if not INCLUDE_REFERENCE_NETWORKS or not blocks:
            return []

        if REFERENCE_NETWORK_NAME_FILTERS:
            filters = [item.lower() for item in REFERENCE_NETWORK_NAME_FILTERS]
            selected = []
            for block in blocks:
                text = f"{block.get('reference_name', '')} {block.get('title_cell_text', '')}".lower()
                if any(item in text for item in filters):
                    selected.append(block)
            return selected[:REFERENCE_NETWORK_LIMIT]

        dataset_groups = set(DATASETS.keys())
        candidates = [block for block in blocks if block.get("reference_group") not in dataset_groups]
        if not candidates:
            candidates = blocks
        return candidates[:REFERENCE_NETWORK_LIMIT]


    def build_reference_color_map(names: List[str]) -> Dict[str, tuple]:
        """Assign stable tab10/tab20 colors to the selected fixed networks."""
        cmap = plt.get_cmap("tab10" if len(names) <= 10 else "tab20")
        return {name: cmap(i % cmap.N) for i, name in enumerate(names)}

    def normalize_cols(A: np.ndarray, eps: float = RELATIVE_EPS) -> Tuple[np.ndarray, np.ndarray]:
        """Column-normalize the original relation matrix to obtain Bp."""
        A = np.asarray(A, dtype=float)
        col_sum = A.sum(axis=0)
        out = np.zeros_like(A, dtype=float)
        valid = np.abs(col_sum) > eps
        out[:, valid] = A[:, valid] / col_sum[None, valid]
        return out, col_sum


    def normalize_rows(A: np.ndarray, eps: float = RELATIVE_EPS) -> Tuple[np.ndarray, np.ndarray]:
        """Row-normalize the original relation matrix to obtain Bpp."""
        A = np.asarray(A, dtype=float)
        row_sum = A.sum(axis=1)
        out = np.zeros_like(A, dtype=float)
        valid = np.abs(row_sum) > eps
        out[valid, :] = A[valid, :] / row_sum[valid, None]
        return out, row_sum


    def safe_inverse(M: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        """Return inv(M), or pinv(M) if M is numerically close to singular."""
        M = np.asarray(M, dtype=float)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"safe_inverse expects a square matrix, got shape {M.shape}")

        singular_values = svd(M, compute_uv=False)
        return pinv(M) if singular_values.min() < tol else inv(M)


    def relative_to_axis_mean(
        A: np.ndarray,
        axis: int,
        eps: float = RELATIVE_EPS,
        output_scale: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert raw cumulative-effect values to relative-average values.

        This matches the validated 1-Fraction calculation:

            Cp_percent[i, j]  = scale * Cp[i, j]  / mean_i_nonzero(Cp[:, j])
            Cpp_percent[i, j] = scale * Cpp[i, j] / mean_j_nonzero(Cpp[i, :])

        Effective zero / near-zero entries, abs(value) <= eps, are ignored when
        calculating the denominator and are kept as 0 in the output. Therefore, if
        a 7-node column or row has only one effective non-zero value, that value is
        100% when PERCENT_AS_100=True, not 700%.
        """
        A = np.asarray(A, dtype=float)
        out = np.full_like(A, np.nan, dtype=float)

        if output_scale is None:
            output_scale = 100.0 if PERCENT_AS_100 else 1.0

        finite = np.isfinite(A)
        effective = finite & (np.abs(A) > eps)

        if axis == 0:
            counts = effective.sum(axis=0)
            sums = np.where(effective, A, 0.0).sum(axis=0)
            means = np.full(A.shape[1], np.nan, dtype=float)
            valid = counts > 0
            means[valid] = sums[valid] / counts[valid]

            out[:, valid] = 0.0
            out[:, valid] = np.where(
                effective[:, valid],
                A[:, valid] / means[None, valid] * output_scale,
                0.0,
            )

        elif axis == 1:
            counts = effective.sum(axis=1)
            sums = np.where(effective, A, 0.0).sum(axis=1)
            means = np.full(A.shape[0], np.nan, dtype=float)
            valid = counts > 0
            means[valid] = sums[valid] / counts[valid]

            out[valid, :] = 0.0
            out[valid, :] = np.where(
                effective[valid, :],
                A[valid, :] / means[valid, None] * output_scale,
                0.0,
            )

        else:
            raise ValueError("axis must be 0 for column mean or 1 for row mean.")

        return out, means, counts


    def relative_to_column_average_nonzero(A: np.ndarray, eps: float = RELATIVE_EPS) -> np.ndarray:
        """Backward-compatible wrapper for Cp relative-average calculation."""
        return relative_to_axis_mean(A, axis=0, eps=eps)[0]


    def relative_to_row_average_nonzero(A: np.ndarray, eps: float = RELATIVE_EPS) -> np.ndarray:
        """Backward-compatible wrapper for Cpp relative-average calculation."""
        return relative_to_axis_mean(A, axis=1, eps=eps)[0]


    def derive_matrices(original_matrix: pd.DataFrame, keep_nodes: List[str] = KEEP_NODES) -> Dict[str, pd.DataFrame]:
        """Derive no-SEA Cp, Cpp, relative-average Cp/Cpp, and Cdelta matrices.

        Calculation order follows the validated 1-Fraction workflow:
        full A -> Bp/Bpp normalization -> no-SEA retained submatrix -> inverse.
        """
        nodes = list(original_matrix.index)
        missing = [node for node in keep_nodes if node not in nodes]
        if missing:
            raise ValueError(f"Missing keep_nodes in matrix: {missing}")

        A = original_matrix.astype(float).values
        Bp_full, _ = normalize_cols(A)
        Bpp_full, _ = normalize_rows(A)

        Bp_full_df = pd.DataFrame(Bp_full, index=nodes, columns=nodes)
        Bpp_full_df = pd.DataFrame(Bpp_full, index=nodes, columns=nodes)

        Bp = Bp_full_df.loc[keep_nodes, keep_nodes].values
        Bpp = Bpp_full_df.loc[keep_nodes, keep_nodes].values

        I = np.eye(len(keep_nodes))
        Cp = safe_inverse(I - Bp)
        Cpp = safe_inverse(I - Bpp)

        Cp_ra, Cp_mean, Cp_count = relative_to_axis_mean(Cp, axis=0)
        Cpp_ra, Cpp_mean, Cpp_count = relative_to_axis_mean(Cpp, axis=1)
        Delta = Cp_ra - Cpp_ra

        return {
            "Cp_percent": pd.DataFrame(Cp_ra, index=keep_nodes, columns=keep_nodes),
            "Cpp_percent": pd.DataFrame(Cpp_ra, index=keep_nodes, columns=keep_nodes),
            "Delta_percent": pd.DataFrame(Delta, index=keep_nodes, columns=keep_nodes),
        }


    def make_alpha_matrix(mid_matrix: pd.DataFrame, alpha: float) -> pd.DataFrame:
        """
        Modify groundwater recharge partitioning.

        alpha controls the SM -> GW share of the combined SM -> GW and
        SM -> SW flux pool. GW -> SW is set equal to SM -> GW to preserve
        groundwater-pathway throughput.
        """
        matrix = mid_matrix.copy()
        required = ["SM", "GW", "SW"]
        missing = [node for node in required if node not in matrix.index or node not in matrix.columns]
        if missing:
            raise ValueError(f"Missing nodes for alpha experiment: {missing}")

        total = float(mid_matrix.loc["SM", "GW"] + mid_matrix.loc["SM", "SW"])
        matrix.loc["SM", "GW"] = alpha * total
        matrix.loc["GW", "SW"] = alpha * total
        matrix.loc["SM", "SW"] = (1.0 - alpha) * total
        return matrix


    def allocation_fractions(matrix: pd.DataFrame) -> Tuple[float, float]:
        """
        Return direct-SW and GW-contribution fractions used for the x-axis.

        For the synthetic alpha experiment, SM -> GW and GW -> SW are forced to
        be equal, so either one gives the same alpha_GW.

        For the six fixed/reference networks, however, SM -> GW and GW -> SW
        are not guaranteed to be equal. If the x-axis is intended to represent
        groundwater contribution to runoff, alpha_GW should be based on GW -> SW:

            alpha_GW = (GW -> SW) / [(SM -> SW) + (GW -> SW)]
        """
        required = [("SM", "SW"), ("GW", "SW")]
        missing = [
            f"{i}->{j}"
            for i, j in required
            if i not in matrix.index or j not in matrix.columns
        ]
        if missing:
            raise ValueError(f"Cannot calculate alpha_GW; missing links: {missing}")

        sw = float(matrix.loc["SM", "SW"])   # direct SM contribution to SW/runoff
        gw = float(matrix.loc["GW", "SW"])     # groundwater contribution to SW/runoff

        denom = sw + gw
        if denom <= 0 or not np.isfinite(denom):
            return np.nan, np.nan

        return sw / denom, gw / denom


    def get_plot_x_from_matrix(matrix: pd.DataFrame) -> Tuple[float, float, float]:
        """Return x-position, SW fraction, and GW fraction."""
        sw_fraction, gw_fraction = allocation_fractions(matrix)
        if PLOT_X_MODE == "SW_OVER_SW_PLUS_GW":
            return sw_fraction, sw_fraction, gw_fraction
        if PLOT_X_MODE == "GW_OVER_SW_PLUS_GW":
            return gw_fraction, sw_fraction, gw_fraction
        raise ValueError("PLOT_X_MODE must be 'SW_OVER_SW_PLUS_GW' or 'GW_OVER_SW_PLUS_GW'.")


    # =========================================================
    # 4. CENTRALITY METRICS
    # =========================================================
    def build_graph_from_matrix(matrix: pd.DataFrame, include_self_loops: bool = INCLUDE_SELF_LOOPS) -> nx.DiGraph:
        """Build a weighted directed graph from the original relation matrix."""
        G = nx.DiGraph()
        for node in matrix.index:
            G.add_node(str(node))

        for source in matrix.index:
            for target in matrix.columns:
                if not include_self_loops and str(source) == str(target):
                    continue
                value = float(matrix.loc[source, target])
                if np.isfinite(value) and value > 0:
                    G.add_edge(str(source), str(target), weight=value)
        return G


    def eigenvector_centrality_weighted(G: nx.DiGraph, node_order: List[str]) -> Dict[str, float]:
        """Robust weighted eigenvector centrality normalized by the maximum value."""
        if not node_order:
            return {}
        if G.number_of_edges() == 0:
            return {node: 0.0 for node in node_order}

        if EIGENVECTOR_MODE == "undirected":
            H = G.to_undirected()
        elif EIGENVECTOR_MODE == "in":
            H = G.copy()
        elif EIGENVECTOR_MODE == "out":
            H = G.reverse(copy=True)
        else:
            raise ValueError("EIGENVECTOR_MODE must be 'undirected', 'in', or 'out'.")

        try:
            values = nx.eigenvector_centrality(H, max_iter=1000, tol=1e-12, weight="weight")
        except Exception:
            try:
                values = nx.eigenvector_centrality_numpy(H, weight="weight")
            except Exception:
                values = {}
                for node in node_order:
                    if node in G:
                        in_w = sum(G[p][node].get("weight", 1.0) for p in G.predecessors(node))
                        out_w = sum(G[node][s].get("weight", 1.0) for s in G.successors(node))
                        values[node] = in_w + out_w
                    else:
                        values[node] = 0.0

        values = {node: max(float(values.get(node, 0.0)), 0.0) for node in node_order}
        max_value = max(values.values()) if values else 0.0
        return {node: value / max_value for node, value in values.items()} if max_value > 0 else values


    def centrality_metrics_original(matrix: pd.DataFrame) -> pd.DataFrame:
        """Calculate five centrality metrics on the original relation matrix."""
        G = build_graph_from_matrix(matrix)
        nodes = [str(node) for node in matrix.index]

        strength = {}
        for node in G.nodes():
            in_weight = sum(G[p][node]["weight"] for p in G.predecessors(node))
            out_weight = sum(G[node][s]["weight"] for s in G.successors(node))
            total_weight = in_weight + out_weight
            total_degree = G.in_degree(node) + G.out_degree(node)
            strength[node] = (
                (total_degree ** STRENGTH_ALPHA) * (total_weight ** (1.0 - STRENGTH_ALPHA))
                if total_degree > 0 and total_weight > 0
                else 0.0
            )

        G_distance = G.copy()
        for _, _, data in G_distance.edges(data=True):
            data["distance"] = 1.0 / (data["weight"] + DISTANCE_EPS)
        closeness = nx.closeness_centrality(G_distance, distance="distance") if G.number_of_edges() else {node: 0.0 for node in nodes}

        eigenvector = eigenvector_centrality_weighted(G, nodes)

        G_undirected = G.to_undirected()
        for u, v in G_undirected.edges():
            weight_uv = G[u][v]["weight"] if G.has_edge(u, v) else 0.0
            weight_vu = G[v][u]["weight"] if G.has_edge(v, u) else 0.0
            G_undirected[u][v]["weight"] = (weight_uv + weight_vu) / 2.0 if weight_uv > 0 and weight_vu > 0 else max(weight_uv, weight_vu)
        clustering = nx.clustering(G_undirected, weight="weight") if G.number_of_edges() else {node: 0.0 for node in nodes}

        rows = []
        for node in nodes:
            rows.append(
                {
                    "Label": node,
                    "Strength": strength.get(node, 0.0),
                    "Closeness": closeness.get(node, 0.0),
                    "Clustering": clustering.get(node, 0.0),
                    "Eigenvector": eigenvector.get(node, 0.0),
                }
            )
        return pd.DataFrame(rows)


    # =========================================================
    # 5. SENSITIVITY DATASET
    # =========================================================
    def build_sensitivity_records(raw: pd.DataFrame) -> pd.DataFrame:
        """Build the long-format plotting table used by the four final figures."""
        records = []

        for dataset, start_row in DATASETS.items():
            mid_matrix = read_matrix(raw, start_row, MID_START_COL)

            for alpha in ALPHAS:
                alpha = float(alpha)
                matrix = make_alpha_matrix(mid_matrix, alpha)
                matrices = derive_matrices(matrix, KEEP_NODES)
                centrality = centrality_metrics_original(matrix)
                plot_x, sw_fraction, gw_fraction = get_plot_x_from_matrix(matrix)

                base = {
                    "Dataset": dataset,
                    "alpha": alpha,
                    "Recharge_Rate_SW_fraction": sw_fraction,
                    "Recharge_Rate_GW_fraction": gw_fraction,
                    "Plot_X": plot_x,
                }

                for source in KEEP_NODES:
                    for target in KEEP_NODES:
                        cp_value = float(matrices["Cp_percent"].loc[source, target])
                        cpp_value = float(matrices["Cpp_percent"].loc[source, target])
                        delta_value = float(matrices["Delta_percent"].loc[source, target])

                        records.append(base | {
                            "Metric_Group": "Contribution_Matrix",
                            "Metric": f"CpPercent_{source}->{target}",
                            "From": source,
                            "To": target,
                            "Node": "",
                            "Value": cp_value,
                        })
                        records.append(base | {
                            "Metric_Group": "Delivery_Matrix",
                            "Metric": f"CppPercent_{source}->{target}",
                            "From": source,
                            "To": target,
                            "Node": "",
                            "Value": cpp_value,
                        })
                        records.append(base | {
                            "Metric_Group": "Delta_Matrix",
                            "Metric": f"DeltaPercent_{source}->{target}",
                            "From": source,
                            "To": target,
                            "Node": "",
                            "Value": delta_value,
                        })

                for _, row in centrality.iterrows():
                    node = str(row["Label"])
                    if node not in KEEP_METRIC_NODES:
                        continue
                    for metric_name in CENTRALITY_METRICS:
                        records.append(base | {
                            "Metric_Group": "Centrality",
                            "Metric": f"{metric_name}_{node}",
                            "From": "",
                            "To": "",
                            "Node": node,
                            "Value": float(row[metric_name]),
                        })

        return pd.DataFrame(records)


    def build_reference_records(raw: pd.DataFrame) -> pd.DataFrame:
        """
        Build long-format records for the six fixed/reference networks.

        These records use the same Metric keys as the sensitivity curves, so they can
        be overlaid directly as star points in all four final figures.
        """
        if not INCLUDE_REFERENCE_NETWORKS:
            return pd.DataFrame()

        blocks = make_unique_reference_names(find_relation_matrix_blocks(raw))
        selected = select_reference_blocks(blocks)
        records = []

        for block in selected:
            reference_name = str(block["reference_name"])
            try:
                matrix = read_matrix(raw, block["start_row"], block["start_col"])
                if any(node not in matrix.index or node not in matrix.columns for node in ["SM", "SW", "GW"]):
                    print(f"Reference network skipped ({reference_name}): missing SM, SW, or GW")
                    continue

                matrices = derive_matrices(matrix, KEEP_NODES)
                centrality = centrality_metrics_original(matrix)
                plot_x, sw_fraction, gw_fraction = get_plot_x_from_matrix(matrix)
                if not np.isfinite(plot_x):
                    print(f"Reference network skipped ({reference_name}): invalid x-position")
                    continue

                base = {
                    "Reference_Network": reference_name,
                    "Reference_Group": block.get("reference_group", "Reference"),
                    "Reference_Title": block.get("title_cell_text", ""),
                    "Recharge_Rate_SW_fraction": sw_fraction,
                    "Recharge_Rate_GW_fraction": gw_fraction,
                    "Plot_X": plot_x,
                }

                for source in KEEP_NODES:
                    for target in KEEP_NODES:
                        cp_value = float(matrices["Cp_percent"].loc[source, target])
                        cpp_value = float(matrices["Cpp_percent"].loc[source, target])
                        delta_value = float(matrices["Delta_percent"].loc[source, target])

                        records.append(base | {
                            "Metric_Group": "Contribution_Matrix",
                            "Metric": f"CpPercent_{source}->{target}",
                            "From": source,
                            "To": target,
                            "Node": "",
                            "Value": cp_value,
                        })
                        records.append(base | {
                            "Metric_Group": "Delivery_Matrix",
                            "Metric": f"CppPercent_{source}->{target}",
                            "From": source,
                            "To": target,
                            "Node": "",
                            "Value": cpp_value,
                        })
                        records.append(base | {
                            "Metric_Group": "Delta_Matrix",
                            "Metric": f"DeltaPercent_{source}->{target}",
                            "From": source,
                            "To": target,
                            "Node": "",
                            "Value": delta_value,
                        })

                for _, row in centrality.iterrows():
                    node = str(row["Label"])
                    if node not in KEEP_METRIC_NODES:
                        continue
                    for metric_name in CENTRALITY_METRICS:
                        records.append(base | {
                            "Metric_Group": "Centrality",
                            "Metric": f"{metric_name}_{node}",
                            "From": "",
                            "To": "",
                            "Node": node,
                            "Value": float(row[metric_name]),
                        })
            except Exception as exc:
                print(f"Reference network skipped ({reference_name}): {exc}")

        return pd.DataFrame(records)


    # =========================================================
    # 6. PLOTTING HELPERS
    # =========================================================
    def clean_metric_name(metric: str) -> str:
        """Clean internal metric keys for figure titles and labels."""
        return (
            str(metric)
            .replace("DeltaPercent_", "ΔRA ")
            .replace("CpPercent_", "CpRA ")
            .replace("CppPercent_", "CppRA ")
            .replace("_", " ")
            .replace("->", "→")
        )


    def row_metric_ylabel(metric: str) -> str:
        """Return compact lower-case y-axis labels for centrality rows."""
        return clean_metric_name(metric).lower()


    def get_series(long_df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """Return one metric series sorted by dataset and x-position."""
        df = long_df[long_df["Metric"] == metric].copy()
        if df.empty:
            return df
        sort_cols = [col for col in ["Dataset", PLOT_X_COLUMN] if col in df.columns]
        return df.sort_values(sort_cols)


    def collect_metric_values(
        long_df: pd.DataFrame,
        metric_list: List[str],
        reference_df: Optional[pd.DataFrame] = None,
    ) -> List[float]:
        """Collect finite y-values for a set of metrics from curves and reference points."""
        values = []
        for metric in metric_list:
            data = long_df.loc[long_df["Metric"] == metric, "Value"]
            values.extend(data.replace([np.inf, -np.inf], np.nan).dropna().astype(float).tolist())

            if reference_df is not None and not reference_df.empty:
                ref_data = reference_df.loc[reference_df["Metric"] == metric, "Value"]
                values.extend(ref_data.replace([np.inf, -np.inf], np.nan).dropna().astype(float).tolist())

        return [float(value) for value in values if np.isfinite(value)]


    def compute_shared_ylim(
        long_df: pd.DataFrame,
        metric_list: List[str],
        reference_df: Optional[pd.DataFrame] = None,
    ) -> Optional[Tuple[float, float]]:
        """Compute one robust y-axis range for a row of subplots."""
        values = collect_metric_values(long_df, metric_list, reference_df)
        if not values:
            return None

        ymin, ymax = min(values), max(values)
        yrange = ymax - ymin
        pad = 0.12 * yrange if yrange > 0 else (0.08 * abs(ymax) if ymax != 0 else 0.05)

        if ymin >= 0:
            return 0.0, ymax + 2.0 * pad
        return ymin - pad, ymax + 2.0 * pad

    def plot_metric_curve(
        ax,
        long_df: pd.DataFrame,
        metric: str,
        add_legend: bool,
        fig_config: dict,
        reference_df: Optional[pd.DataFrame] = None,
        reference_color_map: Optional[Dict[str, tuple]] = None,
    ) -> None:
        """Plot one sensitivity curve panel and optional six-network star overlays."""
        df = get_series(long_df, metric)

        for dataset in DATASETS:
            data = df[df["Dataset"] == dataset].copy()
            if data.empty:
                continue
            data = data.sort_values(PLOT_X_COLUMN)
            ax.plot(
                data[PLOT_X_COLUMN],
                data["Value"],
                marker=COMMON_STYLE["markers"].get(dataset, "o"),
                markersize=COMMON_STYLE["dataset_markersize"],
                linewidth=COMMON_STYLE["dataset_linewidth"],
                label=dataset if add_legend else "_nolegend_",
                color=COMMON_STYLE["colors"].get(dataset),
                zorder=3,
            )

        if reference_df is not None and not reference_df.empty:
            ref_data = reference_df[reference_df["Metric"] == metric].copy()
            if not ref_data.empty:
                ref_data = ref_data.sort_values("Reference_Network")
                for _, row in ref_data.iterrows():
                    name = str(row["Reference_Network"])
                    display_name = name.replace("_", "-")
                    color = reference_color_map.get(name, "black") if reference_color_map else "black"

                    # Reference networks should use the same x-axis definition as the dataset curves.
                    # For alpha_GW plots, this is Recharge_Rate_GW_fraction.
                    x_value = row[PLOT_X_COLUMN]

                    ax.scatter(
                        x_value,
                        row["Value"],
                        s=COMMON_STYLE["reference_size"],
                        marker=COMMON_STYLE["reference_marker"],
                        facecolor=color,
                        edgecolor=color,
                        linewidth=COMMON_STYLE["reference_linewidth"],
                        alpha=COMMON_STYLE["reference_alpha"],
                        label=display_name if add_legend else "_nolegend_",
                        zorder=7,
                    )

        ax.grid(True, alpha=COMMON_STYLE["grid_alpha"], linestyle="--", linewidth=COMMON_STYLE["grid_linewidth"])
        ax.set_xlim(0, 1)
        ax.set_xticks(fig_config["xticks"])
        ax.tick_params(axis="both", labelsize=fig_config["font_tick"])

    def apply_ylim(ax, ylim: Optional[Tuple[float, float]]) -> None:
        """Apply precomputed y-limits if available."""
        if ylim is not None and np.isfinite(ylim).all() and ylim[0] < ylim[1]:
            ax.set_ylim(*ylim)


    def collect_legend(axes) -> Tuple[list, list]:
        """Collect unique legend entries from a grid of axes."""
        handles, labels, seen = [], [], set()
        for ax in np.asarray(axes).ravel():
            current_handles, current_labels = ax.get_legend_handles_labels()
            for handle, label in zip(current_handles, current_labels):
                if label == "_nolegend_" or label in seen:
                    continue
                handles.append(handle)
                labels.append(label)
                seen.add(label)
        return handles, labels


    def add_figure_legend(fig, axes, fig_config: dict) -> None:
        """Add one shared legend for a figure."""
        handles, labels = collect_legend(axes)
        if not handles:
            return
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=fig_config["legend_bbox"],
            ncol=5,
            frameon=False,
            fontsize=fig_config["font_legend"],
            handletextpad=0.35,
            columnspacing=0.75,
            markerscale=COMMON_STYLE["legend_markerscale"],
        )


    # =========================================================
    # 7. FOUR FINAL FIGURES
    # =========================================================
    def make_centrality_figure(
        long_df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None,
        output_dir: Path = OUTDIR,
    ) -> Path:
        """Export the centrality-metrics sensitivity figure."""
        nrows = len(CENTRALITY_METRICS)
        ncols = len(KEEP_METRIC_NODES)
        grid = [[f"{metric}_{node}" for node in KEEP_METRIC_NODES] for metric in CENTRALITY_METRICS]

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(
                CENTRALITY_FIG["figsize_per_col"] * ncols,
                CENTRALITY_FIG["figsize_per_row"] * nrows + CENTRALITY_FIG["figsize_extra_y"],
            ),
            dpi=CENTRALITY_FIG["dpi"],
            sharex=True,
            constrained_layout=False,
        )
        axes = np.asarray(axes).reshape(nrows, ncols)

        reference_color_map = build_reference_color_map(reference_df["Reference_Network"].drop_duplicates().astype(str).tolist()) if reference_df is not None and not reference_df.empty else {}
        ylims = [compute_shared_ylim(long_df, grid[row], reference_df) for row in range(nrows)]

        for row_id, metric_name in enumerate(CENTRALITY_METRICS):
            for col_id, node in enumerate(KEEP_METRIC_NODES):
                ax = axes[row_id, col_id]
                plot_metric_curve(
                    ax,
                    long_df,
                    grid[row_id][col_id],
                    add_legend=(row_id == 0 and col_id == 0),
                    fig_config=CENTRALITY_FIG,
                    reference_df=reference_df,
                    reference_color_map=reference_color_map,
                )

                ax.set_title(node if row_id == 0 else "", pad=3.5, fontsize=CENTRALITY_FIG["font_title"])
                apply_ylim(ax, ylims[row_id])

                if col_id == 0:
                    ax.set_ylabel(row_metric_ylabel(metric_name), fontsize=CENTRALITY_FIG["font_axis_label"])
                    ax.tick_params(axis="y", left=True, labelleft=True)
                else:
                    ax.set_ylabel("")
                    if CENTRALITY_FIG["show_y_ticks_first_col_only"]:
                        ax.tick_params(axis="y", left=False, labelleft=False)

                if row_id == nrows - 1:
                    ax.set_xlabel(PLOT_X_LABEL, fontsize=CENTRALITY_FIG["font_axis_label"])
                else:
                    ax.tick_params(axis="x", labelbottom=False)

        fig.subplots_adjust(**CENTRALITY_FIG["layout"])
        add_figure_legend(fig, axes, CENTRALITY_FIG)

        out_path = output_dir / f"{CENTRALITY_FIG['basename']}.tiff"
        save_figure_both(fig, out_path, dpi=CENTRALITY_FIG["dpi"], bbox_inches="tight")
        plt.close(fig)
        return out_path


    def make_matrix_figure(
        spec: dict,
        long_df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None,
        output_dir: Path = OUTDIR,
    ) -> Path:
        """Export one matrix-link sensitivity figure: contribution, delivery, or delta."""
        nrows = len(KEEP_NODES)
        ncols = len(KEEP_NODES)
        metric_prefix = spec["metric_prefix"]
        metric_group = spec["metric_group"]
        metric_grid = [[f"{metric_prefix}_{source}->{target}" for target in KEEP_NODES] for source in KEEP_NODES]

        plot_df = long_df[long_df["Metric_Group"] == metric_group].copy()
        if plot_df.empty:
            raise ValueError(f"No plotting records found for metric group: {metric_group}")

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(
                MATRIX_FIG["figsize_per_col"] * ncols,
                MATRIX_FIG["figsize_per_row"] * nrows + MATRIX_FIG["figsize_extra_y"],
            ),
            dpi=MATRIX_FIG["dpi"],
            sharex=True,
            constrained_layout=False,
        )
        axes = np.asarray(axes).reshape(nrows, ncols)

        reference_plot_df = reference_df[reference_df["Metric_Group"] == metric_group].copy() if reference_df is not None and not reference_df.empty else pd.DataFrame()
        reference_color_map = build_reference_color_map(reference_df["Reference_Network"].drop_duplicates().astype(str).tolist()) if reference_df is not None and not reference_df.empty else {}
        ylims = [compute_shared_ylim(plot_df, metric_grid[row], reference_plot_df) for row in range(nrows)]

        for row_id, source in enumerate(KEEP_NODES):
            for col_id, target in enumerate(KEEP_NODES):
                ax = axes[row_id, col_id]
                metric = metric_grid[row_id][col_id]
                plot_metric_curve(
                    ax,
                    plot_df,
                    metric,
                    add_legend=(row_id == 0 and col_id == 0),
                    fig_config=MATRIX_FIG,
                    reference_df=reference_plot_df,
                    reference_color_map=reference_color_map,
                )

                ax.set_title(f"To {target}" if row_id == 0 else "", pad=3.5, fontsize=MATRIX_FIG["font_title"])
                apply_ylim(ax, ylims[row_id])

                if col_id == 0:
                    ax.set_ylabel(f"{source}", fontsize=MATRIX_FIG["font_axis_label"])
                    ax.tick_params(axis="y", left=True, labelleft=True)
                else:
                    ax.set_ylabel("")
                    if MATRIX_FIG["show_y_ticks_first_col_only"]:
                        ax.tick_params(axis="y", left=False, labelleft=False)

                if row_id == nrows - 1:
                    ax.set_xlabel(PLOT_X_LABEL, fontsize=MATRIX_FIG["font_axis_label"])
                else:
                    ax.tick_params(axis="x", labelbottom=False)

        fig.subplots_adjust(**MATRIX_FIG["layout"])
        add_figure_legend(fig, axes, MATRIX_FIG)

        out_path = output_dir / f"{spec['basename']}.tiff"
        save_figure_both(fig, out_path, dpi=MATRIX_FIG["dpi"], bbox_inches="tight")
        plt.close(fig)
        return out_path


    def main() -> None:
        """Run the complete sensitivity plotting workflow and export exactly four plots."""
        if not INPUT.exists():
            raise FileNotFoundError(
                f"Input workbook not found: {INPUT.resolve()}\n"
                "Put 'All global networks.xlsx' in the same folder as this script, or edit INPUT."
            )

        raw = pd.read_excel(INPUT, sheet_name=SHEET, header=None, engine="openpyxl")
        long_df = build_sensitivity_records(raw)
        reference_df = build_reference_records(raw)

        if not reference_df.empty:
            selected_names = reference_df["Reference_Network"].drop_duplicates().astype(str).tolist()
            print("Included six-network overlays:")
            for name in selected_names:
                print(f"  {name}")

        output_paths = [make_centrality_figure(long_df, reference_df, OUTDIR)]
        output_paths.extend(make_matrix_figure(spec, long_df, reference_df, OUTDIR) for spec in MATRIX_PLOTS)

        print("Sensitivity plotting complete. Exactly four plots were exported:")
        for path in output_paths:
            print(f"  {path}")

    _public = {k: v for k, v in locals().items() if not k.startswith("__")}
    return types.SimpleNamespace(**_public)


def create_main_figures_workflow(input_file: str | Path, output_root: str | Path, rgw0_file: str | Path | None, sensitivity_workflow, network_graph_workflow):
    """Create the integrated main-figure workflow namespace."""
    """
    Main figures for global water-system network analysis.

    This refactored script removes repeated imports, repeated path-loading utilities,
    repeated sensitivity-data loading, repeated CV/share helpers, and repeated
    `if __name__ == "__main__"` blocks from the original combined script.

    Outputs
    -------
    Fig. 1 : alternative networks and node-level centrality sensitivity.
    Fig. 2 : Delta matrices and ranked pathway uncertainty.
    Fig. 3 : groundwater-routing structure, node sensitivity, and variance attribution.
    Fig. 4 : catchment-scale alphaGW map and a separate distribution panel.

    The four ordered manuscript figures are written to ``main_figures``. The
    supplementary-figure workflows and their existing output files are unchanged.
    """

    import importlib.util
    import os
    from pathlib import Path
    from typing import Dict, Iterable, List, Optional, Sequence, Tuple

    import numpy as np
    import pandas as pd

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from matplotlib.gridspec import GridSpec
    from matplotlib import gridspec
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.text import Text

    from mpl_toolkits.basemap import Basemap



    # =========================================================
    # GLOBAL FIGURE FONT CONFIGURATION
    # =========================================================
    # Apply Arial to ordinary text and to Matplotlib math-text notation.
    FIGURE_FONT_FAMILY = "Arial"

    plt.rcParams.update({
        "font.family": FIGURE_FONT_FAMILY,
        "font.sans-serif": [FIGURE_FONT_FAMILY, "Arimo", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": FIGURE_FONT_FAMILY,
        "mathtext.it": f"{FIGURE_FONT_FAMILY}:italic",
        "mathtext.bf": f"{FIGURE_FONT_FAMILY}:bold",
        "mathtext.sf": FIGURE_FONT_FAMILY,
        "mathtext.tt": FIGURE_FONT_FAMILY,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


    def _apply_arial_font(fig) -> None:
        """Force Arial on every Matplotlib Text artist before export."""
        for text_artist in fig.findobj(match=Text):
            text_artist.set_fontfamily(FIGURE_FONT_FAMILY)


    def _scale_figure_fonts(fig, factor: float) -> None:
        """Scale every text artist in a completed figure by one factor."""
        for text_artist in fig.findobj(match=Text):
            text_artist.set_fontsize(text_artist.get_fontsize() * factor)

    # =============================================================================
    # Global switches
    # =============================================================================
    RUN_FIG1 = True
    RUN_FIG2 = True
    RUN_FIG3 = True
    RUN_FIG4 = True

    DPI = 300
    EPS = 1e-4


    # =============================================================================
    # Shared configuration
    # =============================================================================
    NETWORK_ORDER = [
        "SS-85", "Ab-19", "OK-06", "Wr-13", "Tr-14", "AD-23"
    ]

    DATASET_ORDER = ["Obs", "Mixed-GLEAM", "GLDAS", "ERA5"]
    DATASET_COLORS = {
        "Obs": "#1f77b4",
        "Mixed-GLEAM": "#2ca02c",
        "GLDAS": "#8c7a00",
        "ERA5": "#c23b7a",
    }
    DATASET_MARKERS = {
        "Obs": "o",
        "Mixed-GLEAM": "D",
        "GLDAS": "s",
        "ERA5": "^",
    }

    WORKFLOW_SCRIPT = Path("3-Sensitivity analysis.py")
    INPUT_WORKBOOK = Path(input_file)
    SHEET_NAME = "Network matrices"
    SENSITIVITY_OUTPUT_DIR = Path(output_root) / "sensitivity_outputs"
    MAIN_FIGURE_DIR = Path(output_root) / "main_figures"
    CENTRALITY_OUTPUT_DIR = Path(output_root) / "centrality_output"

    REVISED_FIG1_OUTPUT = MAIN_FIGURE_DIR / "Fig1.tiff"
    REVISED_FIG2_OUTPUT = MAIN_FIGURE_DIR / "Fig2.tiff"
    REVISED_FIG2_RANKING_TABLE = MAIN_FIGURE_DIR / "Fig2_delta_std_ranking.csv"
    SUPP_FIG2G_EXCL_SS85_OUTPUT = CENTRALITY_OUTPUT_DIR / "FigS_delta_std_ranking_excluding_SS85.tiff"
    SUPP_FIG2G_EXCL_SS85_TABLE = CENTRALITY_OUTPUT_DIR / "FigS_delta_std_ranking_excluding_SS85.csv"
    REVISED_FIG3_OUTPUT = MAIN_FIGURE_DIR / "Fig3.tiff"
    REVISED_FIG3_SLOPE_TABLE = MAIN_FIGURE_DIR / "Fig3_panelB_slopes.csv"
    REVISED_FIG4_OUTPUT = MAIN_FIGURE_DIR / "Fig4.tiff"
    REVISED_FIG4_SUMMARY_TABLE = MAIN_FIGURE_DIR / "Fig4_panelB_summary.csv"
    REVISED_FIG4_IMPLIED_TABLE = MAIN_FIGURE_DIR / "Fig4_implied_alphaGW.csv"
    REVISED_FIG4_CAPTION_NOTE = MAIN_FIGURE_DIR / "Fig4_caption_note.txt"

    INDEPENDENT_ALPHA_GW_MEAN = 0.59
    INDEPENDENT_ALPHA_GW_UNCERTAINTY = 0.07
    OBS_ALPHA_GW_LOWER = INDEPENDENT_ALPHA_GW_MEAN - INDEPENDENT_ALPHA_GW_UNCERTAINTY
    OBS_ALPHA_GW_UPPER = INDEPENDENT_ALPHA_GW_MEAN + INDEPENDENT_ALPHA_GW_UNCERTAINTY
    OBS_ALPHA_GW_BAND_COLOR = "#f4a261"
    OBS_ALPHA_GW_BAND_ALPHA = 0.20
    NON_ANTARCTIC_LAND_AREA_KM2 = 134_900_000.0

    FIG4_FIGSIZE = (11.8, 9.0)
    FIG4_OUTER_HEIGHT_RATIOS = [1.0, 1.0]
    FIG4_OUTER_HSPACE = 0.30
    FIG4_FONT_SCALE = 1.28

    FIG4_CAPTION_NOTE = (
        "Fig. 4 | Catchment-scale groundwater contributions constrain their representation "
        "in global water-cycle assessments. a, Global distribution of the relative groundwater "
        "contribution to surface-water runoff across {n_units:,} evaluated, non-overlapping "
        "catchment units, expressed as Rgw0/Ravg (equivalent to αGW in this study). In ref. 16, "
        "Rgw0/Ravg denotes the mean relative groundwater contribution to surface-water runoff "
        "under zero monthly precipitation (Supplementary Fig. S5 of ref. 16). The mapped units "
        "cover {mapped_area_million:.2f} million km², or approximately {land_fraction_pct:.1f}% "
        "of global non-Antarctic land area (134.9 million km²). b, Comparison of implied global "
        "αGW values from six water-cycle assessments with observation-based estimates. Implied "
        "αGW is the assessed GW→SW flux divided by total runoff SW→SEA. Filled circles denote "
        "defined values, including Tr-14 at αGW = 0; open circles indicate αGW undefined "
        "(groundwater not linked to surface water; Ab-19 and Wr-13). The box and density summarize "
        "the area-weighted distribution across the {n_units:,} evaluated catchment units. The "
        "dashed purple connector indicates that the AD-23 value (αGW = 0.58) is the rounded "
        "area-weighted mean (0.583) of these catchment-scale values. The orange point and error "
        "bar show the independent observation-based estimate of 0.59 ± 0.07, and the corresponding "
        "0.52–0.66 interval is shaded."
    )


    # =============================================================================
    # Shared helpers
    # =============================================================================
    def save_figure_both(fig, output_path, dpi=300, **savefig_kwargs):
        """
        Save the same figure as both TIFF and PNG with the same dpi.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure object.
        output_path : str or Path
            Usually a .tiff path. The PNG path will use the same stem.
        dpi : int
            Export dpi for both formats.
        **savefig_kwargs
            Extra keyword arguments passed to fig.savefig().
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = output_path.suffix.lower()
        if suffix not in [".tif", ".tiff", ".png"]:
            tiff_path = output_path.with_suffix(".tiff")
        elif suffix == ".png":
            tiff_path = output_path.with_suffix(".tiff")
        else:
            tiff_path = output_path

        png_path = tiff_path.with_suffix(".png")

        _apply_arial_font(fig)

        fig.savefig(tiff_path, dpi=dpi, **savefig_kwargs)
        fig.savefig(png_path, dpi=dpi, **savefig_kwargs)

        return tiff_path, png_path

    def ensure_parent(path: Path | str) -> Path:
        """Create the parent folder of a file path and return the path as Path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


    def ensure_dir(path: Path | str) -> Path:
        """Create a directory and return it as Path."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path


    def import_python_module(script_path: Path | str, module_name: str = "external_workflow"):
        """Import a Python file as a module."""
        script_path = Path(script_path)
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import Python script: {script_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


    def load_sensitivity_workflow(script_path: Path = WORKFLOW_SCRIPT):
        """Return the integrated sensitivity-analysis workflow."""
        return sensitivity_workflow, Path("<integrated:sensitivity>")


    def build_sensitivity_long_dataframe(
        workflow,
        workbook: Path = INPUT_WORKBOOK,
        sheet_name: str = SHEET_NAME,
    ) -> Tuple[pd.DataFrame, Path]:
        """Build the long-format sensitivity table using the external workflow."""
        workbook = Path(workbook)
        raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None, engine="openpyxl")
        long_df = workflow.build_sensitivity_records(raw)
        long_df["alpha"] = pd.to_numeric(long_df["alpha"], errors="coerce")
        long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
        return long_df, workbook


    def safe_cv(
        values,
        zero_tol: float = 1e-6,
        min_valid: int = 2,
        use_abs: bool = False,
        signed_delta: bool = False,
        return_zero_if_invalid: bool = False,
    ):
        """
        Robust coefficient of variation.

        Parameters
        ----------
        values : array-like
            Input values.
        zero_tol : float
            Absolute values <= zero_tol are treated as structural zeros.
        min_valid : int
            Minimum number of nonzero valid values required.
        use_abs : bool
            If True, compute CV from absolute values.
        signed_delta : bool
            If True, compute std(signed values) / mean(abs(values)). This is more
            stable for signed delta matrices whose signed mean can approach zero.
        return_zero_if_invalid : bool
            If True, invalid CV values are returned as 0.0; otherwise np.nan.
        """
        x = (
            pd.to_numeric(pd.Series(values), errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy(dtype=float)
        )

        x = x[np.abs(x) > zero_tol]
        if len(x) < min_valid:
            return 0.0 if return_zero_if_invalid else np.nan

        if signed_delta:
            denominator = float(np.mean(np.abs(x)))
            numerator = float(np.std(x, ddof=0))
        else:
            stat = np.abs(x) if use_abs else x
            denominator = float(np.mean(stat))
            numerator = float(np.std(stat, ddof=0))

        if not np.isfinite(denominator) or np.isclose(denominator, 0.0, atol=zero_tol):
            return 0.0 if return_zero_if_invalid else np.nan

        return float(numerator / denominator)


    def annotate_heatmap(ax, df: pd.DataFrame, threshold: Optional[float] = None, fontsize: float = 9):
        """Add numeric annotations to a heatmap."""
        values = df.to_numpy(dtype=float)
        if threshold is None:
            threshold = np.nanmax(values) * 0.55

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                val = df.iat[i, j]
                if pd.isna(val):
                    continue
                text_color = "white" if val > threshold else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=fontsize, color=text_color)


    def setup_clean_axis(ax):
        """Apply a light clean style to line/bar axes."""
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.22)


    def nice_ylim(values: Sequence[float], force_zero: bool = True) -> Tuple[float, float]:
        """Return padded y-limits."""
        arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
        if arr.size == 0:
            return 0.0, 1.0

        ymin, ymax = float(arr.min()), float(arr.max())
        pad = 0.12 * (ymax - ymin) if not np.isclose(ymin, ymax) else (0.08 * abs(ymax) if ymax != 0 else 0.05)
        if force_zero and ymin >= 0:
            return 0.0, ymax + pad
        return ymin - pad, ymax + pad


    def add_panel_title(
        fig,
        axes,
        label: str,
        title: str = "",
        label_dx: float = 0.045,
        title_dy: float = 0.026,
    ):
        """Add a panel label and optional centered title above a group of axes."""
        flat_axes = np.asarray(axes, dtype=object).ravel().tolist()
        boxes = [ax.get_position() for ax in flat_axes]
        x0 = min(b.x0 for b in boxes)
        x1 = max(b.x1 for b in boxes)
        y1 = max(b.y1 for b in boxes)

        fig.text(x0 - label_dx, y1 + title_dy, label, fontsize=18, fontweight="bold", va="top")
        if title:
            fig.text((x0 + x1) / 2, y1 + title_dy * 0.75, title, ha="center", va="bottom", fontsize=13, fontweight="bold")


    def safe_fraction(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """Safely calculate numerator / denominator."""
        out = numerator / denominator.replace(0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0)


    # =============================================================================
    # Fig. 2: CV-based sensitivity of node-level metrics across six networks
    # =============================================================================
    FIG2_EXCEL_PATH = Path(output_root) / "All global networks-results-2.xlsx"
    FIG2_OUTPUT_FIG = Path(output_root) / "centrality_output" / "Fig2.tiff"
    FIG2_REFERENCE_NETWORK_FIG = Path(output_root) / "network_plots" / "Reference_networks.tiff"

    FIG2_METRIC_ORDER = [
        "Strength", "Closeness", "Clustering", "Eigenvector",
    ]
    FIG2_METRIC_LABELS = ["Strength", "Closeness", "Clustering", "Eigenvector"]
    FIG2_NODE_ORDER = ["Aw", "VEG", "SM", "GW", "SW", "SP/ENG", "SEA", "GL"]
    FIG2_FOCUS_NODES = ["Aw", "SM", "GW", "SW", "SP/ENG"]
    FIG2_RADAR_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    ASSESSMENT_COLORS = dict(zip(NETWORK_ORDER, FIG2_RADAR_COLORS))


    def FIG2_load_metrics_long(excel_path: Path) -> pd.DataFrame:
        df = pd.read_excel(excel_path, sheet_name="Metrics_long")
        for col in ["Model", "Metric", "Node"]:
            df[col] = df[col].astype(str)
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df["Normalized"] = pd.to_numeric(df["Normalized"], errors="coerce")
        return df


    def FIG2_build_panel_a_matrix(
        df_long: pd.DataFrame,
        networks: Sequence[str] = NETWORK_ORDER,
        metrics: Sequence[str] = FIG2_METRIC_ORDER,
        nodes: Sequence[str] = FIG2_NODE_ORDER,
    ) -> pd.DataFrame:
        """Node-level sensitivity represented by CV across six networks."""
        sub = df_long[df_long["Model"].isin(networks) & df_long["Metric"].isin(metrics)].copy()
        cv_df = sub.groupby(["Node", "Metric"])["Value"].agg(safe_cv).unstack()
        return cv_df.reindex(index=nodes, columns=metrics)


    def FIG2_build_panel_b_data(
        df_long: pd.DataFrame,
        networks: Sequence[str] = NETWORK_ORDER,
        metrics: Sequence[str] = FIG2_METRIC_ORDER,
        focus_nodes: Sequence[str] = FIG2_FOCUS_NODES,
    ) -> pd.DataFrame:
        return df_long[
            df_long["Model"].isin(networks)
            & df_long["Metric"].isin(metrics)
            & df_long["Node"].isin(focus_nodes)
        ].copy()


    def radar_angles(n: int) -> List[float]:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        return angles + angles[:1]


    def plot_radar(ax, values: np.ndarray, labels: Sequence[str], title: str, colors: Sequence[str], line_labels: Optional[Sequence[str]] = None):
        """Plot a radar chart with manually positioned category labels.

        The custom label placement avoids overlap between the long labels
        ``SP/ENG`` and ``SM`` by assigning them different radial distances.
        """
        n = len(labels)
        angles = radar_angles(n)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])
        ax.set_title(title, fontsize=12, pad=6)
        ax.grid(alpha=0.35)

        vmax_data = np.nanmax(values)
        if not np.isfinite(vmax_data) or vmax_data <= 0:
            vmax_data = 1.0

        # Leave extra radial space for the manually placed long labels.
        ax.set_ylim(0, vmax_data * 1.24)

        for k, row in enumerate(values):
            plot_values = list(row) + [row[0]]
            ax.plot(
                angles,
                plot_values,
                linewidth=1.6,
                color=colors[k],
                label=line_labels[k] if line_labels is not None else None,
            )
            ax.fill(angles, plot_values, color=colors[k], alpha=0.04)

        # Custom label radii to separate the long labels SP/ENG and SM.
        label_radius_factor = {
            "Aw": 1.12,
            "SM": 1.08,
            "GW": 1.10,
            "SW": 1.10,
            "SP/ENG": 1.20,
        }

        for angle, label in zip(angles[:-1], labels):
            radius = vmax_data * label_radius_factor.get(label, 1.12)
            if np.isclose(angle, 0.0) or np.isclose(angle, np.pi):
                ha = "center"
            elif 0.0 < angle < np.pi:
                ha = "left"
            else:
                ha = "right"
            ax.text(angle, radius, label, fontsize=12, ha=ha, va="center", clip_on=False)


    def FIG2_load_network_graph_module():
        """Return the integrated six-network drawing workflow."""
        return network_graph_workflow, Path("<integrated:network_graph>")


    def FIG2_load_network_items(
        workbook: Path = INPUT_WORKBOOK,
        sheet_name: str = SHEET_NAME,
        keep_first_n: int = 6,
    ):
        """Read the six relation matrices used for the network-structure panel."""
        graph_module, graph_script = FIG2_load_network_graph_module()

        raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None, engine="openpyxl")
        blocks = graph_module.find_relation_matrix_blocks(raw)
        if len(blocks) < keep_first_n:
            raise ValueError(f"Expected at least {keep_first_n} relation matrices, found {len(blocks)}.")

        network_items = []
        for block in blocks[:keep_first_n]:
            case_name = graph_module.clean_title(block["title"])
            matrix = graph_module.read_relation_matrix(raw, block)
            network_items.append((case_name, matrix))

        return network_items, graph_module, workbook, graph_script


    def FIG2_plot_cv_heatmap(fig, ax, cax, cv_df: pd.DataFrame):
        """Plot the node-metric CV heatmap used as Fig. 2B."""
        vmax = np.nanmax(cv_df.to_numpy(dtype=float))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

        im = ax.imshow(cv_df.values, cmap="Reds", aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(np.arange(cv_df.shape[1]))
        ax.set_xticklabels(FIG2_METRIC_LABELS, rotation=35, ha="right", fontsize=12)
        ax.set_yticks(np.arange(cv_df.shape[0]))
        ax.set_yticklabels(cv_df.index, fontsize=12)
        annotate_heatmap(ax, cv_df, threshold=vmax * 0.55, fontsize=12)

        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("CV across 6 networks", fontsize=12)
        cbar.ax.tick_params(labelsize=12)
        return im


    def find_existing_figure_path(path: Path | str, alternatives: Optional[Sequence[Path | str]] = None) -> Path:
        """
        Resolve an existing figure file from common local/sandbox locations.
        This is used for inserting the pre-generated Reference_networks.tiff into Fig. 1.
        """
        path = Path(path)
        candidates = [path]

        if alternatives:
            candidates.extend(Path(item) for item in alternatives)

        # Also search in the current working directory and /mnt/data by filename.
        candidates.extend([
            Path(path.name),
            Path("/mnt/data") / path.name,
            Path("/mnt/data") / "network_plots" / path.name,
        ])

        seen = set()
        unique_candidates = []
        for candidate in candidates:
            key = str(candidate)
            if key not in seen:
                unique_candidates.append(candidate)
                seen.add(key)

        for candidate in unique_candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "Cannot find the reference-network figure. Tried:\n  "
            + "\n  ".join(str(candidate) for candidate in unique_candidates)
        )


    def FIG2_insert_reference_network_panel(
        ax,
        reference_fig_path: Path = FIG2_REFERENCE_NETWORK_FIG,
    ):
        """
        Insert the already-generated Reference_networks.tiff as Fig. 1 panel A.

        This avoids redrawing the six networks inside the main Fig. 1 script, which
        can make the network labels and pathways unclear when compressed.
        """
        reference_fig_path = find_existing_figure_path(
            reference_fig_path,
            alternatives=[
                "Reference_networks.tiff",
                Path("network_plots") / "Reference_networks.tiff",
            ],
        )

        img = plt.imread(reference_fig_path)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_anchor("C")
        return reference_fig_path


    def make_FIG2(
        panel_a_df: pd.DataFrame,
        panel_b_df: Optional[pd.DataFrame] = None,
        networks: Sequence[str] = NETWORK_ORDER,
        metrics: Sequence[str] = FIG2_METRIC_ORDER,
        focus_nodes: Sequence[str] = FIG2_FOCUS_NODES,
        output_path: Path = FIG2_OUTPUT_FIG,
        reference_network_fig: Path = FIG2_REFERENCE_NETWORK_FIG,
        figure_label: str = "Fig. 2",
    ):
        """
        Create Fig. 2.

        Panel A is inserted from the pre-generated Reference_networks.tiff.
        Panel B is the node-metric CV heatmap.
        Panel C contains radar plots of node-level centrality metrics.
        """
        output_path = ensure_parent(output_path)

        if panel_b_df is None:
            raise ValueError("panel_b_df is required for the radar-plot panel in Fig. 2.")

        # Four radar plots now form a compact 2 x 2 panel, so the overall
        # figure height can be reduced without compressing panel content.
        fig = plt.figure(figsize=(12.5, 15), dpi=DPI)
        outer = GridSpec(
            2,
            2,
            height_ratios=[1.1, 0.9],
            width_ratios=[0.8, 1.15],
            hspace=0.05,
            wspace=0.30,
            figure=fig,
        )

        # ------------------------------------------------------------------
        # Panel A: use the pre-generated Reference_networks.tiff
        # ------------------------------------------------------------------
        ax_ref = fig.add_subplot(outer[0, :])
        used_reference_fig = FIG2_insert_reference_network_panel(
            ax_ref,
            reference_fig_path=reference_network_fig,
        )
        print(f"{figure_label} panel A inserted from: {used_reference_fig}")

        # ------------------------------------------------------------------
        # Panel B: heatmap at lower-left
        # ------------------------------------------------------------------
        gs_heat = outer[1, 0].subgridspec(1, 2, width_ratios=[1.0, 0.05], wspace=0.05)
        ax_heat = fig.add_subplot(gs_heat[0, 0])
        cax_heat = fig.add_subplot(gs_heat[0, 1])
        FIG2_plot_cv_heatmap(fig, ax_heat, cax_heat, panel_a_df)

        # ------------------------------------------------------------------
        # Panel C: radar plots at lower-right
        # ------------------------------------------------------------------
        gs_rad = outer[1, 1].subgridspec(2, 2, wspace=0.38, hspace=0.42)
        positions = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]

        radar_axes = []
        for metric, pos in zip(metrics, positions):
            ax = fig.add_subplot(gs_rad[pos], polar=True)
            radar_axes.append(ax)

            block = panel_b_df[panel_b_df["Metric"] == metric].copy()
            value_mat = []
            for net in networks:
                arr = []
                for node in focus_nodes:
                    s = block[(block["Model"] == net) & (block["Node"] == node)]["Value"]
                    arr.append(float(s.iloc[0]) if len(s) else np.nan)
                value_mat.append(arr)
            value_mat = np.asarray(value_mat, dtype=float)

            plot_radar(
                ax=ax,
                values=value_mat,
                labels=focus_nodes,
                title=metric,
                colors=FIG2_RADAR_COLORS,
                line_labels=networks,
            )
            ax.tick_params(axis="x", labelsize=12, pad=0.5)
            ax.tick_params(axis="y", labelsize=12)
            ax.title.set_fontsize(12)

        # ------------------------------------------------------------------
        # Unified panel labels: align a/b left, and keep b/c at same height
        # ------------------------------------------------------------------
        label_fontsize = 17
        label_dx = 0.030
        label_dy = 0.020

        a_box = ax_ref.get_position()
        b_box = ax_heat.get_position()
        rad_boxes = [ax.get_position() for ax in radar_axes]
        c_x0 = min(box.x0 for box in rad_boxes)
        c_y1 = max(box.y1 for box in rad_boxes)

        bc_y = max(b_box.y1, c_y1) + label_dy
        ab_x = min(a_box.x0, b_box.x0) - label_dx

        fig.text(
            ab_x,
            a_box.y1 + label_dy,
            "a",
            fontsize=label_fontsize,
            fontweight="bold",
            va="top",
            ha="left",
        )

        fig.text(
            ab_x,
            bc_y,
            "b",
            fontsize=label_fontsize,
            fontweight="bold",
            va="top",
            ha="left",
        )

        fig.text(
            c_x0 - label_dx,
            bc_y,
            "c",
            fontsize=label_fontsize,
            fontweight="bold",
            va="top",
            ha="left",
        )

        handles, labels = radar_axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.68, 0.05),
            ncol=3,
            frameon=False,
            fontsize=12.5,
            handlelength=2.5,
            columnspacing=1.5,
        )

        save_figure_both(fig, output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return output_path


    def run_FIG2():
        excel_path = Path(FIG2_EXCEL_PATH)
        df_long = FIG2_load_metrics_long(excel_path)
        panel_a_df = FIG2_build_panel_a_matrix(df_long)
        panel_b_df = FIG2_build_panel_b_data(df_long)
        output = make_FIG2(panel_a_df=panel_a_df, panel_b_df=panel_b_df)
        print(f"Fig. 2 saved to: {output}")


    # =============================================================================
    # Fig. 3: CV-based sensitivity of node-to-node pathways across six networks
    # =============================================================================
    FIG3_EXCEL_PATH = Path(output_root) / "All global networks-results-1.xlsx"
    FIG3_OUTPUT_FIG = Path(output_root) / "matrix_output" / "Fig3.tiff"
    FIG3_OUTPUT_CSV = Path(output_root) / "matrix_output" / "delta_std_ranking.csv"

    FIG3_NODE_ORDER = ["Aw", "VEG", "SM", "SW", "GW", "SP/ENG", "GL"]
    FIG3_RECTANGLE_NODES = ["SM", "SW", "GW", "SP/ENG"]  # 4 x 4 block from SM to SP/ENG
    FIG3_DECIMAL_SCALE = 0.01  # original delta block is stored as percent; display/use decimal fractions
    FIG3_DELTA_BLOCK_TITLE = "Delta_Cp_relative_average_minus_Cpp_relative_average"
    FIG3_TOP_N = 10
    FIG3_USE_ABS_FOR_STD = False  # False = signed STD of Δ; True = STD of |Δ|
    FIG3_MIN_NONZERO = 2
    FIG3_IGNORE_DIAGONAL = True
    FIG3_FIGSIZE = (14, 15)
    FIG3_ANNOT_FONTSIZE = 12.5


    def find_block_start(df_raw: pd.DataFrame, block_title: str) -> int:
        """Find the row index where a matrix block starts."""
        col0 = df_raw.iloc[:, 0].astype(str).str.strip()
        hits = np.where(col0.eq(block_title))[0]
        if len(hits) == 0:
            raise ValueError(f"Cannot find block '{block_title}'.")
        return int(hits[0])


    def extract_matrix_block(df_raw: pd.DataFrame, block_title: str) -> pd.DataFrame:
        """
        Extract a matrix block from one Excel sheet.

        Expected pattern:
            block_title
            header row starting with 'index'
            data rows until the first blank row in column 0
        """
        start_row = find_block_start(df_raw, block_title)
        header_row = start_row + 1

        headers = [h for h in df_raw.iloc[header_row, 1:].tolist() if pd.notna(h)]

        row_labels = []
        values = []
        r = header_row + 1
        while r < len(df_raw):
            row_name = df_raw.iat[r, 0]
            if pd.isna(row_name) or str(row_name).strip() == "":
                break
            row_labels.append(str(row_name).strip())
            values.append(df_raw.iloc[r, 1 : 1 + len(headers)].tolist())
            r += 1

        out = pd.DataFrame(values, index=row_labels, columns=headers).apply(pd.to_numeric, errors="coerce")
        out = out.loc[
            [idx for idx in out.index if str(idx).strip().lower() not in {"total", "total_row"}],
            [col for col in out.columns if str(col).strip().lower() not in {"total", "total_column"}],
        ]
        return out


    def load_delta_matrices(
        excel_path: Path,
        network_order: Sequence[str] = NETWORK_ORDER,
        node_order: Sequence[str] = FIG3_NODE_ORDER,
        block_title: str = FIG3_DELTA_BLOCK_TITLE,
    ) -> Dict[str, pd.DataFrame]:
        """Load delta matrices from the required network sheets."""
        delta_mats = {}
        xls = pd.ExcelFile(excel_path)

        missing_sheets = [sheet for sheet in network_order if sheet not in xls.sheet_names]
        if missing_sheets:
            raise ValueError(f"Missing sheets in workbook: {missing_sheets}")

        for sheet in network_order:
            df_raw = pd.read_excel(excel_path, sheet_name=sheet, header=None)
            delta = extract_matrix_block(df_raw, block_title)

            missing_rows = [node for node in node_order if node not in delta.index]
            missing_cols = [node for node in node_order if node not in delta.columns]
            if missing_rows or missing_cols:
                raise ValueError(
                    f"Sheet '{sheet}' is missing nodes.\n"
                    f"Missing rows: {missing_rows}\n"
                    f"Missing cols: {missing_cols}"
                )

            delta_mats[sheet] = delta.loc[node_order, node_order] * FIG3_DECIMAL_SCALE

        return delta_mats


    def compute_pathway_std(
        delta_mats: Dict[str, pd.DataFrame],
        use_abs: bool = False,
        min_nonzero: int = FIG3_MIN_NONZERO,
        ignore_diagonal: bool = FIG3_IGNORE_DIAGONAL,
        zero_tol: float = 1e-12,
    ) -> pd.DataFrame:
        """
        Compute pathway-level standard deviation of Delta across networks.

        Zero values are ignored when calculating the statistics.
        This is useful when zeros represent absent pathways rather than real
        zero-valued Delta contrasts.

        Parameters
        ----------
        delta_mats : dict
            Dictionary of network-name -> Delta matrix.
        use_abs : bool
            If False, calculate STD from signed Delta values.
            If True, calculate STD from absolute Delta values.
        min_nonzero : int
            Minimum number of non-zero pathway values required.
        ignore_diagonal : bool
            If True, exclude self-to-self pathways.
        zero_tol : float
            Values with absolute magnitude <= zero_tol are treated as zero.
        """
        network_names = list(delta_mats.keys())
        node_order = list(next(iter(delta_mats.values())).index)
        rows = []

        for source in node_order:
            for target in node_order:
                if ignore_diagonal and source == target:
                    continue

                vals = np.array(
                    [delta_mats[network].loc[source, target] for network in network_names],
                    dtype=float,
                )

                # Ignore structural zeros
                nonzero_mask = ~np.isclose(vals, 0.0, atol=zero_tol)
                vals_nonzero = vals[nonzero_mask]
                nonzero_count = int(nonzero_mask.sum())

                if nonzero_count < min_nonzero:
                    continue

                stat_vals = np.abs(vals_nonzero) if use_abs else vals_nonzero

                mean_value = float(np.mean(stat_vals))
                std_value = float(np.std(stat_vals, ddof=0))

                rows.append(
                    {
                        "pathway": f"{source}→{target}",
                        "source": source,
                        "target": target,
                        "mean_value_nonzero": mean_value,
                        "std_value": std_value,
                        "nonzero_count": nonzero_count,
                        **{
                            network: delta_mats[network].loc[source, target]
                            for network in network_names
                        },
                    }
                )

        std_df = pd.DataFrame(rows)

        if not std_df.empty:
            std_df = std_df.sort_values(
                ["std_value", "mean_value_nonzero"],
                ascending=[False, False],
            ).reset_index(drop=True)

        return std_df

    def add_circulation_rectangle(ax, node_order: Sequence[str], block_nodes: Sequence[str] = FIG3_RECTANGLE_NODES):
        """Add one rectangle around the selected 4 x 4 SM-to-SP/ENG submatrix block."""
        idx = sorted(node_order.index(node) for node in block_nodes)
        start = min(idx) - 0.5
        end = max(idx) - 0.5
        rect = patches.Rectangle(
            (start, start),
            end - start + 1,
            end - start + 1,
            fill=False,
            linewidth=2.5,
            edgecolor="red",
            linestyle="-",
        )
        ax.add_patch(rect)


    def plot_delta_heatmap(
        ax,
        df: pd.DataFrame,
        title: str,
        panel_label: str,
        vlim: float,
        annot_fontsize: float = FIG3_ANNOT_FONTSIZE,
    ):
        """Plot one signed delta heatmap with text annotations."""
        im = ax.imshow(df.values, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="equal")

        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=12.5)
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_yticklabels(df.index, fontsize=12.5)

        ax.set_xticks(np.arange(-0.5, len(df.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(df.index), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                val = df.iat[i, j]
                if np.isnan(val):
                    continue
                text = f"{val:.1f}"
                text_color = "white" if abs(val) > 0.55 * vlim else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=annot_fontsize, color=text_color)

        ax.set_title(title, fontsize=12, pad=8)
        ax.text(0, 1.12, panel_label, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top", ha="left")

        ax.set_xlabel("To" if panel_label in ["d", "e", "f"] else "", weight="bold", fontsize=12)
        ax.set_ylabel("From" if panel_label in ["a", "d"] else "", weight="bold", fontsize=12)
        return im


    def plot_delta_ranking_bar(
        ax,
        std_df: pd.DataFrame,
        top_n: int = FIG3_TOP_N,
        network_count: int = 6,
        ranking_note: Optional[str] = None,
        panel_label: Optional[str] = "g",
    ):
        """Plot the top uncertain pathways by standard deviation of Delta."""
        rank_df = std_df.head(top_n).copy() if top_n is not None else std_df.copy()
        rank_df = rank_df.iloc[::-1]

        y = np.arange(len(rank_df))
        bar_values = rank_df["std_value"].to_numpy(float)

        norm = mpl.colors.Normalize(vmin=bar_values.min(), vmax=bar_values.max())
        bar_colors = plt.cm.Blues(norm(bar_values))

        ax.barh(y, bar_values, color=bar_colors, edgecolor="black", linewidth=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(rank_df["pathway"], fontsize=12)

        for tick_label in ax.get_yticklabels():
            if "GW" in tick_label.get_text():
                tick_label.set_fontweight("bold")

        ax.tick_params(axis="x", labelsize=12)
        ax.set_xlabel(rf"SD of $\Delta$ across {network_count} networks", fontsize=15)
        ax.grid(axis="x", alpha=0.3)
        if panel_label:
            ax.text(
                0.02,
                1.12,
                panel_label,
                transform=ax.transAxes,
                fontsize=16,
                fontweight="bold",
                va="top",
                ha="right",
            )
        if ranking_note:
            ax.text(
                0.5,
                1.06,
                ranking_note,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=11.5,
                fontstyle="italic",
            )

        x_offset = 0.01 * max(rank_df["std_value"].max(), 1.0)

        for yi, val in zip(y, rank_df["std_value"].values):
            ax.text(val + x_offset, yi, f"{val:.2f}", va="center", fontsize=12.5)

    def plot_delta_figure(
        delta_mats: Dict[str, pd.DataFrame],
        std_df: pd.DataFrame,
        output_fig: Path = FIG3_OUTPUT_FIG,
        output_csv: Optional[Path] = FIG3_OUTPUT_CSV,
        top_n: int = FIG3_TOP_N,
        figsize: Tuple[float, float] = FIG3_FIGSIZE,
        dpi: int = DPI,
        annot_fontsize: float = FIG3_ANNOT_FONTSIZE,
        ranking_network_count: Optional[int] = None,
        ranking_note: Optional[str] = None,
    ):
        """Create the 3-row figure: six heatmaps plus one STD-ranking bar plot."""
        output_fig = ensure_parent(output_fig)
        if output_csv is not None:
            output_csv = ensure_parent(output_csv)

        network_order = list(delta_mats.keys())
        node_order = list(next(iter(delta_mats.values())).index)
        vmax = max(np.nanmax(np.abs(df.to_numpy(dtype=float))) for df in delta_mats.values())

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(
            nrows=3,
            ncols=3,
            height_ratios=[1.0, 1.0, 1.05],
            hspace=0.35,
            wspace=0.3,
            figure=fig,
        )

        panel_labels = list("abcdef")
        heat_axes = []
        im = None
        for k, network in enumerate(network_order):
            row, col = divmod(k, 3)
            ax = fig.add_subplot(gs[row, col])
            heat_axes.append(ax)

            im = plot_delta_heatmap(
                ax,
                delta_mats[network],
                title=network,
                panel_label=panel_labels[k],
                vlim=vmax,
                annot_fontsize=annot_fontsize,
            )
            add_circulation_rectangle(ax, node_order, block_nodes=FIG3_RECTANGLE_NODES)

        cbar = fig.colorbar(im, ax=heat_axes, fraction=0.028, pad=0.02)
        cbar.set_label(r"$\Delta$ value", fontsize=14)

        ax_bar = fig.add_subplot(gs[2, :])
        if ranking_network_count is None:
            ranking_network_count = len(delta_mats)
        plot_delta_ranking_bar(
            ax_bar,
            std_df,
            top_n=top_n,
            network_count=ranking_network_count,
            ranking_note=ranking_note,
        )

        save_figure_both(fig, output_fig, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        if output_csv is not None:
            std_df.to_csv(output_csv, index=False)

        return output_fig, output_csv


    def plot_delta_ranking_supplement(
        std_df: pd.DataFrame,
        output_fig: Path = SUPP_FIG2G_EXCL_SS85_OUTPUT,
        output_csv: Optional[Path] = SUPP_FIG2G_EXCL_SS85_TABLE,
        top_n: int = FIG3_TOP_N,
        network_count: int = 5,
    ):
        """Export the SS-85-excluded SD ranking as a standalone supplementary figure."""
        output_fig = ensure_parent(output_fig)
        if output_csv is not None:
            output_csv = ensure_parent(output_csv)

        fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=DPI)
        plot_delta_ranking_bar(
            ax,
            std_df,
            top_n=top_n,
            network_count=network_count,
            ranking_note=None,
            panel_label=None,
        )
        fig.subplots_adjust(left=0.20, right=0.97, top=0.97, bottom=0.14)
        save_figure_both(fig, output_fig, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        if output_csv is not None:
            std_df.to_csv(output_csv, index=False)
        return output_fig, output_csv


    def run_FIG3():
        excel_path = Path(FIG3_EXCEL_PATH)
        delta_mats = load_delta_matrices(excel_path)

        std_df = compute_pathway_std(
            delta_mats,
            use_abs=False,          # signed Delta STD
            min_nonzero=FIG3_MIN_NONZERO,
            ignore_diagonal=FIG3_IGNORE_DIAGONAL,
        )

        output_fig, output_csv = plot_delta_figure(delta_mats, std_df)

        print(f"Fig. 3 saved to: {output_fig}")
        print(f"Fig. 3 ranking table saved to: {output_csv}")


    # =============================================================================
    # Fig. 1: Sensitivity of pathways and node metrics to alpha-GW
    # =============================================================================
    FIG1_OUTPUT_FIG = SENSITIVITY_OUTPUT_DIR / "Fig1.tiff"
    FIG1_OUTPUT_TABLE_A = SENSITIVITY_OUTPUT_DIR / "Fig1_panelA.csv"
    FIG1_OUTPUT_TABLE_B = SENSITIVITY_OUTPUT_DIR / "Fig1_panelB.csv"

    FIG1_RGW0_GPKG = Path(rgw0_file) if rgw0_file is not None else Path(output_root) / "Rgw0_data.gpkg"
    # Value mode for mapping the non-overlapping GeoPackage fragments.
    # Options: "overlap_only_formula", "fragment_mean", "source_area_weighted",
    #          "user_formula_all_fragments"
    FIG1_RGW0_VALUE_MODE = "overlap_only_formula"
    FIG1_RGW0_COLUMN = "rgw0_map"
    FIG1_VALID_MIN = 0.0
    FIG1_VALID_MAX = 1.0

    # Internal node names used by the sensitivity workflow.
    FIG1_FOCUS_NODES = ["GW", "SM", "SW", "SP/ENG"]
    FIG1_PANEL_B_METRICS = ["Strength", "Closeness", "Clustering"]
    FIG1_X_COL = "alpha"
    FIG1_X_LABEL = r"$\alpha_{GW}$"

    # Fig. 1 is arranged as: map on top; Obs structure and response curves below.
    FIG1_FIGSIZE = (12.5, 9.2)
    FIG1_OUTER_HEIGHT_RATIOS = [1.0, 1.05]
    FIG1_OUTER_HSPACE = 0.20
    FIG1_BOTTOM_WIDTH_RATIOS = [0.36, 0.64]
    FIG1_BOTTOM_WSPACE = 0.12
    FIG1_PANEL_B_WSPACE = 0.20
    FIG1_PANEL_B_HSPACE = 0.15

    # Inset position inside the Fig. 1A map axes: [left, bottom, width, height].
    # Keep the area-weighted Rgw0 distribution inside the map, bottom-left.
    FIG1_RGW0_DIST_INSET_BOUNDS = [0.075, 0.080, 0.150, 0.420]


    # =============================================================================
    # General helpers
    # =============================================================================
    def metric_key(metric: str, node: str) -> str:
        return f"{metric}_{node}"


    def display_node_label(node: str) -> str:
        if node == "SP/ENG":
            return "SP/ENG"
        return node


    def clean_id(x):
        """
        Standardize catchment IDs from both the Rgw0 table and the shapefile.
        Keeps alphanumeric IDs and converts numeric-looking IDs such as 1634500.0
        into 1634500.
        """
        if pd.isna(x):
            return np.nan

        s = str(x).strip()

        if s.endswith(".0"):
            s = s[:-2]

        return s


    def clean_dataset_name(x):
        """
        Standardize dataset/source names for composite-key merging.
        """
        if pd.isna(x):
            return np.nan

        s = str(x).strip()

        mapping = {
            "camels": "CAMELS",
            "CAMELS": "CAMELS",
            "grdc": "GRDC",
            "GRDC": "GRDC",
        }

        return mapping.get(s, s)


    def format_lon(x):
        if x < 0:
            return f"{abs(int(x))}°W"
        if x > 0:
            return f"{int(x)}°E"
        return "0°"


    def format_lat(y):
        if y < 0:
            return f"{abs(int(y))}°S"
        if y > 0:
            return f"{int(y)}°N"
        return "0°"


    # =============================================================================
    # Sensitivity line-plot data
    # =============================================================================
    def summarize_delta_matrix(
        long_df: pd.DataFrame,
        datasets: Iterable[str] = DATASET_ORDER,
        nodes: Iterable[str] = FIG1_FOCUS_NODES,
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Summarize DeltaPercent values across all alpha scenarios.
        This is retained for compatibility with the broader workflow.
        """
        datasets = list(datasets)
        nodes = list(nodes)

        delta_df = long_df[
            (long_df["Metric_Group"] == "Delta_Matrix")
            & (long_df["Dataset"].isin(datasets))
            & (long_df["From"].isin(nodes))
            & (long_df["To"].isin(nodes))
        ].copy()

        summary = (
            delta_df.groupby(["Dataset", "From", "To"])["Value"]
            .agg(
                vmin="min",
                vmean="mean",
                vmax="max",
                vcv=lambda x: safe_cv(
                    x,
                    zero_tol=1e-6,
                    min_valid=2,
                    signed_delta=True,
                    return_zero_if_invalid=True,
                ),
            )
            .reset_index()
        )

        summary_by_dataset: Dict[str, pd.DataFrame] = {}
        for dataset in datasets:
            sub = summary[summary["Dataset"] == dataset].copy()
            rows = []
            for source in nodes:
                row = []
                for target in nodes:
                    hit = sub[(sub["From"] == source) & (sub["To"] == target)]
                    if hit.empty:
                        row.append((np.nan, np.nan, np.nan, np.nan))
                    else:
                        item = hit.iloc[0]
                        row.append(
                            (
                                float(item["vmin"]),
                                float(item["vmean"]),
                                float(item["vmax"]),
                                float(item["vcv"]),
                            )
                        )
                rows.append(row)
            summary_by_dataset[dataset] = pd.DataFrame(rows, index=nodes, columns=nodes)

        return summary_by_dataset, summary


    def compute_slopes(
        long_df: pd.DataFrame,
        datasets: Iterable[str] = DATASET_ORDER,
        metrics: Iterable[str] = FIG1_PANEL_B_METRICS,
        nodes: Iterable[str] = FIG1_FOCUS_NODES,
        x_col: str = FIG1_X_COL,
    ) -> pd.DataFrame:
        """
        Compute linear slopes for each dataset x metric x node series.
        Slopes are saved to CSV, but not annotated on Fig. 1.
        """
        rows = []

        for dataset in datasets:
            for metric in metrics:
                for node in nodes:
                    key = metric_key(metric, node)

                    sub = long_df[
                        (long_df["Dataset"] == dataset)
                        & (long_df["Metric_Group"] == "Centrality")
                        & (long_df["Metric"] == key)
                    ].copy()

                    sub = (
                        sub[[x_col, "Value"]]
                        .replace([np.inf, -np.inf], np.nan)
                        .dropna()
                        .sort_values(x_col)
                    )

                    if len(sub) >= 2 and sub[x_col].nunique() >= 2:
                        k, b = np.polyfit(
                            sub[x_col].to_numpy(float),
                            sub["Value"].to_numpy(float),
                            1,
                        )
                    else:
                        k, b = np.nan, np.nan

                    rows.append(
                        {
                            "Dataset": dataset,
                            "Metric": metric,
                            "Node": node,
                            "slope_k": k,
                            "intercept": b,
                            "n": len(sub),
                        }
                    )

        return pd.DataFrame(rows)


    # =============================================================================
    # Rgw0 GeoPackage loading
    # =============================================================================
    def load_rgw0_geodata(
        gpkg_path: Path = FIG1_RGW0_GPKG,
        value_mode: str = FIG1_RGW0_VALUE_MODE,
    ):
        """
        Load the non-overlapping Rgw0 fragment GeoPackage for Fig. 1A.

        The GeoPackage is expected to contain non-overlapping fragments with
        fragment_area_km2 and one or more Rgw0 estimates. This replaces the older
        workflow that read Rgw0 from a text file and joined it to the original
        overlapping catchment shapefile.

        value_mode options
        ------------------
        overlap_only_formula : use rgw0_area_ratio_mean_user_formula for fragments
            covered by more than one original catchment, and rgw0_mean for fragments
            covered by only one catchment. This corresponds to the final corrected
            estimate discussed for the manuscript.
        fragment_mean : use rgw0_mean for all fragments.
        source_area_weighted : use rgw0_source_area_weighted for all fragments.
        user_formula_all_fragments : use rgw0_area_ratio_mean_user_formula for all
            fragments. This is generally not recommended for mapping because it
            downweights even non-overlapping fragments.
        """
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise ImportError("Fig. 1A requires geopandas to read and plot the GeoPackage.") from exc

        gpkg_path = Path(gpkg_path)
        if not gpkg_path.exists():
            raise FileNotFoundError(f"Cannot find Rgw0 GeoPackage: {gpkg_path}")

        gdf_equal_area = gpd.read_file(gpkg_path)
        if gdf_equal_area.empty:
            raise ValueError(f"No geometries were read from {gpkg_path}.")

        if gdf_equal_area.crs is None:
            # The generated Rgw0_data.gpkg should be EPSG:6933. If CRS metadata is
            # missing, set it explicitly rather than treating the coordinates as lon/lat.
            gdf_equal_area = gdf_equal_area.set_crs("EPSG:6933", allow_override=True)

        required_cols = [
            "fragment_area_km2",
            "n_covering",
            "rgw0_mean",
            "rgw0_source_area_weighted",
            "rgw0_area_ratio_mean_user_formula",
        ]
        missing = [col for col in required_cols if col not in gdf_equal_area.columns]
        if missing:
            raise ValueError(f"Rgw0 GeoPackage is missing required columns: {missing}")

        gdf_equal_area = gdf_equal_area.copy()
        gdf_equal_area["area_km2"] = pd.to_numeric(gdf_equal_area["fragment_area_km2"], errors="coerce")
        gdf_equal_area["rgw0_fragment_mean"] = pd.to_numeric(gdf_equal_area["rgw0_mean"], errors="coerce")
        gdf_equal_area["rgw0_source_area_weighted"] = pd.to_numeric(
            gdf_equal_area["rgw0_source_area_weighted"], errors="coerce"
        )
        gdf_equal_area["rgw0_user_formula_all_fragments"] = pd.to_numeric(
            gdf_equal_area["rgw0_area_ratio_mean_user_formula"], errors="coerce"
        )

        gdf_equal_area["rgw0_overlap_only_formula"] = np.where(
            pd.to_numeric(gdf_equal_area["n_covering"], errors="coerce") > 1,
            gdf_equal_area["rgw0_user_formula_all_fragments"],
            gdf_equal_area["rgw0_fragment_mean"],
        )

        value_column_by_mode = {
            "overlap_only_formula": "rgw0_overlap_only_formula",
            "fragment_mean": "rgw0_fragment_mean",
            "source_area_weighted": "rgw0_source_area_weighted",
            "user_formula_all_fragments": "rgw0_user_formula_all_fragments",
        }
        if value_mode not in value_column_by_mode:
            raise ValueError(
                f"Unknown FIG1_RGW0_VALUE_MODE={value_mode!r}. "
                f"Choose from {list(value_column_by_mode)}."
            )

        selected_col = value_column_by_mode[value_mode]
        gdf_equal_area[FIG1_RGW0_COLUMN] = pd.to_numeric(gdf_equal_area[selected_col], errors="coerce")
        gdf_equal_area["Rgw0_valid"] = (
            gdf_equal_area[FIG1_RGW0_COLUMN].between(FIG1_VALID_MIN, FIG1_VALID_MAX, inclusive="both")
            & gdf_equal_area["area_km2"].gt(0)
        )

        valid_equal_area = gdf_equal_area[gdf_equal_area["Rgw0_valid"]].copy()
        invalid_equal_area = gdf_equal_area[~gdf_equal_area["Rgw0_valid"]].copy()

        # Plotting is done in lon/lat, but statistics remain based on the
        # equal-area fragment_area_km2 field.
        all_plot = gdf_equal_area.to_crs("EPSG:4326")
        valid_plot = valid_equal_area.to_crs("EPSG:4326")
        invalid_plot = invalid_equal_area.to_crs("EPSG:4326")

        # Preserve equal-area fragment areas after reprojection for the inset and CSV.
        for plot_df, src_df in [
            (all_plot, gdf_equal_area),
            (valid_plot, valid_equal_area),
            (invalid_plot, invalid_equal_area),
        ]:
            if len(plot_df) > 0:
                plot_df["area_km2"] = src_df["area_km2"].to_numpy()

        area_weighted_mean = np.average(
            valid_equal_area[FIG1_RGW0_COLUMN].to_numpy(float),
            weights=valid_equal_area["area_km2"].to_numpy(float),
        )

        print(f"Rgw0 GeoPackage: {gpkg_path}")
        print(f"Selected Rgw0 value mode: {value_mode} ({selected_col})")
        print(f"Total non-overlapping fragments: {len(gdf_equal_area)}")
        print(f"Valid fragments: {len(valid_equal_area)}")
        print(f"Total valid area: {valid_equal_area['area_km2'].sum():,.0f} km2")
        print(f"Area-weighted mean Rgw0: {area_weighted_mean:.4f}")

        return all_plot, valid_plot, invalid_plot, gdf_equal_area

    # =============================================================================
    # Rgw0 statistics and inset plot
    # =============================================================================
    def summarize_rgw0_values(valid_gdf: pd.DataFrame) -> pd.DataFrame:
        """
        Create summary statistics for valid Rgw0 values only.
        Invalid values are excluded from all area-weighted statistics.
        """
        values = valid_gdf[FIG1_RGW0_COLUMN].replace([np.inf, -np.inf], np.nan).astype(float)
        areas = valid_gdf["area_km2"].replace([np.inf, -np.inf], np.nan).astype(float)

        mask = np.isfinite(values) & np.isfinite(areas) & (areas > 0)
        values = values[mask].to_numpy(float)
        areas = areas[mask].to_numpy(float)

        if values.size == 0:
            return pd.DataFrame(
                [{
                    "n_valid_0_1": 0,
                    "valid_area_km2": np.nan,
                    "area_weighted_mean": np.nan,
                    "arithmetic_mean": np.nan,
                    "ordinary_median": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "q25": np.nan,
                    "q75": np.nan,
                    "max": np.nan,
                }]
            )

        aw_mean = float(np.average(values, weights=areas))

        return pd.DataFrame(
            [{
                "n_valid_0_1": int(values.size),
                "valid_area_km2": float(np.sum(areas)),
                "area_weighted_mean": aw_mean,
                "arithmetic_mean": float(np.mean(values)),
                "ordinary_median": float(np.median(values)),
                "std": float(np.std(values, ddof=0)),
                "min": float(np.min(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "max": float(np.max(values)),
            }]
        )

    import numpy as np

    def weighted_quantile(values, quantiles, sample_weight=None):
        """
        Compute weighted quantiles.

        Parameters
        ----------
        values : array-like
        quantiles : float or array-like in [0, 1]
        sample_weight : array-like, optional

        Returns
        -------
        quantile values
        """
        values = np.asarray(values, dtype=float)
        quantiles = np.atleast_1d(quantiles)

        if sample_weight is None:
            sample_weight = np.ones_like(values, dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)

        mask = np.isfinite(values) & np.isfinite(sample_weight) & (sample_weight > 0)
        values = values[mask]
        sample_weight = sample_weight[mask]

        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

        cumulative_weight = np.cumsum(sample_weight)
        cumulative_weight = cumulative_weight / cumulative_weight[-1]

        q_values = np.interp(quantiles, cumulative_weight, values)
        return q_values if len(q_values) > 1 else q_values[0]


    def add_rgw0_distribution_inset(ax, valid_gdf, value_col="rgw0_map", area_col="fragment_area_km2", bounds=None):
        """
        Area-weighted inset distribution for Rgw0.

        Shows:
        - weighted half-violin (via weighted KDE)
        - weighted box statistics (Q1, median, Q3)
        - weighted mean line
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde

        values = pd.to_numeric(valid_gdf[value_col], errors="coerce").to_numpy(dtype=float)
        areas = pd.to_numeric(valid_gdf[area_col], errors="coerce").to_numpy(dtype=float)

        mask = np.isfinite(values) & np.isfinite(areas) & (areas > 0)
        values = values[mask]
        areas = areas[mask]

        if values.size == 0:
            return None

        # normalize weights
        weights = areas / areas.sum()

        # weighted summary stats
        wmean = np.sum(values * weights)
        q25, q50, q75 = weighted_quantile(values, [0.25, 0.50, 0.75], sample_weight=areas)

        # inset axis
        if bounds is None:
            bounds = FIG1_RGW0_DIST_INSET_BOUNDS
        inset = ax.inset_axes(bounds)

        # ---------------------------------------------------------
        # Weighted KDE for half-violin
        # ---------------------------------------------------------
        y_grid = np.linspace(0, 1, 400)
        kde = gaussian_kde(values, weights=weights, bw_method="scott")
        density = kde(y_grid)

        # normalize width
        density = density / density.max() * 0.32
        x0 = 0.25

        inset.fill_betweenx(
            y_grid,
            x0,
            x0 + density,
            facecolor="#2c7fb8",
            edgecolor="#1a355f",
            alpha=0.55,
            linewidth=1.0,
            zorder=2
        )

        # ---------------------------------------------------------
        # Weighted box-like summary
        # ---------------------------------------------------------
        box_x = 0.05
        box_w = 0.10

        # box
        rect = plt.Rectangle(
            (box_x, q25),
            box_w,
            q75 - q25,
            facecolor="#f7faff",
            edgecolor="#08306b",
            linewidth=1.0,
            alpha=0.9,
            zorder=3,
        )
        inset.add_patch(rect)

        # median
        inset.plot([box_x, box_x + box_w], [q50, q50], color="#08306b", lw=1.5, zorder=4)

        # whiskers using min/max
        vmin = values.min()
        vmax = values.max()
        inset.plot([box_x + box_w/2, box_x + box_w/2], [vmin, q25], color="#08306b", lw=1.0, zorder=3)
        inset.plot([box_x + box_w/2, box_x + box_w/2], [q75, vmax], color="#08306b", lw=1.0, zorder=3)
        inset.plot([box_x + 0.02, box_x + box_w - 0.02], [vmin, vmin], color="#08306b", lw=1.0, zorder=3)
        inset.plot([box_x + 0.02, box_x + box_w - 0.02], [vmax, vmax], color="#08306b", lw=1.0, zorder=3)

        # ---------------------------------------------------------
        # Weighted mean
        # ---------------------------------------------------------
        inset.axhline(
            wmean,
            color="#1f77b4",
            linestyle="--",
            linewidth=1.5,
            zorder=5,
        )

        inset.text(
            0.33,
            min(wmean + 0.05, 0.98),
            f"Mean={wmean:.2f}",
            color="#1f77b4",
            fontsize=10.5,
            va="center",
            ha="left",
        )

        # ---------------------------------------------------------
        # cosmetics
        # ---------------------------------------------------------
        inset.set_ylim(0, 1)
        inset.set_xlim(0, 0.62)
        inset.set_xticks([])
        inset.tick_params(axis="y", labelsize=11, length=2)
        inset.set_facecolor((1, 1, 1, 0.88))

        for spine_name, spine in inset.spines.items():
            if spine_name == "right":
                spine.set_visible(False)
            else:
                spine.set_linewidth(0.7)
                spine.set_color("0.35")

        return inset


    def plot_rgw0_distribution_panel(
        ax,
        valid_gdf,
        value_col: str = "rgw0_map",
        area_col: str = "fragment_area_km2",
        independent_mean: float = INDEPENDENT_ALPHA_GW_MEAN,
        independent_uncertainty: float = INDEPENDENT_ALPHA_GW_UNCERTAINTY,
    ):
        """Plot the catchment distribution and an independent global estimate."""
        from scipy.stats import gaussian_kde

        values = pd.to_numeric(valid_gdf[value_col], errors="coerce").to_numpy(dtype=float)
        areas = pd.to_numeric(valid_gdf[area_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(values) & np.isfinite(areas) & (areas > 0)
        values = values[mask]
        areas = areas[mask]
        if values.size == 0:
            raise ValueError("No valid catchment alphaGW values are available for Fig. 4b.")

        weights = areas / areas.sum()
        weighted_mean = float(np.sum(values * weights))
        q25, q50, q75 = weighted_quantile(values, [0.25, 0.50, 0.75], sample_weight=areas)

        ax.axhspan(
            OBS_ALPHA_GW_LOWER,
            OBS_ALPHA_GW_UPPER,
            facecolor=OBS_ALPHA_GW_BAND_COLOR,
            alpha=OBS_ALPHA_GW_BAND_ALPHA,
            edgecolor="none",
            zorder=0,
        )

        y_grid = np.linspace(FIG1_VALID_MIN, FIG1_VALID_MAX, 400)
        density = gaussian_kde(values, weights=weights, bw_method="scott")(y_grid)
        density = density / density.max() * 0.38

        box_x = 0.18
        estimate_x = 0.36
        violin_x = 0.53
        ax.fill_betweenx(
            y_grid,
            violin_x,
            violin_x + density,
            facecolor="#6baed6",
            edgecolor="#1a355f",
            linewidth=1.35,
            alpha=0.70,
            zorder=2,
        )

        box_width = 0.13
        ax.add_patch(
            patches.Rectangle(
                (box_x - box_width / 2, q25),
                box_width,
                q75 - q25,
                facecolor="white",
                edgecolor="#08306b",
                linewidth=1.5,
                zorder=4,
            )
        )
        ax.plot(
            [box_x - box_width / 2, box_x + box_width / 2],
            [q50, q50],
            color="#08306b",
            linewidth=2.0,
            zorder=5,
        )
        whisker_low = float(values.min())
        whisker_high = float(values.max())
        ax.plot([box_x, box_x], [whisker_low, q25], color="#08306b", linewidth=1.2, zorder=3)
        ax.plot([box_x, box_x], [q75, whisker_high], color="#08306b", linewidth=1.2, zorder=3)
        cap_half_width = box_width * 0.42
        ax.plot([box_x - cap_half_width, box_x + cap_half_width], [whisker_low, whisker_low], color="#08306b", linewidth=1.2, zorder=3)
        ax.plot([box_x - cap_half_width, box_x + cap_half_width], [whisker_high, whisker_high], color="#08306b", linewidth=1.2, zorder=3)

        independent_lower = independent_mean - independent_uncertainty
        independent_upper = independent_mean + independent_uncertainty
        ax.errorbar(
            [estimate_x],
            [independent_mean],
            yerr=[[independent_uncertainty], [independent_uncertainty]],
            fmt="o",
            markersize=4.8,
            color="#d95f02",
            ecolor="#d95f02",
            elinewidth=1.4,
            capsize=3.5,
            capthick=1.2,
            zorder=6,
        )
        ax.text(
            estimate_x,
            independent_mean + independent_uncertainty + 0.025,
            r"$0.59 \pm 0.07$",
            color="#d95f02",
            fontsize=8.3,
            fontweight="normal",
            ha="center",
            va="bottom",
            zorder=7,
        )
        ax.set_xlim(0.06, 0.98)
        ax.set_ylim(FIG1_VALID_MIN, FIG1_VALID_MAX)
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
        ax.tick_params(axis="y", labelsize=10.5, width=1.0)
        ax.set_facecolor("#eef7fb")
        ax.grid(axis="y", color="white", linestyle="-", linewidth=0.8, alpha=0.85)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)

        return {
            "area_weighted_mean": weighted_mean,
            "weighted_q25": float(q25),
            "weighted_median": float(q50),
            "weighted_q75": float(q75),
            "independent_mean": float(independent_mean),
            "independent_uncertainty": float(independent_uncertainty),
            "independent_lower": float(independent_lower),
            "independent_upper": float(independent_upper),
        }


    def load_implied_alpha_gw_values(workbook: Path = INPUT_WORKBOOK) -> pd.DataFrame:
        """Derive assessment-level implied alphaGW values from the reference matrices."""
        raw = pd.read_excel(workbook, sheet_name=SHEET_NAME, header=None, engine="openpyxl")
        blocks = []
        for row in range(raw.shape[0]):
            for col in range(raw.shape[1]):
                title = str(raw.iat[row, col]).strip()
                if "RELATION MATRIX" in title.upper():
                    assessment = next((name for name in NETWORK_ORDER if name in title), None)
                    if assessment is not None:
                        blocks.append((assessment, row, max(col - 1, 0)))

        block_by_assessment = {assessment: (row, col) for assessment, row, col in blocks}
        missing = [name for name in NETWORK_ORDER if name not in block_by_assessment]
        if missing:
            raise ValueError(f"Reference relation matrices missing for: {missing}")

        records = []
        for assessment in NETWORK_ORDER:
            start_row, start_col = block_by_assessment[assessment]
            node_row = start_row + 3
            data_row = start_row + 4
            col0 = start_col + 2
            nodes = [str(value).strip() for value in raw.iloc[node_row, col0 : col0 + 8].tolist()]
            matrix = raw.iloc[data_row : data_row + 8, col0 : col0 + 8].copy()
            matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            matrix.index = nodes
            matrix.columns = nodes

            gw_to_sw = float(matrix.loc["GW", "SW"])
            total_runoff = float(matrix.loc["SW", "SEA"])
            represented = bool(gw_to_sw > EPS and total_runoff > EPS)
            alpha_gw_assessed = assessment not in {"Ab-19", "Wr-13"}
            implied = gw_to_sw / total_runoff if alpha_gw_assessed and total_runoff > EPS else 0.0
            records.append(
                {
                    "assessment": assessment,
                    "gw_to_sw": gw_to_sw,
                    "total_runoff_sw_to_sea": total_runoff,
                    "implied_alpha_gw": implied,
                    "gw_to_sw_represented": represented,
                    "alpha_gw_assessed": alpha_gw_assessed,
                }
            )

        return pd.DataFrame.from_records(records)


    def plot_implied_alpha_gw_assessments(ax, implied_df: pd.DataFrame) -> None:
        """Plot the six implied assessment values on the Fig. 4 shared alphaGW axis."""
        ordered = implied_df.set_index("assessment").loc[NETWORK_ORDER].reset_index()
        x_values = np.arange(len(ordered), dtype=float)

        ax.axhspan(
            OBS_ALPHA_GW_LOWER,
            OBS_ALPHA_GW_UPPER,
            facecolor=OBS_ALPHA_GW_BAND_COLOR,
            alpha=OBS_ALPHA_GW_BAND_ALPHA,
            edgecolor="none",
            zorder=0,
        )
        for x, row in zip(x_values, ordered.itertuples(index=False)):
            assessment_color = ASSESSMENT_COLORS[row.assessment]
            face_color = assessment_color if row.alpha_gw_assessed else "white"
            ax.scatter(
                x,
                row.implied_alpha_gw,
                s=88,
                marker="o",
                facecolor=face_color,
                edgecolor=assessment_color,
                linewidth=1.8,
                clip_on=False,
                zorder=4,
            )
            label_y = row.implied_alpha_gw + (0.045 if row.implied_alpha_gw < 0.92 else -0.07)
            va = "bottom" if row.implied_alpha_gw < 0.92 else "top"
            label = (
                "0" if row.alpha_gw_assessed and abs(row.implied_alpha_gw) <= EPS
                else f"{row.implied_alpha_gw:.2f}" if row.alpha_gw_assessed
                else "undefined"
            )
            ax.text(x, label_y, label, ha="center", va=va, fontsize=10.0, color=assessment_color)

        ax.set_xlim(-0.65, len(ordered) - 0.35)
        ax.set_ylim(FIG1_VALID_MIN, FIG1_VALID_MAX)
        ax.set_xticks(x_values)
        ax.set_xticklabels(ordered["assessment"], rotation=45, ha="right", fontsize=8.5)
        ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
        ax.set_ylabel(r"$\alpha_{GW}$", fontsize=13, labelpad=5)
        ax.set_title("Global assessments", fontsize=10.1, pad=8)
        ax.tick_params(axis="y", labelsize=10.5, width=1.0)
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_facecolor("#eef7fb")
        ax.grid(axis="y", color="white", linestyle="-", linewidth=0.8, alpha=0.85)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)


    def plot_combined_alpha_gw_panel(
        ax,
        valid_gdf,
        implied_df: pd.DataFrame,
        value_col="rgw0_map",
        area_col="fragment_area_km2",
        independent_mean=INDEPENDENT_ALPHA_GW_MEAN,
        independent_uncertainty=INDEPENDENT_ALPHA_GW_UNCERTAINTY,
    ):
        """Plot assessments and independent estimates together as one Fig. 4b axis."""
        from scipy.stats import gaussian_kde

        ordered = implied_df.set_index("assessment").loc[NETWORK_ORDER].reset_index()
        assessment_x = np.arange(len(ordered), dtype=float)

        values = pd.to_numeric(valid_gdf[value_col], errors="coerce").to_numpy(dtype=float)
        areas = pd.to_numeric(valid_gdf[area_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(values) & np.isfinite(areas) & (areas > 0)
        values = values[mask]
        areas = areas[mask]
        if values.size == 0:
            raise ValueError("No valid catchment alphaGW values are available for Fig. 4b.")

        weights = areas / areas.sum()
        weighted_mean = float(np.sum(values * weights))
        q25, q50, q75 = weighted_quantile(values, [0.25, 0.50, 0.75], sample_weight=areas)

        ax.axhspan(
            OBS_ALPHA_GW_LOWER,
            OBS_ALPHA_GW_UPPER,
            facecolor=OBS_ALPHA_GW_BAND_COLOR,
            alpha=OBS_ALPHA_GW_BAND_ALPHA,
            edgecolor="none",
            zorder=0,
        )

        for x, row in zip(assessment_x, ordered.itertuples(index=False)):
            assessment_color = ASSESSMENT_COLORS[row.assessment]
            face_color = assessment_color if row.alpha_gw_assessed else "white"
            ax.scatter(
                x,
                row.implied_alpha_gw,
                s=88,
                marker="o",
                facecolor=face_color,
                edgecolor=assessment_color,
                linewidth=1.8,
                clip_on=False,
                zorder=5,
            )
            label_y = row.implied_alpha_gw + (0.045 if row.implied_alpha_gw < 0.92 else -0.07)
            va = "bottom" if row.implied_alpha_gw < 0.92 else "top"
            label = (
                "0" if row.alpha_gw_assessed and abs(row.implied_alpha_gw) <= EPS
                else f"{row.implied_alpha_gw:.2f}" if row.alpha_gw_assessed
                else "undefined"
            )
            ax.text(x, label_y, label, ha="center", va=va, fontsize=8.3, color=assessment_color)

        separator_x = 6.05
        box_x = 6.55
        estimate_x = 7.35
        violin_x = 7.95
        ax.axvline(separator_x, color="white", linewidth=2.0, zorder=1)

        y_grid = np.linspace(FIG1_VALID_MIN, FIG1_VALID_MAX, 400)
        density = gaussian_kde(values, weights=weights, bw_method="scott")(y_grid)
        density = density / density.max() * 0.72
        ax.fill_betweenx(
            y_grid,
            violin_x,
            violin_x + density,
            facecolor="#6baed6",
            edgecolor="#1a355f",
            linewidth=1.35,
            alpha=0.70,
            zorder=2,
        )

        box_width = 0.38
        ax.add_patch(
            patches.Rectangle(
                (box_x - box_width / 2, q25),
                box_width,
                q75 - q25,
                facecolor="white",
                edgecolor="#08306b",
                linewidth=1.5,
                zorder=4,
            )
        )
        ax.plot([box_x - box_width / 2, box_x + box_width / 2], [q50, q50], color="#08306b", linewidth=2.0, zorder=5)
        whisker_low = float(values.min())
        whisker_high = float(values.max())
        ax.plot([box_x, box_x], [whisker_low, q25], color="#08306b", linewidth=1.2, zorder=3)
        ax.plot([box_x, box_x], [q75, whisker_high], color="#08306b", linewidth=1.2, zorder=3)
        cap_half_width = box_width * 0.42
        ax.plot([box_x - cap_half_width, box_x + cap_half_width], [whisker_low, whisker_low], color="#08306b", linewidth=1.2, zorder=3)
        ax.plot([box_x - cap_half_width, box_x + cap_half_width], [whisker_high, whisker_high], color="#08306b", linewidth=1.2, zorder=3)

        # AD-23 is the rounded area-weighted aggregation of the catchment-scale
        # distribution. Point the connector toward AD-23 to show the direction of
        # aggregation without linking it to the independent orange estimate.
        ad23_position = int(np.flatnonzero(ordered["assessment"].to_numpy() == "AD-23")[0])
        ad23_x = float(assessment_x[ad23_position])
        ad23_y = float(ordered.loc[ad23_position, "implied_alpha_gw"])
        aggregation_color = ASSESSMENT_COLORS["AD-23"]
        aggregation_arrow = patches.FancyArrowPatch(
            (box_x - box_width / 2 - 0.03, weighted_mean + 0.01),
            (ad23_x + 0.12, ad23_y + 0.005),
            connectionstyle="arc3,rad=0.24",
            arrowstyle="-|>",
            mutation_scale=10.5,
            linewidth=1.25,
            linestyle=(0, (3, 2)),
            color=aggregation_color,
            alpha=0.90,
            clip_on=False,
            zorder=6,
        )
        ax.add_patch(aggregation_arrow)
        ax.text(
            (box_x + ad23_x) / 2 + 0.03,
            max(weighted_mean, ad23_y) + 0.105,
            "Area-weighted\naggregation",
            ha="center",
            va="bottom",
            fontsize=8.4,
            color=aggregation_color,
            linespacing=0.92,
            zorder=7,
        )

        ax.errorbar(
            [estimate_x],
            [independent_mean],
            yerr=[[independent_uncertainty], [independent_uncertainty]],
            fmt="o",
            markersize=4.8,
            color="#d95f02",
            ecolor="#d95f02",
            elinewidth=1.4,
            capsize=3.5,
            capthick=1.2,
            zorder=6,
        )
        ax.text(
            estimate_x,
            independent_mean + independent_uncertainty + 0.025,
            r"$0.59 \pm 0.07$",
            color="#d95f02",
            fontsize=10.5,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=7,
        )
        ax.text(2.5, 1.035, "Groundwater contribution assessments", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=12.0)
        ax.text(7.45, 1.035, "Cross-scale observational estimate", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=12.0)
        ax.set_xlim(-0.65, 8.95)
        ax.set_ylim(FIG1_VALID_MIN, FIG1_VALID_MAX)
        ax.set_xticks(assessment_x)
        ax.set_xticklabels(ordered["assessment"], rotation=0, ha="center", fontsize=10.5)
        ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
        ax.set_ylabel(r"$\alpha_{GW}$", fontsize=15, labelpad=7)
        ax.tick_params(axis="y", labelsize=12.0, width=1.0)
        ax.tick_params(axis="x", length=0, pad=7)
        ax.set_facecolor("#eef7fb")
        ax.grid(axis="y", color="white", linestyle="-", linewidth=0.8, alpha=0.85)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        return {
            "area_weighted_mean": weighted_mean,
            "weighted_q25": float(q25),
            "weighted_median": float(q50),
            "weighted_q75": float(q75),
            "independent_mean": float(independent_mean),
            "independent_uncertainty": float(independent_uncertainty),
            "independent_lower": float(independent_mean - independent_uncertainty),
            "independent_upper": float(independent_mean + independent_uncertainty),
        }


    # =============================================================================
    # Fig. 1 Obs structural network inset
    # =============================================================================
    def _draw_network_node(ax, xy, label, edgecolor, facecolor, width=0.18, height=0.11):
        """Draw one labeled elliptical node for the Fig. 1A structural network inset."""
        node = patches.Ellipse(
            xy,
            width=width,
            height=height,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.2,
            zorder=4,
        )
        ax.add_patch(node)
        ax.text(
            xy[0],
            xy[1],
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            zorder=5,
        )


    def _draw_network_arrow(
        ax,
        p1,
        p2,
        color,
        rad=0.0,
        label=None,
        label_xy=None,
        label_color=None,
        arrowstyle="-|>",
        linewidth=1.8,
        shrink_a=14,
        shrink_b=14,
        linestyle="-",
        mutation_scale=12,
        zorder=2,
    ):
        patch = FancyArrowPatch(
            p1,
            p2,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle=arrowstyle,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            shrinkA=shrink_a,
            shrinkB=shrink_b,
            mutation_scale=mutation_scale,
            zorder=zorder,
        )
        ax.add_patch(patch)

        if label is not None and label_xy is not None:
            ax.text(
                label_xy[0],
                label_xy[1],
                label,
                color=(label_color if label_color is not None else color),
                fontsize=11,
                ha="center",
                va="center",
                zorder=zorder + 2,
            )


    def plot_obs_structure_network_panel(ax, style_scale: float = 1.0):
        """
        Plot the value-free Obs network structure as an independent Fig. 1 panel.

        Only the three alpha-dependent routing links are labeled:
        SM→GW = alpha_GW, SM→SW = 1-alpha_GW, and GW→SW = alpha_GW.
        All fixed links are shown structurally without numeric annotations.
        """
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor((1, 1, 1, 1.0))

        # No enclosing frame for panel b; keep only the network elements.
        for spine in ax.spines.values():
            spine.set_visible(False)

        pos = {
            "SM":   (0.14, 0.64),
            "VEG": (0.12, 0.30),
            "GW":     (0.36, 0.10),
            "SP/ENG": (0.44, 0.84),
            "SW":     (0.64, 0.34),
            "SEA":    (0.85, 0.14),
            "Aw":     (0.78, 0.66),
        }

        colors = {
            "Aw": "#1597ff",
            "SM": "#c75a00",
            "VEG": "#3d9a2a",
            "GW": "#ef2f2f",
            "SW": "#256bff",
            "SEA": "#5e2ee6",
            "SP/ENG": "#df36f5",
        }
        faces = {
            "Aw": "#dff1ff",
            "SM": "#fff4e8",
            "VEG": "#e9f8e4",
            "GW": "#ffeaea",
            "SW": "#e9f0ff",
            "SEA": "#eeeaff",
            "SP/ENG": "#fde8ff",
        }

        # --------------------------------------------------
        # Structural links
        # Solid lines: only P, E, R
        # Dashed lines: all other pathways
        # --------------------------------------------------

        # --- Dashed structural exchanges (no numeric labels) ---
        _draw_network_arrow(
            ax, pos["SP/ENG"], pos["SM"], colors["SP/ENG"],
            rad=0.05, arrowstyle="<|-|>", linewidth=1.55,
            shrink_a=15, shrink_b=15, linestyle="--"
        )

        _draw_network_arrow(
            ax, pos["SP/ENG"], pos["Aw"], colors["SP/ENG"],
            rad=0.05, arrowstyle="<|-|>", linewidth=1.55,
            shrink_a=15, shrink_b=15, linestyle="--"
        )

        _draw_network_arrow(
            ax, pos["SEA"], pos["Aw"], colors["SEA"],
            rad=-0.06, linewidth=1.6, linestyle="--"
        )

        # Groundwater-routing links (all dashed)
        _draw_network_arrow(
            ax, pos["SM"], pos["SW"], colors["SM"],
            rad=-0.18, linewidth=1.8, linestyle="--",
            label=r"$(1-\alpha_{GW}) \times R$",
            label_xy=(0.70, 0.50),
            label_color=colors["SM"],
        )

        _draw_network_arrow(
            ax, pos["SM"], pos["GW"], colors["SM"],
            rad=0.02, linewidth=1.8, linestyle="--",
            label=r"$\alpha_{GW} \times R$",
            label_xy=(0.06, 0.50),
            label_color=colors["SM"],
        )

        _draw_network_arrow(
            ax, pos["GW"], pos["SW"], colors["GW"],
            rad=0.16, linewidth=1.8, linestyle="--",
            label=r"$\alpha_{GW} \times R$",
            label_xy=(0.60, 0.15),
            label_color=colors["GW"],
        )

        # --- Solid fluxes only: P, E, R ---
        _draw_network_arrow(
            ax, pos["Aw"], pos["SM"], colors["Aw"],
            rad=0.16, linewidth=2.0, linestyle="-",
            label=r"$P$",
            label_xy=(0.47, 0.72),
            label_color=colors["Aw"],
        )

        _draw_network_arrow(
            ax, pos["VEG"], pos["Aw"], colors["VEG"],
            rad=-0.18, linewidth=2.0, linestyle="-",
            label=r"$E$",
            label_xy=(0.45, 0.58),
            label_color=colors["VEG"],
        )

        _draw_network_arrow(
            ax, pos["SW"], pos["SEA"], colors["SW"],
            rad=0.02, linewidth=2.0, linestyle="-",
            label=r"$R$",
            label_xy=(0.77, 0.26),
            label_color=colors["SW"],
        )

        # Draw nodes after arrows so node borders remain clean.
        for node in ["SP/ENG", "SM", "Aw", "VEG", "GW", "SW", "SEA"]:
            _draw_network_node(ax, pos[node], node, colors[node], faces[node])

        if style_scale != 1.0:
            for text_artist in ax.texts:
                text_artist.set_fontsize(text_artist.get_fontsize() * style_scale)
            for patch_artist in ax.patches:
                patch_artist.set_linewidth(patch_artist.get_linewidth() * style_scale)

        return ax


    def add_obs_structure_network_inset(ax, bounds):
        """Backward-compatible wrapper for adding the Obs structure as an inset."""
        return plot_obs_structure_network_panel(ax.inset_axes(bounds))


    # =============================================================================
    # Map plot
    # =============================================================================
    def plot_rgw0_world_map(
        fig,
        ax,
        all_gdf,
        valid_gdf,
        invalid_gdf,
        include_distribution: bool = True,
        colorbar_side: str = "right",
    ):
        """
        Plot Fig. 1A: Rgw0 world map.
        Valid catchments are colored by Rgw0.
        Invalid or unmatched catchments are shown in grey.
        """

        ax.set_facecolor("#eef7fb")

        basemap_obj = None
        try:

            basemap_obj = Basemap(
                projection="cyl",
                llcrnrlon=-180,
                urcrnrlon=180,
                llcrnrlat=-60,
                urcrnrlat=85,
                resolution="c",
                ax=ax,
            )
            basemap_obj.drawmapboundary(fill_color="#eef7fb", linewidth=0.0)
            basemap_obj.fillcontinents(color="#f1efe3", lake_color="#eef7fb", zorder=0)
            basemap_obj.drawcoastlines(color="0.55", linewidth=0.22, zorder=1)
            basemap_obj.drawcountries(color="0.75", linewidth=0.18, zorder=1)
        except Exception:
            pass

        # Valid catchments: continuous colormap.
        if not valid_gdf.empty:
            valid_gdf.plot(
                ax=ax,
                column=FIG1_RGW0_COLUMN,
                cmap="viridis",
                vmin=FIG1_VALID_MIN,
                vmax=FIG1_VALID_MAX,
                linewidth=0.12,
                edgecolor=(1, 1, 1, 0.22),
                legend=False,
                zorder=1,
                alpha=1.0,
                rasterized=True,
            )

        if basemap_obj is not None:
            basemap_obj.drawcoastlines(color="black", linewidth=0.45, zorder=5)

        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)

        lon_ticks = [-180, -120, -60, 0, 60, 120, 180]
        lat_ticks = [-60, -30, 0, 30, 60]

        ax.set_xticks(lon_ticks)
        ax.set_yticks(lat_ticks)
        ax.set_xticklabels([format_lon(x) for x in lon_ticks], fontsize=12)
        ax.set_yticklabels([format_lat(y) for y in lat_ticks], fontsize=12)

        ax.grid(
            which="major",
            color="0.78",
            linestyle="--",
            linewidth=0.40,
            alpha=0.50,
            zorder=1,
        )

        ax.tick_params(axis="both", length=3, width=0.7)

        # Visible outer frame around map panel.
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.25)
            spine.set_edgecolor("0.25")

        # Colorbar for valid values only.
        sm = mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=FIG1_VALID_MIN, vmax=FIG1_VALID_MAX),
            cmap="viridis",
        )
        sm.set_array([])

        if colorbar_side not in {"left", "right", "inset"}:
            raise ValueError("colorbar_side must be 'left', 'right', or 'inset'.")
        if colorbar_side == "inset":
            cbar_ax = ax.inset_axes([0.025, 0.12, 0.018, 0.58])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.ax.set_title(r"$\alpha_{GW}$", fontsize=12, pad=5)
            cbar.ax.yaxis.set_ticks_position("right")
        else:
            colorbar_pad = 0.100 if colorbar_side == "left" else 0.012
            cbar = fig.colorbar(
                sm,
                ax=ax,
                location=colorbar_side,
                fraction=0.018,
                pad=colorbar_pad,
            )
        if colorbar_side == "left":
            cbar.ax.yaxis.set_ticks_position("left")
            cbar.ax.yaxis.set_label_position("left")
            cbar.set_label(r"$\alpha_{GW}$", fontsize=13, rotation=90, labelpad=9)
        elif colorbar_side == "right":
            cbar.ax.set_title(r"$\alpha_{GW}$", fontsize=15, pad=7)
        cbar.ax.tick_params(labelsize=12)

        if include_distribution:
            add_rgw0_distribution_inset(ax, valid_gdf, bounds=FIG1_RGW0_DIST_INSET_BOUNDS)



    def _rescale_axes_group_x(axes, target_x0, target_x1):
        """
        Rescale a group of axes horizontally so the whole group has the same
        left/right extent as a reference axes, while preserving relative widths
        and horizontal gaps inside the group.
        """
        flat_axes = np.asarray(axes, dtype=object).ravel().tolist()
        boxes = [ax.get_position() for ax in flat_axes]
        current_x0 = min(box.x0 for box in boxes)
        current_x1 = max(box.x1 for box in boxes)
        current_width = current_x1 - current_x0
        target_width = target_x1 - target_x0

        if current_width <= 0 or target_width <= 0:
            return

        scale = target_width / current_width
        for ax, box in zip(flat_axes, boxes):
            new_x0 = target_x0 + (box.x0 - current_x0) * scale
            new_width = box.width * scale
            ax.set_position([new_x0, box.y0, new_width, box.height])

    # =============================================================================
    # Fig. 1 main function
    # =============================================================================
    def make_FIG1(
        long_df: pd.DataFrame,
        output_fig: Path = FIG1_OUTPUT_FIG,
        output_table_a: Path = FIG1_OUTPUT_TABLE_A,
        output_table_b: Path = FIG1_OUTPUT_TABLE_B,
    ):
        output_fig = ensure_parent(output_fig)
        output_table_a = ensure_parent(output_table_a)
        output_table_b = ensure_parent(output_table_b)

        all_gdf, valid_gdf, invalid_gdf, rgw0_df = load_rgw0_geodata()

        # Summary statistics use valid catchments only.
        rgw0_summary = summarize_rgw0_values(valid_gdf)
        rgw0_summary.to_csv(output_table_a, index=False)

        slopes = compute_slopes(long_df)
        slopes.to_csv(output_table_b, index=False)

        fig = plt.figure(figsize=FIG1_FIGSIZE, dpi=DPI)
        outer = gridspec.GridSpec(
            nrows=2,
            ncols=1,
            height_ratios=FIG1_OUTER_HEIGHT_RATIOS,
            hspace=FIG1_OUTER_HSPACE,
            figure=fig,
        )

        # ------------------------------------------------------------------
        # Top row: Panel A is Obs structure; Panel B is response curves.
        # Bottom row: Panel C is the Rgw0 world map.
        # ------------------------------------------------------------------
        top = outer[0, 0].subgridspec(
            nrows=1,
            ncols=2,
            width_ratios=FIG1_BOTTOM_WIDTH_RATIOS,
            wspace=FIG1_BOTTOM_WSPACE,
        )

        ax_network = fig.add_subplot(top[0, 0])
        plot_obs_structure_network_panel(ax_network)

        gs_b_wrap = top[0, 1].subgridspec(
            5,
            1,
            height_ratios=[0.06, 1, 1, 1, 0.10],
            hspace=0.0,
        )

        gs_b = gs_b_wrap[1:4, 0].subgridspec(
            nrows=len(FIG1_PANEL_B_METRICS),
            ncols=len(FIG1_FOCUS_NODES),
            wspace=FIG1_PANEL_B_WSPACE,
            hspace=FIG1_PANEL_B_HSPACE,
        )

        axes_b = np.empty((len(FIG1_PANEL_B_METRICS), len(FIG1_FOCUS_NODES)), dtype=object)

        row_ylims = {}
        for metric in FIG1_PANEL_B_METRICS:
            vals = []
            for node in FIG1_FOCUS_NODES:
                key = metric_key(metric, node)
                sub = long_df[
                    (long_df["Metric_Group"] == "Centrality")
                    & (long_df["Metric"] == key)
                    & (long_df["Dataset"].isin(DATASET_ORDER))
                ]
                vals.extend(
                    sub["Value"]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                    .astype(float)
                    .tolist()
                )

            row_ylims[metric] = nice_ylim(vals, force_zero=True)

        for row_id, metric in enumerate(FIG1_PANEL_B_METRICS):
            for col_id, node in enumerate(FIG1_FOCUS_NODES):
                ax = fig.add_subplot(gs_b[row_id, col_id])
                axes_b[row_id, col_id] = ax
                key = metric_key(metric, node)

                for dataset in DATASET_ORDER:
                    sub = long_df[
                        (long_df["Dataset"] == dataset)
                        & (long_df["Metric_Group"] == "Centrality")
                        & (long_df["Metric"] == key)
                    ].copy()

                    if sub.empty:
                        continue

                    sub = sub.sort_values(FIG1_X_COL)

                    ax.plot(
                        sub[FIG1_X_COL],
                        sub["Value"],
                        color=DATASET_COLORS.get(dataset),
                        marker=DATASET_MARKERS.get(dataset, "o"),
                        markersize=2.5,
                        linewidth=1,
                        label=dataset if (row_id == 0 and col_id == 0) else "_nolegend_",
                        zorder=3,
                    )

                ax.set_xlim(0, 1)
                ax.set_xticks([0, 0.5, 1.0])
                ax.set_ylim(row_ylims[metric])
                setup_clean_axis(ax)

                if row_id == 0:
                    ax.set_title(
                        display_node_label(node),
                        fontsize=10.5,
                        fontweight="bold",
                        pad=4,
                    )

                if col_id == 0:
                    ax.set_ylabel(metric, fontsize=11.5, labelpad=6)
                else:
                    ax.set_yticklabels([])
                    ax.tick_params(axis="y", length=0)

                if row_id == len(FIG1_PANEL_B_METRICS) - 1:
                    ax.set_xlabel(FIG1_X_LABEL, fontsize=10.5)
                else:
                    ax.set_xticklabels([])

                ax.tick_params(axis="both", labelsize=9.5)

        # Bottom full-width map (new panel C).
        ax_map = fig.add_subplot(outer[1, 0])
        plot_rgw0_world_map(fig, ax_map, all_gdf, valid_gdf, invalid_gdf)

        # ------------------------------------------------------------------
        # Match the full top-row width (a+b) to the actual map-panel width.
        # GeoPandas/Basemap can impose an aspect-constrained map axes; therefore
        # the map panel is used as the horizontal reference.
        # ------------------------------------------------------------------
        map_box = ax_map.get_position()
        _rescale_axes_group_x(
            [ax_network] + axes_b.ravel().tolist(),
            target_x0=map_box.x0,
            target_x1=map_box.x1,
        )

        # ------------------------------------------------------------------
        # Unified panel labels: a = Obs structure, b = response curves,
        # c = map.
        # ------------------------------------------------------------------
        net_box = ax_network.get_position()
        curve_boxes = [ax.get_position() for ax in axes_b.ravel()]
        curve_x0 = min(box.x0 for box in curve_boxes)
        curve_y1 = max(box.y1 for box in curve_boxes)
        map_box = ax_map.get_position()

        label_fontsize = 18
        fig.text(
            net_box.x0 - 0.025,
            net_box.y1 + 0.030,
            "a",
            fontsize=label_fontsize,
            fontweight="bold",
            va="top",
            ha="left",
        )

        fig.text(
            curve_x0 - 0.025,
            curve_y1 + 0.030,
            "b",
            fontsize=label_fontsize,
            fontweight="bold",
            va="top",
            ha="left",
        )

        fig.text(
            map_box.x0 - 0.025,
            map_box.y1 + 0.030,
            "c",
            fontsize=label_fontsize,
            fontweight="bold",
            va="top",
            ha="left",
        )

        handles, labels = axes_b[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center",
            bbox_to_anchor=(0.66, 0.49),
            ncol=4,
            frameon=False,
            fontsize=10.5,
            handlelength=2.3,
            columnspacing=1.0,
        )

        save_figure_both(fig, output_fig, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return output_fig, output_table_a, output_table_b

    def run_FIG1(long_df: Optional[pd.DataFrame] = None, workflow=None):
        if long_df is None:
            workflow = workflow or load_sensitivity_workflow()[0]
            long_df, _workbook_path = build_sensitivity_long_dataframe(workflow)

        outputs = make_FIG1(long_df)

        print("Fig. 1 exported:")
        for item in outputs:
            print(f"  {item}")


    # =============================================================================
    # Fig. 4: Variance decomposition across alpha-GW scenarios
    # =============================================================================
    FIG4_OUTPUT_FIG = SENSITIVITY_OUTPUT_DIR / "Fig4.tiff"

    OUT_NODE_METRICS = SENSITIVITY_OUTPUT_DIR / "alphaGW_variance_share_node_metrics.csv"
    OUT_DELTA_PATHWAYS = SENSITIVITY_OUTPUT_DIR / "alphaGW_variance_share_delta_pathways.csv"
    OUT_NODE_METRICS_WIDE = SENSITIVITY_OUTPUT_DIR / "alphaGW_variance_share_node_metrics_wide.csv"
    OUT_DELTA_PATHWAYS_WIDE = SENSITIVITY_OUTPUT_DIR / "alphaGW_variance_share_delta_pathways_wide.csv"
    OUT_NODE_METRICS_SS = SENSITIVITY_OUTPUT_DIR / "alphaGW_variance_components_node_metrics.csv"
    OUT_DELTA_PATHWAYS_SS = SENSITIVITY_OUTPUT_DIR / "alphaGW_variance_components_delta_pathways.csv"

    INCLUDE_ALPHA_DATASET_INTERACTION = True
    EXCLUDE_ALPHA_ENDPOINTS = True
    ALPHA_LOWER = 0.0
    ALPHA_UPPER = 1.0

    FIG4_NODE_ORDER = ["SM", "GW", "SW"]
    FIG4_PATHWAY_NODE_ORDER = ["Aw", "VEG", "SM", "GW", "SW", "SP/ENG"]
    FIG4_METRIC_ORDER = [
        "Strength",
        "Closeness",
        "Clustering",
        "Eigenvector",
    ]
    FIG4_METRIC_LABELS = {
        "Strength": "Strength",
        "Closeness": "Closeness",
        "Clustering": "Clustering",
        "Eigenvector": "Eigenvector",
    }
    BAR_COLORS = {
        "alpha": "#ed6a5a",
        "interaction": "#f4f1bb",
        "dataset": "#9bc1bc",
    }
    FIG4_TOP_N = 10
    FIG4_PANEL_RELATED_NODES = ["SM", "GW", "SW"]


    def filter_alpha_endpoints(long_df: pd.DataFrame) -> pd.DataFrame:
        """Remove alphaGW = 0 and alphaGW = 1 from the analysis."""
        if not EXCLUDE_ALPHA_ENDPOINTS:
            return long_df.copy()
        return long_df[(long_df["alpha"] > ALPHA_LOWER + EPS) & (long_df["alpha"] < ALPHA_UPPER - EPS)].copy()


    def get_analysis_alphas(long_df: pd.DataFrame, workflow) -> List[float]:
        """Return alphaGW scenarios used for variance decomposition."""
        raw_alphas = list(getattr(workflow, "ALPHAS", sorted(long_df["alpha"].dropna().unique())))
        alphas = pd.to_numeric(pd.Series(raw_alphas), errors="coerce").dropna().astype(float)

        if EXCLUDE_ALPHA_ENDPOINTS:
            alphas = alphas[(alphas > ALPHA_LOWER + EPS) & (alphas < ALPHA_UPPER - EPS)]

        return sorted(alphas.unique().tolist())


    def variance_decomposition_balanced(
        sub: pd.DataFrame,
        datasets: Iterable[str],
        alphas: Iterable[float],
        dataset_col: str = "Dataset",
        alpha_col: str = "alpha",
        value_col: str = "Value",
        eps: float = EPS,
    ) -> dict:
        """
        Two-way balanced decomposition over alphaGW and dataset.

        SS_total       = sum((y[a,d] - grand_mean)^2)
        SS_alpha       = n_dataset * sum((mean_d(y[a,d]) - grand_mean)^2)
        SS_dataset     = n_alpha   * sum((mean_a(y[a,d]) - grand_mean)^2)
        SS_interaction = sum((y[a,d] - mean_alpha[a] - mean_dataset[d] + grand_mean)^2)
        """
        pivot = (
            sub.pivot_table(index=alpha_col, columns=dataset_col, values=value_col, aggfunc="mean")
            .reindex(index=list(alphas), columns=list(datasets))
        )

        if pivot.isna().any().any():
            pivot = pivot.dropna(axis=0, how="any").dropna(axis=1, how="any")

        arr = pivot.to_numpy(dtype=float)
        n_alpha, n_dataset = arr.shape if arr.ndim == 2 else (0, 0)

        invalid_result = {
            "n_alpha": n_alpha,
            "n_dataset": n_dataset,
            "mean_value": np.nan,
            "total_variance": np.nan,
            "ss_total": np.nan,
            "ss_alpha": np.nan,
            "ss_dataset": np.nan,
            "ss_alpha_dataset_interaction": np.nan,
            "alpha_main_share": np.nan,
            "dataset_share": np.nan,
            "alpha_dataset_interaction_share": np.nan,
            "alphaGW_response_share": np.nan,
            "dataset_only_share": np.nan,
        }
        if n_alpha < 2 or n_dataset < 2 or arr.size == 0:
            return invalid_result

        grand = float(arr.mean())
        alpha_means = arr.mean(axis=1)
        dataset_means = arr.mean(axis=0)
        ss_total = float(((arr - grand) ** 2).sum())

        if ss_total <= eps:
            return {
                "n_alpha": n_alpha,
                "n_dataset": n_dataset,
                "mean_value": grand,
                "total_variance": 0.0,
                "ss_total": 0.0,
                "ss_alpha": 0.0,
                "ss_dataset": 0.0,
                "ss_alpha_dataset_interaction": 0.0,
                "alpha_main_share": 0.0,
                "dataset_share": 0.0,
                "alpha_dataset_interaction_share": 0.0,
                "alphaGW_response_share": 0.0,
                "dataset_only_share": 0.0,
            }

        ss_alpha = float(n_dataset * ((alpha_means - grand) ** 2).sum())
        ss_dataset = float(n_alpha * ((dataset_means - grand) ** 2).sum())
        ss_interaction = float(((arr - alpha_means[:, None] - dataset_means[None, :] + grand) ** 2).sum())

        alpha_main_share = ss_alpha / ss_total
        dataset_share = ss_dataset / ss_total
        interaction_share = ss_interaction / ss_total
        alpha_response_share = (ss_alpha + ss_interaction) / ss_total if INCLUDE_ALPHA_DATASET_INTERACTION else alpha_main_share

        return {
            "n_alpha": n_alpha,
            "n_dataset": n_dataset,
            "mean_value": grand,
            "total_variance": float(np.var(arr, ddof=0)),
            "ss_total": ss_total,
            "ss_alpha": ss_alpha,
            "ss_dataset": ss_dataset,
            "ss_alpha_dataset_interaction": ss_interaction,
            "alpha_main_share": alpha_main_share,
            "dataset_share": dataset_share,
            "alpha_dataset_interaction_share": interaction_share,
            "alphaGW_response_share": alpha_response_share,
            "dataset_only_share": dataset_share,
        }


    def calculate_node_metric_variance_shares(long_df: pd.DataFrame, workflow) -> pd.DataFrame:
        """Calculate alphaGW variance shares for node-level centrality metrics."""
        datasets = list(getattr(workflow, "DATASETS", {}).keys()) or sorted(long_df["Dataset"].dropna().unique())
        alphas = get_analysis_alphas(long_df, workflow)

        available_metrics = list(
            getattr(
                workflow,
                "CENTRALITY_METRICS",
                sorted(long_df.loc[long_df["Metric_Group"] == "Centrality", "Metric"].dropna().unique()),
            )
        )
        # Figure 4 uses only the selected metrics; Ranking_Betweenness is excluded
        # from both the variance calculation and the plotted/output tables.
        metric_order = [metric for metric in FIG4_METRIC_ORDER if metric in available_metrics]
        node_order = list(
            getattr(
                workflow,
                "KEEP_METRIC_NODES",
                sorted(long_df.loc[long_df["Metric_Group"] == "Centrality", "Node"].dropna().unique()),
            )
        )

        rows = []
        cent = long_df[long_df["Metric_Group"] == "Centrality"].copy()
        for metric_name in metric_order:
            for node in node_order:
                key = f"{metric_name}_{node}"
                sub = cent[cent["Metric"] == key].copy()
                if sub.empty:
                    continue

                row = {
                    "Metric_Group": "Centrality",
                    "Metric": metric_name,
                    "Node": node,
                    "Metric_Key": key,
                }
                row.update(variance_decomposition_balanced(sub, datasets=datasets, alphas=alphas))
                rows.append(row)

        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values(["alphaGW_response_share", "Metric", "Node"], ascending=[False, True, True])
        return out


    def calculate_delta_pathway_variance_shares(long_df: pd.DataFrame, workflow) -> pd.DataFrame:
        """Calculate alphaGW variance shares for delta matrix pathways."""
        datasets = list(getattr(workflow, "DATASETS", {}).keys()) or sorted(long_df["Dataset"].dropna().unique())
        alphas = get_analysis_alphas(long_df, workflow)
        nodes = list(getattr(workflow, "KEEP_NODES", sorted(set(long_df["From"].dropna()) | set(long_df["To"].dropna()))))

        rows = []
        delta = long_df[long_df["Metric_Group"] == "Delta_Matrix"].copy()
        for source in nodes:
            for target in nodes:
                sub = delta[(delta["From"] == source) & (delta["To"] == target)].copy()
                if sub.empty:
                    continue

                row = {
                    "Metric_Group": "Delta_Matrix",
                    "From": source,
                    "To": target,
                    "Pathway": f"{source}→{target}",
                    "Metric_Key": f"DeltaPercent_{source}->{target}",
                }
                row.update(variance_decomposition_balanced(sub, datasets=datasets, alphas=alphas))
                rows.append(row)

        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values(["alphaGW_response_share", "From", "To"], ascending=[False, True, True])
        return out


    def make_wide_tables(node_share: pd.DataFrame, delta_share: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Make wide-format variance-share summary tables."""
        node_wide = node_share.pivot(index="Node", columns="Metric", values="alphaGW_response_share")
        delta_wide = delta_share.pivot(index="From", columns="To", values="alphaGW_response_share")
        return node_wide, delta_wide


    def save_variance_outputs(
        node_share: pd.DataFrame,
        delta_share: pd.DataFrame,
        node_wide: pd.DataFrame,
        delta_wide: pd.DataFrame,
    ):
        """Save long, wide, and SS-component variance-decomposition tables."""
        ensure_dir(SENSITIVITY_OUTPUT_DIR)

        component_cols_node = [
            "Metric_Group",
            "Metric",
            "Node",
            "Metric_Key",
            "n_alpha",
            "n_dataset",
            "mean_value",
            "total_variance",
            "ss_total",
            "ss_alpha",
            "ss_alpha_dataset_interaction",
            "ss_dataset",
            "alpha_main_share",
            "alpha_dataset_interaction_share",
            "dataset_share",
            "alphaGW_response_share",
        ]
        component_cols_delta = [
            "Metric_Group",
            "From",
            "To",
            "Pathway",
            "Metric_Key",
            "n_alpha",
            "n_dataset",
            "mean_value",
            "total_variance",
            "ss_total",
            "ss_alpha",
            "ss_alpha_dataset_interaction",
            "ss_dataset",
            "alpha_main_share",
            "alpha_dataset_interaction_share",
            "dataset_share",
            "alphaGW_response_share",
        ]

        node_components = node_share[[col for col in component_cols_node if col in node_share.columns]].copy()
        delta_components = delta_share[[col for col in component_cols_delta if col in delta_share.columns]].copy()

        node_share.to_csv(OUT_NODE_METRICS, index=False)
        delta_share.to_csv(OUT_DELTA_PATHWAYS, index=False)
        node_components.to_csv(OUT_NODE_METRICS_SS, index=False)
        delta_components.to_csv(OUT_DELTA_PATHWAYS_SS, index=False)
        node_wide.to_csv(OUT_NODE_METRICS_WIDE)
        delta_wide.to_csv(OUT_DELTA_PATHWAYS_WIDE)


    def ensure_variance_shares(df: pd.DataFrame) -> pd.DataFrame:
        """Recalculate variance shares from SS components for plotting."""
        df = df.copy()
        df["alpha_main_share"] = safe_fraction(df["ss_alpha"], df["ss_total"])
        df["dataset_share"] = safe_fraction(df["ss_dataset"], df["ss_total"])
        df["interaction_share"] = safe_fraction(df["ss_alpha_dataset_interaction"], df["ss_total"])

        if INCLUDE_ALPHA_DATASET_INTERACTION:
            df["alphaGW_response_share"] = safe_fraction(df["ss_alpha"] + df["ss_alpha_dataset_interaction"], df["ss_total"])
        else:
            df["alphaGW_response_share"] = df["alpha_main_share"]

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        return df


    def stacked_barh(
        ax,
        df: pd.DataFrame,
        label_col: str,
        title: str = "",
        label_fontsize: float = 14.5,
        axis_fontsize: float = 15,
        tick_fontsize: float = 14,
    ):
        """Horizontal stacked bar plot for variance components."""
        plot_df = df.copy()
        y = np.arange(len(plot_df))
        alpha = plot_df["alpha_main_share"].to_numpy(float)
        interaction = plot_df["interaction_share"].to_numpy(float)
        dataset = plot_df["dataset_share"].to_numpy(float)

        ax.barh(y, alpha, color=BAR_COLORS["alpha"], label=r"$SS_{\alpha_{GW}}$")
        ax.barh(y, interaction, left=alpha, color=BAR_COLORS["interaction"], label=r"$SS_{\alpha_{GW} \times dataset}$")
        ax.barh(y, dataset, left=alpha + interaction, color=BAR_COLORS["dataset"], label=r"$SS_{dataset}$")

        ax.set_yticks(y)
        ax.set_yticklabels(plot_df[label_col], fontsize=label_fontsize)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraction of total variance", fontsize=axis_fontsize)
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=label_fontsize)
        ax.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.5)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)


    def build_variance_panel_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return the ranked node and pathway tables used in Fig. 3c-d."""
        node_df = ensure_variance_shares(pd.read_csv(OUT_NODE_METRICS_SS))
        delta_df = ensure_variance_shares(pd.read_csv(OUT_DELTA_PATHWAYS_SS))

        node_panel = node_df[
            node_df["Node"].isin(FIG4_NODE_ORDER)
            & node_df["Metric"].isin(FIG4_METRIC_ORDER)
        ].copy()
        top_node = node_panel.sort_values("alphaGW_response_share", ascending=False).head(FIG4_TOP_N).copy()
        top_node["Label"] = top_node["Node"] + " | " + top_node["Metric"]
        top_node = top_node.iloc[::-1]

        delta_panel = delta_df[
            (delta_df["From"].isin(FIG4_PANEL_RELATED_NODES) | delta_df["To"].isin(FIG4_PANEL_RELATED_NODES))
            & (delta_df["From"] != delta_df["To"])
        ].copy()
        top_delta = delta_panel.sort_values("alphaGW_response_share", ascending=False).head(FIG4_TOP_N).copy()
        top_delta["Label"] = top_delta["Pathway"]
        top_delta = top_delta.iloc[::-1]
        return top_node, top_delta


    def plot_fig4(output_fig: Path = FIG4_OUTPUT_FIG):
        """
        Generate Fig. 4 with two panels.

        Panel A: top node metrics by alphaGW response share.
        Panel B: top SM-, GW-, and SW-related delta pathways, excluding self-links.
        """
        output_fig = ensure_parent(output_fig)

        top_node, top_delta = build_variance_panel_data()

        fig = plt.figure(figsize=(13, 6), dpi=DPI)
        gs = GridSpec(nrows=1, ncols=2, width_ratios=[1, 1], wspace=0.4, figure=fig)

        ax_a = fig.add_subplot(gs[0, 0])
        stacked_barh(ax_a, top_node, label_col="Label", title="")
        ax_a.text(-0.12, 1.05, "a", transform=ax_a.transAxes, fontsize=16, fontweight="bold", va="top")

        ax_b = fig.add_subplot(gs[0, 1])
        stacked_barh(ax_b, top_delta, label_col="Label", title="")
        ax_b.text(-0.12, 1.05, "b", transform=ax_b.transAxes, fontsize=16, fontweight="bold", va="top")

        handles, labels = ax_a.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=15, frameon=False, bbox_to_anchor=(0.5, -0.09))

        save_figure_both(fig, output_fig, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Fig. 4 saved to: {output_fig}")
        return output_fig


    def run_fig4(long_df: Optional[pd.DataFrame] = None, workflow=None):
        if long_df is None or workflow is None:
            workflow, workflow_path = load_sensitivity_workflow()
            long_df, workbook_path = build_sensitivity_long_dataframe(workflow)
            print(f"Sensitivity workflow script: {workflow_path}")
            print(f"Sensitivity workbook: {workbook_path}")

        long_df = filter_alpha_endpoints(long_df)

        node_share = calculate_node_metric_variance_shares(long_df, workflow)
        delta_share = calculate_delta_pathway_variance_shares(long_df, workflow)
        node_wide, delta_wide = make_wide_tables(node_share, delta_share)
        save_variance_outputs(node_share, delta_share, node_wide, delta_wide)
        plot_fig4()


    # =============================================================================
    # Revised manuscript figure assembly
    # =============================================================================
    def _add_routing_structure_and_sensitivity_panels(
        fig,
        network_spec,
        sensitivity_spec,
        long_df: pd.DataFrame,
    ):
        """Add revised Fig. 3a-b and return the created axes."""
        ax_network = fig.add_subplot(network_spec)
        plot_obs_structure_network_panel(ax_network, style_scale=1.16)

        gs_b_wrap = sensitivity_spec.subgridspec(
            5,
            1,
            height_ratios=[0.02, 1, 1, 1, 0.18],
            hspace=0.0,
        )
        gs_b = gs_b_wrap[1:4, 0].subgridspec(
            nrows=len(FIG1_PANEL_B_METRICS),
            ncols=len(FIG1_FOCUS_NODES),
            wspace=FIG1_PANEL_B_WSPACE,
            hspace=FIG1_PANEL_B_HSPACE,
        )
        axes_b = np.empty((len(FIG1_PANEL_B_METRICS), len(FIG1_FOCUS_NODES)), dtype=object)

        row_ylims = {}
        for metric in FIG1_PANEL_B_METRICS:
            values = []
            for node in FIG1_FOCUS_NODES:
                key = metric_key(metric, node)
                sub = long_df[
                    (long_df["Metric_Group"] == "Centrality")
                    & (long_df["Metric"] == key)
                    & (long_df["Dataset"].isin(DATASET_ORDER))
                ]
                values.extend(
                    sub["Value"].replace([np.inf, -np.inf], np.nan).dropna().astype(float).tolist()
                )
            row_ylims[metric] = nice_ylim(values, force_zero=True)

        for row_id, metric in enumerate(FIG1_PANEL_B_METRICS):
            for col_id, node in enumerate(FIG1_FOCUS_NODES):
                ax = fig.add_subplot(gs_b[row_id, col_id])
                axes_b[row_id, col_id] = ax
                key = metric_key(metric, node)

                for dataset in DATASET_ORDER:
                    sub = long_df[
                        (long_df["Dataset"] == dataset)
                        & (long_df["Metric_Group"] == "Centrality")
                        & (long_df["Metric"] == key)
                    ].copy()
                    if sub.empty:
                        continue
                    sub = sub.sort_values(FIG1_X_COL)
                    ax.plot(
                        sub[FIG1_X_COL],
                        sub["Value"],
                        color=DATASET_COLORS.get(dataset),
                        marker=DATASET_MARKERS.get(dataset, "o"),
                        markersize=3.1,
                        linewidth=1.35,
                        label=dataset if (row_id == 0 and col_id == 0) else "_nolegend_",
                        zorder=3,
                    )

                ax.set_xlim(0, 1)
                ax.set_xticks([0, 0.5, 1.0])
                ax.set_ylim(row_ylims[metric])
                setup_clean_axis(ax)
                if row_id == 0:
                    ax.set_title(display_node_label(node), fontsize=11.0, fontweight="bold", pad=4)
                if col_id == 0:
                    ax.set_ylabel(metric, fontsize=11.5, labelpad=5)
                else:
                    ax.set_yticklabels([])
                    ax.tick_params(axis="y", length=0)
                if row_id == len(FIG1_PANEL_B_METRICS) - 1:
                    ax.set_xlabel(FIG1_X_LABEL, fontsize=10.8)
                else:
                    ax.set_xticklabels([])
                ax.tick_params(axis="both", labelsize=9.6, width=1.0)

        return ax_network, axes_b


    def _calculate_and_save_variance_outputs(long_df: pd.DataFrame, workflow):
        """Calculate the variance tables used by revised Fig. 3c-d."""
        analysis_df = filter_alpha_endpoints(long_df)
        node_share = calculate_node_metric_variance_shares(analysis_df, workflow)
        delta_share = calculate_delta_pathway_variance_shares(analysis_df, workflow)
        node_wide, delta_wide = make_wide_tables(node_share, delta_share)
        save_variance_outputs(node_share, delta_share, node_wide, delta_wide)
        return build_variance_panel_data()


    def run_revised_FIG1():
        """Generate revised Fig. 1 from the network and centrality panels."""
        df_long = FIG2_load_metrics_long(Path(FIG2_EXCEL_PATH))
        panel_a_df = FIG2_build_panel_a_matrix(df_long)
        panel_b_df = FIG2_build_panel_b_data(df_long)
        output = make_FIG2(
            panel_a_df=panel_a_df,
            panel_b_df=panel_b_df,
            output_path=REVISED_FIG1_OUTPUT,
            figure_label="Fig. 1",
        )
        print(f"Fig. 1 saved to: {output}")
        return output


    def run_revised_FIG2():
        """Generate main Fig. 2 and the SS-85-excluded supplementary SD ranking."""
        delta_mats = load_delta_matrices(Path(FIG3_EXCEL_PATH))
        std_df = compute_pathway_std(
            delta_mats,
            use_abs=False,
            min_nonzero=FIG3_MIN_NONZERO,
            ignore_diagonal=FIG3_IGNORE_DIAGONAL,
        )
        output_fig, output_csv = plot_delta_figure(
            delta_mats,
            std_df,
            output_fig=REVISED_FIG2_OUTPUT,
            output_csv=REVISED_FIG2_RANKING_TABLE,
        )
        print(f"Fig. 2 saved to: {output_fig}")
        print(f"Fig. 2 ranking table saved to: {output_csv}")

        supplementary_fig, supplementary_csv = run_supplementary_delta_ranking(delta_mats)
        return (output_fig, output_csv), (supplementary_fig, supplementary_csv)


    def run_supplementary_delta_ranking(delta_mats: Optional[Dict[str, pd.DataFrame]] = None):
        """Generate Supplementary Fig. S7 without requiring the other main figures."""
        if delta_mats is None:
            delta_mats = load_delta_matrices(Path(FIG3_EXCEL_PATH))

        reduced_mats = {name: matrix for name, matrix in delta_mats.items() if name != "SS-85"}
        reduced_std_df = compute_pathway_std(
            reduced_mats,
            use_abs=False,
            min_nonzero=FIG3_MIN_NONZERO,
            ignore_diagonal=FIG3_IGNORE_DIAGONAL,
        )
        supplementary_fig, supplementary_csv = plot_delta_ranking_supplement(
            reduced_std_df,
            output_fig=SUPP_FIG2G_EXCL_SS85_OUTPUT,
            output_csv=SUPP_FIG2G_EXCL_SS85_TABLE,
            network_count=len(reduced_mats),
        )
        print(f"Supplementary SS-85-excluded SD ranking saved to: {supplementary_fig}")
        print(f"Supplementary ranking table saved to: {supplementary_csv}")
        return supplementary_fig, supplementary_csv


    def plot_revised_FIG3(
        long_df: pd.DataFrame,
        top_node: pd.DataFrame,
        top_delta: pd.DataFrame,
        output_fig: Path = REVISED_FIG3_OUTPUT,
    ):
        """Combine routing panels with node and pathway variance attribution."""
        output_fig = ensure_parent(output_fig)
        fig = plt.figure(figsize=(10.6, 8.6), dpi=DPI)
        outer = GridSpec(
            nrows=2,
            ncols=2,
            width_ratios=[1, 1],
            height_ratios=[1, 1],
            wspace=0.38,
            hspace=0.34,
            figure=fig,
        )

        ax_network, axes_b = _add_routing_structure_and_sensitivity_panels(
            fig,
            outer[0, 0],
            outer[0, 1],
            long_df,
        )
        ax_b_frame = fig.add_subplot(outer[0, 1], frame_on=False)
        ax_b_frame.set_axis_off()

        ax_c = fig.add_subplot(outer[1, 0])
        stacked_barh(
            ax_c,
            top_node,
            label_col="Label",
            label_fontsize=10.4,
            axis_fontsize=12.3,
            tick_fontsize=10.8,
        )
        ax_d = fig.add_subplot(outer[1, 1])
        stacked_barh(
            ax_d,
            top_delta,
            label_col="Label",
            label_fontsize=10.4,
            axis_fontsize=12.3,
            tick_fontsize=10.8,
        )

        label_fontsize = 17
        for panel_ax, label in [(ax_network, "a"), (ax_b_frame, "b"), (ax_c, "c"), (ax_d, "d")]:
            panel_ax.text(
                -0.12,
                1.06,
                label,
                transform=panel_ax.transAxes,
                fontsize=label_fontsize,
                fontweight="bold",
                va="top",
                ha="left",
            )

        curve_handles, curve_labels = axes_b[0, 0].get_legend_handles_labels()
        ax_b_frame.legend(
            curve_handles,
            curve_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=4,
            frameon=False,
            fontsize=9.6,
            handlelength=2.0,
            columnspacing=0.8,
        )
        variance_handles, variance_labels = ax_c.get_legend_handles_labels()
        fig.legend(
            variance_handles,
            variance_labels,
            loc="lower center",
            ncol=3,
            fontsize=10.8,
            frameon=False,
            bbox_to_anchor=(0.5, 0.005),
        )
        _scale_figure_fonts(fig, 1.12)
        fig.subplots_adjust(left=0.105, right=0.985, top=0.965, bottom=0.10)
        save_figure_both(fig, output_fig, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return output_fig


    def run_revised_FIG3(long_df: pd.DataFrame, workflow):
        """Generate revised Fig. 3 and its slope/variance tables."""
        ensure_parent(REVISED_FIG3_SLOPE_TABLE)
        compute_slopes(long_df).to_csv(REVISED_FIG3_SLOPE_TABLE, index=False)
        top_node, top_delta = _calculate_and_save_variance_outputs(long_df, workflow)
        output = plot_revised_FIG3(long_df, top_node, top_delta)
        print(f"Fig. 3 saved to: {output}")
        return output


    def plot_revised_FIG4(
        all_gdf,
        valid_gdf,
        invalid_gdf,
        output_fig: Path = REVISED_FIG4_OUTPUT,
    ):
        """Plot the map, implied assessment values, and catchment distribution."""
        output_fig = ensure_parent(output_fig)
        implied_df = load_implied_alpha_gw_values()
        fig = plt.figure(figsize=FIG4_FIGSIZE, dpi=DPI)
        gs = GridSpec(
            2,
            1,
            height_ratios=FIG4_OUTER_HEIGHT_RATIOS,
            hspace=FIG4_OUTER_HSPACE,
            figure=fig,
        )
        ax_map = fig.add_subplot(gs[0, 0])
        plot_rgw0_world_map(
            fig,
            ax_map,
            all_gdf,
            valid_gdf,
            invalid_gdf,
            include_distribution=False,
            colorbar_side="inset",
        )
        ax_combined = fig.add_subplot(gs[1, 0])
        panel_stats = plot_combined_alpha_gw_panel(ax_combined, valid_gdf, implied_df)

        fig.subplots_adjust(left=0.075, right=0.985, top=0.955, bottom=0.105)
        fig.canvas.draw()
        map_position = ax_map.get_position()
        combined_position = ax_combined.get_position()
        ax_combined.set_position([map_position.x0, combined_position.y0, map_position.width, map_position.height])
        ax_map.text(0.0, 1.035, "a", transform=ax_map.transAxes, fontsize=17, fontweight="bold", va="bottom", ha="left")
        ax_combined.text(0.0, 1.035, "b", transform=ax_combined.transAxes, fontsize=17, fontweight="bold", va="bottom", ha="left")
        _scale_figure_fonts(fig, FIG4_FONT_SCALE)
        save_figure_both(fig, output_fig, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return output_fig, panel_stats, implied_df


    def run_revised_FIG4():
        """Generate revised Fig. 4 and save its distribution statistics."""
        all_gdf, valid_gdf, invalid_gdf, _rgw0_df = load_rgw0_geodata()
        summary = summarize_rgw0_values(valid_gdf)
        output, panel_stats, implied_df = plot_revised_FIG4(all_gdf, valid_gdf, invalid_gdf)
        for key, value in panel_stats.items():
            summary[key] = value
        mapped_area_km2 = float(summary.loc[0, "valid_area_km2"])
        n_units = int(summary.loc[0, "n_valid_0_1"])
        land_fraction = mapped_area_km2 / NON_ANTARCTIC_LAND_AREA_KM2
        summary["non_antarctic_land_area_km2"] = NON_ANTARCTIC_LAND_AREA_KM2
        summary["non_antarctic_land_fraction_covered"] = land_fraction
        ensure_parent(REVISED_FIG4_SUMMARY_TABLE)
        summary.to_csv(REVISED_FIG4_SUMMARY_TABLE, index=False)
        implied_df.to_csv(ensure_parent(REVISED_FIG4_IMPLIED_TABLE), index=False)
        caption_note = FIG4_CAPTION_NOTE.format(
            n_units=n_units,
            mapped_area_million=mapped_area_km2 / 1_000_000.0,
            land_fraction_pct=100.0 * land_fraction,
        )
        ensure_parent(REVISED_FIG4_CAPTION_NOTE).write_text(caption_note + "\n", encoding="utf-8")
        print(f"Fig. 4 saved to: {output}")
        print(f"Fig. 4 summary table saved to: {REVISED_FIG4_SUMMARY_TABLE}")
        print(f"Fig. 4 implied alphaGW table saved to: {REVISED_FIG4_IMPLIED_TABLE}")
        print(f"Fig. 4 caption note saved to: {REVISED_FIG4_CAPTION_NOTE}")
        return output, REVISED_FIG4_SUMMARY_TABLE


    def write_main_figure_order_manifest(include_fig4: bool = True):
        """Write the manuscript figure order and panel provenance."""
        rows = [
            {
                "figure": "Fig1",
                "file": "Fig1.tiff",
                "panels": "a-c",
                "content": "Global networks and node-level centrality sensitivity",
            },
            {
                "figure": "Fig2",
                "file": "Fig2.tiff",
                "panels": "a-g",
                "content": "Delta matrices and ranked pathway uncertainty",
            },
            {
                "figure": "Fig3",
                "file": "Fig3.tiff",
                "panels": "a-d",
                "content": "Groundwater routing, node sensitivity, and variance attribution",
            },
        ]
        if include_fig4:
            rows.append(
                {
                    "figure": "Fig4",
                    "file": "Fig4.tiff",
                    "panels": "a-b",
                    "content": "Catchment alphaGW map, implied assessment values, and distribution",
                }
            )
        manifest_path = ensure_parent(MAIN_FIGURE_DIR / "figure_order.tsv")
        pd.DataFrame(rows).to_csv(manifest_path, sep="\t", index=False)
        return manifest_path


    # =============================================================================
    # Main entry point
    # =============================================================================
    def main():
        """Run the revised manuscript figures in numerical order."""
        if RUN_FIG1:
            run_revised_FIG1()

        if RUN_FIG2:
            run_revised_FIG2()

        workflow = None
        long_df = None
        if RUN_FIG3:
            workflow, workflow_path = load_sensitivity_workflow()
            long_df, workbook_path = build_sensitivity_long_dataframe(workflow)
            print(f"Sensitivity workflow script: {workflow_path}")
            print(f"Sensitivity workbook: {workbook_path}")

            run_revised_FIG3(long_df=long_df.copy(), workflow=workflow)

        if RUN_FIG4:
            run_revised_FIG4()

        manifest_path = write_main_figure_order_manifest(include_fig4=RUN_FIG4)
        print(f"Main-figure order manifest: {manifest_path}")

    _public = {k: v for k, v in locals().items() if not k.startswith("__")}
    return types.SimpleNamespace(**_public)
