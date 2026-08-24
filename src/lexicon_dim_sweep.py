"""Lexicon dimensionality sweep control for encoding probe.

Usage
-----
    python src/lexicon_dim_sweep.py --librispeech_split train-clean-100 \
        --modelname facebook/wav2vec2-base --probe_name ridge \
        --k_values 10 25 50 100 --random_seed 42
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

import matplotlib  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402

# Make sibling modules importable regardless of the current working directory
# (the repo's scripts are normally run from inside src/).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plotnine as p9  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from load_probe_data import get_section_shapes, load_data  # noqa: E402
from probe_runner import (  # noqa: E402
    build_feature_lookup,
    drop_feature_groups,
    evaluate_probe,
    fit_probe,
    split_train_test,
)
from utils import FIGURES_ROOT, RESULTS_ROOT, pick_probe  # noqa: E402

# Set up logger with time, name, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# Name of the lexicon block as it appears in data_shape / section_shapes.
# format_data stores both "word_embedding" (fastText, 100-dim) and
# "dnn_word_embedding" (DNN), but only "word_embedding" is part of
# selected_input_components, so it is the only one that survives
# further_process's final filtering and appears in get_section_shapes.
LEXICON_NAME = "word_embedding"

RESULT_COLUMNS = [
    "k",
    "layer",
    "uv_full",
    "uv_ablated",
    "gap",
    "modelname",
    "librispeech_split",
]


def run_sweep(
    feature_sets: np.ndarray,
    model_hidden_states: np.ndarray,
    section_shapes: np.ndarray,
    speaker_ids: list[str],
    k_values: list[int],
    estimator,
    param_grid: dict,
    *,
    lexicon_name: str = LEXICON_NAME,
    random_state: int = 42,
    modelname: str = "",
    librispeech_split: str = "",
) -> pd.DataFrame:
    """Run the lexicon dimensionality sweep.

    Args:
        feature_sets: 2D input feature matrix ``[n_samples, n_features]``.
        model_hidden_states: 3D target array ``[n_samples, n_layers, hidden]``.
        section_shapes: Array of ``[start, end, name]`` rows (see
            ``get_section_shapes``).
        speaker_ids: Speaker id per sample, used to stratify the single
            deterministic train/test split.
        k_values: PCA dimensionalities to sweep for the lexicon block.
        estimator: Probe estimator (e.g. from ``pick_probe``).
        param_grid: GridSearchCV parameter grid for the estimator.
        lexicon_name: Name of the lexicon block in ``section_shapes``.
        random_state: Seed for the single train/test split.
        modelname: Model name recorded in the result rows.
        librispeech_split: Split recorded in the result rows.

    Returns:
        DataFrame with columns ``k, layer, uv_full, uv_ablated, gap,
        modelname, librispeech_split`` where ``uv_* = 1 - R2`` (test) and
        ``gap = uv_ablated - uv_full``.
    """
    feature_sets = np.asarray(feature_sets)
    model_hidden_states = np.asarray(model_hidden_states)
    section_shapes = np.asarray(section_shapes)

    if feature_sets.ndim != 2:
        raise ValueError(f"feature_sets must be 2D, got shape {feature_sets.shape}")
    if model_hidden_states.ndim != 3:
        raise ValueError(
            "model_hidden_states must be 3D (n_samples, n_layers, hidden), "
            f"got shape {model_hidden_states.shape}"
        )
    if feature_sets.shape[0] != model_hidden_states.shape[0]:
        raise ValueError(
            "feature_sets and model_hidden_states must share the sample axis: "
            f"{feature_sets.shape} vs {model_hidden_states.shape}"
        )

    lookup = build_feature_lookup(section_shapes)
    if lexicon_name not in lookup:
        raise KeyError(
            f"Lexicon block '{lexicon_name}' not found in section_shapes. "
            f"Available blocks: {sorted(lookup)}"
        )
    start, end = lookup[lexicon_name]
    lexicon_dim = end - start

    k_values = sorted({int(k) for k in k_values})
    for k in k_values:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k > lexicon_dim:
            raise ValueError(
                f"k={k} exceeds the lexicon block dimensionality "
                f"({lexicon_dim} for '{lexicon_name}')."
            )
    logger.info(
        "Lexicon block '%s' spans columns %d:%d (%d dims); sweeping k over %s",
        lexicon_name,
        start,
        end,
        lexicon_dim,
        k_values,
    )

    # One deterministic split on sample indices; the same indices are applied
    # to the hidden states and to the ablated matrix so every k value and both
    # probe variants share identical train/test rows.
    sample_idx = np.arange(feature_sets.shape[0])
    _, _, idx_train, idx_test = split_train_test(
        feature_sets,
        sample_idx,
        stratify_labels=speaker_ids,
        random_state=random_state,
    )
    logger.info(
        "Split: %d train / %d test samples (stratified by speaker, seed=%d)",
        idx_train.size,
        idx_test.size,
        random_state,
    )

    X_train_full = feature_sets[idx_train]
    X_test_full = feature_sets[idx_test]

    # Full-minus-lexicon matrix, computed once; the lexicon block is removed
    # entirely so this does not depend on k.
    X_ablated = drop_feature_groups(feature_sets, lookup, [lexicon_name])
    X_train_ablated = X_ablated[idx_train]
    X_test_ablated = X_ablated[idx_test]

    y_train_all = model_hidden_states[idx_train]
    y_test_all = model_hidden_states[idx_test]
    n_layers = model_hidden_states.shape[1]

    # Column indices of the non-lexicon blocks, in their original order.
    other_cols = np.concatenate(
        [np.arange(0, start), np.arange(end, feature_sets.shape[1])]
    )

    rows: list[dict] = []
    # Ablated test scores per layer: identical for every k (same X, same y,
    # deterministic GridSearchCV), so fit once per layer and reuse.
    ablated_test_score: dict[int, float] = {}

    for k in k_values:
        # Fit StandardScaler + PCA on the TRAIN lexicon columns only, then
        # transform both train and test lexicon columns (no leakage). k ==
        # lexicon_dim is kept as a full-rank rotation: all k values get the
        # identical StandardScaler+PCA treatment, only dimensionality varies.
        lexicon_pca = make_pipeline(StandardScaler(), PCA(n_components=k))
        lex_train_pca = lexicon_pca.fit_transform(X_train_full[:, start:end])
        lex_test_pca = lexicon_pca.transform(X_test_full[:, start:end])

        # Assemble Full(k): PCA'd lexicon block (k columns) + other blocks.
        X_train_k = np.concatenate([lex_train_pca, X_train_full[:, other_cols]], axis=1)
        X_test_k = np.concatenate([lex_test_pca, X_test_full[:, other_cols]], axis=1)
        logger.info("Assembled Full(k=%d) with shape %s", k, X_train_k.shape)

        for layer in range(n_layers):
            y_train = y_train_all[:, layer, :]
            y_test = y_test_all[:, layer, :]

            grid_full = fit_probe(estimator, param_grid, X_train_k, y_train)
            _, test_score_full = evaluate_probe(
                grid_full.best_estimator_, X_train_k, y_train, X_test_k, y_test
            )

            if layer not in ablated_test_score:
                grid_ablated = fit_probe(
                    estimator, param_grid, X_train_ablated, y_train
                )
                _, test_score_ablated = evaluate_probe(
                    grid_ablated.best_estimator_,
                    X_train_ablated,
                    y_train,
                    X_test_ablated,
                    y_test,
                )
                ablated_test_score[layer] = test_score_ablated
            test_score_ablated = ablated_test_score[layer]

            uv_full = 1.0 - test_score_full
            uv_ablated = 1.0 - test_score_ablated
            rows.append(
                {
                    "k": k,
                    "layer": layer,
                    "uv_full": uv_full,
                    "uv_ablated": uv_ablated,
                    "gap": uv_ablated - uv_full,
                    "modelname": modelname,
                    "librispeech_split": librispeech_split,
                }
            )
        logger.info(
            "Finished k=%d (%d/%d layers done)",
            k,
            n_layers,
            n_layers,
        )

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def plot_sweep(
    results_df: pd.DataFrame,
    *,
    modelname: str,
    librispeech_split: str,
    out_dir: str = FIGURES_ROOT,
) -> str | None:
    """Plot the lexicon dimensionality sweep as a two-panel figure.

    Left panel: absolute unexplained variance (1 - R^2) against layer, with one
    line per PCA dimensionality k for the Full probe and a dashed
    Full-minus-lexicon baseline. Right panel: the lexicon ablation gap as a
    percentage of the full (maximum-k) gap, against layer, one line per k.
    Saves to
    ``lexicon_dim_sweep_{modelname-slug}_{split}.png`` inside ``out_dir``
    (defaults to FIGURES_ROOT, i.e. ``<repo>/figures``).
    """
    if results_df is None or results_df.empty:
        logger.warning("Empty results DataFrame; skipping sweep plot.")
        return None

    df = results_df.copy()
    ks = sorted(df["k"].unique(), key=int)
    k_labels = [str(int(k)) for k in ks]
    ablated_label = "Full \u2216 lexicon"
    uv_panel = "Unexplained variance (1 \u2212 R\u00b2)"
    gap_panel = f"Lexicon contribution (% of {int(max(ks))}-dim)"

    # Left panel: absolute UV. One Full(k) line per k plus the ablated baseline.
    full_rows = []
    for k in ks:
        sub = df.loc[df["k"] == k, ["layer", "uv_full"]].copy()
        sub["k"] = str(int(k))
        sub["value"] = sub["uv_full"]
        sub["linetype"] = "solid"
        full_rows.append(sub[["layer", "k", "value", "linetype"]])
    full = pd.concat(full_rows, ignore_index=True)

    ablated = df[["layer", "uv_ablated"]].drop_duplicates().copy()
    ablated["k"] = ablated_label
    ablated["value"] = ablated["uv_ablated"]
    ablated["linetype"] = "dashed"
    ablated = ablated[["layer", "k", "value", "linetype"]]

    panel_uv = pd.concat([full, ablated], ignore_index=True)
    panel_uv["panel"] = uv_panel

    # Right panel: ablation gap per k, as a percentage of the full (maximum-k)
    # gap, so each reduced dimensionality is shown relative to the unreduced
    # (100-dim) lexicon contribution.
    ref_gap = df.loc[df["k"] == max(ks), ["layer", "gap"]].rename(
        columns={"gap": "ref_gap"}
    )
    gap_rows = []
    for k in ks:
        sub = df.loc[df["k"] == k, ["layer", "gap"]].merge(ref_gap, on="layer")
        sub["value"] = sub["gap"] / sub["ref_gap"] * 100.0
        sub["k"] = str(int(k))
        sub["linetype"] = "solid"
        gap_rows.append(sub[["layer", "k", "value", "linetype"]])
    gap = pd.concat(gap_rows, ignore_index=True)
    gap["panel"] = gap_panel

    plot_df = pd.concat([panel_uv, gap], ignore_index=True)
    plot_df["k"] = pd.Categorical(
        plot_df["k"], categories=k_labels + [ablated_label], ordered=True
    )
    plot_df["panel"] = pd.Categorical(
        plot_df["panel"], categories=[uv_panel, gap_panel], ordered=True
    )

    # Sequential palette for k (higher k = deeper color) + grey for the ablated
    # baseline.
    n_k = len(ks)
    cmap = matplotlib.colormaps["viridis"]
    k_colors = [
        mcolors.to_hex(cmap(i / max(n_k - 1, 1))) for i in range(n_k)
    ]
    colors = k_colors + ["#404040"]

    slug = modelname.replace("/", "-")
    filename = f"lexicon_dim_sweep_{slug}_{librispeech_split}.png"
    savepath = os.path.join(out_dir, filename)
    os.makedirs(out_dir, exist_ok=True)

    figure = (
        p9.ggplot(
            plot_df,
            p9.aes(
                x="layer",
                y="value",
                color="k",
                linetype="linetype",
                group="k",
            ),
        )
        + p9.geom_line(size=0.8)
        + p9.geom_point(size=1.3)
        + p9.scale_color_manual(values=colors)
        + p9.scale_linetype_manual(
            values={"solid": "solid", "dashed": "dashed"}
        )
        + p9.facet_wrap("~ panel", scales="free_y", nrow=1)
        + p9.scale_x_continuous(breaks=list(range(0, 13, 2)))
        + p9.labs(x="Layer", y="", color="PCA components (k)")
        + p9.theme_minimal()
        + p9.theme(
            figure_size=(9.5, 4),
            dpi=300,
            legend_position="bottom",
            legend_title=p9.element_text(size=11),
            legend_text=p9.element_text(size=11),
            axis_title=p9.element_text(size=11),
            axis_text=p9.element_text(size=11),
            strip_text=p9.element_text(size=11),
        )
        + p9.guides(
            color=p9.guide_legend(nrow=1, byrow=True),
            linetype=False,
        )
    )
    figure.save(savepath)
    logger.info("Saved sweep plot to %s", savepath)
    return savepath


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Lexicon dimensionality sweep control: measure the encoding "
            "probe's lexicon-block contribution (ablation UV gap) as a "
            "function of the block's PCA dimensionality."
        )
    )
    parser.add_argument(
        "--librispeech_split",
        type=str,
        default="train-clean-100",
        help="LibriSpeech split to use (default: train-clean-100).",
    )
    parser.add_argument(
        "--modelname",
        type=str,
        default="facebook/wav2vec2-base",
        help="Transformer model name used to extract hidden states.",
    )
    parser.add_argument(
        "--probe_name",
        type=str,
        default="ridge",
        help="Probe name passed to pick_probe (default: ridge).",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100],
        help="PCA dimensionalities to sweep for the lexicon block.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for frame sampling and the train/test split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite an existing results CSV.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Re-plot from an existing results CSV without rerunning the sweep.",
    )
    args = parser.parse_args()

    modelname_slug = args.modelname.replace("/", "-")
    csv_path = os.path.join(
        RESULTS_ROOT,
        "lexicon_dim_sweep",
        f"{args.librispeech_split}_{modelname_slug}_{args.probe_name}.csv",
    )

    if args.plot_only:
        if not os.path.exists(csv_path):
            logger.error("No results CSV at %s to plot; run the sweep first.", csv_path)
            sys.exit(1)
        results_df = pd.read_csv(csv_path)
        plot_sweep(
            results_df,
            modelname=args.modelname,
            librispeech_split=args.librispeech_split,
        )
        return

    if os.path.exists(csv_path) and not args.overwrite:
        logger.info(
            "Results CSV already exists at %s; skipping run "
            "(use --overwrite to recompute).",
            csv_path,
        )
        return

    # Load data ONCE with the word-embedding reduction DISABLED so we get the
    # full 100-dim lexicon block (mirrors get_default_load_kwargs in
    # experiment_pipeline.py except for reduce_dnn_word_embedding and
    # normalize_features).
    feature_sets, model_hidden_states, filename_timestamp, data_shape = load_data(
        librispeech_split=args.librispeech_split,
        modelname=args.modelname,
        selected_input_components=[
            "eGeMAPSv02",
            "syntax_feature",
            "ppg_feature",
            "metadata",
            "word_embedding",
        ],
        seq_sampling="random_frames",
        one_hot_encode_syntax=True,
        one_hot_encode_syntax_separate=False,
        one_hot_encode_metadata=True,
        argmax_ppg=False,
        normalize_features=False,
        reduce_dnn_word_embedding=False,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
    )

    # Note on block names: format_data stores both "word_embedding" (fastText)
    # and "dnn_word_embedding" (DNN), but only "word_embedding" is in
    # selected_input_components, so further_process's final filtering keeps
    # only "word_embedding" and it is the only lexicon key that appears in
    # data_shape / section_shapes. The "dnn_word_embedding" PCA branch
    # (load_probe_data.py lines ~519-541) never runs here and its key never
    # reaches get_section_shapes.
    speaker_ids = [x[0].split("-")[0] for x in filename_timestamp]
    section_shapes = np.array(get_section_shapes(data_shape))
    logger.info("Section shapes: %s", section_shapes)
    logger.info("Lexicon block name in section_shapes: %s", LEXICON_NAME)

    estimator, param_grid = pick_probe(probe_name=args.probe_name)

    results_df = run_sweep(
        feature_sets,
        model_hidden_states,
        section_shapes,
        speaker_ids,
        k_values=args.k_values,
        estimator=estimator,
        param_grid=param_grid,
        random_state=args.random_seed,
        modelname=args.modelname,
        librispeech_split=args.librispeech_split,
    )

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    logger.info("Saved %d result rows to %s", len(results_df), csv_path)

    plot_sweep(
        results_df,
        modelname=args.modelname,
        librispeech_split=args.librispeech_split,
    )


if __name__ == "__main__":
    main()
