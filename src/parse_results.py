import argparse
import glob
import logging
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
import plotnine as p9

from utils import FIGURES_ROOT, RESULTS_ROOT

# Set up logger with time, name, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


# Rename config_name
CONFIG_NAME_RENAME: dict = {
    "AllFeatures": r"$\mathit{Full}$",
    "AcousticOnly": "Acoustics Only",
    "eGeMAPSv02": r"$\mathit{Full} \setminus \mathit{Acoustics}$",
    "ChapterID-OH": r"$\mathit{Full} \setminus\mathit{Chapter}$",
    "SpeakerID-OH": r"$\mathit{Full} \setminus \mathit{Speaker}$",
    "word_embedding": r"$\mathit{Full} \setminus \mathit{Lexicon}$",
    "ppg_feature": r"$\mathit{Full} \setminus \mathit{Phonetics}$",
    "syntax_feature": r"$\mathit{Full} \setminus \mathit{Syntax}$",
    "syntax_feature+word_embedding": r"$\mathit{Full} \setminus \mathit{Syntax} \setminus \mathit{Lexicon}$",
    "ppg_feature+eGeMAPSv02": r"$\mathit{Full} \setminus \mathit{Phonetics} \setminus \mathit{Acoustics}$",
    "ppg_feature+SpeakerID-OH": r"$\mathit{Full} \setminus \mathit{Phonetics} \setminus \mathit{Speaker}$",
    "SpeakerID-OH+eGeMAPSv02": r"$\mathit{Full} \setminus \mathit{Acoustics} \setminus \mathit{Speaker}$",
    "SpeakerID-OH+eGeMAPSv02+ppg_feature": r"$\mathit{Full} \setminus \mathit{Acoustics} \setminus \mathit{Phonetics} \setminus \mathit{Speaker}$",
}

ALLFEATURE_NAME = CONFIG_NAME_RENAME.get("AllFeatures", "AllFeatures")
SYNTAX_NAME = CONFIG_NAME_RENAME.get("syntax_feature", "syntax_feature")
PHONETIC_NAME = CONFIG_NAME_RENAME.get("ppg_feature", "ppg_feature")
ACOUSTIC_NAME = CONFIG_NAME_RENAME.get("eGeMAPSv02", "eGeMAPSv02")
SPEAKERID_NAME = CONFIG_NAME_RENAME.get("SpeakerID-OH", "SpeakerID-OH")
LEXICON_NAME = CONFIG_NAME_RENAME.get("word_embedding", "word_embedding")
SYNTAX_LEXICON_NAME = CONFIG_NAME_RENAME.get(
    "syntax_feature+word_embedding", "syntax_feature+word_embedding"
)
PHONETIC_ACOUSTIC_NAME = CONFIG_NAME_RENAME.get(
    "ppg_feature+eGeMAPSv02", "ppg_feature+eGeMAPSv02"
)
SPEAKERID_ACOUSTIC_NAME = CONFIG_NAME_RENAME.get(
    "SpeakerID-OH+eGeMAPSv02", "SpeakerID-OH+eGeMAPSv02"
)

PHONETIC_SPEAKERID_NAME = CONFIG_NAME_RENAME.get(
    "ppg_feature+SpeakerID-OH", "ppg_feature+SpeakerID-OH"
)

# JOINT_REMOVAL_CONFIGS: set[str] = {
#     "—Syntax —Lexicon",
#     "—Acoustics —Speaker",
#     "—Phonetics —Speaker",
#     "—Acoustics —Phonetics —Speaker",
# }
# JOINT_REMOVAL_LABEL = "Joint removal"

SHARED_TOPLINE_RUN_GROUP = "combined_topdown_shared_topline"
RUN_GROUP_BY_EXPERIMENT: dict[str, str] = {
    "single_feat_removal": SHARED_TOPLINE_RUN_GROUP,
    "syntax_feat_removal": SHARED_TOPLINE_RUN_GROUP,
    "speakerid_phonetic_acoustic_removal": SHARED_TOPLINE_RUN_GROUP,
}

RESULTS_DIRS: list[str] = [
    "single_feat_removal",
    "syntax_feat_removal",
    "syntax_lexicon_decomposition",
    "speakerid_phonetic_acoustic_removal",
]

FOCUS_CONFIGS: dict[str, dict[str, list[str] | str]] = {
    "syntax_lexical": {
        "single_configs": [LEXICON_NAME, SYNTAX_NAME],
        "joint_config": SYNTAX_LEXICON_NAME,
    },
    "acoustic_speaker": {
        "single_configs": [ACOUSTIC_NAME, SPEAKERID_NAME],
        "joint_config": SPEAKERID_ACOUSTIC_NAME,
    },
    "phonetic_speaker": {
        "single_configs": [PHONETIC_NAME, SPEAKERID_NAME],
        "joint_config": PHONETIC_SPEAKERID_NAME,
    },
}


MODELNAME_ORDER: list = [
    "wav2vec2-base",
    "wav2vec2-base-960h",
    "wav2vec2-large",
    "hubert-base-ls960",
    "hubert-large-ll60k",
    "wavlm-base",
    "wav2vec2-base-superb-sid",
    "wav2vec2-ls100-sid",
    "bert-base-uncased",
    "roberta-base",
    "ModernBERT-base",
]


MODELNAME_RENAME: dict[str, str] = {
    "wav2vec2-base": "wav2vec2 (base)",
    "wav2vec2-base-960h": "wav2vec2 (ASR)",
    "wav2vec2-large": "wav2vec2 (large)",
    "hubert-base-ls960": "HuBERT (base)",
    "hubert-large-ll60k": "HuBERT (large)",
    "wavlm-base": "WavLM (base)",
    "roberta-base": "RoBERTa (base)",
    "bert-base-uncased": "BERT (base)",
    "ModernBERT-base": "ModernBERT (base)",
    "wav2vec2-base-superb-sid": "wav2vec2 (SID-superb)",
    "wav2vec2-ls100-sid": "wav2vec2 (SID)",
}

MODELNAME_RENAME_BACKWARD: dict[str, str] = {v: k for k, v in MODELNAME_RENAME.items()}

# Fixed color mapping for model names to ensure consistency across plots
# Uses the display names from MODELNAME_RENAME
# IBM Design Library colorblind-safe palette
# Primary focus models (wav2vec2-sid, wav2vec2-base, wav2vec2-960h, bert-base)
# use the most distinct colors
MODEL_COLOR_MAPPING: dict[str, str] = {
    # Primary focus models - most distinct colors
    "wav2vec2 (base)": "#648FFF",  # Blue
    "wav2vec2 (ASR)": "#DC267F",  # Magenta
    "wav2vec2 (SID)": "#FE6100",  # Orange
    "BERT (base)": "#009E73",  # Teal
    # Secondary models
    "wav2vec2 (large)": "#785EF0",  # Purple
    "HuBERT (base)": "#FFB000",  # Gold
    "HuBERT (large)": "#E69F00",  # Amber
    "WavLM (base)": "#56B4E9",  # Sky blue
    "wav2vec2 (SID-superb)": "#CC79A7",  # Pink
    "RoBERTa (base)": "#D55E00",  # Vermillion
    "ModernBERT (base)": "#0072B2",  # Dark blue
}

SYNTAX_COMPONENT_RENAME: dict[str, str] = {
    "syntax_POS_OH": r"\setminus \mathit{Syntax-POS}",
    "syntax_Dependency_Label_OH": r"\setminus \mathit{Syntax-Dependency}",
    "syntax_Tree_Depth": r"\setminus \mathit{Syntax-Tree-Depth}",
    "syntax_Word_Position": r"\setminus \mathit{Syntax-Word-Position}",
    "syntax_Total_Tree_Depth": r"\setminus \mathit{Syntax-Total-Tree-Depth}",
    "syntax_Total_Word_Count": r"\setminus \mathit{Syntax-Total-Word-Count}",
}


def _rename_syntax_component_config(config_name: str) -> str:
    base_str = r"\mathit{Full} \setminus \mathit{lexicon}"
    for component, label in SYNTAX_COMPONENT_RENAME.items():
        token = f"word_embedding+{component}"
        if config_name == token:
            return "$" + base_str + " " + label + "$"
    return config_name


CONFIG_NAME_ORDER: list = [
    ALLFEATURE_NAME,
    "Acoustics Only",
]
CONFIG_NAME_ORDER += sorted(
    [
        config
        for config in CONFIG_NAME_RENAME.values()
        if config not in CONFIG_NAME_ORDER
    ],
    key=lambda x: (x.count("_"), x),
)
CONFIG_NAME_ORDER += ["Sum of Individual Effects"]

# Remove duplicates while preserving order
CONFIG_NAME_ORDER = list(dict.fromkeys(CONFIG_NAME_ORDER))

PLOT_COLOR_MAPPING: dict = {
    ALLFEATURE_NAME: "#808080",
    "Acoustics Only": "#A9A9A9",
    ACOUSTIC_NAME: "#4E79A7",
    SPEAKERID_NAME: "#E15759",
    LEXICON_NAME: "#59A14F",
    PHONETIC_NAME: "#F28E2B",
    SYNTAX_NAME: "#B07AA1",
    "Joint removal": "#2F2F2F",
    SYNTAX_LEXICON_NAME: "#2F2F2F",
    SPEAKERID_ACOUSTIC_NAME: "#2F2F2F",
    PHONETIC_SPEAKERID_NAME: "#2F2F2F",
    "Sum of Individual Effects": "#76B7B2",
}

Y_COL_NAME_MAPPING = {
    "test_score": r"HRS ($R^2$) Score",
    "unexplained_variance": r"Unexplained Variance (1 - $R^2$)",
    "departure_from_topline": r"Departure from Topline ($R^2$ difference)",
}

PLOTTING_CONFIGS: dict[str, dict] = {
    "syntax_lexical": {
        "target_configs": [
            LEXICON_NAME,
            SYNTAX_NAME,
            SYNTAX_LEXICON_NAME,
        ],
        "target_models": [
            "bert-base-uncased",
            "wav2vec2-base",
        ],
        "y_col": "unexplained_variance",
    },
    "syntax_lexical_2": {
        "target_configs": [
            LEXICON_NAME,
            SYNTAX_NAME,
            SYNTAX_LEXICON_NAME,
        ],
        "target_models": [
            "wav2vec2-base",
            "wav2vec2-base-960h",
        ],
        "y_col": "unexplained_variance",
    },
    "acoustics_speaker_id": {
        "target_configs": [
            ACOUSTIC_NAME,
            SPEAKERID_NAME,
            SPEAKERID_ACOUSTIC_NAME,
        ],
        "target_models": [
            "wav2vec2-base",
            "wav2vec2-ls100-sid",
        ],
        "y_col": "unexplained_variance",
    },
    "phonetic_speaker_id": {
        "target_configs": [
            PHONETIC_NAME,
            SPEAKERID_NAME,
            PHONETIC_SPEAKERID_NAME,
        ],
        "target_models": [
            "wav2vec2-base",
            "wav2vec2-ls100-sid",
        ],
        "y_col": "unexplained_variance",
    },
    "acoustics_speaker_id_2": {
        "target_configs": [
            ACOUSTIC_NAME,
            SPEAKERID_NAME,
            SPEAKERID_ACOUSTIC_NAME,
        ],
        "target_models": [
            "wav2vec2-base",
            "wav2vec2-base-960h",
        ],
        "y_col": "unexplained_variance",
    },
    "phonetic_speaker_id_2": {
        "target_configs": [
            PHONETIC_NAME,
            SPEAKERID_NAME,
            PHONETIC_SPEAKERID_NAME,
        ],
        "target_models": [
            "wav2vec2-base",
            "wav2vec2-base-960h",
        ],
        "y_col": "unexplained_variance",
    },
    "all_models_syntax_lexical": {
        "target_configs": [
            LEXICON_NAME,
            SYNTAX_NAME,
            SYNTAX_LEXICON_NAME,
        ],
        "target_models": None,
        "exclude_models": ["wav2vec2-ls100-sid"],
        "x_col": "normalized_layer",
        "figure_size": (8, 8),
    },
    "all_models_acoustic_speaker": {
        "target_configs": [
            ACOUSTIC_NAME,
            SPEAKERID_NAME,
            SPEAKERID_ACOUSTIC_NAME,
        ],
        "target_models": None,
        "exclude_models": ["wav2vec2-ls100-sid"],
        "x_col": "normalized_layer",
        "figure_size": (8, 8),
    },
    "all_models_phonetic_speaker": {
        "target_configs": [
            PHONETIC_NAME,
            SPEAKERID_NAME,
            PHONETIC_SPEAKERID_NAME,
        ],
        "target_models": None,
        "exclude_models": ["wav2vec2-ls100-sid"],
        "x_col": "normalized_layer",
        "figure_size": (8, 8),
    },
    "syntax_lexical_wav2vec2": {
        "target_configs": [
            LEXICON_NAME,
            SYNTAX_NAME,
            SYNTAX_LEXICON_NAME,
        ],
        "target_models": ["wav2vec2-base"],
        "x_col": "layer",
        "y_col": "test_score",
        "figure_size": (4, 4),
        "legend_n_row": 2,
    },
    "acoustics_speaker_id_wav2vec2": {
        "target_configs": [
            ACOUSTIC_NAME,
            SPEAKERID_NAME,
            SPEAKERID_ACOUSTIC_NAME,
        ],
        "target_models": ["wav2vec2-base"],
        "x_col": "layer",
        "y_col": "test_score",
        "figure_size": (4, 4),
        "legend_n_row": 2,
    },
    "phonetic_speaker_id_wav2vec2": {
        "target_configs": [
            PHONETIC_NAME,
            SPEAKERID_NAME,
            PHONETIC_SPEAKERID_NAME,
        ],
        "target_models": ["wav2vec2-base"],
        "x_col": "layer",
        "y_col": "test_score",
        "figure_size": (4, 4),
        "legend_n_row": 2,
    },
    "syntax_lexicon_decomposition_wav2vec2": {
        "target_configs": [
            LEXICON_NAME,
            # "—Lexicon —Syntax POS",
            # "—Lexicon —Syntax Dependency",
            # "—Lexicon —Syntax Tree Depth",
            # "—Lexicon —Syntax Position",
            # "—Lexicon —Syntax Total Tree Depth",
            # "—Lexicon —Syntax Total Word Count",
            SYNTAX_LEXICON_NAME,
        ]
        + [
            _rename_syntax_component_config("word_embedding+" + x)
            for x in SYNTAX_COMPONENT_RENAME.keys()
        ],
        "target_models": ["wav2vec2-base"],
        "x_col": "layer",
        "y_col": "test_score",
        "figure_size": (8, 8),
        "legend_n_row": 5,
        "facet": "plot_config_name",
        "color_mapping": None,
    },
}


def _get_run_group(experiment: str) -> str:
    return RUN_GROUP_BY_EXPERIMENT.get(experiment, experiment)


def _safe_list(value, default: list[str]) -> list[str]:
    return value if isinstance(value, list) else default


def _safe_str(value, default: str) -> str:
    return value if isinstance(value, str) else default


def _safe_int(value, default: int | None = None) -> int | None:
    if value is None:
        return default
    return value if isinstance(value, int) else default


def _safe_tuple2(value, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple) and len(value) == 2:
        return value
    return default


def _shorten_modelname(modelname: str) -> str:
    return modelname.split("/")[-1]


def _read_results_impl(
    librispeech_split: str,
    modelnames: list[str] | None = None,
    probename: str = "ridge",
    require_random_seed: bool = False,
) -> pd.DataFrame:
    all_results = []
    for results_dir in RESULTS_DIRS:
        if modelnames is None:
            target_models = [
                "facebook/wav2vec2-base",
                "facebook/wav2vec2-base-960h",
                "superb/wav2vec2-base-superb-sid",
                "facebook/wav2vec2-large",
                "facebook/hubert-base-ls960",
                "facebook/hubert-large-ll60k",
                "microsoft/wavlm-base",
                "FacebookAI/roberta-base",
                "google-bert/bert-base-uncased",
                "answerdotai/ModernBERT-base",
                "techsword/wav2vec2-ls100-sid",
            ]
        else:
            target_models = modelnames
        for target_model in target_models:
            savepath = os.path.join(
                RESULTS_ROOT,
                results_dir,
                f"{librispeech_split}_{target_model.replace('/', '-')}_{probename}_results.pkl",
            )
            if require_random_seed:
                savefiles = glob.glob(savepath.replace(".pkl", "-*.pkl"))
                for savefile in savefiles:
                    if os.path.exists(savefile):
                        with open(savefile, "rb") as f:
                            results = pickle.load(f)
                        results_df = pd.DataFrame(results)
                        seed = savefile.split("-")[-1].split(".")[0]
                        results_df["random_seed"] = int(seed.replace("seed", ""))
                        results_df["experiment"] = results_dir
                        results_df["run_group"] = _get_run_group(results_dir)
                        all_results.append(results_df)
            if os.path.isfile(savepath):
                with open(savepath, "rb") as f:
                    results = pickle.load(f)
                results_df = pd.DataFrame(results)
                if require_random_seed:
                    results_df["random_seed"] = 42
                results_df["experiment"] = results_dir
                results_df["run_group"] = _get_run_group(results_dir)
                all_results.append(results_df)
            elif not require_random_seed:
                logger.warning(
                    "Results not found for %s and %s under %s. Skipping.",
                    librispeech_split,
                    target_model,
                    results_dir,
                )
    if not all_results:
        if require_random_seed:
            logger.warning(
                "No random-seed result files found for split '%s'.",
                librispeech_split,
            )
        else:
            logger.warning("No result files found for split '%s'.", librispeech_split)
        return pd.DataFrame()

    all_results_df = pd.concat(all_results, ignore_index=True)

    # Make new column for plotting the amount of variance unexplained by probe reconstruction
    # Since we use R^2 as the metric, we can calculate this as 1 - R^2
    all_results_df["unexplained_variance"] = 1 - all_results_df["test_score"]

    # Make new column for plotting the amount of departure from the topline score
    # We calculate this by taking the difference between the topline score and the current score.
    # Topline results are the ones with config_name AllFeatures, which represents the full feature set without any removals.
    topline_score_per_model_per_experiment = all_results_df[
        all_results_df["config_name"] == ALLFEATURE_NAME
    ]
    # Calculate departure from topline for each model, config_name, layer, and experiment
    all_results_df["departure_from_topline"] = all_results_df.apply(
        lambda row: (
            row["test_score"]
            - topline_score_per_model_per_experiment[
                (
                    topline_score_per_model_per_experiment["modelname"]
                    == row["modelname"]
                )
                & (
                    topline_score_per_model_per_experiment["experiment"]
                    == row["experiment"]
                )
            ]["test_score"].values[0]
            if not topline_score_per_model_per_experiment[
                (
                    topline_score_per_model_per_experiment["modelname"]
                    == row["modelname"]
                )
                & (
                    topline_score_per_model_per_experiment["experiment"]
                    == row["experiment"]
                )
            ].empty
            else np.nan
        ),
        axis=1,
    )

    all_results_df["plot_config_name"] = all_results_df["config_name"].map(
        lambda x: CONFIG_NAME_RENAME.get(x, x)
    )
    all_results_df["plot_config_name"] = all_results_df["plot_config_name"].map(
        _rename_syntax_component_config
    )
    all_results_df["modelname"] = all_results_df["modelname"].map(_shorten_modelname)
    all_results_df["normalized_layer"] = all_results_df.groupby("modelname")[
        "layer"
    ].transform(lambda x: x / x.max())

    order_config_name = all_results_df["config_name"].unique().tolist()
    order_config_name.sort(key=lambda x: (x.count("mathit"), x))
    all_results_df["config_name"] = pd.Categorical(
        all_results_df["config_name"], categories=order_config_name, ordered=True
    )

    return all_results_df


def _build_main_figure_filename(
    *,
    mode: str,
    featname: str,
    librispeech_split: str,
) -> str:
    filename = f"{featname.lower()}_{mode}_results.png"
    if librispeech_split != "train-clean-100":
        filename = f"{featname.lower()}_{mode}_{librispeech_split}_results.png"
    return filename


def _build_random_seed_figure_filename(
    *,
    modelname: str,
    focus: str,
    librispeech_split: str,
) -> str:
    filename = f"{modelname}_random_seed_{focus}_results.png"
    if librispeech_split != "train-clean-100":
        filename = f"{modelname}_random_seed_{focus}_{librispeech_split}_results.png"
    return filename


def _build_decoding_speakerid_figure_filename(
    *, librispeech_split: str, figure_name_suffix: str = "speakerid_decoding_by_layer"
) -> str:
    filename = f"{figure_name_suffix}.png"
    if librispeech_split != "train-clean-100":
        filename = f"{figure_name_suffix}_{librispeech_split}.png"
    return filename


def read_decoding_results(
    librispeech_split: str = "train-clean-100",
    decoding_csv_path: str | None = None,
    config_filter: dict | None = None,
) -> pd.DataFrame:
    target_variable = _safe_str(
        config_filter.get("target_variable", "SpeakerID")
        if config_filter
        else "SpeakerID",
        "SpeakerID",
    )
    x_filter_pattern = _safe_str(
        config_filter.get("x_filter_pattern", "hidden_state_L")
        if config_filter
        else "hidden_state_L",
        "hidden_state_L",
    )
    target_models = config_filter.get("target_models") if config_filter else None

    csv_path = decoding_csv_path or os.path.join(
        RESULTS_ROOT, f"combined_all_decoding_{librispeech_split}.csv"
    )
    if not os.path.exists(csv_path):
        logger.warning(
            "decoding CSV not found at '%s'. Skipping decoding plot.", csv_path
        )
        return pd.DataFrame()

    decoding_df = pd.read_csv(csv_path)
    if decoding_df.empty:
        logger.warning("decoding CSV '%s' is empty. Skipping decoding plot.", csv_path)
        return pd.DataFrame()

    unnamed_cols = [col for col in decoding_df.columns if col.startswith("Unnamed:")]
    if unnamed_cols:
        decoding_df = decoding_df.drop(columns=unnamed_cols)

    if "librispeech_split" in decoding_df.columns:
        decoding_df = decoding_df[
            decoding_df["librispeech_split"] == librispeech_split
        ].copy()

    decoding_df = decoding_df[
        decoding_df["config_name"]
        .astype(str)
        .str.contains(f"->{target_variable}", regex=False)
    ].copy()
    decoding_df = decoding_df[
        decoding_df["config_name"]
        .astype(str)
        .str.contains(x_filter_pattern, regex=False)
    ].copy()

    if decoding_df.empty:
        logger.warning(
            "No rows matching pattern '%s' -> '%s' found in '%s'. Skipping decoding plot.",
            x_filter_pattern,
            target_variable,
            csv_path,
        )
        return pd.DataFrame()

    decoding_df["modelname"] = decoding_df["modelname"].map(lambda x: x.split("/")[-1])
    decoding_df["layer"] = pd.to_numeric(decoding_df["layer"], errors="coerce")
    decoding_df = decoding_df.dropna(subset=["layer", "test_score"]).copy()
    if decoding_df.empty:
        logger.warning("No valid layer/test_score rows found for decoding plotting.")
        return pd.DataFrame()

    if target_models is not None:
        decoding_df = decoding_df[decoding_df["modelname"].isin(target_models)].copy()
        if decoding_df.empty:
            logger.warning(
                "No rows found for target models %s. Skipping decoding plot.",
                target_models,
            )
            return pd.DataFrame()

    decoding_df["normalized_layer"] = decoding_df.groupby("modelname")[
        "layer"
    ].transform(lambda x: x / x.max() if x.max() > 0 else 0)

    decoding_df["modelname"] = pd.Categorical(
        decoding_df["modelname"],
        categories=[
            m for m in MODELNAME_ORDER if m in decoding_df["modelname"].unique()
        ],
        ordered=True,
    )
    # Rename modelname for better display
    decoding_df["modelname"] = decoding_df["modelname"].map(
        lambda x: MODELNAME_RENAME.get(x, x)
    )
    return decoding_df


def read_classification_baselines(
    librispeech_split: str = "train-clean-100",
    baseline_csv_path: str | None = None,
    target_variable: str | None = None,
    exact_match: bool = False,
    modelname: str | None = None,
) -> pd.DataFrame:
    if baseline_csv_path:
        baseline_files = [baseline_csv_path]
    elif modelname:
        baseline_files = glob.glob(
            os.path.join(
                RESULTS_ROOT,
                "decoding_results",
                f"classification_baselines_*{modelname}_{librispeech_split}.csv",
            )
        )
        if not baseline_files:
            logger.info(
                "No baseline file found for modelname '%s', falling back to any available.",
                modelname,
            )
            baseline_files = glob.glob(
                os.path.join(
                    RESULTS_ROOT,
                    "decoding_results",
                    f"classification_baselines_*_{librispeech_split}.csv",
                )
            )
    else:
        baseline_files = glob.glob(
            os.path.join(
                RESULTS_ROOT,
                "decoding_results",
                f"classification_baselines_*_{librispeech_split}.csv",
            )
        )

    if not baseline_files:
        logger.warning(
            "No classification baseline files found for split '%s'.", librispeech_split
        )
        return pd.DataFrame()

    baseline_file = sorted(baseline_files)[0]
    if len(baseline_files) > 1:
        logger.info(
            "Multiple baseline files found, using: %s",
            os.path.basename(baseline_file),
        )

    if not os.path.exists(baseline_file):
        return pd.DataFrame()

    baselines_df = pd.read_csv(baseline_file)

    unnamed_cols = [col for col in baselines_df.columns if col.startswith("Unnamed:")]
    if unnamed_cols:
        baselines_df = baselines_df.drop(columns=unnamed_cols)

    if "librispeech_split" in baselines_df.columns:
        baselines_df = baselines_df[
            baselines_df["librispeech_split"] == librispeech_split
        ].copy()

    baselines_df["target_variable"] = (
        baselines_df["config_name"]
        .astype(str)
        .str.extract(r"MajorityClass->(\w+)", expand=False)
    )

    if target_variable is not None:
        if exact_match:
            baselines_df = baselines_df[
                baselines_df["target_variable"] == target_variable
            ].copy()
        else:
            baselines_df = baselines_df[
                baselines_df["target_variable"]
                .astype(str)
                .str.contains(target_variable, regex=True, na=False)
            ].copy()

    baselines_df = baselines_df.rename(columns={"test_score": "baseline_score"})
    return baselines_df[
        ["target_variable", "baseline_score", "n_samples"]
    ].drop_duplicates()


def plot_decoding_by_layer(
    decoding_df: pd.DataFrame,
    *,
    show_plot: bool = False,
    librispeech_split: str = "train-clean-100",
    plot_config: dict | None = None,
    config_filter: dict | None = None,
    use_facet_grid: bool = False,
    facet_row: str | None = None,
    facet_col: str | None = None,
    baseline_target_variables: dict[str, str] | None = None,
) -> None:
    if decoding_df.empty:
        return
    decoding_df["modelname"] = decoding_df["modelname"].cat.remove_unused_categories()

    if decoding_df.empty:
        return

    y_label = _safe_str(
        plot_config.get("y_label", "Speaker-ID Accuracy")
        if plot_config
        else "Speaker-ID Accuracy",
        "Speaker-ID Accuracy",
    )
    x_label = _safe_str(
        plot_config.get("x_label", "Layer") if plot_config else "Layer",
        "Layer",
    )
    figure_name_suffix = _safe_str(
        plot_config.get("figure_name_suffix", "speakerid_decoding_by_layer")
        if plot_config
        else "speakerid_decoding_by_layer",
        "speakerid_decoding_by_layer",
    )
    figure_size = _safe_tuple2(
        plot_config.get("figure_size", (6, 3)) if plot_config else (6, 3),
        (6, 3),
    )
    # Remove 'hidden_state_L*->' from config_name using regex
    decoding_df["config_name"] = (
        decoding_df["config_name"]
        .astype(str)
        .map(lambda x: re.sub(r"hidden_state_L.*->", "", x))
    )
    # Remove syntax_ from config_name using the helper function
    decoding_df["config_name"] = decoding_df["config_name"].map(
        lambda x: x.replace("syntax_", "")
    )
    decoding_df["config_name"] = decoding_df["config_name"].map(
        lambda x: x.replace("OH", "")
    )
    decoding_df["config_name"] = decoding_df["config_name"].map(
        lambda x: x.replace("_", " ")
    )
    decoding_df["is_baseline"] = False

    include_baseline = (
        plot_config.get("include_baseline", False) if plot_config else False
    )

    # Fix the order of the facets to be the same as the order of the models in MODELNAME_ORDER
    decoding_df["modelname"] = pd.Categorical(
        decoding_df["modelname"],
        categories=[
            MODELNAME_RENAME.get(m, m)
            for m in MODELNAME_ORDER
            if MODELNAME_RENAME.get(m, m) in decoding_df["modelname"].unique()
        ],
        ordered=True,
    )

    # Determine color variable and faceting based on use_facet_grid
    if use_facet_grid:
        # Models as colors, facet_grid layout
        color_var = "modelname"
        shape_var = "modelname"
        # Build facet formula
        row_formula = facet_row if facet_row else "."
        col_formula = facet_col if facet_col else "."
        facet_formula = f"{row_formula} ~ {col_formula}"
        # Use fixed model color mapping for consistency
        color_mapping = MODEL_COLOR_MAPPING
    else:
        # Original behavior: models as facets, config_name as color
        color_var = "config_name"
        shape_var = "config_name"
        # If there are multiple "config_names"
        if len(decoding_df["config_name"].unique()) > 1:
            color_mapping = None
        else:
            # Keep all colors black
            color_mapping = {
                config_name: "#000000"
                for config_name in decoding_df["config_name"].unique()
            }

    figure = (
        p9.ggplot(decoding_df)
        + p9.geom_line(
            p9.aes(x="layer", y="test_score", color=color_var, linetype="is_baseline")
        )
        + p9.geom_point(
            p9.aes(
                x="layer",
                y="test_score",
                color=color_var,
                shape=shape_var,
            ),
            size=0.8,
        )
        + p9.scale_x_continuous(
            breaks=np.arange(
                decoding_df["layer"].min(), decoding_df["layer"].max() + 1, 6
            )
        )
        + p9.theme_minimal()
        + p9.theme(
            figure_size=figure_size,
            dpi=300,
            legend_position="bottom",
            legend_justification="center",
            legend_title=p9.element_blank(),
        )
        + p9.labs(
            x=x_label,
            y=y_label,
        )
    )

    # Add faceting based on mode
    if use_facet_grid:
        figure += p9.facet_grid(facet_formula)
    else:
        figure += p9.facet_wrap("modelname")

    if color_mapping is not None:
        figure += p9.scale_color_manual(values=color_mapping)

    # Hide legend if only one unique value for color variable
    unique_color_values = decoding_df[color_var].nunique()
    if unique_color_values == 1:
        figure += p9.theme(legend_position="none")

    if include_baseline:
        exact_match = (
            plot_config.get("baseline_exact_match", False) if plot_config else False
        )
        plot_models = list(decoding_df["modelname"].unique())
        # Revert plot_models back to their original modelnames for baseline lookup
        original_modelnames = [MODELNAME_RENAME_BACKWARD.get(x, x) for x in plot_models]

        # Determine if we need to handle multiple decoding types
        has_multiple_decoding_types = (
            use_facet_grid
            and "decoding_type" in decoding_df.columns
            and baseline_target_variables is not None
        )

        baselines_per_model = []
        for model in original_modelnames:
            plot_model = MODELNAME_RENAME.get(model, model)
            model_layers = decoding_df[decoding_df["modelname"] == plot_model][
                "layer"
            ].unique()

            if has_multiple_decoding_types:
                # Handle multiple decoding types with different target variables
                assert baseline_target_variables is not None  # Already checked above
                for decoding_type, target_var in baseline_target_variables.items():
                    model_baseline = read_classification_baselines(
                        librispeech_split=librispeech_split,
                        target_variable=target_var,
                        exact_match=exact_match,
                        modelname=model,
                    )
                    if model_baseline.empty:
                        logger.warning(
                            "No baseline found for target variable '%s' and model '%s'.",
                            target_var,
                            model,
                        )
                        continue
                    model_baseline["modelname"] = plot_model
                    model_baseline["config_name"] = model_baseline[
                        "target_variable"
                    ].map(
                        lambda x: (
                            x.replace("syntax_", "").replace("OH", "").replace("_", " ")
                        )
                    )
                    # Duplicate baseline rows for each layer
                    model_baseline = model_baseline.loc[
                        model_baseline.index.repeat(len(model_layers))
                    ].copy()
                    model_baseline["layer"] = np.tile(
                        model_layers, len(model_baseline) // len(model_layers)
                    )
                    model_baseline["is_baseline"] = True
                    model_baseline["decoding_type"] = decoding_type
                    baselines_per_model.append(model_baseline)
            else:
                # Original single target variable handling
                target_variable = _safe_str(
                    config_filter.get("target_variable", "SpeakerID")
                    if config_filter
                    else "SpeakerID",
                    "SpeakerID",
                )
                model_baseline = read_classification_baselines(
                    librispeech_split=librispeech_split,
                    target_variable=target_variable,
                    exact_match=exact_match,
                    modelname=model,
                )
                if model_baseline.empty:
                    logger.warning(
                        "No baseline found for target variable '%s' and model '%s'.",
                        target_variable,
                        model,
                    )
                    continue
                model_baseline["modelname"] = plot_model
                model_baseline["config_name"] = model_baseline["target_variable"].map(
                    lambda x: (
                        x.replace("syntax_", "").replace("OH", "").replace("_", " ")
                    )
                )
                # Duplicate baseline rows for each layer
                model_baseline = model_baseline.loc[
                    model_baseline.index.repeat(len(model_layers))
                ].copy()
                model_baseline["layer"] = np.tile(
                    model_layers, len(model_baseline) // len(model_layers)
                )
                model_baseline["is_baseline"] = True
                # Propagate decoding_type column if present
                if "decoding_type" in decoding_df.columns:
                    decoding_types = decoding_df[
                        decoding_df["modelname"] == plot_model
                    ]["decoding_type"].unique()
                    if len(decoding_types) > 0:
                        model_baseline["decoding_type"] = decoding_types[0]
                baselines_per_model.append(model_baseline)

        if baselines_per_model:
            model_baseline = pd.concat(baselines_per_model, ignore_index=True)
            # Fix the order of the facets to be the same as the order of the models in MODELNAME_ORDER
            model_baseline["modelname"] = pd.Categorical(
                model_baseline["modelname"],
                categories=[
                    MODELNAME_RENAME.get(m, m)
                    for m in MODELNAME_ORDER
                    if MODELNAME_RENAME.get(m, m)
                    in model_baseline["modelname"].unique()
                ],
                ordered=True,
            )
            # Use color_var for baseline as well to match the main plot
            figure += p9.geom_line(
                data=model_baseline,
                mapping=p9.aes(
                    x="layer",
                    y="baseline_score",
                    color=color_var,
                    linetype="is_baseline",
                ),
            )
        else:
            logger.warning("No baselines found for any model. Skipping baseline lines.")
    figure += p9.scale_linetype_manual(values={True: "dashed", False: "solid"})
    # remove linetype from legend
    figure += p9.guides(linetype="none")
    if show_plot:
        figure.show()

    filename = _build_decoding_speakerid_figure_filename(
        librispeech_split=librispeech_split,
        figure_name_suffix=figure_name_suffix,
    )
    figure.save(os.path.join(FIGURES_ROOT, filename))


def plot_helper(
    results_df: pd.DataFrame,
    comparison_results_df: pd.DataFrame,
    facet: str = "modelname",
    color_mapping: dict | None = None,
    x_col: str = "normalized_layer",
    y_col: str = "test_score",
    legend_n_row: int | None = 2,
    include_sum_of_individual: bool = True,
    order_facet_by_topline: bool = False,
    topline_config_name: str = ALLFEATURE_NAME,
) -> p9.ggplot:
    if not include_sum_of_individual:
        col_name = "Sum of Individual Effects"
        results_df = results_df[results_df["config_name"] != col_name].copy()
        comparison_results_df = comparison_results_df[
            comparison_results_df["config_name"] != col_name
        ].copy()

    # Order the modelname by a set list
    results_df["modelname"] = pd.Categorical(
        results_df["modelname"], categories=MODELNAME_ORDER, ordered=True
    )
    results_df["modelname"] = results_df["modelname"].map(
        lambda x: MODELNAME_RENAME.get(x, x)
    )

    comparison_results_df["modelname"] = pd.Categorical(
        comparison_results_df["modelname"], categories=MODELNAME_ORDER, ordered=True
    )
    comparison_results_df["modelname"] = comparison_results_df["modelname"].map(
        lambda x: MODELNAME_RENAME.get(x, x)
    )

    all_config_names = list(results_df["config_name"].unique()) + list(
        comparison_results_df["config_name"].unique()
    )

    config_name_order = [x for x in CONFIG_NAME_ORDER if x in all_config_names]

    if order_facet_by_topline and facet == "config_name":
        layer_key = "normalized_layer"
        if (
            layer_key not in results_df.columns
            or layer_key not in comparison_results_df.columns
        ):
            layer_key = x_col

        key_cols = ["modelname", layer_key]
        if (
            "run_group" in results_df.columns
            and "run_group" in comparison_results_df.columns
        ):
            key_cols.append("run_group")
        elif (
            "experiment" in results_df.columns
            and "experiment" in comparison_results_df.columns
        ):
            key_cols.append("experiment")
        if (
            "random_seed" in results_df.columns
            and "random_seed" in comparison_results_df.columns
        ):
            key_cols.append("random_seed")

        topline_df = comparison_results_df[
            comparison_results_df["config_name"] == topline_config_name
        ]
        if not topline_df.empty:
            topline_lookup = (
                topline_df[key_cols + ["test_score"]]
                .drop_duplicates(subset=key_cols)
                .rename(columns={"test_score": "topline_test_score"})
            )
            facet_rank_df = results_df.merge(topline_lookup, on=key_cols, how="left")
            facet_rank_df = facet_rank_df.dropna(subset=["topline_test_score"]).copy()
            if not facet_rank_df.empty:
                facet_rank_df["abs_gap_to_topline"] = (
                    facet_rank_df["test_score"] - facet_rank_df["topline_test_score"]
                ).abs()
                ranked_configs = (
                    facet_rank_df.groupby("config_name")["abs_gap_to_topline"]
                    .mean()
                    .sort_values(ascending=True)
                    .index.tolist()
                )
                config_name_order = ranked_configs + [
                    config
                    for config in config_name_order
                    if config not in ranked_configs
                ]

    results_df["config_name"] = pd.Categorical(
        results_df["config_name"],
        categories=config_name_order,
        ordered=True,
    )
    comparison_results_df["config_name"] = pd.Categorical(
        comparison_results_df["config_name"],
        categories=config_name_order,
        ordered=True,
    )

    all_plot_config_names = list(results_df["plot_config_name"].unique()) + list(
        comparison_results_df["plot_config_name"].unique()
    )

    def _sort_labels_by_amount_of_mathit(x: list[str]) -> list[str]:
        return sorted(x, key=lambda label: (label.count("mathit"), label))

    plot_config_name_order = _sort_labels_by_amount_of_mathit(
        [x for x in all_plot_config_names]
    )

    results_df["plot_config_name"] = pd.Categorical(
        results_df["plot_config_name"],
        categories=plot_config_name_order,
        ordered=True,
    )
    comparison_results_df["plot_config_name"] = pd.Categorical(
        comparison_results_df["plot_config_name"],
        categories=plot_config_name_order,
        ordered=True,
    )

    linetype_mapping = {
        ALLFEATURE_NAME: "dashed",
        "Acoustics Only": "dashed",
    }
    linetype_mapping = {
        config: linetype_mapping.get(config, "solid")
        for config in all_plot_config_names
    }

    # Drop all nan columns
    results_df = results_df.dropna(axis=1, how="any")
    comparison_results_df = comparison_results_df.dropna(axis=1, how="any")

    y_lim_max = max(results_df[y_col].max(), comparison_results_df[y_col].max()) * 1.05
    y_lim_min = min(results_df[y_col].min(), comparison_results_df[y_col].min()) * 0.95
    combi_df = pd.concat([results_df, comparison_results_df], ignore_index=True)

    p = (
        p9.ggplot()
        + p9.geom_line(
            data=combi_df,
            mapping=p9.aes(
                x=x_col,
                y=y_col,
                color="plot_config_name",
                group="plot_config_name",
                linetype="plot_config_name",
            ),
        )
        + p9.geom_point(
            data=results_df,
            mapping=p9.aes(
                x=x_col,
                y=y_col,
                color="plot_config_name",
                shape="plot_config_name",
            ),
            size=0.7,
        )
        + p9.facet_wrap(facet)
        + p9.theme_minimal()
        + p9.scale_y_continuous(limits=(y_lim_min, y_lim_max))
        + p9.scale_linetype_manual(values=linetype_mapping)
    )
    if legend_n_row is not None:
        p += p9.guides(
            color=p9.guide_legend(nrow=legend_n_row, byrow=True),
            linetype=p9.guide_legend(nrow=legend_n_row, byrow=True),
            shape=p9.guide_legend(nrow=legend_n_row, byrow=True),
        )
    if color_mapping is not None:
        p += p9.scale_color_manual(values=color_mapping)
    if x_col == "normalized_layer":
        # Set the x-axis to be from 0 to 1 with breaks at every 0.25
        p += p9.scale_x_continuous(breaks=(0, 0.25, 0.5, 0.75, 1))
    else:
        p += p9.scale_x_continuous(
            breaks=np.arange(
                results_df["layer"].min(), results_df["layer"].max() + 1, 3
            )
        )
    return p


def read_all_results(librispeech_split: str = "dev-clean") -> pd.DataFrame:
    return _read_results_impl(librispeech_split=librispeech_split)


def read_random_seed_results(
    librispeech_split: str = "dev-clean",
    modelname: str = "facebook/wav2vec2-base",
    probename: str = "ridge",
) -> pd.DataFrame:
    return _read_results_impl(
        librispeech_split=librispeech_split,
        modelnames=[modelname],
        probename=probename,
        require_random_seed=True,
    )


def get_combined_single_results(
    mode_results_df,
    mode_comparison_results_df,
    target_configs: list,
    new_config_name: str | None,
) -> pd.DataFrame:
    if mode_results_df.empty:
        return mode_results_df

    if not target_configs:
        logger.warning("No target configs provided for combined single results.")
        return mode_results_df.copy()

    target_config_set = set(target_configs)

    # Compute departure to topline using baseline rows from the same run/directory.
    key_cols = ["modelname"]
    if (
        "layer" in mode_results_df.columns
        and "layer" in mode_comparison_results_df.columns
    ):
        key_cols.append("layer")
    if (
        "normalized_layer" in mode_results_df.columns
        and "normalized_layer" in mode_comparison_results_df.columns
    ):
        key_cols.append("normalized_layer")
    if (
        "random_seed" in mode_results_df.columns
        and "random_seed" in mode_comparison_results_df.columns
    ):
        key_cols.append("random_seed")
    if (
        "run_group" in mode_results_df.columns
        and "run_group" in mode_comparison_results_df.columns
    ):
        key_cols.append("run_group")
    elif (
        "experiment" in mode_results_df.columns
        and "experiment" in mode_comparison_results_df.columns
    ):
        key_cols.append("experiment")

    comparison_df = mode_comparison_results_df.copy()
    if "config_name" in comparison_df.columns:
        all_features_rows = comparison_df[
            comparison_df["config_name"] == ALLFEATURE_NAME
        ]
        if not all_features_rows.empty:
            comparison_df = all_features_rows

    if comparison_df.empty:
        logger.warning(
            "No baseline rows available for combined single-results calculation."
        )
        return mode_results_df.copy()

    comparison_lookup = (
        comparison_df[key_cols + ["test_score"]]
        .drop_duplicates(subset=key_cols, keep="first")
        .rename(columns={"test_score": "topline_test_score"})
    )

    mode_results_df = mode_results_df.merge(
        comparison_lookup,
        on=key_cols,
        how="left",
    )
    mode_results_df["departure_to_topline"] = (
        mode_results_df["test_score"] - mode_results_df["topline_test_score"]
    )
    before_drop = len(mode_results_df)
    mode_results_df = mode_results_df.dropna(subset=["departure_to_topline"]).copy()
    dropped_rows = before_drop - len(mode_results_df)
    if dropped_rows > 0:
        logger.warning(
            "Dropped %s rows without same-run topline matches.",
            dropped_rows,
        )

    if mode_results_df.empty:
        return mode_results_df

    # Add the departure_to_topline from -Lexical and -Syntactic together and append that to the df under a new config_name
    combined_lex_syntx_departure = mode_results_df[
        mode_results_df["config_name"].isin(target_configs)
    ].copy()

    if combined_lex_syntx_departure.empty:
        mode_results_df = mode_results_df.drop(
            columns=["topline_test_score"], errors="ignore"
        )
        return mode_results_df

    group_cols = key_cols.copy()
    coverage_by_group = (
        combined_lex_syntx_departure.groupby(group_cols)["config_name"]
        .nunique()
        .reset_index(name="n_unique_configs")
    )
    complete_groups = coverage_by_group[
        coverage_by_group["n_unique_configs"] == len(target_config_set)
    ][group_cols]

    if complete_groups.empty:
        logger.warning(
            "No groups contain all target configs (%s). Skipping '%s' line.",
            sorted(target_config_set),
            new_config_name or "Combined singles",
        )
        mode_results_df = mode_results_df.drop(
            columns=["topline_test_score"], errors="ignore"
        )
        return mode_results_df

    n_incomplete_groups = int(coverage_by_group.shape[0] - complete_groups.shape[0])
    if n_incomplete_groups > 0:
        logger.warning(
            "Skipping %s incomplete group(s) missing one or more target configs.",
            n_incomplete_groups,
        )

    combined_lex_syntx_departure = combined_lex_syntx_departure.merge(
        complete_groups,
        on=group_cols,
        how="inner",
    )

    combined_lex_syntx_departure = (
        combined_lex_syntx_departure.groupby(group_cols)
        .agg({"departure_to_topline": "sum"})
        .reset_index()
    )
    combined_lex_syntx_departure["config_name"] = (
        new_config_name
        if new_config_name is not None
        else "Combined " + " + ".join(target_configs)
    )
    # Use the departure to calculate the theoretical test_score from same-run topline.
    combined_lex_syntx_departure = combined_lex_syntx_departure.merge(
        comparison_lookup,
        on=[k for k in key_cols if k in combined_lex_syntx_departure.columns],
        how="left",
    )
    combined_lex_syntx_departure["test_score"] = (
        combined_lex_syntx_departure["departure_to_topline"]
        + combined_lex_syntx_departure["topline_test_score"]
    )
    combined_lex_syntx_departure = combined_lex_syntx_departure.dropna(
        subset=["test_score"]
    )
    combined_lex_syntx_departure = combined_lex_syntx_departure.drop(
        columns=["topline_test_score"],
        errors="ignore",
    )

    mode_results_df = mode_results_df.drop(
        columns=["topline_test_score"], errors="ignore"
    )

    mode_results_df = pd.concat(
        [mode_results_df, combined_lex_syntx_departure], ignore_index=True
    )

    return mode_results_df


def _filter_focus_to_complete_experiments(
    subset_df: pd.DataFrame,
    subset_comparison_df: pd.DataFrame,
    *,
    single_configs: list[str],
    joint_config: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if subset_df.empty:
        return subset_df, subset_comparison_df

    group_col = (
        "run_group"
        if "run_group" in subset_df.columns
        else "experiment"
        if "experiment" in subset_df.columns
        else None
    )
    if group_col is None:
        return subset_df.copy(), subset_comparison_df.copy()

    required_configs = set(single_configs)
    if joint_config is not None:
        required_configs.add(joint_config)

    if not required_configs:
        return subset_df.copy(), subset_comparison_df.copy()

    group_coverage = (
        subset_df.groupby(group_col)["plot_config_name"].apply(set).to_dict()
    )

    valid_groups = [
        group_name
        for group_name, configs in group_coverage.items()
        if required_configs.issubset(configs)
    ]

    if subset_comparison_df.empty:
        logger.warning("No baseline rows available for focus filtering.")
        return subset_df.iloc[0:0].copy(), subset_comparison_df.iloc[0:0].copy()

    comparison_group_coverage = (
        subset_comparison_df.groupby(group_col)["plot_config_name"].apply(set).to_dict()
    )
    valid_groups = [
        group_name
        for group_name in valid_groups
        if ALLFEATURE_NAME in comparison_group_coverage.get(group_name, set())
    ]

    if not valid_groups:
        logger.warning(
            "No run group contains full focus coverage with topline (%s).",
            sorted(required_configs),
        )
        return subset_df.iloc[0:0].copy(), subset_comparison_df.iloc[0:0].copy()

    filtered_subset = subset_df[subset_df[group_col].isin(valid_groups)].copy()
    filtered_comparison = subset_comparison_df[
        subset_comparison_df[group_col].isin(valid_groups)
    ].copy()

    return filtered_subset, filtered_comparison


def plot_main_figures(
    all_results_df: pd.DataFrame,
    show_plots: bool = False,
    librispeech_split: str = "train-clean-100",
):
    mode = "top-down"
    comparison_configs = [ALLFEATURE_NAME, "Acoustics Only"]
    mode_results_df = all_results_df[all_results_df["mode"] == mode]
    mode_comparison_results_df = mode_results_df[
        mode_results_df["plot_config_name"].isin(comparison_configs)
    ].copy()
    mode_results_df = mode_results_df[
        ~mode_results_df["plot_config_name"].isin(comparison_configs)
    ].copy()

    for featname, plotting_config in PLOTTING_CONFIGS.items():
        target_configs = _safe_list(plotting_config["target_configs"], [])
        target_models_raw = plotting_config.get("target_models")
        if target_models_raw is None:
            target_models = list(mode_results_df["modelname"].unique())
        else:
            target_models = _safe_list(target_models_raw, [])
        exclude_models_raw = plotting_config.get("exclude_models", None)
        if exclude_models_raw is not None:
            exclude_models = _safe_list(exclude_models_raw, [])
            target_models = [m for m in target_models if m not in exclude_models]
        x_col = _safe_str(plotting_config.get("x_col", "layer"), "layer")
        y_col = _safe_str(plotting_config.get("y_col", "test_score"), "test_score")
        legend_n_row = _safe_int(plotting_config.get("legend_n_row", None), None)
        figure_size = _safe_tuple2(plotting_config.get("figure_size", (6, 3)), (6, 3))
        facet = _safe_str(plotting_config.get("facet", "modelname"), "modelname")
        color_mapping_value = plotting_config.get("color_mapping", PLOT_COLOR_MAPPING)
        if color_mapping_value is not None and not isinstance(
            color_mapping_value, dict
        ):
            color_mapping = PLOT_COLOR_MAPPING
        else:
            color_mapping = color_mapping_value

        plot_df = mode_results_df.copy()
        plot_df = plot_df[
            (plot_df["plot_config_name"].isin(target_configs))
            & (plot_df["modelname"].isin(target_models))
        ]
        if plot_df.empty:
            logger.warning("No rows for plotting config '%s'. Skipping.", featname)
            continue

        if (
            "run_group" in plot_df.columns
            and "run_group" in mode_comparison_results_df.columns
        ):
            target_run_groups = plot_df["run_group"].unique().tolist()
            plot_compare_df = mode_comparison_results_df[
                (mode_comparison_results_df["modelname"].isin(target_models))
                & (mode_comparison_results_df["run_group"].isin(target_run_groups))
            ]
        else:
            target_experiments = plot_df["experiment"].unique().tolist()
            plot_compare_df = mode_comparison_results_df[
                (mode_comparison_results_df["modelname"].isin(target_models))
                & (mode_comparison_results_df["experiment"].isin(target_experiments))
            ]
        if plot_compare_df.empty:
            logger.warning(
                "No same-run baseline rows for plotting config '%s'. Skipping.",
                featname,
            )
            continue

        plot_df = get_combined_single_results(
            plot_df,
            plot_compare_df,
            target_configs=[
                x for x in target_configs if x.count("-") == 1
            ],  # Only include the configs with single feature removal for calculating the sum of individual effects
            new_config_name="Sum of Individual Effects",
        )
        if plot_df.empty:
            logger.warning(
                "No rows remain after baseline matching for '%s'. Skipping.",
                featname,
            )
            continue

        # Plot the results in mode_results_df and use mode_comparison_results_df as the baseline with dashed gray line
        p = plot_helper(
            plot_df,
            plot_compare_df,
            facet=facet,
            color_mapping=color_mapping,
            x_col=x_col,
            y_col=y_col,
            legend_n_row=legend_n_row,
            include_sum_of_individual=False,
            order_facet_by_topline=(facet == "plot_config_name"),
        )

        p += p9.labs(
            x="Layer (From bottom to top)",
            y=Y_COL_NAME_MAPPING.get(y_col, y_col),
            color="Feature Group",
            shape="Feature Group",
            linetype="Feature Group",
        )
        p += p9.theme(
            legend_position="bottom",
            legend_justification="center",
            figure_size=figure_size,
            dpi=300,
            legend_title=p9.element_blank(),
        )
        if show_plots:
            p.show()

        filename = _build_main_figure_filename(
            mode=mode,
            featname=featname,
            librispeech_split=librispeech_split,
        )

        p.save(os.path.join(FIGURES_ROOT, filename))


def plot_focus_random_seed(
    random_seed_results_df: pd.DataFrame,
    focus: str = "syntax_lexical",
    y_col: str = "test_score",
    show_plot: bool = False,
    print_ttest: bool = False,
    librispeech_split: str = "train-clean-100",
):
    focus_spec = FOCUS_CONFIGS[focus]
    single_configs = _safe_list(focus_spec["single_configs"], [])
    joint_config = _safe_str(focus_spec["joint_config"], "")
    focus_configs = [*single_configs, joint_config]

    subset_results_df = random_seed_results_df[
        random_seed_results_df["plot_config_name"].isin(focus_configs)
    ].copy()

    subset_results_comparison_df = random_seed_results_df[
        random_seed_results_df["plot_config_name"].isin(
            [ALLFEATURE_NAME, "Acoustics Only"]
        )
    ].copy()
    if subset_results_comparison_df.empty:
        logger.warning(
            "No baseline rows for focus '%s'; skipping random-seed plot.",
            focus,
        )
        return

    subset_results_df, subset_results_comparison_df = (
        _filter_focus_to_complete_experiments(
            subset_results_df,
            subset_results_comparison_df,
            single_configs=single_configs,
            joint_config=joint_config,
        )
    )
    if subset_results_df.empty:
        logger.warning(
            "No complete experiments for focus '%s'; skipping random-seed plot.",
            focus,
        )
        return

    # subset_results_df = get_combined_single_results(
    #     subset_results_df,
    #     subset_results_comparison_df,
    #     target_configs=single_configs,  # Exclude the combined config
    #     new_config_name="Sum of Individual Effects",
    # )

    if subset_results_df.empty:
        logger.warning(
            "No rows remain after sum-of-individual computation for focus '%s'.",
            focus,
        )
        return

    subset_results_df["layer"] = subset_results_df["normalized_layer"] * 12
    subset_results_comparison_df["layer"] = (
        subset_results_comparison_df["normalized_layer"] * 12
    )  # Assuming wav2vec2-base has 12 layers

    group_col = (
        "run_group"
        if "run_group" in subset_results_df.columns
        else "experiment"
        if "experiment" in subset_results_df.columns
        else None
    )
    mean_group_cols = ["plot_config_name", "modelname", "config_name"]
    if group_col is not None:
        mean_group_cols.append(group_col)
    mean_group_cols.append("layer")

    # Compute the mean and std of the test_score for each config_name and layer across different random seeds
    subset_results_df_mean = (
        subset_results_df.groupby(mean_group_cols, observed=True)
        .agg(test_score_mean=(y_col, "mean"), test_score_std=(y_col, "std"))
        .reset_index()
    )
    subset_results_comparison_df_mean = (
        subset_results_comparison_df.groupby(mean_group_cols, observed=True)
        .agg(test_score_mean=(y_col, "mean"), test_score_std=(y_col, "std"))
        .reset_index()
    )
    line_group_col = (
        "run_group"
        if "run_group" in subset_results_df_mean.columns
        else "experiment"
        if "experiment" in subset_results_df_mean.columns
        else None
    )
    if line_group_col is None:
        subset_results_df_mean["line_group"] = subset_results_df_mean[
            "plot_config_name"
        ].astype(str)
        subset_results_comparison_df_mean["line_group"] = (
            subset_results_comparison_df_mean["plot_config_name"].astype(str)
        )
    else:
        subset_results_df_mean["line_group"] = (
            subset_results_df_mean["plot_config_name"].astype(str)
            + "__"
            + subset_results_df_mean[line_group_col].astype(str)
        )
        subset_results_comparison_df_mean["line_group"] = (
            subset_results_comparison_df_mean["plot_config_name"].astype(str)
            + "__"
            + subset_results_comparison_df_mean[line_group_col].astype(str)
        )

    subset_results_comparison_df_mean = subset_results_comparison_df_mean.dropna(
        subset=[f"{y_col}_mean"]
    )

    subset_results_comparison_df_mean["plot_config_name"] = pd.Categorical(
        subset_results_comparison_df_mean["plot_config_name"],
        categories=CONFIG_NAME_ORDER,
        ordered=True,
    )

    # Print the mean std of the test_score for each config_name and layer across different random seeds
    logger.info("Subset Results with Random Seeds for focus '%s':", focus)
    logger.info(
        "\n%s",
        subset_results_df_mean.groupby(["config_name"])[
            [f"{y_col}_mean", f"{y_col}_std"]
        ]
        .mean()
        .dropna(),
    )
    logger.info(
        "\n%s",
        subset_results_comparison_df_mean.groupby(["config_name"])[
            [f"{y_col}_mean", f"{y_col}_std"]
        ]
        .mean()
        .dropna(),
    )

    # Calculate the confidence interval for the test_score_mean using the test_score_std and the number of random seeds
    num_random_seeds = random_seed_results_df["random_seed"].nunique()
    logger.info("Number of random seeds: %s", num_random_seeds)
    subset_results_df_mean["ci_lower"] = subset_results_df_mean[
        f"{y_col}_mean"
    ] - 1.96 * subset_results_df_mean[f"{y_col}_std"] / np.sqrt(num_random_seeds)
    subset_results_df_mean["ci_upper"] = subset_results_df_mean[
        f"{y_col}_mean"
    ] + 1.96 * subset_results_df_mean[f"{y_col}_std"] / np.sqrt(num_random_seeds)

    subset_results_comparison_df_mean["ci_lower"] = subset_results_comparison_df_mean[
        f"{y_col}_mean"
    ] - 1.96 * subset_results_comparison_df_mean[f"{y_col}_std"] / np.sqrt(
        num_random_seeds
    )
    subset_results_comparison_df_mean["ci_upper"] = subset_results_comparison_df_mean[
        f"{y_col}_mean"
    ] + 1.96 * subset_results_comparison_df_mean[f"{y_col}_std"] / np.sqrt(
        num_random_seeds
    )

    mean_ci_per_config = subset_results_df_mean.groupby("config_name").apply(
        lambda x: np.mean(x["ci_upper"] - x["ci_lower"])
    )
    mean_ci_width = np.mean(mean_ci_per_config)
    logger.info("Mean confidence interval width: %.4f", mean_ci_width)

    if print_ttest:
        # Calculate the pairwise t-test between the different config_names for each layer to see if the difference is statistically significant
        from scipy.stats import ttest_ind

        config_names = subset_results_df["config_name"].unique()
        for i in range(len(config_names)):
            for j in range(i + 1, len(config_names)):
                config_name_i = config_names[i]
                config_name_j = config_names[j]
                for layer in subset_results_df["layer"].unique():
                    scores_i = subset_results_df[
                        (subset_results_df["config_name"] == config_name_i)
                        & (subset_results_df["layer"] == layer)
                    ]["test_score"]
                    scores_j = subset_results_df[
                        (subset_results_df["config_name"] == config_name_j)
                        & (subset_results_df["layer"] == layer)
                    ]["test_score"]
                    t_stat, p_value = ttest_ind(scores_i, scores_j)
                    logger.info(
                        "T-test between %s and %s at layer %s: t-statistic=%.4f, p-value=%.4f",
                        config_name_i,
                        config_name_j,
                        layer,
                        t_stat,
                        p_value,
                    )
    linetype_mapping = {
        ALLFEATURE_NAME: "dashed",
        "Acoustics Only": "dashed",
    }
    all_config_names = list(subset_results_df_mean["plot_config_name"].unique()) + list(
        subset_results_comparison_df_mean["plot_config_name"].unique()
    )
    linetype_mapping = {
        config: linetype_mapping.get(config, "solid") for config in all_config_names
    }
    plot_df = pd.concat(
        [subset_results_df_mean, subset_results_comparison_df_mean], ignore_index=True
    )
    # Plot the random seed results with error bars using plotnine
    p = (
        p9.ggplot()
        + p9.geom_line(
            data=plot_df,
            mapping=p9.aes(
                x="layer",
                y=f"{y_col}_mean",
                color="plot_config_name",
                group="line_group",
                linetype="plot_config_name",
            ),
        )
        + p9.geom_point(
            data=plot_df,
            mapping=p9.aes(
                x="layer",
                y=f"{y_col}_mean",
                color="plot_config_name",
                shape="plot_config_name",
            ),
            size=0.7,
        )
        + p9.geom_errorbar(
            data=plot_df,
            mapping=p9.aes(
                x="layer",
                ymin="ci_lower",
                ymax="ci_upper",
                color="plot_config_name",
            ),
            width=0.02,
        )
        + p9.theme_minimal()
        + p9.theme(
            figure_size=(5, 4),
            dpi=300,
            legend_position="bottom",
            legend_justification="center",
            legend_title=p9.element_blank(),
        )
        + p9.scale_x_continuous(
            breaks=np.arange(
                subset_results_df["layer"].min(),
                subset_results_df["layer"].max() + 1,
                3,
            )
        )
        + p9.labs(
            x="Layer (From bottom to top)",
            y=Y_COL_NAME_MAPPING.get(y_col, y_col),
        )
        + p9.scale_color_manual(values=PLOT_COLOR_MAPPING)
        + p9.scale_linetype_manual(values=linetype_mapping)
        + p9.guides(
            color=p9.guide_legend(nrow=2, byrow=True),
            shape=p9.guide_legend(nrow=2, byrow=True),
            linetype=p9.guide_legend(nrow=2, byrow=True),
        )
    )

    if show_plot:
        p.show()
    modelname = random_seed_results_df["modelname"].iloc[0]
    modelname = modelname.split(": ")[-1]

    filename = _build_random_seed_figure_filename(
        modelname=modelname,
        focus=focus,
        librispeech_split=librispeech_split,
    )
    p.save(
        os.path.join(
            FIGURES_ROOT,
            filename,
        )
    )


def summarize_random_seed_line_differences(
    random_seed_results_df: pd.DataFrame,
    focus_lookup: dict,
    output_dir: str = FIGURES_ROOT,
    y_col: str = "test_score",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize pairwise line differences across random seeds and layers.

    Args:
        random_seed_results_df: DataFrame containing random-seed runs.
        focus_lookup: Mapping of focus names to either:
            - ordered config list (single removals + joint removal), or
            - dict with keys `single_configs`, `joint_config`, and optional
              `comparison_configs` / `line_configs`.
        output_dir: Directory to save CSV summaries.

    Returns:
        Tuple of (summary_df, layerwise_df).
    """

    os.makedirs(output_dir, exist_ok=True)
    summary_rows: list[dict] = []
    layerwise_rows: list[dict] = []

    for focus, focus_spec in focus_lookup.items():

        def _norm_key(value: float) -> float:
            return round(float(value), 8)

        if isinstance(focus_spec, dict):
            single_configs = _safe_list(focus_spec.get("single_configs", []), [])
            joint_config = focus_spec.get("joint_config")
            if isinstance(joint_config, str):
                line_configs = _safe_list(
                    focus_spec.get("line_configs", [*single_configs, joint_config]),
                    [*single_configs, joint_config],
                )
            else:
                line_configs = _safe_list(
                    focus_spec.get("line_configs", single_configs), single_configs
                )
            comparison_configs = _safe_list(
                focus_spec.get(
                    "comparison_configs", [ALLFEATURE_NAME, "Acoustics Only"]
                ),
                [ALLFEATURE_NAME, "Acoustics Only"],
            )
        else:
            line_configs = list(focus_spec)
            single_configs = (
                line_configs[:-1] if len(line_configs) > 1 else line_configs
            )
            joint_config = line_configs[-1] if line_configs else None
            comparison_configs = [ALLFEATURE_NAME, "Acoustics Only"]

        subset = random_seed_results_df[
            random_seed_results_df["plot_config_name"].isin(line_configs)
        ].copy()
        if subset.empty:
            logger.warning("No rows found for focus '%s'; skipping summary.", focus)
            continue

        subset_comparison = random_seed_results_df[
            random_seed_results_df["plot_config_name"].isin(comparison_configs)
        ].copy()

        subset, subset_comparison = _filter_focus_to_complete_experiments(
            subset,
            subset_comparison,
            single_configs=single_configs,
            joint_config=joint_config if isinstance(joint_config, str) else None,
        )
        if subset.empty:
            logger.warning(
                "No complete experiment coverage for focus '%s'; skipping summary.",
                focus,
            )
            continue

        subset["normalized_layer"] = pd.to_numeric(
            subset["normalized_layer"], errors="coerce"
        )
        subset = subset.dropna(subset=["normalized_layer"]).copy()
        if subset.empty:
            logger.warning(
                "No valid normalized_layer values for focus '%s'; skipping summary.",
                focus,
            )
            continue

        group_col = (
            "run_group"
            if "run_group" in subset.columns
            else "experiment"
            if "experiment" in subset.columns
            else None
        )

        layer_lookup: dict[tuple[str, float], int] = {}
        if "layer" in subset.columns and group_col is not None:
            layer_rows = subset[[group_col, "normalized_layer", "layer"]].copy()
            layer_rows["normalized_layer"] = pd.to_numeric(
                layer_rows["normalized_layer"], errors="coerce"
            )
            layer_rows["layer"] = pd.to_numeric(layer_rows["layer"], errors="coerce")
            layer_rows = layer_rows.dropna(
                subset=[group_col, "normalized_layer", "layer"]
            )
            if not layer_rows.empty:
                grouped_layer_rows = (
                    layer_rows.groupby([group_col, "normalized_layer"], as_index=False)[
                        "layer"
                    ]
                    .median()
                    .reset_index(drop=True)
                )
                layer_lookup = {
                    (
                        str(row[group_col]),
                        _norm_key(row["normalized_layer"]),
                    ): int(round(float(row["layer"])))
                    for _, row in grouped_layer_rows.iterrows()
                }

        if (
            len(single_configs) >= 2
            and isinstance(joint_config, str)
            and joint_config in subset["plot_config_name"].unique()
            and not subset_comparison.empty
        ):
            subset = get_combined_single_results(
                subset,
                subset_comparison,
                target_configs=single_configs,
                new_config_name="Sum of Individual Effects",
            )

        pivot_index_cols = ["random_seed", "normalized_layer"]
        if group_col is not None:
            pivot_index_cols = [group_col] + pivot_index_cols

        pivot_df = subset.pivot_table(
            index=pivot_index_cols,
            columns="plot_config_name",
            values=y_col,
            aggfunc="mean",
        )

        compare_order = [*line_configs, "Sum of Individual Effects"]
        present_configs = [cfg for cfg in compare_order if cfg in pivot_df.columns]
        if len(present_configs) < 2:
            logger.warning(
                "Focus '%s' has fewer than 2 present config lines (%s); skipping.",
                focus,
                present_configs,
            )
            continue

        for i in range(len(present_configs)):
            for j in range(i + 1, len(present_configs)):
                cfg_a = present_configs[i]
                cfg_b = present_configs[j]
                pair_name = f"{cfg_a} - {cfg_b}"

                delta_series = (pivot_df[cfg_a] - pivot_df[cfg_b]).dropna()
                if delta_series.empty:
                    continue

                n_points = int(delta_series.shape[0])
                mean_delta = float(delta_series.mean())
                std_delta = float(delta_series.std(ddof=1)) if n_points > 1 else 0.0
                sem_delta = std_delta / np.sqrt(n_points) if n_points > 1 else 0.0
                ci_low = mean_delta - 1.96 * sem_delta
                ci_high = mean_delta + 1.96 * sem_delta
                mean_abs_delta = float(np.abs(delta_series).mean())
                pct_positive = float((delta_series > 0).mean())

                # Seed-level aggregation to avoid overweighting layers
                seed_group_cols = ["random_seed"]
                if group_col is not None:
                    seed_group_cols = [group_col] + seed_group_cols
                seed_level_delta = (
                    delta_series.reset_index()
                    .groupby(seed_group_cols)[0]
                    .mean()
                    .dropna()
                )
                n_seed = int(seed_level_delta.shape[0])
                seed_mean_delta = float(seed_level_delta.mean())
                seed_std_delta = (
                    float(seed_level_delta.std(ddof=1)) if n_seed > 1 else 0.0
                )
                seed_sem_delta = seed_std_delta / np.sqrt(n_seed) if n_seed > 1 else 0.0
                seed_ci_low = seed_mean_delta - 1.96 * seed_sem_delta
                seed_ci_high = seed_mean_delta + 1.96 * seed_sem_delta

                from scipy.stats import ttest_1samp

                t_stat, p_value = (
                    ttest_1samp(seed_level_delta, popmean=0.0)
                    if n_seed > 1
                    else (np.nan, np.nan)
                )

                summary_rows.append(
                    {
                        "focus": focus,
                        "pair": pair_name,
                        "y_col": y_col,
                        "config_a": cfg_a,
                        "config_b": cfg_b,
                        "joint_config": joint_config,
                        "single_configs": "+".join(single_configs),
                        "is_joint_vs_sum": int(
                            {
                                cfg_a,
                                cfg_b,
                            }
                            == {joint_config, "Sum of Individual Effects"}
                        )
                        if isinstance(joint_config, str)
                        else 0,
                        "n_points": n_points,
                        "mean_delta": mean_delta,
                        "std_delta": std_delta,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "mean_abs_delta": mean_abs_delta,
                        "pct_positive": pct_positive,
                        "n_seed": n_seed,
                        "seed_mean_delta": seed_mean_delta,
                        "seed_std_delta": seed_std_delta,
                        "seed_ci_low": seed_ci_low,
                        "seed_ci_high": seed_ci_high,
                        "t_stat_seed_mean_delta": float(t_stat)
                        if not np.isnan(t_stat)
                        else np.nan,
                        "p_value_seed_mean_delta": float(p_value)
                        if not np.isnan(p_value)
                        else np.nan,
                    }
                )

                for idx, delta_val in delta_series.items():
                    if group_col is not None:
                        group_name, seed, normalized_layer = idx
                        group_name = str(group_name)
                    else:
                        seed, normalized_layer = idx
                        group_name = ""
                    norm_key = _norm_key(normalized_layer)
                    layerwise_rows.append(
                        {
                            "focus": focus,
                            "pair": pair_name,
                            "y_col": y_col,
                            "config_a": cfg_a,
                            "config_b": cfg_b,
                            "joint_config": joint_config,
                            "single_configs": "+".join(single_configs),
                            "is_joint_vs_sum": int(
                                {
                                    cfg_a,
                                    cfg_b,
                                }
                                == {joint_config, "Sum of Individual Effects"}
                            )
                            if isinstance(joint_config, str)
                            else 0,
                            group_col if group_col is not None else "group": group_name,
                            "random_seed": int(seed),
                            "normalized_layer": float(normalized_layer),
                            "layer": layer_lookup.get((group_name, norm_key), np.nan),
                            "delta": float(delta_val),
                        }
                    )

    summary_df = pd.DataFrame(summary_rows)
    layerwise_df = pd.DataFrame(layerwise_rows)

    if summary_df.empty:
        logger.warning("No random-seed difference summaries were generated.")
        return summary_df, layerwise_df

    modelname = random_seed_results_df["modelname"].iloc[0].split(": ")[-1]
    split_name = random_seed_results_df["librispeech_split"].iloc[0]
    summary_path = os.path.join(
        output_dir,
        f"{modelname}_{split_name}_random_seed_line_diff_summary.csv",
    )
    layerwise_path = os.path.join(
        output_dir,
        f"{modelname}_{split_name}_random_seed_line_diff_layerwise.csv",
    )
    summary_df.to_csv(summary_path, index=False)
    layerwise_df.to_csv(layerwise_path, index=False)

    logger.info("Saved random-seed difference summary to %s", summary_path)
    logger.info("Saved random-seed layerwise deltas to %s", layerwise_path)

    for focus in focus_lookup:
        focus_rows = summary_df[summary_df["focus"] == focus]
        if focus_rows.empty:
            continue
        interaction_rows = focus_rows[focus_rows["is_joint_vs_sum"] == 1]
        if not interaction_rows.empty:
            logger.info(
                "Joint vs Sum-of-individuals for focus '%s':\n%s",
                focus,
                interaction_rows[
                    [
                        "pair",
                        "y_col",
                        "mean_delta",
                        "ci_low",
                        "ci_high",
                        "seed_mean_delta",
                        "seed_ci_low",
                        "seed_ci_high",
                        "p_value_seed_mean_delta",
                    ]
                ].to_string(index=False),
            )
        logger.info(
            "Top pairwise differences for focus '%s':\n%s",
            focus,
            focus_rows.sort_values("mean_abs_delta", ascending=False)
            .head(3)[
                [
                    "pair",
                    "y_col",
                    "mean_delta",
                    "ci_low",
                    "ci_high",
                    "mean_abs_delta",
                    "pct_positive",
                    "p_value_seed_mean_delta",
                ]
            ]
            .to_string(index=False),
        )

    return summary_df, layerwise_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Display generated plots interactively. If unset, plots are only saved.",
    )
    parser.add_argument(
        "--print_ttest",
        action="store_true",
        help="Print per-layer pairwise t-test results to the console.",
    )
    parser.add_argument(
        "--librispeech_split",
        type=str,
        default="train-clean-100",
        help="Specify the LibriSpeech split to analyze (e.g., 'dev-clean', 'train-clean-100').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    librispeech_split = "train-clean-100"

    librispeech_split = args.librispeech_split
    all_results_df = read_all_results(librispeech_split)
    plot_main_figures(
        all_results_df, show_plots=args.show_plots, librispeech_split=librispeech_split
    )
    decoding_plot_configs: dict[str, dict] = {
        "speakerid_hidden": {
            "config_filter": {
                "target_variable": "SpeakerID",
                "x_filter_pattern": "hidden_state_L",
                "target_models": [
                    "wav2vec2-base",
                    "wav2vec2-base-960h",
                    "wav2vec2-ls100-sid",
                ],
            },
            "plot_config": {
                "y_label": "Speaker Label: Accuracy",
                "x_label": "Layer",
                "figure_name_suffix": "speakerid_decoding_by_layer",
                "figure_size": (6, 3),
                "include_baseline": True,
            },
        },
        "phoneid_hidden": {
            "config_filter": {
                "target_variable": "PhoneID",
                "x_filter_pattern": "hidden_state_L",
                "target_models": [
                    # "bert-base-uncased",
                    "wav2vec2-base",
                    "wav2vec2-base-960h",
                    "wav2vec2-ls100-sid",
                ],
            },
            "plot_config": {
                "y_label": "Phone Identity: Accuracy",
                "x_label": "Layer",
                "figure_name_suffix": "phoneid_decoding_by_layer",
                "figure_size": (6, 3),
                "include_baseline": True,
            },
        },
        "syntax_decomp_hidden": {
            "config_filter": {
                "target_variable": "syntax_",
                "x_filter_pattern": "hidden_state_L",
                "target_models": [
                    "bert-base-uncased",
                    "wav2vec2-base",
                    "wav2vec2-base-960h",
                ],
            },
            "plot_config": {
                "y_label": "Decomposed Syntax Decoding Metrics",
                "x_label": "Layer",
                "figure_name_suffix": "syntax_decomposition_decoding_by_layer",
                "figure_size": (6, 3),
                "include_baseline": True,
            },
        },
        "syntax_full_hidden": {
            "config_filter": {
                "target_variable": "syntax_feature",
                "x_filter_pattern": "hidden_state_L",
                "target_models": [
                    "bert-base-uncased",
                    "wav2vec2-base",
                    "wav2vec2-base-960h",
                ],
            },
            "plot_config": {
                "y_label": r"Syntax Decoding $R^2$ Score",
                "x_label": "Layer",
                "figure_name_suffix": "syntax_full_decoding_by_layer",
                "figure_size": (6, 3),
            },
        },
        "lexicon_hidden": {
            "config_filter": {
                "target_variable": "word_embedding",
                "x_filter_pattern": "hidden_state_L",
                "target_models": [
                    "bert-base-uncased",
                    "wav2vec2-base",
                    "wav2vec2-base-960h",
                ],
            },
            "plot_config": {
                "y_label": r"Lexicon Decoding $R^2$ Score",
                "x_label": "Layer",
                "figure_name_suffix": "lexicon_decoding_by_layer",
                "figure_size": (6, 3),
            },
        },
    }

    for plot_name, decoding_config in decoding_plot_configs.items():
        config_filter = decoding_config.get("config_filter")
        plot_config = decoding_config.get("plot_config")
        decoding_df = read_decoding_results(
            librispeech_split,
            config_filter=config_filter,
        )
        plot_decoding_by_layer(
            decoding_df,
            show_plot=args.show_plots,
            librispeech_split=librispeech_split,
            plot_config=plot_config,
            config_filter=config_filter,
        )

    # Combined Syntax + Lexicon decoding plot with facet_grid
    # Read syntax decoding results
    syntax_config_filter = {
        "target_variable": "syntax_feature",
        "x_filter_pattern": "hidden_state_L",
        "target_models": [
            "bert-base-uncased",
            "wav2vec2-base",
            "wav2vec2-base-960h",
        ],
    }
    syntax_df = read_decoding_results(
        librispeech_split,
        config_filter=syntax_config_filter,
    )
    if not syntax_df.empty:
        syntax_df["decoding_type"] = "Syntax Decoding Probe"

    # Read lexicon decoding results
    lexicon_config_filter = {
        "target_variable": "word_embedding",
        "x_filter_pattern": "hidden_state_L",
        "target_models": [
            "bert-base-uncased",
            "wav2vec2-base",
            "wav2vec2-base-960h",
        ],
    }
    lexicon_df = read_decoding_results(
        librispeech_split,
        config_filter=lexicon_config_filter,
    )
    if not lexicon_df.empty:
        lexicon_df["decoding_type"] = "Lexicon Decoding Probe"

    # Combine and plot if both have data
    if not syntax_df.empty and not lexicon_df.empty:
        combined_df = pd.concat([syntax_df, lexicon_df], ignore_index=True)
        combined_plot_config = {
            "y_label": r"$R^2$ Score",
            "x_label": "Layer",
            "figure_name_suffix": "syntax_lexicon_combined_decoding_by_layer",
            "figure_size": (6, 3),
        }
        plot_decoding_by_layer(
            combined_df,
            show_plot=args.show_plots,
            librispeech_split=librispeech_split,
            plot_config=combined_plot_config,
            config_filter=None,
            use_facet_grid=True,
            # facet_row="config_name",
            facet_col="decoding_type",
        )
    elif syntax_df.empty:
        logger.warning("No syntax decoding results found for combined plot.")
    elif lexicon_df.empty:
        logger.warning("No lexicon decoding results found for combined plot.")

    # Combined Speaker + Phonetics decoding plot with facet_grid
    # Read speaker decoding results
    speaker_config_filter = {
        "target_variable": "SpeakerID",
        "x_filter_pattern": "hidden_state_L",
        "target_models": [
            "wav2vec2-base",
            "wav2vec2-base-960h",
            "wav2vec2-ls100-sid",
        ],
    }
    speaker_df = read_decoding_results(
        librispeech_split,
        config_filter=speaker_config_filter,
    )
    if not speaker_df.empty:
        speaker_df["decoding_type"] = "Speaker Decoding Probe"

    # Read phonetics decoding results
    phonetics_config_filter = {
        "target_variable": "PhoneID",
        "x_filter_pattern": "hidden_state_L",
        "target_models": [
            "wav2vec2-base",
            "wav2vec2-base-960h",
            "wav2vec2-ls100-sid",
        ],
    }
    phonetics_df = read_decoding_results(
        librispeech_split,
        config_filter=phonetics_config_filter,
    )
    if not phonetics_df.empty:
        phonetics_df["decoding_type"] = "Phonetics Decoding Probe"

    # Combine and plot if both have data
    if not speaker_df.empty and not phonetics_df.empty:
        combined_speaker_phonetics_df = pd.concat(
            [speaker_df, phonetics_df], ignore_index=True
        )
        combined_speaker_phonetics_plot_config = {
            "y_label": "Accuracy",
            "x_label": "Layer",
            "figure_name_suffix": "speaker_phonetics_combined_decoding_by_layer",
            "figure_size": (6, 3),
            "include_baseline": True,
        }
        # Map decoding_type to target_variable for baseline lookup
        speaker_phonetics_baseline_targets = {
            "Speaker Decoding Probe": "SpeakerID",
            "Phonetics Decoding Probe": "PhoneID",
        }
        plot_decoding_by_layer(
            combined_speaker_phonetics_df,
            show_plot=args.show_plots,
            librispeech_split=librispeech_split,
            plot_config=combined_speaker_phonetics_plot_config,
            config_filter=None,
            use_facet_grid=True,
            facet_col="decoding_type",
            baseline_target_variables=speaker_phonetics_baseline_targets,
        )
    elif speaker_df.empty:
        logger.warning("No speaker decoding results found for combined plot.")
    elif phonetics_df.empty:
        logger.warning("No phonetics decoding results found for combined plot.")

    if librispeech_split != "train-clean-100":
        logger.info(
            "Random seed analysis is currently only set up for 'train-clean-100' split. Skipping random seed plots."
        )
        return
    modelname = "facebook/wav2vec2-base"
    random_seed_results_df = read_random_seed_results(
        librispeech_split=librispeech_split,
        modelname=modelname,
    )

    summarize_random_seed_line_differences(
        random_seed_results_df=random_seed_results_df,
        focus_lookup=FOCUS_CONFIGS,
    )

    for focus in FOCUS_CONFIGS:
        plot_focus_random_seed(
            random_seed_results_df,
            focus=focus,
            show_plot=args.show_plots,
            print_ttest=args.print_ttest,
            librispeech_split=librispeech_split,
        )

    modelname = "google-bert/bert-base-uncased"
    random_seed_results_df = read_random_seed_results(
        librispeech_split=librispeech_split, modelname=modelname
    )
    summarize_random_seed_line_differences(
        random_seed_results_df=random_seed_results_df,
        focus_lookup=FOCUS_CONFIGS,
    )
    plot_focus_random_seed(
        random_seed_results_df,
        show_plot=args.show_plots,
        print_ttest=args.print_ttest,
        librispeech_split=librispeech_split,
    )


if __name__ == "__main__":
    main()
