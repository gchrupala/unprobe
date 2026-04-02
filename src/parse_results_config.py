"""
Configuration constants and mappings for parse_results.py.

This module contains all the configuration dictionaries, rename mappings,
color palettes, and ordering definitions used in the results parsing
and visualization pipeline.
"""

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

# Derived feature name constants
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

# Run group configuration
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

# Model name ordering and renaming
MODELNAME_ORDER: list = [
    "wav2vec2-base",
    "wav2vec2-base-960h",
    "wav2vec2-large",
    "wav2vec2-large-960h",
    "wav2vec2-large-xlsr-53",
    "hubert-base-ls960",
    "hubert-large-ll60k",
    "hubert-large-ls960-ft",
    "wavlm-base",
    "wav2vec2-base-superb-sid",
    "wav2vec2-ls100-sid",
    "roberta-base",
    "bert-base-uncased",
    "ModernBERT-base",
]

MODELNAME_RENAME: dict[str, str] = {
    "wav2vec2-base": "wav2vec2 (base)",
    "wav2vec2-base-960h": "wav2vec2 (ASR)",
    "wav2vec2-large": "wav2vec2 (large)",
    "wav2vec2-large-960h": "wav2vec2 (large-ASR)",
    "hubert-base-ls960": "HuBERT (base)",
    "hubert-large-ll60k": "HuBERT (large)",
    "hubert-large-ls960-ft": "HuBERT (large-ASR)",
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

# Syntax component renaming
SYNTAX_COMPONENT_RENAME: dict[str, str] = {
    "syntax_POS_OH": r"\setminus \mathit{Syntax-POS}",
    "syntax_Dependency_Label_OH": r"\setminus \mathit{Syntax-Dependency}",
    "syntax_Tree_Depth": r"\setminus \mathit{Syntax-Tree-Depth}",
    "syntax_Word_Position": r"\setminus \mathit{Syntax-Word-Position}",
    "syntax_Total_Tree_Depth": r"\setminus \mathit{Syntax-Total-Tree-Depth}",
    "syntax_Total_Word_Count": r"\setminus \mathit{Syntax-Total-Word-Count}",
}


def _rename_syntax_component_config(config_name: str) -> str:
    """Rename syntax component config names to LaTeX format.

    Args:
        config_name: The config name to rename (e.g., "word_embedding+syntax_POS_OH")

    Returns:
        LaTeX-formatted string if matched, otherwise the original config_name
    """
    base_str = r"\mathit{Full} \setminus \mathit{lexicon}"
    for component, label in SYNTAX_COMPONENT_RENAME.items():
        token = f"word_embedding+{component}"
        if config_name == token:
            return "$" + base_str + " " + label + "$"
    return config_name


# Config name ordering (computed from CONFIG_NAME_RENAME)
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

# Plot color mapping for config names
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

# Y column name mappings for display
Y_COL_NAME_MAPPING = {
    "test_score": r"HRS ($R^2$) Score",
    "unexplained_variance": r"Unexplained Variance (1 - $R^2$)",
    "departure_from_topline": r"Departure from Topline ($R^2$ difference)",
}

# Plotting configurations for different experiment views
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
        "exclude_models": [
            "wav2vec2-ls100-sid",
            "wav2vec2-large-xlsr-53",
            "wav2vec2-base-superb-sid",
        ],
        "x_col": "normalized_layer",
        "y_col": "unexplained_variance",
        "figure_size": (8, 8),
    },
    "all_models_acoustic_speaker": {
        "target_configs": [
            ACOUSTIC_NAME,
            SPEAKERID_NAME,
            SPEAKERID_ACOUSTIC_NAME,
        ],
        "target_models": [
            "wav2vec2-base",
            "wav2vec2-base-960h",
            "wav2vec2-large",
            "wav2vec2-large-960h",
            "hubert-base-ls960",
            "hubert-large-ll60k",
            "hubert-large-ls960-ft",
            "wavlm-base",
            "wav2vec2-base-superb-sid",
        ],
        "exclude_models": ["wav2vec2-ls100-sid"],
        "x_col": "normalized_layer",
        "y_col": "unexplained_variance",
        "figure_size": (8, 6),
    },
    "all_models_phonetic_speaker": {
        "target_configs": [
            PHONETIC_NAME,
            SPEAKERID_NAME,
            PHONETIC_SPEAKERID_NAME,
        ],
        "target_models": [
            "wav2vec2-base",
            "wav2vec2-base-960h",
            "wav2vec2-large",
            "wav2vec2-large-960h",
            "hubert-base-ls960",
            "hubert-large-ll60k",
            "hubert-large-ls960-ft",
            "wavlm-base",
            "wav2vec2-base-superb-sid",
        ],
        "exclude_models": ["wav2vec2-ls100-sid"],
        "x_col": "normalized_layer",
        "y_col": "unexplained_variance",
        "figure_size": (8, 6),
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
    "syntax_lexicon_decomposition": {
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
        "target_models": ["wav2vec2-base", "wav2vec2-base-960h", "bert-base-uncased"],
        "x_col": "layer",
        "y_col": "unexplained_variance",
        "figure_size": (6, 4),
        "legend_n_row": 5,
        "color_mapping": None,
    },
}

# Decoding plot configurations
# Individual plots have: config_filter, plot_config
# Combined plots have: is_combined=True, constituent_configs, decoding_type_labels, plot_config
# Combined plots may also have: baseline_target_variables (for classification tasks)
DECODING_PLOT_CONFIGS: dict[str, dict] = {
    # Individual decoding plots
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
            "y_label": "Syntax Decoding Metric",
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
    # Combined decoding plots
    "syntax_lexicon_combined": {
        "is_combined": True,
        "constituent_configs": ["syntax_full_hidden", "lexicon_hidden"],
        "decoding_type_labels": {
            "syntax_full_hidden": "Syntax Decoding Probe",
            "lexicon_hidden": "Lexicon Decoding Probe",
        },
        "plot_config": {
            "y_label": r"$R^2$ Score",
            "x_label": "Layer",
            "figure_name_suffix": "syntax_lexicon_combined_decoding_by_layer",
            "figure_size": (6, 3),
            "use_facet_grid": True,
            "facet_col": "decoding_type",
        },
    },
    "speaker_phonetics_combined": {
        "is_combined": True,
        "constituent_configs": ["speakerid_hidden", "phoneid_hidden"],
        "decoding_type_labels": {
            "speakerid_hidden": "Speaker Decoding Probe",
            "phoneid_hidden": "Phonetics Decoding Probe",
        },
        "baseline_target_variables": {
            "Speaker Decoding Probe": "SpeakerID",
            "Phonetics Decoding Probe": "PhoneID",
        },
        "plot_config": {
            "y_label": "Accuracy",
            "x_label": "Layer",
            "figure_name_suffix": "speaker_phonetics_combined_decoding_by_layer",
            "figure_size": (6, 3),
            "include_baseline": True,
            "use_facet_grid": True,
            "facet_col": "decoding_type",
        },
    },
}
