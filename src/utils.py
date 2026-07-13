import argparse
import logging
import os
import sys

import numpy as np
from sklearn.linear_model import Ridge

# Set up logger with time, name, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # We want the logging info to be saved to stdout not stderr
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)
# Get the hostname of the machine running the code
hostname = os.uname().nodename
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


if "snellius" in hostname:
    # If running on Snellius, use the Snellius dataset root
    DATASET_ROOT = os.path.realpath("/projects/prjs1586/corpora/LibriSpeech")
    ALIGNMENT_ROOT = DATASET_ROOT.replace("LibriSpeech", "librispeech_textgrids")
    SAVEPATH = "/projects/prjs1586/experimental_data"
    RESULTS_ROOT = "/projects/prjs1586/experimental_results"
    # FIGURES_ROOT = "/projects/prjs1586/experimental_figures"
    FIGURES_ROOT = os.path.join(PROJECT_ROOT, "figures")


else:
    # If running on local machine, use the local dataset root
    DATASET_ROOT = os.path.realpath("/corpora/LibriSpeech/LibriSpeech")
    # ALIGNMENT_ROOT = os.path.expanduser(f"~/corpora/librispeech_alignment/")
    ALIGNMENT_ROOT = os.path.join(PROJECT_ROOT, "data")
    SAVEPATH = os.path.join(PROJECT_ROOT, "experimental_data")
    RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
    FIGURES_ROOT = os.path.join(PROJECT_ROOT, "figures")


def r2_score(
    y_true, y_pred, multioutput="variance_weighted", sklearn_=False, train_data=None
):
    if sklearn_:
        from sklearn.metrics import r2_score as r2_score_

        logger.warning(
            "Using sklearn's r2_score. Note that this uses test data mean for r2 calculation."
        )
        return r2_score_(y_true, y_pred, multioutput=multioutput)
    else:
        # Use custom implementation due to sklearn using test data mean in r2 calculation
        # We want to use train data mean for both train and test r2 calculation
        assert train_data is not None, (
            "train_data must be provided for custom r2_score calculation"
        )
        ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
        y_train_mean = np.mean(train_data, axis=0)
        ss_tot = np.sum((y_true - y_train_mean) ** 2, axis=0)
        r2 = 1 - ss_res / ss_tot
        if multioutput == "variance_weighted":
            var = np.var(y_true, axis=0)
            weights = var / np.sum(var)
            r2 = np.sum(r2 * weights)
        elif multioutput == "uniform_average":
            r2 = np.mean(r2)
        return r2


def pick_probe(probe_name: str = "ridge", n_components: None | int = None):
    if probe_name == "ridge":
        model = Ridge()
        param_grid = {
            "alpha": [10**x for x in range(-3, 5)],
            # "solver": ["auto", "sag", "saga", "lsqr", "cholesky"],
            # "max_iter": [1000, 2000,  3000],
        }
    elif probe_name == "random-forest":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor()  # n_jobs=-1, verbose=1)
        param_grid = {
            "max_depth": [10, 15, 20],  # 5, 7,
            # "min_samples_split": [10, 20, 40, 80],
            # "min_samples_leaf": [5, 10, 20, 40],
            "max_features": [
                "sqrt",
                0.7,
            ],
            # "ccp_alpha": [0.0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
        }
    elif probe_name == "ridge-transformed-target":
        from sklearn.compose import TransformedTargetRegressor

        # We scale the Y values using standardscaler and apply PCA as the y_transformer
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        logger.info(
            f"Using Ridge regression with TransformedTargetRegressor with PCA n_components={n_components}"
        )

        pca = PCA(n_components=n_components, svd_solver="full")
        scaler = StandardScaler()
        y_transformer = Pipeline([("scaler", scaler), ("pca", pca)])
        base_model = Ridge()
        model = TransformedTargetRegressor(
            regressor=base_model, transformer=y_transformer
        )
        param_grid = {
            "regressor__alpha": [10**x for x in range(-3, 5)],
            # "regressor__solver": ["auto", "sag", "saga", "lsqr", "cholesky"],
            # "regressor__max_iter": [1000, 2000,  3000],
        }
    elif probe_name == "ridge_classifier":
        from sklearn.linear_model import RidgeClassifier

        model = RidgeClassifier()
        param_grid = {
            "alpha": [10**x for x in range(-3, 5)],
            # "solver": ["auto", "sag", "saga", "lsqr", "cholesky"],
            # "max_iter": [1000, 2000,  3000],
        }

    else:
        raise ValueError(f"Probe {probe_name} not supported")

    return model, param_grid


def parse_args(use_default=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--librispeech_split",
        type=str,
        default="dev-clean",
        help="The LibriSpeech split to use.",
    )
    parser.add_argument(
        "--modelname",
        type=str,
        default="facebook/wav2vec2-base",
        help="The name of the model to use. Choose from 'facebook/wav2vec2-base', 'facebook/wav2vec2-large-960h', 'answerdotai/ModernBERT-base'",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for selecting randomly sampled dnn_hidden_states frames.",
    )
    parser.add_argument(
        "--probe_name",
        type=str,
        default="ridge",
        help="The name of the probe to use. Options are 'ridge' and 'random_forest'.",
    )
    parser.add_argument(
        "--select_layers",
        type=int,
        nargs="+",
        default=None,
        help="The layers to use for probing.",
    )
    parser.add_argument(
        "--zeroing",
        action="store_true",
        help="Whether to perform zeroing manipulation.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Whether to perform ablation manipulation.",
    )
    parser.add_argument(
        "--permutation",
        action="store_true",
        help="Whether to perform permutation manipulation.",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Whether to save the test set predictions.",
    )
    parser.add_argument(
        "--normalize_features",
        action="store_true",
        help="Whether to normalize the input features.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing extracted features.",
    )
    parser.add_argument(
        "--dim_reduction",
        type=str,
        default="False",
        help="The number of dimensions to reduce the target features to using PCA.",
    )
    parser.add_argument(
        "--feature_groups_config",
        type=str,
        default=f"{SAVEPATH}/default_feature_groups.json",
        help="The feature groups to use for probing.",
    )
    parser.add_argument(
        "--syntax_deep_dive",
        action="store_true",
        help="Whether to run the syntax deep dive analysis.",
    )
    parser.add_argument(
        "--bottom_up_probe",
        action="store_true",
        help="Whether to run the bottom-up probe analysis.",
    )
    parser.add_argument(
        "--spec_name",
        type=str,
        default=None,
        help=(
            "If set, run only the ExperimentSpec with this name. "
            "When unset, all default specs plus the syntax lexicon "
            "decomposition are run (existing behavior)."
        ),
    )
    parser.add_argument(
        "--manipulation",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Manipulation mode(s) to apply to targeted feature blocks: "
            "'drop' (ablation, default), 'shuffle' (per-column independent "
            "permutation), 'zero' (set block to 0.0). Defaults to ['drop']."
        ),
    )
    parser.add_argument(
        "--result_subdir",
        type=str,
        default=None,
        help=(
            "Override the ExperimentSpec's result_subdir (and experiment "
            "name) so results land in an isolated directory without "
            "overwriting existing outputs. When unset, the spec's default "
            "result_subdir is used (existing behavior)."
        ),
    )

    if use_default:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    return args


def get_opensmile_feature_names() -> dict:
    import opensmile

    feature_level = opensmile.FeatureLevel.LowLevelDescriptors
    smile = opensmile.Smile(
        feature_set="eGeMAPSv02",
        feature_level=feature_level,
        verbose=True,
        num_workers=8,
        sampling_rate=16000,
        resample=True,
    )

    feature_names = smile.feature_names

    feature_groups = {
        "eGeMAPSv02": feature_names,
    }

    # Grab the corresponding index for each feature within each group
    grouped_feature_indices = {}
    for group_name, features in feature_groups.items():
        indices = [
            feature_names.index(feat) for feat in features if feat in feature_names
        ]
        grouped_feature_indices[group_name] = {
            "feature_names": features,
            "indices": indices,
        }

    return grouped_feature_indices
