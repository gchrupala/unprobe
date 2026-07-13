"""Shared helpers for running probe experiments.

The original codebase duplicated train/test splitting, GridSearchCV setup,
feature masking, and scoring logic across multiple scripts.  This module
centralizes those utilities so that encoding, decoding, and targeted
experiments can build on the same primitives.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split

from utils import r2_score

DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2


def build_feature_lookup(section_shapes: np.ndarray) -> dict[str, tuple[int, int]]:
    """Create a fast lookup dictionary from section metadata."""

    return {str(name): (int(start), int(end)) for start, end, name in section_shapes}


def indices_for_features(
    lookup: dict[str, tuple[int, int]], feature_names: Iterable[str]
) -> np.ndarray:
    """Return absolute column indices for the requested feature groups."""

    indices: list[int] = []
    for name in feature_names:
        if name not in lookup:
            raise KeyError(f"Feature group '{name}' not present in section shapes")
        start, end = lookup[name]
        indices.extend(range(start, end))
    return np.asarray(indices, dtype=int)


def select_feature_groups(
    feature_sets: np.ndarray,
    lookup: dict[str, tuple[int, int]],
    feature_names: Iterable[str],
) -> np.ndarray:
    """Return a view containing only the requested feature groups."""

    group_indices = indices_for_features(lookup, feature_names)
    return feature_sets[:, group_indices]


def drop_feature_groups(
    feature_sets: np.ndarray,
    lookup: dict[str, tuple[int, int]],
    feature_names: Iterable[str],
) -> np.ndarray:
    """Return a copy with the requested feature groups removed."""

    mask = np.ones(feature_sets.shape[1], dtype=bool)
    drop_idx = indices_for_features(lookup, feature_names)
    mask[drop_idx] = False
    return feature_sets[:, mask]


def shuffle_feature_groups(
    feature_sets: np.ndarray,
    lookup: dict[str, tuple[int, int]],
    feature_names: Iterable[str],
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> np.ndarray:
    """Return a copy with the requested feature groups per-column shuffled.

    Each dimension inside the targeted blocks is independently permuted along
    the sample axis (axis=0). This preserves every dimension's marginal
    distribution while destroying its alignment with the target and with the
    other dimensions inside the block, yielding a dimensionality-preserving
    control for ``drop_feature_groups``. The total column count is unchanged
    so the result stays directly comparable to the full-feature topline.

    Args:
        feature_sets: 2D array of shape ``[n_samples, n_features]``.
        lookup: Feature-group name to ``(start, end)`` column range.
        feature_names: Groups whose columns should be shuffled.
        random_state: Seed for the per-column permutations; defaults to
            ``DEFAULT_RANDOM_STATE`` for reproducibility.
    """

    shuffled = feature_sets.copy()
    block_idx = indices_for_features(lookup, feature_names)
    if block_idx.size == 0:
        return shuffled
    block = shuffled[:, block_idx]
    # Independent permutation per column via argsort of random keys: each
    # column of ``perms`` is a distinct permutation of range(n_samples).
    perms = np.argsort(np.random.default_rng(random_state).random(block.shape), axis=0)
    shuffled[:, block_idx] = np.take_along_axis(block, perms, axis=0)
    return shuffled


def zero_feature_groups(
    feature_sets: np.ndarray,
    lookup: dict[str, tuple[int, int]],
    feature_names: Iterable[str],
) -> np.ndarray:
    """Return a copy with the requested feature groups set to zero.

    All values inside the targeted blocks are replaced with ``0.0`` while the
    columns themselves are retained, so the total feature dimensionality is
    unchanged and the result stays directly comparable to the full-feature
    topline. This is most meaningful when features are mean-centered
    (``normalize_features=True``), where zero coincides with the feature mean.
    """

    zeroed = feature_sets.copy()
    block_idx = indices_for_features(lookup, feature_names)
    zeroed[:, block_idx] = 0.0
    return zeroed


def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    *,
    metadata: Sequence | None = None,
    stratify_labels: Sequence | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    test_size: float = DEFAULT_TEST_SIZE,
):
    """Split arrays (and optional metadata) with consistent shuffling."""

    arrays: list = [X, y]
    if metadata is not None:
        arrays.append(metadata)

    split = train_test_split(
        *arrays,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    if metadata is None:
        x_train, x_test, y_train, y_test = split
        return x_train, x_test, y_train, y_test

    x_train, x_test, y_train, y_test, meta_train, meta_test = split
    return x_train, x_test, y_train, y_test, meta_train, meta_test


def fit_probe(
    estimator,
    param_grid,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    scoring: str = "r2",
    n_jobs: int = -1,
    cv: int = 5,
    verbose: int = 0,
):
    """Run GridSearchCV with the provided configuration and data."""

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=n_jobs,
        cv=cv,
        verbose=verbose,
    )
    grid.fit(X_train, y_train)
    return grid


def evaluate_probe(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float]:
    """Compute custom train/test R² using training statistics for stability."""

    train_score = r2_score(
        y_train,
        model.predict(X_train),
        multioutput="variance_weighted",
        train_data=y_train,
    )
    test_score = r2_score(
        y_test,
        model.predict(X_test),
        multioutput="variance_weighted",
        train_data=y_train,
    )
    return train_score, test_score


def run_standard_probe(
    estimator,
    param_grid,
    X: np.ndarray,
    y: np.ndarray,
    *,
    stratify_labels: Sequence | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    return_splits: bool = False,
):
    """Convenience wrapper that splits, fits, and scores a probe."""

    split = split_train_test(
        X,
        y,
        stratify_labels=stratify_labels,
        random_state=random_state,
    )
    x_train, x_test, y_train, y_test = split
    grid = fit_probe(estimator, param_grid, x_train, y_train)
    train_score, test_score = evaluate_probe(
        grid.best_estimator_, x_train, y_train, x_test, y_test
    )
    result = {
        "train_score": train_score,
        "test_score": test_score,
        "best_params": grid.best_params_,
        "best_estimator": grid.best_estimator_,
    }
    if return_splits:
        result.update(
            {
                "x_train": x_train,
                "x_test": x_test,
                "y_train": y_train,
                "y_test": y_test,
            }
        )
    return result


__all__ = [
    "build_feature_lookup",
    "drop_feature_groups",
    "evaluate_probe",
    "fit_probe",
    "indices_for_features",
    "run_standard_probe",
    "select_feature_groups",
    "shuffle_feature_groups",
    "split_train_test",
    "zero_feature_groups",
]
