"""Stratified train/val/test splitting with duplicate-aware leakage prevention.

See REPRODUCIBILITY.md: splits are computed once, saved as index files, and reused
identically by every model (A/B/C) so comparisons are fair.
"""
from collections import defaultdict
from typing import Hashable, Sequence

from sklearn.model_selection import train_test_split


def stratified_split(
    ids: Sequence[Hashable],
    labels: Sequence[Hashable],
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> dict:
    """Split `ids` into train/val/test, stratified by `labels`.

    Returns a dict with keys "train", "val", "test", each a list of ids.
    Raises ValueError if a class has too few members to stratify into all three splits.
    """
    if len(ids) != len(labels):
        raise ValueError("ids and labels must be the same length")

    class_counts = defaultdict(int)
    for lbl in labels:
        class_counts[lbl] += 1
    min_count = min(class_counts.values())
    if min_count < 3:
        raise ValueError(
            f"At least one class has only {min_count} sample(s); cannot stratify into "
            "train/val/test. Inspect the raw label distribution before splitting."
        )

    ids_trainval, ids_test, labels_trainval, _ = train_test_split(
        list(ids),
        list(labels),
        test_size=test_size,
        random_state=seed,
        stratify=list(labels),
    )
    relative_val_size = val_size / (1.0 - test_size)
    ids_train, ids_val = train_test_split(
        ids_trainval,
        test_size=relative_val_size,
        random_state=seed,
        stratify=labels_trainval,
    )
    return {"train": ids_train, "val": ids_val, "test": ids_test}


def assert_no_overlap(splits: dict) -> None:
    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    overlaps = {
        "train/val": train & val,
        "train/test": train & test,
        "val/test": val & test,
    }
    leaking = {k: v for k, v in overlaps.items() if v}
    if leaking:
        raise AssertionError(f"Data leakage detected between splits: {leaking}")
