import pandas as pd
import pytest

from src.data.mvsa_dataset import LABEL_NAMES, LABEL_TO_INT, get_split, load_manifest


def test_manifest_loads_and_has_expected_columns():
    df = load_manifest()
    for col in ["id", "text", "image_path", "label", "label_int", "split"]:
        assert col in df.columns


def test_manifest_has_expected_total_size():
    df = load_manifest()
    assert len(df) == 4511  # real, confirmed count — see DATASET_SELECTION.md


def test_splits_have_expected_sizes():
    df = load_manifest()
    assert len(get_split(df, "train")) == 3611
    assert len(get_split(df, "val")) == 450
    assert len(get_split(df, "test")) == 450


def test_labels_are_from_fixed_set():
    df = load_manifest()
    assert set(df["label"].unique()) <= set(LABEL_NAMES)


def test_label_to_int_mapping_consistent_with_label_int_column():
    df = load_manifest()
    for _, row in df.head(20).iterrows():
        assert LABEL_TO_INT[row["label"]] == row["label_int"]


def test_image_paths_are_absolute_and_exist():
    df = load_manifest()
    sample = df.sample(n=10, random_state=1)
    for path in sample["image_path"]:
        from pathlib import Path

        assert Path(path).is_absolute()
        assert Path(path).exists()


def test_get_split_rejects_invalid_split_name():
    df = load_manifest()
    with pytest.raises(AssertionError):
        get_split(df, "not_a_real_split")
