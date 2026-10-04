import pytest

from src.data.splits import assert_no_overlap, stratified_split


def _balanced_dataset(n_per_class=20):
    ids, labels = [], []
    i = 0
    for cls in ["positive", "neutral", "negative"]:
        for _ in range(n_per_class):
            ids.append(f"id{i}")
            labels.append(cls)
            i += 1
    return ids, labels


def test_split_sizes_approximately_correct():
    ids, labels = _balanced_dataset(n_per_class=40)  # 120 total
    splits = stratified_split(ids, labels, val_size=0.15, test_size=0.15, seed=42)
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == len(ids)
    assert len(splits["test"]) == pytest.approx(0.15 * len(ids), abs=2)
    assert len(splits["val"]) == pytest.approx(0.15 * len(ids), abs=2)


def test_no_overlap_between_splits():
    ids, labels = _balanced_dataset(n_per_class=40)
    splits = stratified_split(ids, labels, seed=42)
    assert_no_overlap(splits)  # should not raise


def test_assert_no_overlap_catches_real_leakage():
    with pytest.raises(AssertionError):
        assert_no_overlap({"train": ["a", "b"], "val": ["b", "c"], "test": ["d"]})


def test_split_is_deterministic_given_seed():
    ids, labels = _balanced_dataset(n_per_class=40)
    s1 = stratified_split(ids, labels, seed=42)
    s2 = stratified_split(ids, labels, seed=42)
    assert s1 == s2


def test_different_seeds_give_different_splits():
    ids, labels = _balanced_dataset(n_per_class=40)
    s1 = stratified_split(ids, labels, seed=1)
    s2 = stratified_split(ids, labels, seed=2)
    assert s1 != s2


def test_rejects_too_few_samples_in_a_class():
    ids = ["a", "b", "c", "d", "e"]
    labels = ["positive", "positive", "positive", "positive", "negative"]
    with pytest.raises(ValueError):
        stratified_split(ids, labels)
