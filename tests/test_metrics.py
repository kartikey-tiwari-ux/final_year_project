from src.evaluation.metrics import compute_metrics


def test_perfect_predictions_give_full_scores():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0


def test_all_wrong_gives_zero_f1_for_rotated_labels():
    y_true = [0, 0, 0]
    y_pred = [1, 1, 1]
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.0


def test_confusion_matrix_shape_is_3x3():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 0, 2, 1, 1, 0]
    m = compute_metrics(y_true, y_pred)
    assert len(m["confusion_matrix"]) == 3
    assert all(len(row) == 3 for row in m["confusion_matrix"])


def test_per_class_keys_present():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    m = compute_metrics(y_true, y_pred)
    for name in ["positive", "neutral", "negative"]:
        assert name in m["per_class"]
