import numpy as np

from forecasting.evaluate import directional_accuracy, make_metrics, price_mape


def test_directional_accuracy_perfect_and_wrong():
    yt = np.array([0.01, -0.02, 0.03])
    assert directional_accuracy(yt, np.array([0.5, -0.1, 0.2])) == 1.0
    assert directional_accuracy(yt, np.array([-0.5, 0.1, -0.2])) == 0.0


def test_price_mape_zero_when_exact():
    prev = np.array([100.0, 200.0])
    ret = np.array([0.01, -0.02])
    assert price_mape(prev, ret, ret) < 1e-9


def test_skill_positive_for_perfect_prediction():
    yt = np.array([0.01, -0.01, 0.02, -0.02])
    prev = np.full(4, 100.0)
    m = make_metrics(yt, yt, prev, baseline_rmse=0.02)
    assert m.rmse < 1e-9
    assert m.skill_vs_baseline > 0.99
    assert m.directional_accuracy == 1.0


def test_skill_negative_for_bad_prediction():
    yt = np.array([0.01, -0.01, 0.02, -0.02])
    yp = -yt * 5  # wrong sign and larger magnitude
    m = make_metrics(yt, yp, np.full(4, 100.0), baseline_rmse=float(np.sqrt(np.mean(yt**2))))
    assert m.skill_vs_baseline < 0
