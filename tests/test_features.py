import numpy as np

from technical_analysis.features import TARGET, build_features, feature_columns, training_frame


def test_target_is_next_day_return(synth_prices):
    """The supervised target must equal the NEXT day's log return (no look-ahead)."""
    feats = build_features(synth_prices)
    expected = np.log(synth_prices["Close"].shift(-1) / synth_prices["Close"])
    valid = feats["target"].dropna().index
    assert np.allclose(feats.loc[valid, "target"], expected.loc[valid], atol=1e-9)


def test_last_row_target_is_nan(synth_prices):
    """The final row has an unknown next-day return — it is used only for inference."""
    feats = build_features(synth_prices)
    assert np.isnan(feats["target"].iloc[-1])


def test_training_frame_clean(synth_prices):
    X, y, feats, cols = training_frame(synth_prices)
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert len(cols) > 10
    assert TARGET not in cols
    assert len(X) == len(y)
