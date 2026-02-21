import pytest
from ml_framework_template.data_analyzer.data_reader import DataReader


def test_split_features_target(dummy_dataframe):
    """Test that the DataReader can safely split features from target."""
    X, y = DataReader.split_features_target(dummy_dataframe, "target_variable")

    assert "target_variable" not in X.columns
    assert len(X) == len(dummy_dataframe)
    assert len(y) == len(dummy_dataframe)
    assert y.name == "target_variable"
    assert "feature1" in X.columns
