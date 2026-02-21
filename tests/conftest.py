import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def dummy_dataframe():
    """Provides a basic dummy DataFrame for testing."""
    return pd.DataFrame(
        {
            "feature1": [1.5, 2.5, 3.5, np.nan, 5.5],
            "feature2": np.random.randint(0, 10, 5),
            "category1": ["A", "A", "B", "B", "C"],
            "target_variable": [0, 1, 0, 1, 0],
        }
    )
