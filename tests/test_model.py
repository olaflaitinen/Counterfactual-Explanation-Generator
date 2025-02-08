"""
test_model.py

Unit tests for the logistic regression model.
"""

import pytest
from src.model import LogisticRegressionModel
from src.data import load_data


@pytest.fixture(scope="module")
def model():
    lr_model = LogisticRegressionModel()
    lr_model.train()
    return lr_model


def test_model_accuracy(model):
    X, y = load_data()
    predictions = model.predict(X)
    accuracy = (predictions == y).mean()
    assert accuracy > 0.7, "Model accuracy should be greater than 70%"


def test_model_predict_proba(model):
    X, _ = load_data()
    proba = model.predict_proba(X)
    for row in proba:
        assert abs(sum(row) - 1.0) < 1e-5, "Predicted probabilities should sum to 1"
