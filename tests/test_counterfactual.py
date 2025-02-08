"""
test_counterfactual.py

Unit tests for the counterfactual generator functionality.
"""

import pytest
import numpy as np
from src.data import load_data
from src.model import LogisticRegressionModel
from src.counterfactual_generator import CounterfactualGenerator


@pytest.fixture(scope="module")
def trained_model():
    model = LogisticRegressionModel()
    model.train()
    return model


@pytest.fixture(scope="module")
def sample_instance():
    X, _ = load_data()
    # Select the first instance for testing
    return X.iloc[0].values


@pytest.fixture(scope="module")
def feature_range():
    # For simplicity, compute feature ranges from the Iris dataset
    X, _ = load_data()
    ranges = {}
    for i in range(X.shape[1]):
        ranges[i] = (float(X.iloc[:, i].min()), float(X.iloc[:, i].max()))
    return ranges


@pytest.fixture(scope="module")
def cf_generator(trained_model, feature_range):
    # Initialize the counterfactual generator with the model's predict function
    shape = (trained_model.model.coef_.shape[1],)
    return CounterfactualGenerator(
        predict_fn=trained_model.predict_proba,
        shape=shape,
        target_feature_range=feature_range,
        target_proba=0.5,
        tol=0.05,
        lam_init=1e-1,
        max_iter=1000,
        early_stop=50
    )


def test_generate_counterfactual(cf_generator, sample_instance):
    # For testing, request a counterfactual for a target class different from the original prediction.
    original_pred = np.argmax(cf_generator.predict_fn(sample_instance.reshape(1, -1)))
    target_class = (original_pred + 1) % 3  # assuming 3 classes in Iris
    explanation = cf_generator.generate_counterfactual(sample_instance, target_class)
    # Check that the explanation contains a counterfactual instance
    assert 'X' in explanation.cf, "Explanation should contain counterfactual instance under key 'X'"
    assert explanation.cf['X'].shape == (1, len(sample_instance)), "Counterfactual instance shape mismatch"
