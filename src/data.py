"""
data.py

Module for loading and preprocessing the Iris dataset.
"""

from sklearn.datasets import load_iris
import pandas as pd


def load_data():
    """
    Load the Iris dataset and return features and target.

    Returns:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target labels.
    """
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    return X, y
