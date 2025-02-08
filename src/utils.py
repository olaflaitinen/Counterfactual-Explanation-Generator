"""
utils.py

Utility functions for the Counterfactual Explanation Generator project.
"""

import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_counterfactual(original, counterfactual, feature_names):
    """
    Plot a comparison between the original instance and its counterfactual.

    Args:
        original (array-like): Original instance features.
        counterfactual (array-like): Counterfactual instance features.
        feature_names (list): List of feature names.
    """
    x = range(len(feature_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, original, width, label='Original', color='skyblue')
    ax.bar([p + width for p in x], counterfactual, width, label='Counterfactual', color='salmon')

    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_ylabel('Feature Value')
    ax.set_title('Comparison: Original vs. Counterfactual Instance')
    ax.legend()

    plt.tight_layout()
    plt.show()
    logger.info("Displayed counterfactual comparison plot.")
