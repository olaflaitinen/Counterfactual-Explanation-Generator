"""
run_counterfactual.py

Example script to train a logistic regression model, generate a counterfactual explanation,
and visualize the results.
"""

import numpy as np
import pandas as pd
from src.data import load_data
from src.model import LogisticRegressionModel
from src.counterfactual_generator import CounterfactualGenerator
from src.utils import plot_counterfactual

def main():
    # Load data and train the model
    X, y = load_data()
    model = LogisticRegressionModel()
    metrics = model.train()
    print(f"Model training metrics: {metrics}")

    # Select a sample instance for explanation
    sample_instance = X.iloc[0].values
    print("Original instance:", sample_instance)
    original_pred = np.argmax(model.predict_proba(sample_instance.reshape(1, -1)))
    print("Original prediction:", original_pred)

    # Compute feature ranges for the counterfactual generator
    feature_range = {}
    for i in range(X.shape[1]):
        feature_range[i] = (float(X.iloc[:, i].min()), float(X.iloc[:, i].max()))

    # Initialize counterfactual generator
    cf_generator = CounterfactualGenerator(
        predict_fn=model.predict_proba,
        shape=(X.shape[1],),
        target_feature_range=feature_range,
        target_proba=0.5,
        tol=0.05,
        lam_init=1e-1,
        max_iter=1000,
        early_stop=50
    )

    # Define desired target class (choose a different one from the original prediction)
    target_class = (original_pred + 1) % 3  # For Iris dataset with 3 classes
    print(f"Requesting counterfactual for target class: {target_class}")

    # Generate counterfactual explanation
    explanation = cf_generator.generate_counterfactual(sample_instance, target_class)
    if 'X' in explanation.cf:
        counterfactual_instance = explanation.cf['X'][0]
        print("Counterfactual instance:", counterfactual_instance)
    else:
        print("No counterfactual generated.")
        return

    # Visualize the difference between the original and counterfactual instances
    feature_names = list(X.columns)
    plot_counterfactual(sample_instance, counterfactual_instance, feature_names)

if __name__ == "__main__":
    main()
