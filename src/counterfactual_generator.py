"""
counterfactual_generator.py

Module for generating counterfactual explanations using the Alibi library.
"""

import numpy as np
import logging
from alibi.explainers import Counterfactual

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CounterfactualGenerator:
    """
    Class for generating counterfactual explanations for a given instance.
    """

    def __init__(self, predict_fn, shape, target_feature_range, target_proba=0.5, tol=0.05,
                 lam_init=1e-1, max_iter=1000, early_stop=50):
        """
        Initialize the counterfactual explainer.

        Args:
            predict_fn (callable): Prediction function for the model.
            shape (tuple): Shape of a single instance (n_features,).
            target_feature_range (dict): Dictionary mapping feature indices to (min, max) tuples.
            target_proba (float): Desired probability for the target class.
            tol (float): Tolerance for stopping criterion.
            lam_init (float): Initial value for regularization parameter.
            max_iter (int): Maximum number of iterations.
            early_stop (int): Early stopping rounds.
        """
        self.predict_fn = predict_fn
        self.shape = shape
        self.target_feature_range = target_feature_range
        self.target_proba = target_proba
        self.tol = tol
        self.lam_init = lam_init
        self.max_iter = max_iter
        self.early_stop = early_stop

        # Initialize the counterfactual explainer
        self.cf = Counterfactual(
            predict_fn,
            shape=self.shape,
            target_proba=self.target_proba,
            tol=self.tol,
            target_class=None,  # This can be set when explaining an instance
            lam_init=self.lam_init,
            max_iterations=self.max_iter,
            early_stop=self.early_stop,
            feature_range=self.target_feature_range,
            verbose=True,
            loss_type='l2'
        )
        logger.info("Initialized Counterfactual explainer.")

    def generate_counterfactual(self, instance, target_class):
        """
        Generate a counterfactual explanation for the given instance.

        Args:
            instance (np.array): The input instance (1D array).
            target_class (int): Desired target class for the counterfactual.

        Returns:
            dict: A dictionary containing the counterfactual explanation.
        """
        # Set the target class for the explanation
        self.cf.target_class = target_class
        explanation = self.cf.explain(instance.reshape(1, -1))
        logger.info("Generated counterfactual explanation.")
        return explanation
