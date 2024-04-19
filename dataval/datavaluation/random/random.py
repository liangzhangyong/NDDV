from typing import Optional

import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_random_state

from dataval.datavaluation.api import DataEvaluator


class RandomEvaluator(DataEvaluator):
    """Completely Random DataEvaluator for baseline comparison purposes.

    Generates Random data values from Uniform[0.0, 1.0].

    Parameters
    ----------
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(self, random_state: Optional[RandomState] = None):
        self.random_state = check_random_state(random_state)

    def train_data_values(self, *args, **kwargs):
        """RandomEval does not train to find the data values."""
        pass

    def evaluate_data_values(self) -> np.ndarray:
        """Return random data values for each training data point."""
        return self.random_state.uniform(size=(len(self.x_train),))
    
    def data_values_trajectory(self):
        pass
    
    def data_state_trajectory(self):
        pass
    
    def save_train_meta_loss(self):
        pass
