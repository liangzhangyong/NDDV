from abc import ABC, abstractmethod
from functools import cached_property
from typing import Callable, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import Dataset

from dataval.dataloader import DataFetcher
from dataval.metrics import accuracy, neg_mse
from dataval.model import Model
from dataval.util import ReprMixin

Self = TypeVar("Self", bound="DataEvaluator")


class DataEvaluator(ABC, ReprMixin):
    """Abstract class of Data Evaluators. Facilitates Data Evaluation computation.

    The following is an example of how the api would work:
    ::
        dataval = (
            DataEvaluator(*args, **kwargs)
            .input_data(x_train, y_train, x_valid, y_valid)
            .train_data_values(batch_size, epochs)
            .evaluate_data_values()
        )

    Parameters
    ----------
    random_state : RandomState, optional
        Random initial state, by default None
    args : tuple[Any]
        DavaEvaluator positional arguments
    kwargs : Dict[str, Any]
        DavaEvaluator key word arguments

    Attributes
    ----------
    pred_model : Model
        Prediction model to find how much each training datum contributes towards it.
    data_values: np.array
        Cached data values, used by :py:mod:`dataval.experiment.exper_methods`
    """

    Evaluators: ClassVar[dict[str, Self]] = {}

    def __init__(self, random_state: Optional[RandomState] = None, *args, **kwargs):
        self.random_state = check_random_state(random_state)

    def __init_subclass__(cls, *args, **kwargs):
        """Registers DataEvaluator types, used as part of the CLI."""
        super().__init_subclass__(*args, **kwargs)
        cls.Evaluators[cls.__name__.lower()] = cls

    def input_data(
        self,
        x_train: Union[torch.Tensor, Dataset],
        y_train: torch.Tensor,
        x_valid: Union[torch.Tensor, Dataset],
        y_valid: torch.Tensor,
        x_test: Union[torch.Tensor, Dataset],
        y_test: torch.Tensor,
        x_meta: Union[torch.Tensor, Dataset],
        y_meta: torch.Tensor,
    ):
        """Store and transform input data for DataEvaluator.

        Parameters
        ----------
        x_train : torch.Tensor
            Data covariates
        y_train : torch.Tensor
            Data labels
        x_valid : torch.Tensor
            Test+Held-out covariates
        y_valid : torch.Tensor
            Test+Held-out labels

        Returns
        -------
        self : object
            Returns a Data Evaluator.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test = x_test
        self.y_test = y_test
        self.x_meta = x_meta
        self.y_meta = y_meta

        return self

    def setup(
        self,
        fetcher: DataFetcher,
        pred_model: Optional[Model] = None,
        metric: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
    ):
        """Inputs model, metric and data into Data Evaluator.

        Parameters
        ----------
        fetcher : DataFetcher
            DataFetcher containing the training and validation data set.
        pred_model : Model, optional
            Prediction model, not required if the DataFetcher is Model less
        metric : Callable[[torch.Tensor, torch.Tensor], float]
            Evaluation function to determine prediction model performance,
            by default None and assigns either -MSE or ACC depending if categorical
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Returns
        -------
        self : object
            Returns a Data Evaluator.
        """
        self.input_fetcher(fetcher)

        if isinstance(self, ModelMixin):
            if metric is None:
                metric = accuracy if fetcher.one_hot else neg_mse
            self.input_model(pred_model).input_metric(metric)
        return self

    def train(
        self,
        fetcher: DataFetcher,
        pred_model: Optional[Model] = None,
        metric: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
        *args,
        **kwargs,
    ):
        """Store and transform data, then train model to predict data values.

        Trains the Data Evaluator and the underlying prediction model. Wrapper for
        ``self.input_data`` and ``self.train_data_values`` under one method.

        Parameters
        ----------
        fetcher : DataFetcher
            DataFetcher containing the training and validation data set.
        pred_model : Model, optional
            Prediction model, not required if the DataFetcher is Model less
        metric : Callable[[torch.Tensor, torch.Tensor], float]
            Evaluation function to determine prediction model performance,
            by default None and assigns either -MSE or ACC depending if categorical
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Returns
        -------
        self : object
            Returns a Data Evaluator.
        """
        self.setup(fetcher, pred_model, metric)
        self.train_data_values(*args, **kwargs)  

        return self

    @abstractmethod
    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Returns
        -------
        self : object
            Returns a trained Data Evaluator.
        """
        return self

    @abstractmethod
    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for training input data point
        """

    @cached_property
    def data_values(self) -> np.ndarray:
        """Cached data values."""
        return self.evaluate_data_values()
    
    @abstractmethod
    def data_state_trajectory(self):
        """Return data state trajectories for each training data point.

        Returns
        -------
        np.ndarray
            Predicted data state trajectories for training input data point
        """
        
    @abstractmethod
    def data_costate_trajectory(self):
        """Return data state trajectories for each training data point.

        Returns
        -------
        np.ndarray
            Predicted data state trajectories for training input data point
        """
    
    @cached_property
    def ext_train_meta_loss(self):
        return self.save_train_meta_loss()
    
    @abstractmethod
    def save_train_meta_loss(self):
        """Return train and meta loss for each training process.

        Returns
        -------
        np.ndarray
            Extracted train and meta loss for training process
        """
        
    @cached_property
    def ext_data_loss(self):
        return self.save_data_loss()
    
    @abstractmethod
    def save_data_loss(self):
        """Return train and meta loss for each training process.

        Returns
        -------
        np.ndarray
            Extracted train and meta loss for training process
        """
        
    @cached_property
    def ext_x_y_state(self):
        return self.save_x_y_state()
    
    @abstractmethod
    def save_x_y_state(self):
        """Return train and meta loss for each training process.

        Returns
        -------
        np.ndarray
            Extracted train and meta loss for training process
        """
        
    @cached_property
    def data_state_traject(self):
        return self.data_state_trajectory()
    
    @cached_property
    def data_costate_traject(self):
        return self.data_costate_trajectory()
    
    @abstractmethod
    def data_values_trajectory(self):
        """Return data state trajectories for each training data point.

        Returns
        -------
        np.ndarray
            Predicted data state trajectories for training input data point
        """
    
    @cached_property
    def data_values_traject(self):
        return self.data_values_trajectory()

    def input_fetcher(self, fetcher: DataFetcher):
        """Input data from a DataFetcher object. Alternative way of adding data."""
        x_train, y_train, x_valid, y_valid, x_test, y_test = fetcher.datapoints
        x_meta, y_meta = fetcher.metadata
        return self.input_data(x_train, y_train, x_valid, y_valid, x_test, y_test, x_meta, y_meta)


class ModelMixin:
    def evaluate(self, y: torch.Tensor, y_hat: torch.Tensor):
        """Evaluate performance of the specified metric between label and predictions.

        Moves input tensors to cpu because of certain bugs/errors that arise when the
        tensors are not on the same device

        Parameters
        ----------
        y : torch.Tensor
            Labels to be evaluate performance of predictions
        y_hat : torch.Tensor
            Predictions of labels

        Returns
        -------
        float
            Performance metric
        """
        return self.metric(y.cpu(), y_hat.cpu())

    def input_model(self, pred_model: Model):
        """Input the prediction model.

        Parameters
        ----------
        pred_model : Model
            Prediction model
        """
        self.pred_model = pred_model.clone()
        return self

    def input_metric(self, metric: Callable[[torch.Tensor, torch.Tensor], float]):
        """Input the evaluation metric.

        Parameters
        ----------
        metric : Callable[[torch.Tensor, torch.Tensor], float]
            Evaluation function to determine prediction model performance
        """
        self.metric = metric
        return self

    def input_model_metric(
        self, pred_model: Model, metric: Callable[[torch.Tensor, torch.Tensor], float]
    ):
        """Input the prediction model and the evaluation metric.

        Parameters
        ----------
        pred_model : Model
            Prediction model
        metric : Callable[[torch.Tensor, torch.Tensor], float]
            Evaluation function to determine prediction model performance

        Returns
        -------
        self : object
            Returns a Data Evaluator.
        """
        return self.input_model(pred_model).input_metric(metric)


class ModelLessMixin:
    """Mixin for DataEvaluators without a prediction model and use embeddings.

    Using embeddings and then predictiong the data values has been used by
    Ruoxi Jia Group with their KNN Shapley and LAVA data evaluators.

    References
    ----------
    .. [1] R. Jia et al.,
        Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1908.08619.

    Attributes
    ----------
    embedding_model : Model
        Embedding model used by model-less DataEvaluator to compute the data values for
        the embeddings and not the raw input.
    pred_model : Model
        The pred_model is unused for training, but to compare a series of models on
        the same algorithim, we compare against a shared prediction algorithim.
    """

    def embeddings(
        self, *tensors: tuple[Union[Dataset, torch.Tensor], ...]
    ) -> tuple[torch.Tensor, ...]:
        """Returns Embeddings for the input tensors

        Returns
        -------
        tuple[torch.Tensor, ...]
            Returns tupple of tensors equal to the number of tensors input
        """
        if hasattr(self, "embedding_model") and self.embedding_model is not None:
            return tuple(self.embedding_model.predict(tensor) for tensor in tensors)

        # No embedding is used
        return tensors
