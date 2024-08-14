"""Create :py:class:`~dataval.dataval.DataEvaluator` to quantify the value of data.

Data Evaluator
==============

.. currentmodule:: dataval.dataval

Provides an ABC for DataEvaluator to inherit from. The work flow is as follows:
:py:class:`~dataval.dataloader.Register`,
:py:class:`~dataval.dataloader.DataFetcher`
-> :py:class:`~dataval.dataval.DataEvaluator`
-> :py:mod:`~dataval.experiment.exper_methods`



Catalog
-------
.. autosummary::
    :toctree: generated/

    DataEvaluator
    ModelMixin
    ModelLessMixin
    AME
    DVRL
    InfluenceFunction
    InfluenceSubsample
    KNNShapley
    DataOob
    DataBanzhaf
    BetaShapley
    DataShapley
    LavaEvaluator
    LeaveOneOut
    ShapEvaluator
    RandomEvaluator
    RobustVolumeShapley
    Sampler
    TMCSampler
    GrTMCSampler
"""
from dataval.datavaluation.ame import AME
from dataval.datavaluation.api import DataEvaluator, ModelLessMixin, ModelMixin
from dataval.datavaluation.csshap import ClassWiseShapley
from dataval.datavaluation.dvrl import DVRL
from dataval.datavaluation.influence import InfluenceFunction, InfluenceSubsample
from dataval.datavaluation.knnshap import KNNShapley
from dataval.datavaluation.lava import LavaEvaluator
from dataval.datavaluation.margcontrib import (
    BetaShapley,
    DataBanzhaf,
    DataBanzhafMargContrib,
    DataShapley,
    GrTMCSampler,
    LeaveOneOut,
    Sampler,
    ShapEvaluator,
    TMCSampler,
)
from dataval.datavaluation.oob import DataOob
from dataval.datavaluation.random import RandomEvaluator
from dataval.datavaluation.volume import RobustVolumeShapley
from dataval.datavaluation.ndsv import NDSV
from dataval.datavaluation.nddv import NDDV
