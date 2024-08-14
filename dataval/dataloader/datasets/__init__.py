"""Data sets registered with :py:class:`~dataval.dataloader.register.Register`.

Data sets
=========
.. autosummary::
    :toctree: generated/

    datasets
    imagesets
    nlpsets

Catalog of registered data sets that can be used with
:py:class:`~dataval.dataloader.fetcher.DataFetcher`. Pass in the ``str`` name
registering the data set to load the data set as needed.
.
"""
from dataval.dataloader.datasets import (
    challenge,
    datasets,
    imagesets,
    nlpsets,
)
