import importlib

import pytest

import towbintools


def test_version_is_resolved_from_metadata():
    assert towbintools.__version__ != "unknown"


@pytest.mark.parametrize(
    "module_name",
    [
        "towbintools.segmentation",
        "towbintools.straightening",
        "towbintools.quantification",
        "towbintools.classification",
        "towbintools.data_analysis",
        "towbintools.deep_learning",
        "towbintools.deep_learning.architectures",
    ],
)
def test_every_name_in_dunder_all_is_importable(module_name):
    module = importlib.import_module(module_name)
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert missing == []
