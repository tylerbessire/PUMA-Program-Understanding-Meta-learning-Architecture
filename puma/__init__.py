"""PUMA meta-learning components."""
from importlib import import_module

__all__ = ["rft"]


def __getattr__(name):
    if name == "rft":
        return import_module("puma.rft")
    raise AttributeError(name)
