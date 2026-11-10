"""Game adapters for the general harness."""

from .astray import AstrayAdapter
from .gzdoom import GzdoomAdapter
from .html5 import HtmlGameAdapter

__all__ = ["AstrayAdapter", "GzdoomAdapter", "HtmlGameAdapter"]
